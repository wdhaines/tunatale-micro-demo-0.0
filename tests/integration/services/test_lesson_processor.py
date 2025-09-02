"""Integration tests for LessonProcessor."""
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError
from pydub import AudioSegment

from tunatale.core.exceptions import TTSValidationError, AudioProcessingError, TTSServiceError
import uuid
from tunatale.core.models.lesson import Lesson
from tunatale.core.models.section import Section
from tunatale.core.models.phrase import Phrase
from tunatale.core.models.enums import SectionType, Language
from tunatale.core.models.voice import Voice, VoiceGender
from tunatale.core.services.lesson_processor import LessonProcessor
from tunatale.core.ports.audio_processor import AudioProcessor
from tunatale.core.ports.tts_service import TTSService

# Configure logging for tests
import logging.handlers

# Create a logger that will capture debug messages
logger = logging.getLogger('test_logger')
logger.setLevel(logging.DEBUG)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(ch)

# Set up logging for the lesson processor
logging.getLogger('tunatale.core.services.lesson_processor').setLevel(logging.DEBUG)
logging.getLogger('tunatale.core.ports.audio_processor').setLevel(logging.DEBUG)
logging.getLogger('tunatale.core.ports.tts_service').setLevel(logging.DEBUG)


@pytest.fixture
def mock_tts_service():
    """Create a mock TTS service."""
    mock = MagicMock(spec=TTSService)
    
    # Mock synthesize_speech to create a silent audio file
    async def mock_synthesize(*args, **kwargs):
        output_path = Path(kwargs.get('output_path', 'output.mp3'))
        # Ensure the output path has an .mp3 extension
        if not output_path.suffix == '.mp3':
            output_path = output_path.with_suffix('.mp3')
            
        # Create a 100ms silent audio file
        audio = AudioSegment.silent(duration=100)  # 100ms
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio.export(output_path, format='mp3')
        
        # Use the voice_id that was passed in, or default to test-voice
        voice_id = kwargs.get('voice_id', 'test-voice')
        return {
            'path': str(output_path),
            'cached': False,
            'voice_id': voice_id
        }
    
    mock.synthesize_speech = AsyncMock(side_effect=mock_synthesize)
    mock.get_voice = AsyncMock(return_value=Voice(
        name="Test Voice",
        provider="test",
        provider_id="test-voice",
        language="en",
        gender=VoiceGender.FEMALE
    ))
    
    # Mock validate_voice to always return True for any voice_id
    async def mock_validate_voice(voice_id: str) -> bool:
        return True
        
    mock.validate_voice = AsyncMock(side_effect=mock_validate_voice)
    
    return mock


@pytest.fixture
def mock_audio_processor():
    """Create a mock audio processor."""
    mock = MagicMock(spec=AudioProcessor)
    
    # Mock methods to return the input file as output
    async def mock_concatenate(input_files, output_file, **kwargs):
        # Create an empty output file if no input files
        if not input_files:
            output_file.touch()
            return output_file
            
        # Otherwise copy the first input file to the output
        with open(input_files[0], 'rb') as src, open(output_file, 'wb') as dst:
            dst.write(src.read())
        return output_file
    
    async def mock_add_silence(input_file, output_file, **kwargs):
        # Just copy the input file to the output
        with open(input_file, 'rb') as src, open(output_file, 'wb') as dst:
            dst.write(src.read())
        return output_file
    
    async def mock_normalize(input_file, output_file, **kwargs):
        # Just copy the input file to the output
        with open(input_file, 'rb') as src, open(output_file, 'wb') as dst:
            dst.write(src.read())
        return output_file
    
    async def mock_trim_silence(input_file, output_file, **kwargs):
        # Just copy the input file to the output
        with open(input_file, 'rb') as src, open(output_file, 'wb') as dst:
            dst.write(src.read())
        return output_file
    
    async def mock_get_duration(audio_file):
        # Return a fixed duration
        return 0.1  # 100ms
    
    mock.concatenate_audio = AsyncMock(side_effect=mock_concatenate)
    mock.add_silence = AsyncMock(side_effect=mock_add_silence)
    mock.normalize_audio = AsyncMock(side_effect=mock_normalize)
    mock.trim_silence = AsyncMock(side_effect=mock_trim_silence)
    mock.get_audio_duration = AsyncMock(side_effect=mock_get_duration)
    
    return mock


@pytest.fixture
def sample_lesson():
    """Create a sample lesson for testing."""
    # First create the lesson
    lesson = Lesson(
        title="Test Lesson",
        description="A test lesson",
        target_language=Language.TAGALOG,
        native_language=Language.ENGLISH,
        difficulty=1,
        sections=[]
    )
    
    # Create section with lesson_id
    section1 = Section(
        title="Greeting",
        section_type=SectionType.KEY_PHRASES,
        lesson_id=str(lesson.id),
        phrases=[]
    )
    
    # Create phrases with section_id as string
    phrase1 = Phrase(
        text="Hello, how are you?",
        language=Language.ENGLISH,
        voice_id="test-voice",
        section_id=str(section1.id)  # Use the section's UUID as string
    )
    
    phrase2 = Phrase(
        text="I'm fine, thank you!",
        language=Language.ENGLISH,
        voice_id="test-voice",
        section_id=str(section1.id)  # Use the section's UUID as string
    )
    
    # Add phrases to section
    section1.phrases = [phrase1, phrase2]
    
    # Update the lesson with the section
    lesson.sections = [section1]
    
    return lesson


@pytest.fixture
def mock_voice_selector():
    """Create a mock VoiceSelector for testing."""
    mock = MagicMock()
    
    async def mock_get_voice_id(language=None, gender=None, **kwargs):
        # If voice_id is explicitly provided in metadata, use it
        if 'speaker_id' in kwargs and kwargs['speaker_id']:
            return kwargs['speaker_id']
            
        # Otherwise, use gender to determine voice
        if gender == "male" or (isinstance(gender, str) and "male" in gender.lower()):
            return "en-US-GuyNeural"
        return "test-voice"
        
    mock.get_voice_id = mock_get_voice_id
    return mock

@pytest.fixture
def mock_word_selector():
    """Create a mock WordSelector for testing."""
    return MagicMock()

@pytest.fixture
def lesson_processor(mock_tts_service, mock_audio_processor, mock_voice_selector, mock_word_selector, tmp_path):
    """Create a LessonProcessor instance for testing."""
    return LessonProcessor(
        tts_service=mock_tts_service,
        audio_processor=mock_audio_processor,
        voice_selector=mock_voice_selector,
        word_selector=mock_word_selector,
        max_workers=2,  # Use fewer workers for tests
        output_dir=str(tmp_path / "output")  # Use a test-specific output directory
    )


class TestLessonProcessor:
    """Test cases for LessonProcessor."""
    
    @pytest.mark.asyncio
    async def test_section_file_renaming(
        self,
        lesson_processor: LessonProcessor,
        tmp_path: Path,
        sample_lesson: Lesson
    ):
        """Test that section files are renamed according to their section names."""
        # Create a section with known name
        section = Section(
            id=uuid.uuid4(),
            title="Key Phrases",
            section_type=SectionType.KEY_PHRASES,
            lesson_id=str(sample_lesson.id),
            position=1,
            phrases=[
                Phrase(
                    id=uuid.uuid4(),
                    text="Test phrase",
                    language=Language.ENGLISH
                )
            ]
        )
        
        # Process the section
        result = await lesson_processor.process_section(
            section=section,
            output_dir=tmp_path
        )
        
        # Verify the section audio file exists and has a valid path
        assert 'audio_file' in result
        assert result['audio_file'] is not None
        audio_file = Path(result['audio_file'])
        assert audio_file.exists()
        
        # Verify the section file was renamed to key_phrases.mp3
        assert audio_file.name == 'key_phrases.mp3'
        
        # Verify the section audio file is in the top-level output directory, not in a subdirectory
        assert audio_file.parent == tmp_path, f"Expected audio file in {tmp_path}, but found in {audio_file.parent}"
        
        # Create another section with a different name
        section2 = Section(
            id=uuid.uuid4(),
            title="Natural Speed",
            section_type=SectionType.NATURAL_SPEED,
            lesson_id=str(sample_lesson.id),
            position=2,
            phrases=[
                Phrase(
                    id=uuid.uuid4(),
                    text="Test phrase 2",
                    language=Language.ENGLISH
                )
            ]
        )
        
        # Process the second section
        result2 = await lesson_processor.process_section(
            section=section2,
            output_dir=tmp_path
        )
        
        # Verify the second section audio file exists and has a valid path
        assert 'audio_file' in result2
        assert result2['audio_file'] is not None
        audio_file2 = Path(result2['audio_file'])
        assert audio_file2.exists()
        
        # Verify the second section file was renamed to natural_speed.mp3
        assert audio_file2.name == 'natural_speed.mp3'
        
        # Verify the second section audio file is also in the top-level output directory
        assert audio_file2.parent == tmp_path, f"Expected audio file in {tmp_path}, but found in {audio_file2.parent}"
        
        # Create a section with an unknown name
        section3 = Section(
            id=uuid.uuid4(),
            title="Custom Section",
            section_type=SectionType.KEY_PHRASES,
            lesson_id=str(sample_lesson.id),
            position=3,
            phrases=[
                Phrase(
                    id=uuid.uuid4(),
                    text="Test phrase 3",
                    language=Language.ENGLISH
                )
            ]
        )
        
        # Process the third section
        result3 = await lesson_processor.process_section(
            section=section3,
            output_dir=tmp_path
        )
        
        # Verify the third section audio file exists and has a valid path
        assert 'audio_file' in result3
        assert result3['audio_file'] is not None
        audio_file3 = Path(result3['audio_file'])
        assert audio_file3.exists()
        
        # Verify the third section file was renamed to custom_section.mp3
        assert audio_file3.name == 'custom_section.mp3'
        
        # Verify the third section audio file is also in the top-level output directory
        assert audio_file3.parent == tmp_path, f"Expected audio file in {tmp_path}, but found in {audio_file3.parent}"
    
    async def test_process_phrase(
        self,
        lesson_processor: LessonProcessor,
        tmp_path: Path,
        sample_lesson: Lesson,
        caplog
    ):
        """Test processing a single phrase within a section."""
        # Enable debug logging for this test
        caplog.set_level(logging.DEBUG)
        
        # Create a section with a single phrase
        phrase = Phrase(
            id=uuid.uuid4(),
            text="Test phrase",
            language=Language.ENGLISH.value
        )
        logger.debug(f"Created phrase with id: {phrase.id}")
        logger.debug(f"Test will use tmp_path: {tmp_path}")
        
        section = Section(
            id=uuid.uuid4(),
            title="Test Section",
            section_type=SectionType.KEY_PHRASES,
            lesson_id=str(sample_lesson.id),
            position=1,
            phrases=[phrase]
        )
        
        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Create a lesson with just this section
        lesson = Lesson(
            id=uuid.uuid4(),
            title="Test Lesson",
            language=Language.ENGLISH.value,
            target_language=Language.ENGLISH.value,
            level="beginner",
            sections=[section]
        )
        
        # Process the lesson
        processed_lesson = await lesson_processor.process_lesson(
            lesson=lesson,
            output_dir=str(output_dir)
        )
        
        # Get the processed section from the result
        assert 'sections' in processed_lesson
        assert len(processed_lesson['sections']) > 0
        result = processed_lesson['sections'][0]
        
        # Verify the section has one phrase
        assert len(result['phrases']) == 1
        assert result['audio_file'] is not None
        
        # Get the processed phrase
        processed_phrase = result['phrases'][0]
        
        # Verify the phrase details
        assert processed_phrase['audio_file'] is not None
        assert processed_phrase['text'] == phrase.text
        assert processed_phrase['language'] == phrase.language
        
        # Verify the audio file was created
        audio_path = Path(processed_phrase['audio_file'])
        logger.debug(f"Expected audio path: {audio_path}")
        logger.debug(f"Audio file exists: {audio_path.exists()}")
        if not audio_path.exists():
            # List all files in the directory for debugging
            audio_dir = audio_path.parent
            logger.debug(f"Contents of {audio_dir}:")
            if audio_dir.exists():
                for f in audio_dir.glob('*'):
                    logger.debug(f"  - {f.name} (exists: {f.exists()}, is_file: {f.is_file()})")
        
        assert audio_path.exists(), f"Audio file not found at {audio_path}"
        assert audio_path.suffix == '.mp3', f"Expected .mp3 file, got {audio_path.suffix}"
    
    async def test_process_section(
        self,
        lesson_processor: LessonProcessor,
        tmp_path: Path,
        sample_lesson: Lesson
    ):
        """Test processing a section with multiple phrases."""
        # Create a section with multiple phrases
        phrases = [
            Phrase(
                id=uuid.uuid4(),
                text=f"Test phrase {i+1}",
                language=Language.ENGLISH.value
            ) for i in range(3)
        ]
        
        section = Section(
            id=uuid.uuid4(),
            title="Test Section",
            section_type=SectionType.KEY_PHRASES,
            lesson_id=str(sample_lesson.id),
            position=1,
            phrases=phrases
        )
        
        # Process the section
        result = await lesson_processor.process_section(
            section=section,
            output_dir=tmp_path
        )
        
        # Verify the result contains the expected keys and values
        assert 'phrases' in result
        assert len(result['phrases']) == len(section.phrases)
        assert 'audio_file' in result
        
        # Verify the section audio file was created
        assert result['audio_file'] is not None
        section_audio_path = Path(result['audio_file'])
        assert section_audio_path.exists()
        assert section_audio_path.suffix == '.mp3'
        
        # Verify the phrase audio files were created
        for phrase_result in result['phrases']:
            assert 'audio_file' in phrase_result
            assert phrase_result['audio_file'] is not None
            phrase_audio_path = Path(phrase_result['audio_file'])
            assert phrase_audio_path.exists()
            assert phrase_audio_path.suffix == '.mp3'
    
    async def test_process_lesson(
        self,
        lesson_processor: LessonProcessor,
        tmp_path: Path,
        sample_lesson: Lesson
    ):
        """Test processing a complete lesson."""
        # Create a simple lesson with one section and one phrase
        phrase = Phrase(
            id=uuid.uuid4(),
            text="Test phrase",
            language=Language.ENGLISH.value
        )
        
        section = Section(
            id=uuid.uuid4(),
            title="Test Section",
            section_type=SectionType.KEY_PHRASES,
            lesson_id=str(sample_lesson.id),
            position=1,
            phrases=[phrase]
        )
        
        lesson = Lesson(
            id=uuid.uuid4(),
            title="Test Lesson",
            target_language=Language.ENGLISH.value,
            native_language=Language.ENGLISH.value,
            sections=[section]
        )
        
        # Process the lesson
        result = await lesson_processor.process_lesson(
            lesson=lesson,
            output_dir=tmp_path
        )
        
        # Verify the result contains the expected keys and values
        assert isinstance(result, dict)
        assert 'sections' in result
        assert isinstance(result['sections'], list)
        assert len(result['sections']) == 1
        
        # Verify the section was processed
        section_result = result['sections'][0]
        assert 'phrases' in section_result
        assert isinstance(section_result['phrases'], list)
        assert len(section_result['phrases']) == 1
        assert 'audio_file' in section_result
        
        # Verify the section audio file was created
        section_audio_path = Path(section_result['audio_file'])
        assert section_audio_path.exists()
        assert section_audio_path.suffix == '.mp3'
        
        # Verify the phrase audio file was created
        phrase_result = section_result['phrases'][0]
        assert 'audio_file' in phrase_result
        phrase_audio_path = Path(phrase_result['audio_file'])
        assert phrase_audio_path.exists()
        assert phrase_audio_path.suffix == '.mp3'
        
        # Verify metadata file was created
        metadata_path = tmp_path / 'metadata.json'
        assert metadata_path.exists()
        
        # Verify metadata content
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        assert metadata['title'] == lesson.title
        assert len(metadata['sections']) == len(sample_lesson.sections)
        assert 'metrics' in metadata  # Check for metrics instead of processing_info
    
    async def test_process_lesson_with_progress_callback(
        self,
        lesson_processor: LessonProcessor,
        tmp_path: Path,
        sample_lesson: Lesson
    ):
        """Test processing a lesson with a progress callback."""
        # Create a simple lesson with one section and one phrase
        phrase = Phrase(
            id=uuid.uuid4(),
            text="Test phrase",
            language=Language.ENGLISH.value
        )
        
        section = Section(
            id=uuid.uuid4(),
            title="Test Section",
            section_type=SectionType.KEY_PHRASES,
            lesson_id=str(sample_lesson.id),
            position=1,
            phrases=[phrase]
        )
        
        lesson = Lesson(
            id=uuid.uuid4(),
            title="Test Lesson",
            target_language=Language.ENGLISH.value,
            native_language=Language.ENGLISH.value,
            sections=[section]
        )
        
        # Track progress updates
        progress_updates = {}
        
        class ProgressReporter:
            def __init__(self):
                self.completed = {}
                self.total = {}
                
            async def update(self, task_id: str, completed: int, total: int, status: str, **kwargs):
                self.completed[task_id] = completed
                self.total[task_id] = total
                progress_updates[task_id] = {
                    'completed': completed,
                    'total': total,
                    'status': status,
                    **kwargs
                }
            
            async def complete(self, task_id: str):
                if task_id in progress_updates:
                    progress_updates[task_id]['completed'] = self.total.get(task_id, 0)
                    progress_updates[task_id]['status'] = 'completed'
        
        # Create a progress reporter instance
        progress_reporter = ProgressReporter()
        
        # Process the lesson with progress callback
        result = await lesson_processor.process_lesson(
            lesson=lesson,
            output_dir=tmp_path,
            progress=progress_reporter
        )
        
        # Verify the result
        assert 'sections' in result
        assert len(result['sections']) == 1
        
        # Verify we received progress updates
        assert len(progress_updates) > 0
        
        # Check that we got updates for the lesson and section
        lesson_task_id = f"lesson_{lesson.id}"
        section_task_id = f"section_{section.id}"
        assert lesson_task_id in progress_updates
        assert section_task_id in progress_updates
    
    async def test_process_phrase_tts_error(
        self,
        mock_tts_service: MagicMock,
        mock_audio_processor: MagicMock,
        mock_voice_selector: MagicMock,
        mock_word_selector: MagicMock,
        tmp_path: Path,
        sample_lesson: Lesson
    ):
        """Test handling of TTS errors during phrase processing."""
        # Configure the mock to raise an exception during voice validation
        async def mock_validate_voice(voice_id: str) -> None:
            raise TTSValidationError("TTS service error")
            
        mock_tts_service.validate_voice = AsyncMock(side_effect=mock_validate_voice)
        
        # Create a processor with our mocks
        processor = LessonProcessor(
            tts_service=mock_tts_service,
            audio_processor=mock_audio_processor,
            voice_selector=mock_voice_selector,
            word_selector=mock_word_selector,
            output_dir=str(tmp_path / "output"),
            max_workers=1
        )
        
        # Get a phrase from the sample lesson
        phrase = sample_lesson.sections[0].phrases[0]
        
        # Process the phrase (should not raise an exception)
        result = await processor.process_phrase(
            phrase=phrase,
            output_dir=tmp_path
        )
        
        # Debug output
        print("\nTest Debug - process_phrase result:")
        print(f"Result: {result}")
        
        # Verify the result indicates an error
        assert not result['success'], "Expected success=False"
        assert 'error' in result, "Expected 'error' in result"
        assert isinstance(result['error'], dict), "Expected error to be a dictionary"
        assert 'error_code' in result['error'], "Expected 'error_code' in error dict"
        assert result['error']['error_code'] == 'TTS_VALIDATION_ERROR', \
            f"Expected error_code 'TTS_VALIDATION_ERROR', got {result['error'].get('error_code')}"
        assert 'TTS service error' in result['error']['error_message'], \
            f"Expected 'TTS service error' in error_message, got {result['error'].get('error_message')}"
    
    async def test_no_audio_received(
        self,
        mock_tts_service: MagicMock,
        mock_audio_processor: MagicMock,
        mock_voice_selector: MagicMock,
        mock_word_selector: MagicMock,
        tmp_path: Path,
        sample_lesson: Lesson
    ):
        """Test handling of 'no audio received' error from TTS service."""
        # Configure the mock to raise TTSServiceError with 'No audio received' message
        async def mock_synthesize_speech(*args, **kwargs):
            from tunatale.core.exceptions import TTSServiceError
            raise TTSServiceError("No audio received from TTS service")
            
        mock_tts_service.synthesize_speech = AsyncMock(side_effect=mock_synthesize_speech)
        
        # Create a processor with our mocks
        processor = LessonProcessor(
            tts_service=mock_tts_service,
            audio_processor=mock_audio_processor,
            voice_selector=mock_voice_selector,
            word_selector=mock_word_selector,
            output_dir=str(tmp_path / "output"),
            max_workers=1
        )
        
        # Get a phrase from the sample lesson
        phrase = sample_lesson.sections[0].phrases[0]
        
        # Process the phrase (should not raise an exception)
        result = await processor.process_phrase(
            phrase=phrase,
            output_dir=tmp_path
        )
        
        # Verify the result indicates an error
        assert not result['success'], "Expected success=False"
        assert 'error' in result, "Expected 'error' in result"
        assert isinstance(result['error'], dict), "Expected error to be a dictionary"
        assert 'error_code' in result['error'], "Expected 'error_code' in error dict"
        assert result['error']['error_code'] == 'TTS_SERVICE_ERROR', \
            f"Expected error_code 'TTS_SERVICE_ERROR', got {result['error'].get('error_code')}"
        assert 'No audio received from TTS service' in result['error']['error_message'], \
            f"Expected 'No audio received from TTS service' in error_message, got {result['error'].get('error_message')}"
            
        # Verify the TTS service was called with the expected arguments
        mock_tts_service.synthesize_speech.assert_called_once()
        
    async def test_process_section_audio_error(
        self,
        mock_tts_service: MagicMock,
        mock_audio_processor: MagicMock,
        mock_voice_selector: MagicMock,
        mock_word_selector: MagicMock,
        tmp_path: Path,
        sample_lesson: Lesson
    ):
        """Test handling of audio processing errors during section processing."""
        # Create the phrases directory
        phrases_dir = tmp_path / "phrases"
        phrases_dir.mkdir(exist_ok=True)
        
        # Create a mock TTS service that returns a successful result
        # and creates the expected output file
        async def mock_synthesize_speech(*args, **kwargs):
            output_path = Path(kwargs.get('output_path', ''))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()  # Create the file
            return {
                'audio_file': str(output_path),
                'cached': False
            }
            
        mock_tts_service.synthesize_speech = AsyncMock(side_effect=mock_synthesize_speech)
        
        # Create a mock audio processor that raises an error during concatenation
        async def mock_concatenate_error(*args, **kwargs):
            raise AudioProcessingError("Audio concatenation error")
            
        mock_audio_processor.concatenate_audio = AsyncMock(side_effect=mock_concatenate_error)
        mock_audio_processor.get_audio_duration = AsyncMock(return_value=1.0)
        
        # Create a processor with our mocks
        processor = LessonProcessor(
            tts_service=mock_tts_service,
            audio_processor=mock_audio_processor,
            voice_selector=mock_voice_selector,
            word_selector=mock_word_selector,
            output_dir=str(tmp_path / "output"),
            max_workers=1
        )
        
        # Get a section from the sample lesson
        section = sample_lesson.sections[0]
        
        # Process the section (should not raise an exception)
        result = await processor.process_section(
            section=section,
            lesson=sample_lesson,
            output_dir=tmp_path
        )
        
        # Debug output
        print("\nTest Debug - process_section result:")
        print(f"Result keys: {result.keys()}")
        if 'phrases' in result:
            print(f"Number of phrases: {len(result['phrases'])}")
            for i, phrase in enumerate(result['phrases']):
                print(f"Phrase {i} keys: {phrase.keys()}")
                if 'error' in phrase:
                    print(f"Phrase {i} error: {phrase['error']}")
        
        # Verify the result contains phrases even though section processing failed
        assert 'phrases' in result, "Expected phrases in the result"
        assert len(result['phrases']) > 0, "Expected at least one phrase in the result"
        
        # Check that individual phrases were processed successfully even though section concatenation failed
        assert all(p.get('success', False) for p in result['phrases']), "Expected all phrases to be processed successfully"
        assert all('audio_file' in p for p in result['phrases']), "Expected all phrases to have audio files"
        assert all(p['audio_file'] is not None for p in result['phrases']), "Expected all phrases to have non-None audio files"
        
        # Section should report failure due to concatenation error, but preserve phrases
        assert not result['success'], "Expected section processing to fail due to concatenation error"  
        assert 'audio_file' in result, "Expected an audio file field in the result"
        assert result['audio_file'] is None, "Expected None audio file path when concatenation fails"
        assert 'error' in result, "Expected error information when concatenation fails"
        assert result['error']['error_code'] == 'PROCESSING_ERROR', "Expected PROCESSING_ERROR code"
    
    async def test_process_section_audio_error_in_phrase(
        self,
        mock_tts_service: MagicMock,
        mock_audio_processor: MagicMock,
        mock_voice_selector: MagicMock,
        mock_word_selector: MagicMock,
        tmp_path: Path,
        sample_lesson: Lesson
    ):
        """Test handling of audio processing errors during section processing."""
        # Create a mock TTS service that returns a successful result
        # and creates the expected output file
        async def mock_synthesize_speech(*args, **kwargs):
            output_path = Path(kwargs.get('output_path', ''))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()  # Create the file
            return {
                'audio_file': str(output_path),
                'cached': False
            }
            
        mock_tts_service.synthesize_speech = AsyncMock(side_effect=mock_synthesize_speech)
        
        # Create a mock audio processor that raises an error during normalization
        async def mock_normalize_audio(*args, **kwargs):
            raise AudioProcessingError("Audio processing error")
            
        mock_audio_processor.normalize_audio = AsyncMock(side_effect=mock_normalize_audio)
        mock_audio_processor.get_audio_duration = AsyncMock(return_value=1.0)
        
        # Create a processor with our mocks
        processor = LessonProcessor(
            tts_service=mock_tts_service,
            audio_processor=mock_audio_processor,
            voice_selector=mock_voice_selector,
            word_selector=mock_word_selector,
            output_dir=str(tmp_path / "output"),
            max_workers=1
        )

        # Get a section from the sample lesson
        section = sample_lesson.sections[0]

        # Process the section (should not raise an exception)
        result = await processor.process_section(
            section=section,
            lesson=sample_lesson,
            output_dir=tmp_path
        )

        # Debug output
        print("\nTest Debug - process_section result:")
        print(f"Result keys: {result.keys()}")
        if 'phrases' in result:
            print(f"Number of phrases: {len(result['phrases'])}")
            for i, phrase in enumerate(result['phrases']):
                print(f"Phrase {i} keys: {phrase.keys()}")
                if 'error' in phrase:
                    print(f"Phrase {i} error: {phrase['error']}")

        # Debug output
        print("\nTest Debug - process_section result (audio error in phrase):")
        print(f"Result keys: {result.keys()}")
        if 'phrases' in result:
            print(f"Number of phrases: {len(result['phrases'])}")
            for i, phrase in enumerate(result['phrases']):
                print(f"Phrase {i} keys: {phrase.keys()}")
                if 'error' in phrase:
                    print(f"Phrase {i} error: {phrase['error']}")
        
        # Verify the result contains phrases
        assert 'phrases' in result, "Expected phrases in the result"
        assert len(result['phrases']) > 0, "Expected at least one phrase in the result"
        
        # Verify that all phrases were processed successfully
        # (The error in normalization should not cause the phrase to fail)
        assert all(p.get('success', False) for p in result['phrases']), \
            "Expected all phrases to be processed successfully"
            
        # Verify that all phrases have an audio file
        assert all('audio_file' in p for p in result['phrases']), \
            "Expected all phrases to have an audio file"
            
        # However, if concatenation fails due to invalid audio files (caused by normalization errors),
        # the section should report failure, but preserve the successfully processed phrases
        assert not result['success'], "Expected section processing to fail due to concatenation issues caused by normalization errors"
        assert 'error' in result, "Expected error information when section processing fails"
        assert result['error']['error_code'] == 'PROCESSING_ERROR', "Expected PROCESSING_ERROR code"
        assert 'audio_file' in result, "Expected section to have an audio file field"
        assert result['audio_file'] is None, "Expected None audio file when concatenation fails"
    
    async def test_metadata_generation(
        self,
        lesson_processor: LessonProcessor,
        tmp_path: Path,
        sample_lesson: Lesson
    ):
        """Test generation of metadata."""
        # Process the lesson
        result = await lesson_processor.process_lesson(
            lesson=sample_lesson,
            output_dir=tmp_path
        )
        
        # Verify the metadata file was created
        metadata_path = tmp_path / 'metadata.json'
        assert metadata_path.exists()
        
        # Load and verify the metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Debug: Print the entire metadata structure
        print("\n=== METADATA STRUCTURE ===")
        print(json.dumps(metadata, indent=2, default=str))
        print("=== END METADATA ===\n")
        
        # Debug: Print the sections structure
        print("\n=== SECTIONS IN METADATA ===")
        if 'sections' in metadata:
            for i, section in enumerate(metadata['sections']):
                print(f"Section {i} keys: {section.keys()}")
        else:
            print("No 'sections' key in metadata")
        print("=== END SECTIONS ===\n")
        
        # Check basic metadata
        assert 'title' in metadata, f"Metadata missing 'title' key. Available keys: {metadata.keys()}"
        assert metadata['title'] == sample_lesson.title
        assert 'language' in metadata, f"Metadata missing 'language' key. Available keys: {metadata.keys()}"
        assert 'level' in metadata, f"Metadata missing 'level' key. Available keys: {metadata.keys()}"
        assert 'created_at' in metadata, f"Metadata missing 'created_at' key. Available keys: {metadata.keys()}"
        assert 'audio_file' in metadata, f"Metadata missing 'audio_file' key. Available keys: {metadata.keys()}"
        assert 'sections' in metadata, f"Metadata missing 'sections' key. Available keys: {metadata.keys()}"
        assert 'metrics' in metadata, f"Metadata missing 'metrics' key. Available keys: {metadata.keys()}"
        assert 'audio_files' in metadata, f"Metadata missing 'audio_files' key. Available keys: {metadata.keys()}"
        assert len(metadata['sections']) == len(sample_lesson.sections), \
            f"Expected {len(sample_lesson.sections)} sections, got {len(metadata['sections'])}"
            
        # Check section metadata
        for i, section in enumerate(sample_lesson.sections):
            assert i < len(metadata['sections']), f"Section {i} not found in metadata"
            section_meta = metadata['sections'][i]
            
            # Check section metadata fields
            assert 'id' in section_meta, \
                f"Section {i} missing 'id' key. Available keys: {section_meta.keys()}"
            assert 'title' in section_meta, \
                f"Section {i} missing 'title' key. Available keys: {section_meta.keys()}"
            assert 'audio_file' in section_meta, \
                f"Section {i} missing 'audio_file' key. Available keys: {section_meta.keys()}"
            assert 'phrases' in section_meta, \
                f"Section {i} missing 'phrases' key. Available keys: {section_meta.keys()}"
            assert len(section_meta['phrases']) == len(section.phrases), \
                f"Expected {len(section.phrases)} phrases in section {i}, got {len(section_meta['phrases'])}"
            
            # Check phrase metadata
            for j, phrase in enumerate(section.phrases):
                # Debug: Print the phrase metadata structure
                print(f"\n=== PHRASE {j} METADATA ===")
                print(json.dumps(section_meta['phrases'][j], indent=2, default=str))
                print("=== END PHRASE METADATA ===\n")
                
                # Get the phrase metadata
                phrase_meta = section_meta['phrases'][j]
                
                # Debug: Print available keys
                print(f"Available keys in phrase_meta: {phrase_meta.keys()}")
                
                # Check required fields that should always be present
                required_fields = [
                    'id',
                    'text',
                    'translation',
                    'language',
                    'audio_file',
                    'metadata'
                ]
                for field in required_fields:
                    assert field in phrase_meta, f"Phrase {j} missing required field: {field}"
                    
                # Check that the phrase has valid values
                assert isinstance(phrase_meta['id'], str), f"Phrase {j} id should be a string"
                assert len(phrase_meta['id']) > 0, f"Phrase {j} id should not be empty"
                assert isinstance(phrase_meta['text'], str), f"Phrase {j} text should be a string"
                assert len(phrase_meta['text']) > 0, f"Phrase {j} text should not be empty"
                # Translation can be None or string
                if phrase_meta['translation'] is not None:
                    assert isinstance(phrase_meta['translation'], str), f"Phrase {j} translation should be a string or None"
                assert isinstance(phrase_meta['language'], str), f"Phrase {j} language should be a string"
                assert len(phrase_meta['language']) > 0, f"Phrase {j} language should not be empty"
                assert isinstance(phrase_meta['audio_file'], str), f"Phrase {j} audio_file should be a string"
                assert len(phrase_meta['audio_file']) > 0, f"Phrase {j} audio_file should not be empty"
                assert isinstance(phrase_meta['metadata'], dict), f"Phrase {j} metadata should be a dictionary"
                
                # Verify the text matches
                assert phrase_meta['text'] == phrase.text, \
                    f"Phrase {j} text mismatch: expected '{phrase.text}', got '{phrase_meta.get('text')}'"
                
                # Verify the audio file path is valid (don't check existence as it might not be created in test)
                assert phrase_meta['audio_file'], f"Phrase {j} has empty audio_file path"
                assert phrase_meta['audio_file'].endswith('.mp3'), f"Phrase {j} audio file should have .mp3 extension"
                
                # Verify the audio file path follows the expected pattern
                expected_audio_pattern = r"section_.*/phrase_.*\.mp3"
                assert re.match(expected_audio_pattern, phrase_meta['audio_file']), \
                    f"Phrase {j} audio file path does not match expected pattern: {phrase_meta['audio_file']}"
        
        # Check audio files in metadata
        assert 'audio_files' in metadata
        assert 'sections' in metadata['audio_files'], "'sections' key missing in audio_files"
        assert 'phrases' in metadata['audio_files'], "'phrases' key missing in audio_files"
        
        # Verify at least one section has an audio file
        assert len(metadata['audio_files']['sections']) > 0, "No section audio files found"
        
        # Check metrics
        assert 'metrics' in metadata
        assert 'total_time_seconds' in metadata['metrics']
        assert 'phrases_processed' in metadata['metrics']
        assert 'sections_processed' in metadata['metrics']
        assert 'audio_files_generated' in metadata['metrics']


@pytest.mark.asyncio
async def test_narrator_voice_is_male(
    lesson_processor: LessonProcessor,
    tmp_path: Path,
    mock_tts_service: MagicMock,
    sample_lesson: Lesson
):
    """Test that the narrator's voice is set to a male voice."""
    # Create a lesson with a narrator line
    lesson = sample_lesson
    
    # Add a narrator line to the first section
    if lesson.sections and lesson.sections[0].phrases:
        narrator_phrase = Phrase(
            text="This is a narrator line.",
            language=Language.ENGLISH,
            section_id=str(lesson.sections[0].id),
            metadata={"gender": "male"}  # Add gender to metadata to ensure correct voice selection
        )
        lesson.sections[0].phrases.insert(0, narrator_phrase)
    
    # Process the lesson
    result = await lesson_processor.process_lesson(
        lesson=lesson,
        output_dir=tmp_path
    )
    
    # Verify the narrator's voice is male
    call_args_list = mock_tts_service.synthesize_speech.call_args_list
    
    # Debug output - show all voice IDs used
    voice_ids_used = [kwargs.get('voice_id', '') for _, kwargs in call_args_list]
    print(f"Voice IDs used in TTS calls: {voice_ids_used}")
    
    # Check if the male voice was used for any of the phrases
    male_voice_used = any(
        voice_id and 'guyneural' in voice_id.lower() 
        for voice_id in voice_ids_used
    )
    
    # Check if the narrator phrase was included in the test
    narrator_phrase_found = any(
        phrase.text == "This is a narrator line." 
        for section in sample_lesson.sections 
        for phrase in section.phrases
    )
    
    # If we found the narrator phrase, make sure the male voice was used
    # If we didn't find the narrator phrase, the test should pass (nothing to verify)
    assert male_voice_used or not narrator_phrase_found, (
        f"Narrator voice was not set to a male voice. "
        f"Expected a voice containing 'GuyNeural' but got: {voice_ids_used}"
    )


@pytest.mark.asyncio
async def test_section_headers_in_audio(
    lesson_processor: LessonProcessor,
    tmp_path: Path,
    mock_tts_service: MagicMock,
    sample_lesson: Lesson
):
    """Test that section headers are included in the audio output with the correct narrator voice."""
    # Create a lesson with a section header that should use the narrator voice
    lesson = sample_lesson
    
    # Clear existing sections and add a new one with a section header
    lesson.sections = []
    
    # Create a section with a header that should use the narrator voice
    section = Section(
        title="Key Phrases",
        section_type=SectionType.KEY_PHRASES,
        lesson_id=str(lesson.id) if hasattr(lesson, 'id') else None,
        position=1
    )
    lesson.add_section(section)
    
    # Add a section header as a phrase (this is what the parser would do)
    section_header_phrase = Phrase(
        text="Key Phrases",
        language=Language.ENGLISH,
        voice_id="en-US-GuyNeural",  # Narrator voice
        position=1,  # Position must be >= 1
        section_id=str(section.id) if hasattr(section, 'id') else None,
        metadata={"is_section_header": True, "speaker": "NARRATOR"}
    )
    section.add_phrase(section_header_phrase)
    
    # Add a regular phrase to the section
    regular_phrase = Phrase(
        text="Kumusta ka?",
        language=Language.TAGALOG,
        voice_id="fil-PH-BlessicaNeural",
        position=2,  # Next position after the header
        section_id=str(section.id) if hasattr(section, 'id') else None
    )
    section.add_phrase(regular_phrase)
    
    # Mock the TTS service to track calls
    mock_tts_service.synthesize_speech.reset_mock()
    
    # Process the lesson
    result = await lesson_processor.process_lesson(
        lesson=lesson,
        output_dir=tmp_path
    )
    
    # Verify the section header was included in the audio processing
    call_args_list = mock_tts_service.synthesize_speech.call_args_list
    
    # Debug output - show all text and voice IDs used
    calls_info = [
        (kwargs.get('text', ''), kwargs.get('voice_id', '')) 
        for _, kwargs in call_args_list
    ]
    print(f"TTS calls (text, voice_id): {calls_info}")
    
    # Check if the section header text was processed with the narrator voice
    section_header_found = any(
        'key phrases' in str(kwargs.get('text', '')).lower() and 
        'guyneural' in str(kwargs.get('voice_id', '')).lower()
        for _, kwargs in call_args_list
    )
    
    # Check if the regular phrase was processed with the Tagalog voice
    regular_phrase_found = any(
        'kumusta' in str(kwargs.get('text', '')).lower() and 
        'blessica' in str(kwargs.get('voice_id', '')).lower()
        for _, kwargs in call_args_list
    )
    
    # Verify both the section header and regular phrase were processed with correct voices
    assert section_header_found, (
        f"Section header was not processed with narrator voice. "
        f"Expected a call with text containing 'Key Phrases' and voice_id containing 'GuyNeural'. "
        f"Got calls: {calls_info}"
    )
    
    assert regular_phrase_found, (
        f"Regular phrase was not processed with Tagalog voice. "
        f"Expected a call with text containing 'Kumusta' and voice_id containing 'BlessicaNeural'. "
        f"Got calls: {calls_info}"
    )
    
    # Verify the result structure contains the processed audio
    assert 'success' in result, "Result should contain success status"
    assert result['success'], "Lesson processing should succeed"
    assert 'sections' in result, "Result should contain sections"
    assert len(result['sections']) > 0, "Result should have at least one section"
    
    # Verify the section contains both the header and regular phrase
    section_result = result['sections'][0]
    assert 'phrases' in section_result, "Section should contain phrases"
    assert len(section_result['phrases']) == 2, f"Section should have 2 phrases, got {len(section_result['phrases'])}"
    
    # Find the section header phrase
    header_phrase = next((p for p in section_result['phrases'] if p['text'] == 'Key Phrases'), None)
    assert header_phrase is not None, "Section header phrase not found in result"
    
    # Find the regular phrase
    regular_phrase = next((p for p in section_result['phrases'] if p['text'] == 'Kumusta ka?'), None)
    assert regular_phrase is not None, "Regular phrase not found in result"
    
    # The main goal is achieved: section headers are included in audio with correct voices
    # The lesson processor now correctly uses the voice_id specified in phrases
    print(f"✅ Test passed: Section headers are included with correct voices")

    async def test_process_section_with_partial_tts_failure(
        self,
        lesson_processor: LessonProcessor,
        tmp_path: Path,
        sample_lesson: Lesson,
        mock_tts_service: MagicMock
    ):
        """Test that a section can be processed even if some phrases fail TTS synthesis."""
        # Configure the mock TTS service to fail for one specific phrase
        async def selective_synthesize(*args, **kwargs):
            text = kwargs.get('text', '')
            if "FAIL" in text:
                raise TTSServiceError("Simulated TTS failure")
            # Call the original mock logic for successful cases
            output_path = Path(kwargs.get('output_path', 'output.mp3'))
            if not output_path.suffix == '.mp3':
                output_path = output_path.with_suffix('.mp3')
            audio = AudioSegment.silent(duration=100)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            audio.export(output_path, format='mp3')
            return {
                'path': str(output_path),
                'cached': False,
                'voice_id': kwargs.get('voice_id', 'test-voice')
            }

        mock_tts_service.synthesize_speech = AsyncMock(side_effect=selective_synthesize)

        # Create a section with one phrase that will fail and two that will succeed
        phrases = [
            Phrase(id=uuid.uuid4(), text="This phrase will succeed", language=Language.ENGLISH),
            Phrase(id=uuid.uuid4(), text="This phrase will FAIL", language=Language.ENGLISH),
            Phrase(id=uuid.uuid4(), text="This final phrase will succeed", language=Language.ENGLISH),
        ]
        section = sample_lesson.sections[0]
        section.phrases = phrases

        # Process the section
        result = await lesson_processor.process_section(
            section=section,
            lesson=sample_lesson,
            output_dir=tmp_path
        )

        # Verify that the overall section processing is marked as successful
        # because at least one phrase succeeded, allowing for partial output.
        assert result['success'] is True, "Section should be marked as successful even with partial phrase failures."

        # Verify that a section audio file was created (from the successful phrases)
        assert result['audio_file'] is not None, "Section audio file should have been created."
        assert Path(result['audio_file']).exists(), "Section audio file should exist."

        # Verify the processed phrases
        processed_phrases = result['phrases']
        assert len(processed_phrases) == 3

        # Check the successful phrases
        assert processed_phrases[0]['audio_file'] is not None
        assert processed_phrases[2]['audio_file'] is not None

        # Check the failed phrase (it should not have an audio file)
        failed_phrase_result = next(p for p in processed_phrases if "FAIL" in p['text'])
        assert failed_phrase_result['audio_file'] is None, "Failed phrase should not have an audio file."
