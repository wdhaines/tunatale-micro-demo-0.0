"""Integration tests for LessonProcessor."""
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from pydub import AudioSegment

from tunatale.core.exceptions import AudioProcessingError, TTSServiceError
import uuid
from tunatale.core.models.lesson import Lesson
from tunatale.core.models.section import Section
from tunatale.core.models.phrase import Phrase
from tunatale.core.models.enums import SectionType, Language
from tunatale.core.models.voice import Voice, VoiceGender
from tunatale.core.services.lesson_processor import LessonProcessor
from tunatale.core.ports.audio_processor import AudioProcessor
from tunatale.core.ports.tts_service import TTSService

# Disable logging during tests
logging.getLogger().setLevel(logging.CRITICAL)


@pytest.fixture
def mock_tts_service():
    """Create a mock TTS service."""
    mock = MagicMock(spec=TTSService)
    
    # Mock synthesize_speech to create a silent audio file
    async def mock_synthesize(*args, **kwargs):
        output_path = Path(kwargs.get('output_path', 'output.mp3'))
        # Get format from file extension or default to 'mp3'
        output_format = output_path.suffix.lstrip('.').lower() or 'mp3'
        # Create a 100ms silent audio file
        audio = AudioSegment.silent(duration=100)  # 100ms
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio.export(output_path, format=output_format)
        return {
            'path': str(output_path),
            'cached': False,
            'voice_id': kwargs.get('voice_id', 'test-voice')
        }
    
    mock.synthesize_speech = AsyncMock(side_effect=mock_synthesize)
    mock.get_voice = AsyncMock(return_value=Voice(
        name="Test Voice",
        provider="test",
        provider_id="test-voice",
        language="en",
        gender=VoiceGender.FEMALE
    ))
    
    return mock


@pytest.fixture
def mock_audio_processor():
    """Create a mock audio processor."""
    mock = MagicMock(spec=AudioProcessor)
    
    # Mock methods to return the input file as output
    async def mock_concatenate(input_files, output_file, **kwargs):
        # Just copy the first input file to the output
        if input_files:
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
def lesson_processor(mock_tts_service, mock_audio_processor):
    """Create a LessonProcessor instance for testing."""
    return LessonProcessor(
        tts_service=mock_tts_service,
        audio_processor=mock_audio_processor,
        config={
            'output_format': 'mp3',
            'silence_between_phrases': 0.5,
            'silence_between_sections': 1.0,
            'cleanup_temp_files': False,  # Keep files for inspection in tests
        }
    )


class TestLessonProcessor:
    """Test cases for LessonProcessor."""
    
    async def test_process_phrase(
        self,
        lesson_processor: LessonProcessor,
        tmp_path: Path,
        sample_lesson: Lesson
    ):
        """Test processing a single phrase."""
        # Get a phrase from the sample lesson
        phrase = sample_lesson.sections[0].phrases[0]
        
        # Process the phrase
        result = await lesson_processor.process_phrase(
            phrase=phrase,
            output_dir=tmp_path
        )
        
        # Verify the result
        assert 'audio_file' in result
        assert result['text'] == phrase.text
        # Compare string values since we convert the enum to string in the result
        assert result['language'] == phrase.language.value
        assert result['voice_id'] == phrase.voice_id
        
        # Verify the audio file was created
        audio_path = Path(result['audio_file'])
        assert audio_path.exists()
        assert audio_path.suffix == '.mp3'
    
    async def test_process_section(
        self,
        lesson_processor: LessonProcessor,
        tmp_path: Path,
        sample_lesson: Lesson
    ):
        """Test processing a section with multiple phrases."""
        # Get a section from the sample lesson
        section = sample_lesson.sections[0]
        
        # Process the section
        result = await lesson_processor.process_section(
            section=section,
            lesson=sample_lesson,  # Add the required lesson parameter
            output_dir=tmp_path
        )
        
        # Verify the result
        assert 'phrases' in result
        assert len(result['phrases']) == len(section.phrases)
        assert 'audio_file' in result
        
        # Verify the section audio file was created
        audio_path = Path(result['audio_file'])
        assert audio_path.exists()
        assert audio_path.suffix == '.mp3'
        
        # Verify the phrase audio files were created
        for phrase_result in result['phrases']:
            assert 'audio_file' in phrase_result
            assert Path(phrase_result['audio_file']).exists()
    
    async def test_process_lesson(
        self,
        lesson_processor: LessonProcessor,
        tmp_path: Path,
        sample_lesson: Lesson
    ):
        """Test processing a complete lesson."""
        # Process the lesson
        result = await lesson_processor.process_lesson(
            lesson=sample_lesson,
            output_dir=tmp_path
        )
        
        # Verify the result
        assert 'sections' in result
        assert len(result['sections']) == len(sample_lesson.sections)
        assert 'final_audio_file' in result
        
        # Verify the final audio file was created
        final_audio_path = Path(result['final_audio_file'])
        assert final_audio_path.exists()
        assert final_audio_path.suffix == '.mp3'
        
        # Verify the metadata file was created
        metadata_path = tmp_path / 'metadata.json'
        assert metadata_path.exists()
        
        # Verify metadata content
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        assert metadata['title'] == sample_lesson.title
        assert len(metadata['sections']) == len(sample_lesson.sections)
        assert 'processing_info' in metadata
    
    async def test_process_lesson_with_progress_callback(
        self,
        lesson_processor: LessonProcessor,
        tmp_path: Path,
        sample_lesson: Lesson
    ):
        """Test processing a lesson with a progress callback."""
        # Create a mock callback
        callback_calls = []
        
        def progress_callback(current: int, total: int, status: str, **kwargs):
            callback_calls.append({
                'current': current,
                'total': total,
                'status': status,
                **kwargs
            })
        
        # Process the lesson with the callback
        await lesson_processor.process_lesson(
            lesson=sample_lesson,
            output_dir=tmp_path,
            progress_callback=progress_callback
        )
        
        # Verify the callback was called
        assert len(callback_calls) > 0
        
        # Verify the callback received progress updates
        for call in callback_calls:
            assert 'current' in call
            assert 'total' in call
            assert 'status' in call
    
    async def test_process_phrase_tts_error(
        self,
        mock_tts_service: MagicMock,
        tmp_path: Path,
        sample_lesson: Lesson
    ):
        """Test handling of TTS errors during phrase processing."""
        # Create a new mock TTS service that raises an error
        error_tts_service = MagicMock(spec=TTSService)
        
        # Set up the mock to raise an error when synthesize_speech is called
        async def mock_synthesize(*args, **kwargs):
            raise TTSServiceError("TTS service error")
            
        error_tts_service.synthesize_speech = mock_synthesize
        
        # Create a mock audio processor
        mock_audio_processor = MagicMock()
        
        # Create a processor with the error-raising mock
        processor = LessonProcessor(
            tts_service=error_tts_service,
            audio_processor=mock_audio_processor,
            config={'output_format': 'mp3'}
        )
        
        # Get a phrase from the sample lesson
        phrase = sample_lesson.sections[0].phrases[0]
        
        # Process the phrase (should not raise an exception)
        result = await processor.process_phrase(
            phrase=phrase,
            output_dir=tmp_path
        )
        
        # Verify the result indicates an error
        assert 'error' in result
        assert 'TTS service error' in result['error']
    
    async def test_process_section_audio_error(
        self,
        mock_tts_service: MagicMock,
        mock_audio_processor: MagicMock,
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
            config={
                'output_format': 'mp3',
                'normalize_audio': False,  # Disable to simplify test
                'trim_silence': False     # Disable to simplify test
            }
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
        
        # Verify the result contains phrases with errors
        assert 'phrases' in result, "Expected phrases in the result"
        assert len(result['phrases']) > 0, "Expected at least one phrase in the result"
        
        # Check that the section has an error and no audio file
        assert not result['success'], "Expected section processing to fail"
        assert 'error' in result, "Expected an error message in the result"
        assert 'AudioProcessingError' in result['error'], "Expected an audio processing error"
        assert result['audio_file'] is None, "Expected no audio file when concatenation fails"
        
        # Verify all phrases were processed successfully
        assert 'phrases' in result, "Expected phrases in the result"
        assert len(result['phrases']) > 0, "Expected at least one phrase in the result"
        assert all(p.get('success', False) for p in result['phrases']), "All phrases should process successfully"
    
    async def test_process_section_audio_error_in_phrase(
        self,
        mock_tts_service: MagicMock,
        mock_audio_processor: MagicMock,
        tmp_path: Path,
        sample_lesson: Lesson
    ):
        """Test handling of audio processing errors during section processing."""
        # Create a mock TTS service that returns a successful result
        mock_tts_service = MagicMock()
        mock_tts_service.synthesize_speech = AsyncMock(return_value={
            'audio_file': str(tmp_path / 'temp_audio.mp3'),
            'cached': False
        })
        
        # Create a mock audio processor that raises an error during normalization
        async def mock_normalize_audio(*args, **kwargs):
            raise AudioProcessingError("Audio processing error")
            
        mock_audio_processor.normalize_audio = AsyncMock(side_effect=mock_normalize_audio)
        mock_audio_processor.get_audio_duration = AsyncMock(return_value=1.0)
        
        # Create a processor with our mocks
        processor = LessonProcessor(
            tts_service=mock_tts_service,
            audio_processor=mock_audio_processor,
            config={
                'output_format': 'mp3',
                'normalize_audio': True,  # Ensure normalization is enabled
                'trim_silence': False    # Disable to simplify test
            }
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

        # Verify the result contains phrases with errors
        assert 'phrases' in result, "Expected phrases in the result"
        assert len(result['phrases']) > 0, "Expected at least one phrase in the result"
        
        # Check that at least one phrase has an error
        phrase_errors = [p.get('error') for p in result['phrases'] if p.get('error')]
        assert len(phrase_errors) > 0, "Expected at least one phrase to have an error"
        
        # Check for any error that indicates audio processing failed
        assert any('AudioProcessingError' in str(e) for e in phrase_errors), \
            "Expected an audio processing error in one of the phrases"
    
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
        assert 'description' in metadata, f"Metadata missing 'description' key. Available keys: {metadata.keys()}"
        assert metadata['description'] == sample_lesson.description
        assert 'sections' in metadata, f"Metadata missing 'sections' key. Available keys: {metadata.keys()}"
        assert len(metadata['sections']) == len(sample_lesson.sections), \
            f"Expected {len(sample_lesson.sections)} sections, got {len(metadata['sections'])}"
        
        # Check section metadata
        for i, section in enumerate(sample_lesson.sections):
            assert i < len(metadata['sections']), f"Section {i} not found in metadata"
            section_meta = metadata['sections'][i]
            
            # Check section metadata fields
            assert 'section_id' in section_meta, \
                f"Section {i} missing 'section_id' key. Available keys: {section_meta.keys()}"
            assert 'section_type' in section_meta, \
                f"Section {i} missing 'section_type' key. Available keys: {section_meta.keys()}"
            assert section_meta['section_type'] == section.section_type.value
            assert 'phrases' in section_meta, \
                f"Section {i} missing 'phrases' key. Available keys: {section_meta.keys()}"
            assert len(section_meta['phrases']) == len(section.phrases), \
                f"Expected {len(section.phrases)} phrases in section {i}, got {len(section_meta['phrases'])}"
            
            # Check phrase metadata
            for j, phrase in enumerate(section.phrases):
                phrase_meta = section_meta['phrases'][j]
                assert phrase_meta['text'] == phrase.text
                assert phrase_meta['language'] == phrase.language.value  # Compare string values
                assert phrase_meta['voice_id'] == phrase.voice_id
                assert 'audio_file' in phrase_meta
                assert 'duration' in phrase_meta
        
        # Check audio files in metadata
        assert 'audio_files' in metadata
        assert 'final' in metadata['audio_files']
        assert 'sections' in metadata['audio_files']
        assert 'phrases' in metadata['audio_files']
        
        # Check processing info
        assert 'processing_info' in metadata
        assert 'output_dir' in metadata['processing_info']
        assert 'timestamp' in metadata['processing_info']
        assert 'duration' in metadata['processing_info']
