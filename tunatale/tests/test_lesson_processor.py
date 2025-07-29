import asyncio
import logging
import os
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import List, Optional

from tunatale.core.models.phrase import Phrase
from tunatale.core.models.section import Section, SectionType
from tunatale.core.models.lesson import Lesson
from tunatale.core.services.lesson_processor import LessonProcessor
from tunatale.core.ports.tts_service import TTSService
from tunatale.core.ports.audio_processor import AudioProcessor
from pydantic import ValidationError
from tunatale.core.exceptions import TTSValidationError

logger = logging.getLogger(__name__)

@pytest.fixture
def mock_tts_service():
    mock = AsyncMock(spec=TTSService)
    mock.name = "mock_tts"
    
    # Configure get_voices to return only voices matching the requested language
    voices = [
        Voice(
            id="en-US-JennyNeural",
            name="Jenny",
            provider="Edge",
            provider_id="en-US-JennyNeural",
            language=Language.ENGLISH,
            gender="Female",
            age="Adult"
        ),
        Voice(
            id="fil-PH-BlessicaNeural",
            name="Blessica",
            provider="Edge",
            provider_id="fil-PH-BlessicaNeural",
            language=Language.TAGALOG,
            gender="Female",
            age="Adult"
        )
    ]
    
    async def get_voices(language: Optional[Language] = None) -> List[Voice]:
        if language is None:
            return voices
        matching_voices = [v for v in voices if v.language == language]
        if not matching_voices:
            raise TTSValidationError(f"No available voice for language: {language}")
        return matching_voices

    # Configure get_voice_id to return correct voice ID
    async def get_voice_id(language: Language) -> str:
        matching_voices = await get_voices(language)
        if not matching_voices:
            raise TTSValidationError(f"No available voice for language: {language}")
        return matching_voices[0].id

    # Configure mock methods
    mock.get_voices = AsyncMock(side_effect=get_voices)
    mock.get_voice_id = AsyncMock(side_effect=get_voice_id)
    
    # Configure mock to return voice_id as string
    mock.get_voice_id.return_value = "en-US-JennyNeural"
    
    # Configure synthesize_speech to create dummy audio files
    async def mock_synthesize_speech(text: str, voice_id: str, output_path: Path, **options):
        """Mock implementation of synthesize_speech that creates dummy audio files."""
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write dummy audio data
        with open(output_path, 'wb') as f:
            f.write(b'dummy audio data')
        
        # Return the output path as a string
        return str(output_path)
    
    mock.synthesize_speech = AsyncMock(side_effect=mock_synthesize_speech)
    
    return mock

@pytest.fixture
def mock_audio_processor():
    mock = AsyncMock(spec=AudioProcessor)
    
    # Configure mock methods
    # Configure mock methods with format parameter
    async def mock_process_audio(input_file: Path, output_file: Path, format: str = 'mp3') -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'wb') as f:
            f.write(b'dummy audio data')
    
    async def mock_concatenate_audio(files: List[Path], output_file: Path, format: str = 'mp3') -> Path:
        """Mock implementation of concatenate_audio that creates dummy audio files."""
        # Create parent directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write dummy audio data
        with open(output_file, 'wb') as f:
            f.write(b'dummy concatenated audio data')
        
        # Return the output file path
        return output_file
    
    mock.process_audio = AsyncMock(side_effect=mock_process_audio)
    mock.concatenate_audio = AsyncMock(side_effect=mock_concatenate_audio)
    
    return mock

import uuid
from tunatale.core.models.enums import Language
from tunatale.core.models.voice import Voice
from tunatale.core.exceptions import AudioProcessingError

@pytest.fixture
def test_phrase():
    return Phrase(
        id=str(uuid.uuid4()),
        text="Hello world",
        language=Language.ENGLISH,
        speaker="user"
    )

@pytest.fixture
def test_section():
    return Section(
        id=str(uuid.uuid4()),
        lesson_id=str(uuid.uuid4()),
        title="Test Section",
        section_type=SectionType.KEY_PHRASES,
        phrases=[
            Phrase(
                id=str(uuid.uuid4()),
                text="Hello world",
                language=Language.ENGLISH,
                speaker="user"
            ),
            Phrase(
                id=str(uuid.uuid4()),
                text="Kamusta mundo",
                language=Language.TAGALOG,
                speaker="user"
            )
        ]
    )

@pytest.mark.asyncio
async def test_process_phrase(mock_tts_service, mock_audio_processor, test_phrase, tmp_path):
    processor = LessonProcessor(mock_tts_service, mock_audio_processor)
    
    # Test English phrase
    result = await processor.process_phrase(test_phrase, tmp_path)
    assert result["success"]
    assert result["voice_id"] == "en-US-JennyNeural"
    assert result["language"] == "english"
    assert Path(result["audio_file"]).exists()
    
    # Test Tagalog phrase with valid UUID and enum
    tag_phrase = Phrase(
        id=str(uuid.uuid4()),
        text="Kamusta mundo",
        language=Language.TAGALOG,
        speaker="user"
    )
    result = await processor.process_phrase(tag_phrase, tmp_path)
    assert result["success"]
    assert result["voice_id"] == "fil-PH-BlessicaNeural"
    assert result["language"] == "tagalog"
    assert Path(result["audio_file"]).exists()

@pytest.mark.asyncio
async def test_process_section(mock_tts_service, mock_audio_processor, test_section, tmp_path):
    processor = LessonProcessor(mock_tts_service, mock_audio_processor)
    
    result = await processor.process_section(test_section, tmp_path)
    assert result["success"]
    assert result["section_id"] == str(test_section.id)
    assert result["title"] == "Test Section"
    assert result["type"] == SectionType.KEY_PHRASES.value
    assert len(result["phrases"]) == 2
    
    # Verify phrase results
    for phrase_result in result["phrases"]:
        assert phrase_result["success"]
        assert Path(phrase_result["audio_file"]).exists()
    
    # Verify section audio file exists and has correct name
    section_dir = tmp_path / 'sections'
    assert section_dir.exists()
    section_title = test_section.title.lower().replace(' ', '_')
    section_file = section_dir / f"{section_title}.mp3"
    assert section_file.exists()
    assert result["audio_file"] == str(section_file)

@pytest.mark.asyncio
async def test_error_handling(mock_tts_service, mock_audio_processor, test_phrase, tmp_path):
    processor = LessonProcessor(mock_tts_service, mock_audio_processor)
    
    # Test invalid language
    with pytest.raises(ValidationError) as exc_info:
        Phrase(
            id=str(uuid.uuid4()),
            text="Test",
            language="xyz-ABC",
            speaker="user"
        )
    
    assert "language" in str(exc_info.value)
    assert "Input should be 'tagalog', 'english' or 'spanish'" in str(exc_info.value)
    
    # Test process_phrase with invalid language
    invalid_phrase = Phrase(
        id=str(uuid.uuid4()),
        text="Test",
        language=Language.ENGLISH,
        speaker="user"
    )
        
    # Mock TTS service to raise error for invalid voice
    mock_tts_service.get_voices.side_effect = TTSValidationError("No available voice for language")
        
    result = await processor.process_phrase(invalid_phrase, tmp_path)
    assert not result["success"]
    assert "error" in result
    assert "error_code" in result["error"]
    assert "error_message" in result["error"]
    assert result["error"]["error_code"] == "TTS_VALIDATION_ERROR"
    assert "No available voice for language" in result["error"]["error_message"]
    # Test process_section with invalid language
    invalid_section = Section(
        id=str(uuid.uuid4()),
        lesson_id=str(uuid.uuid4()),
        title="Invalid Section",
        section_type=SectionType.KEY_PHRASES,
        phrases=[invalid_phrase]
    )
    
    result = await processor.process_section(invalid_section, tmp_path)
    assert not result["success"]
    assert "error" in result
    assert "error_code" in result["error"]
    assert "error_message" in result["error"]
    assert result["error"]["error_code"] == "SECTION_PROCESSING_ERROR"
    assert "One or more phrases failed to process" in result["error"]["error_message"]

@pytest.mark.asyncio
async def test_audio_normalization_error(mock_tts_service, mock_audio_processor, test_phrase, tmp_path):
    """Test error handling for audio normalization failures."""
    processor = LessonProcessor(mock_tts_service, mock_audio_processor)
    
    # Create a mock audio file in the phrases directory
    phrases_dir = tmp_path / "phrases"
    phrases_dir.mkdir(exist_ok=True)
    audio_file = phrases_dir / "test_phrase_en.mp3"
    audio_file.write_bytes(b"mock audio data")  # Create a mock audio file
    
    # Mock audio_processor.normalize_audio to raise error
    async def mock_normalize_audio(input_file: Path, output_file: Path, **options):
        raise AudioProcessingError("Failed to normalize audio")
    
    mock_audio_processor.normalize_audio.side_effect = mock_normalize_audio
    
    # Create a valid phrase
    valid_phrase = Phrase(
        id=str(uuid.uuid4()),
        text="Hello world",
        language=Language.ENGLISH,
        speaker="user"
    )
    
    # Mock synthesize_speech to return the mock audio file path
    mock_tts_service.synthesize_speech.return_value = {"audio_file": str(audio_file)}
    
    # Process phrase and verify error handling
    result = await processor.process_phrase(valid_phrase, tmp_path)
    assert not result["success"]
    assert "error" in result
    assert "error_code" in result["error"]
    assert "error_message" in result["error"]
    assert result["error"]["error_code"] == "AUDIO_PROCESSING_ERROR"
    assert "Failed to normalize audio" in result["error"]["error_message"]
    
    # Test section processing with normalization error
    valid_section = Section(
        id=str(uuid.uuid4()),
        lesson_id=str(uuid.uuid4()),
        title="Test Section",
        section_type=SectionType.KEY_PHRASES,
        phrases=[valid_phrase]
    )
    
    result = await processor.process_section(valid_section, tmp_path)
    assert not result["success"]
    assert "error" in result
    assert "error_code" in result["error"]
    assert "error_message" in result["error"]
    assert result["error"]["error_code"] == "SECTION_PROCESSING_ERROR"
    assert "One or more phrases failed to process" in result["error"]["error_message"]
