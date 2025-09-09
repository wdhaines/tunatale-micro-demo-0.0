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
from tunatale.core.models.enums import Language
from tunatale.core.models.voice import Voice
from tunatale.core.services.lesson_processor import LessonProcessor
from tunatale.core.ports.tts_service import TTSService
from tunatale.core.ports.audio_processor import AudioProcessor
from pydantic import ValidationError
from tunatale.core.exceptions import TTSValidationError, AudioProcessingError
from tunatale.core.models.audio_config import AudioConfig
import uuid

logger = logging.getLogger(__name__)

@pytest.fixture
def mock_tts_service():
    mock = AsyncMock(spec=TTSService)
    mock.name = "mock_tts"
    
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

    async def get_voice_id(language: Language) -> str:
        matching_voices = await get_voices(language)
        if not matching_voices:
            raise TTSValidationError(f"No available voice for language: {language}")
        return matching_voices[0].id

    mock.get_voices = AsyncMock(side_effect=get_voices)
    mock.get_voice_id = AsyncMock(side_effect=get_voice_id)
    mock.get_voice_id.return_value = "en-US-JennyNeural"
    
    async def mock_synthesize_speech(text: str, voice_id: str, output_path: Path, **options):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(b'dummy audio data')
        return str(output_path)
    
    mock.synthesize_speech = AsyncMock(side_effect=mock_synthesize_speech)
    
    return mock

@pytest.fixture
def mock_audio_processor():
    mock = AsyncMock(spec=AudioProcessor)
    
    async def mock_process_audio(input_file: Path, output_file: Path, format: str = 'mp3') -> None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'wb') as f:
            f.write(b'dummy audio data')
    
    async def mock_concatenate_audio(files: List[Path], output_file: Path, format: str = 'mp3') -> Path:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'wb') as f:
            f.write(b'dummy concatenated audio data')
        return output_file
    
    mock.process_audio = AsyncMock(side_effect=mock_process_audio)
    mock.concatenate_audio = AsyncMock(side_effect=mock_concatenate_audio)
    
    return mock

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
    
    result = await processor.process_phrase(test_phrase, tmp_path)
    assert result["success"]
    assert result["language"] == Language.ENGLISH
    assert Path(result["audio_file"]).exists()
    
    tag_phrase = Phrase(
        id=str(uuid.uuid4()),
        text="Kamusta mundo",
        language=Language.TAGALOG,
        speaker="user"
    )
    result = await processor.process_phrase(tag_phrase, tmp_path)
    assert result["success"]
    assert result["language"] == Language.TAGALOG
    assert Path(result["audio_file"]).exists()

@pytest.mark.asyncio
async def test_process_section(mock_tts_service, mock_audio_processor, test_section, tmp_path):
    processor = LessonProcessor(mock_tts_service, mock_audio_processor)
    
    result = await processor.process_section(test_section, tmp_path)
    assert result["success"]
    assert result["id"] == str(test_section.id)
    assert result["title"] == "Test Section"
    assert len(result["phrases"]) == 2
    
    for phrase_result in result["phrases"]:
        assert phrase_result["success"]
        assert Path(phrase_result["audio_file"]).exists()
    
    section_title = test_section.title.lower().replace(' ', '_')
    section_file = tmp_path / f"{section_title}.mp3"
    assert section_file.exists()
    assert result["audio_file"] == str(section_file)

@pytest.mark.asyncio
async def test_error_handling(mock_tts_service, mock_audio_processor, test_phrase, tmp_path):
    processor = LessonProcessor(mock_tts_service, mock_audio_processor)
    
    with pytest.raises(ValidationError):
        Phrase(
            id=str(uuid.uuid4()),
            text="Test",
            language="xyz-ABC",
            speaker="user"
        )
    
    invalid_phrase = Phrase(
        id=str(uuid.uuid4()),
        text="Test",
        language=Language.ENGLISH,
        speaker="user"
    )
        
    mock_tts_service.synthesize_speech.side_effect = TTSValidationError("Simulated TTS error")
        
    result = await processor.process_phrase(invalid_phrase, tmp_path)
    assert not result["success"]
    assert result["error"]["error_code"] == "TTS_VALIDATION_ERROR"
    assert "Simulated TTS error" in result["error"]["error_message"]

@pytest.mark.asyncio
async def test_audio_normalization_error(mock_tts_service, mock_audio_processor, test_phrase, tmp_path):
    processor = LessonProcessor(mock_tts_service, mock_audio_processor)
    
    phrases_dir = tmp_path / "phrases"
    phrases_dir.mkdir(exist_ok=True)
    audio_file = phrases_dir / "test_phrase_en.mp3"
    audio_file.write_bytes(b"mock audio data")
    
    async def mock_normalize(input_file: str, **options):
        raise AudioProcessingError("Failed to normalize audio")
    
    mock_audio_processor.normalize = AsyncMock(side_effect=mock_normalize)
    
    valid_phrase = Phrase(
        id=str(uuid.uuid4()),
        text="Hello world",
        language=Language.ENGLISH,
        speaker="user",
        metadata={"audio_config": AudioConfig(normalize=True)}
    )
    
    mock_tts_service.synthesize_speech.return_value = str(audio_file)
    
    result = await processor.process_phrase(valid_phrase, tmp_path)
    assert not result["success"]
    assert result["error"]["error_code"] == "AUDIO_PROCESSING_ERROR"
    assert "Failed to normalize audio" in result["error"]["error_message"]