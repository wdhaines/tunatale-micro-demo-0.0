"""Tests for ellipsis handling in lesson processor."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from pathlib import Path
import tempfile
import os

from tunatale.core.services.lesson_processor import LessonProcessor
from tunatale.core.models.phrase import Phrase
from tunatale.core.models.enums import Language, SectionType
from tunatale.core.models.voice import Voice, VoiceGender


@pytest.fixture
def mock_tts_service():
    """Create a mock TTS service."""
    tts_service = AsyncMock()
    tts_service.synthesize_speech = AsyncMock()
    tts_service.synthesize_speech_with_pauses = AsyncMock()
    tts_service.provider_name = 'edge_tts'
    tts_service.supports_ssml = True
    return tts_service


@pytest.fixture
def mock_audio_processor():
    """Create a mock audio processor."""
    audio_processor = AsyncMock()
    audio_processor.combine_audio_files = AsyncMock()
    audio_processor.add_silence = AsyncMock()
    return audio_processor


@pytest.fixture
def mock_voice_selector():
    """Create a mock voice selector."""
    voice_selector = AsyncMock()
    voice_selector.get_voice_id.return_value = "test-voice-id"
    return voice_selector


@pytest.fixture 
def mock_word_selector():
    """Create a mock word selector."""
    word_selector = AsyncMock()
    word_selector.get_words.return_value = []
    return word_selector


@pytest.fixture
def ssml_lesson_processor(mock_tts_service, mock_audio_processor, mock_voice_selector, mock_word_selector):
    """Create a lesson processor with SSML support enabled."""
    with tempfile.TemporaryDirectory() as temp_dir:
        processor = LessonProcessor(
            tts_service=mock_tts_service,
            audio_processor=mock_audio_processor,
            voice_selector=mock_voice_selector,
            word_selector=mock_word_selector,
            output_dir=str(temp_dir),
            ellipsis_pause_duration_ms=800,
            use_natural_pauses=True
        )
        yield processor


@pytest.fixture
def non_ssml_lesson_processor(mock_tts_service, mock_audio_processor, mock_voice_selector, mock_word_selector):
    """Create a lesson processor with SSML support disabled."""
    mock_tts_service.supports_ssml = False
    with tempfile.TemporaryDirectory() as temp_dir:
        processor = LessonProcessor(
            tts_service=mock_tts_service,
            audio_processor=mock_audio_processor,
            voice_selector=mock_voice_selector,
            word_selector=mock_word_selector,
            output_dir=str(temp_dir),
            ellipsis_pause_duration_ms=800,
            use_natural_pauses=False
        )
        yield processor


class TestEllipsisHandling:
    """Test ellipsis handling for creating pauses in audio."""
    
    def test_preprocess_text_with_ssml_ellipsis(self, ssml_lesson_processor):
        """Test that ellipsis in text are converted to pause markers with SSML support."""
        text_with_ellipsis = "Excuse... me... po."
        processed_text = ssml_lesson_processor._preprocess_text(text_with_ellipsis, "tagalog")
        
        # Should convert to pause markers for SSML providers
        assert "[PAUSE:0.5s]" in processed_text
        assert "..." not in processed_text
        assert processed_text == "Excuse[PAUSE:0.5s] me[PAUSE:0.5s] po."
    
    def test_preprocess_text_with_non_ssml_ellipsis(self, non_ssml_lesson_processor):
        """Test that ellipsis in text are converted to pause markers without SSML support."""
        text_with_ellipsis = "Excuse... me... po."
        processed_text = non_ssml_lesson_processor._preprocess_text(text_with_ellipsis, "tagalog")
        
        # Should convert to pause markers for non-SSML providers
        assert "[PAUSE:0.5s]" in processed_text
        assert "..." not in processed_text
        assert processed_text == "Excuse[PAUSE:0.5s] me[PAUSE:0.5s] po."

    @pytest.mark.parametrize("input_text,expected_ssml,expected_non_ssml", [
        ("Hello... world", "Hello[PAUSE:0.5s] world", "Hello[PAUSE:0.5s] world"),
        ("One... two... three", "One[PAUSE:0.5s] two[PAUSE:0.5s] three", "One[PAUSE:0.5s] two[PAUSE:0.5s] three"),
        ("Start...middle...end", "Start[PAUSE:0.5s]middle[PAUSE:0.5s]end", "Start[PAUSE:0.5s]middle[PAUSE:0.5s]end"),
        ("Wait... ... continue", "Wait[PAUSE:0.5s] [PAUSE:0.5s] continue", "Wait[PAUSE:0.5s] [PAUSE:0.5s] continue"),
        ("No ellipsis", "No ellipsis", "No ellipsis"),
        ("...", "[PAUSE:0.5s]", "[PAUSE:0.5s]"),
        ("Text...", "Text[PAUSE:0.5s]", "Text[PAUSE:0.5s]"),
        ("...Text", "[PAUSE:0.5s]Text", "[PAUSE:0.5s]Text"),
    ])
    def test_multiple_ellipsis_patterns(self, ssml_lesson_processor, non_ssml_lesson_processor, 
                                      input_text, expected_ssml, expected_non_ssml):
        """Test various ellipsis patterns with both SSML and non-SSML providers."""
        # Test with SSML support
        ssml_processed = ssml_lesson_processor._preprocess_text(input_text, "tagalog")
        assert ssml_processed == expected_ssml
        
        # Test without SSML support
        non_ssml_processed = non_ssml_lesson_processor._preprocess_text(input_text, "tagalog")
        assert non_ssml_processed == expected_non_ssml
    
    @pytest.mark.asyncio
    async def test_phrase_processing_with_ssml_ellipsis(self, ssml_lesson_processor, mock_tts_service, mock_audio_processor):
        """Test that phrase processing handles SSML ellipsis correctly."""
        phrase = Phrase(
            text="Tubig?... Opo... Malamig... o... normal?",
            language=Language.TAGALOG,
            speaker_id="TAGALOG-FEMALE-2"
        )
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            mock_tts_service.synthesize_speech_with_pauses.return_value = None
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir)
                
                await ssml_lesson_processor._process_phrase(
                    phrase=phrase,
                    output_path=output_path
                )
                
                # Should be a single TTS call with pause markers using pause-aware synthesis
                assert mock_tts_service.synthesize_speech_with_pauses.call_count == 1
                
                call_args = mock_tts_service.synthesize_speech_with_pauses.call_args_list[0]
                tts_text = call_args.kwargs['text']
                
                # Should have pause markers, not semicolons
                assert "[PAUSE:" in tts_text
                assert "..." not in tts_text
                
                # Audio processor should be called to add silence for pause-aware synthesis
                # (This is expected behavior when using pause markers)
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_phrase_processing_with_non_ssml_ellipsis(self, non_ssml_lesson_processor, mock_tts_service, mock_audio_processor):
        """Test that phrase processing handles non-SSML ellipsis correctly."""
        phrase = Phrase(
            text="Tubig?... Opo... Malamig... o... normal?",
            language=Language.TAGALOG,
            speaker_id="TAGALOG-FEMALE-2"
        )
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            mock_tts_service.synthesize_speech_with_pauses.return_value = None
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir)
                
                await non_ssml_lesson_processor._process_phrase(
                    phrase=phrase,
                    output_path=output_path
                )
                
                # Should be a single TTS call with pause markers using pause-aware synthesis
                assert mock_tts_service.synthesize_speech_with_pauses.call_count == 1
                
                call_args = mock_tts_service.synthesize_speech_with_pauses.call_args_list[0]
                tts_text = call_args.kwargs['text']
                
                # Should have pause markers, not semicolons
                assert "[PAUSE:" in tts_text
                assert "..." not in tts_text
                
                # Audio processor should be called to handle pause markers
                mock_audio_processor.add_silence.assert_called()
                
                # Verify the silence duration (800ms = 0.8s)
                silence_call = mock_audio_processor.add_silence.call_args
                assert silence_call.kwargs['duration'] == 0.8
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_configurable_pause_duration(self, mock_tts_service, mock_audio_processor, mock_voice_selector, mock_word_selector):
        """Test that ellipsis pause duration is configurable."""
        test_durations = [200, 800, 1500]  # ms
        
        for duration_ms in test_durations:
            with tempfile.TemporaryDirectory() as temp_dir:
                processor = LessonProcessor(
                    tts_service=mock_tts_service,
                    audio_processor=mock_audio_processor,
                    voice_selector=mock_voice_selector,
                    word_selector=mock_word_selector,
                    output_dir=str(temp_dir),
                    ellipsis_pause_duration_ms=duration_ms,
                    use_natural_pauses=False
                )
                
                assert processor.ellipsis_pause_duration_ms == duration_ms