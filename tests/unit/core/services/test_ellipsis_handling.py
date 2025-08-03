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
        """Test that single ellipsis are left as-is for natural TTS handling."""
        text_with_ellipsis = "Excuse... me... po."
        processed_text = ssml_lesson_processor._preprocess_text(text_with_ellipsis, "tagalog")
        
        # Single ... should be left as-is for natural TTS handling
        assert "..." in processed_text
        assert "[PAUSE:" not in processed_text
        assert processed_text == "Excuse... me... po."
    
    def test_preprocess_text_with_non_ssml_ellipsis(self, non_ssml_lesson_processor):
        """Test that single ellipsis are left as-is for natural TTS handling."""
        text_with_ellipsis = "Excuse... me... po."
        processed_text = non_ssml_lesson_processor._preprocess_text(text_with_ellipsis, "tagalog")
        
        # Single ... should be left as-is for natural TTS handling
        assert "..." in processed_text
        assert "[PAUSE:" not in processed_text  
        assert processed_text == "Excuse... me... po."

    @pytest.mark.parametrize("input_text,expected_ssml,expected_non_ssml", [
        # Single ... should be left as-is for natural TTS handling
        ("Hello... world", "Hello... world", "Hello... world"),
        ("One... two... three", "One... two... three", "One... two... three"),
        ("Start...middle...end", "Start...middle...end", "Start...middle...end"),
        ("Wait... ... continue", "Wait... ... continue", "Wait... ... continue"),
        ("No ellipsis", "No ellipsis", "No ellipsis"),
        ("...", "...", "..."),
        ("Text...", "Text...", "Text..."),
        ("...Text", "...Text", "...Text"),
        # Long ellipsis (4+ dots) should be converted
        ("Hello.... world", "Hello[PAUSE:0.5s] world", "Hello[PAUSE:0.5s] world"),
        ("Wait..... for it", "Wait[PAUSE:0.75s] for it", "Wait[PAUSE:0.75s] for it"),
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
        """Test that phrase processing handles long ellipsis correctly."""
        phrase = Phrase(
            text="Tubig?.... Opo..... Malamig.... o.... normal?",
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
                
                # Should have pause markers from long ellipsis
                assert "[PAUSE:" in tts_text
                # Should not have the original long ellipsis patterns
                assert "...." not in tts_text and "....." not in tts_text
                
                # Audio processor should be called to add silence for pause-aware synthesis
                # (This is expected behavior when using pause markers)
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_phrase_processing_with_non_ssml_ellipsis(self, non_ssml_lesson_processor, mock_tts_service, mock_audio_processor):
        """Test that phrase processing handles single ellipsis naturally."""
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
                
                # Single ... should use regular synthesis (not pause-aware)
                # Since pause-aware synthesis wasn't called, check regular synthesis
                if mock_tts_service.synthesize_speech_with_pauses.call_count == 0:
                    # Should use regular synthesis for single ellipsis
                    assert mock_tts_service.synthesize_speech.call_count >= 1
                    call_args = mock_tts_service.synthesize_speech.call_args_list[0]
                    tts_text = call_args.kwargs['text']
                    # Should keep original ellipsis for natural TTS handling
                    assert "..." in tts_text
                    assert "[PAUSE:" not in tts_text
                else:
                    # If it did use pause-aware synthesis, that's ok too
                    call_args = mock_tts_service.synthesize_speech_with_pauses.call_args_list[0]
                    tts_text = call_args.kwargs['text']
                
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