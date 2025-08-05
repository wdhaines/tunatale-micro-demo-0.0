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
        """Test that single ellipsis are converted to semicolons."""
        text = "Excuse me... do you have the time?"
        processed = ssml_lesson_processor._preprocess_text(text, "en-US")
        assert ";" in processed or "[PAUSE:" in processed, "Ellipsis should be converted to semicolons or pause markers for natural TTS handling"
    
    def test_preprocess_text_with_non_ssml_ellipsis(self, non_ssml_lesson_processor):
        """Test that single ellipsis are converted to semicolons."""
        text = "Tubig? ... Oo, tubig."
        processed = non_ssml_lesson_processor._preprocess_text(text, "fil-PH")
        assert ";" in processed or "[PAUSE:" in processed, "Ellipsis should be converted to semicolons or pause markers for natural TTS handling"

    @pytest.mark.parametrize("input_text,expected_ssml,expected_non_ssml", [
        # Single ... should be converted to semicolons
        ("Hello... world", "Hello; world", "Hello; world"),
        ("One... two... three", "One; two; three", "One; two; three"),
        ("Start...middle...end", "Start;middle;end", "Start;middle;end"),
        ("Wait... ... continue", "Wait; ; continue", "Wait; ; continue"),
        (";", ";", ";"),
        ("Text...", "Text;", "Text;"),
        (";Text", ";Text", ";Text"),
        # Multiple .... should be preserved as-is
        ("Hello.... world", "Hello.... world", "Hello.... world"),
        ("Wait..... for it", "Wait..... for it", "Wait..... for it"),
    ])
    def test_multiple_ellipsis_patterns(self, input_text, expected_ssml, expected_non_ssml, ssml_lesson_processor, non_ssml_lesson_processor):
        """Test various patterns of ellipsis with both SSML and non-SSML processors."""
        # Test with SSML processor
        processed_ssml = ssml_lesson_processor._preprocess_text(input_text, "en-US")
        assert expected_ssml in processed_ssml, f"SSML preprocessing failed for: {input_text} (got: {processed_ssml})"
        
        # Test with non-SSML processor
        processed_non_ssml = non_ssml_lesson_processor._preprocess_text(input_text, "en-US")
        assert expected_non_ssml in processed_non_ssml, f"Non-SSML preprocessing failed for: {input_text} (got: {processed_non_ssml})"

    @pytest.mark.asyncio
    async def test_phrase_processing_with_ssml_ellipsis(self, ssml_lesson_processor, mock_tts_service, mock_audio_processor):
        """Test that phrase processing handles ellipsis correctly."""
        # Setup mock to return a dummy audio file
        mock_tts_service.synthesize_speech.return_value = (b"dummy_audio", False)
        
        # Create an AsyncMock for _synthesize_speech_with_retry
        mock_synthesize_with_retry = AsyncMock(return_value=(b"dummy_audio", False))
        ssml_lesson_processor._synthesize_speech_with_retry = mock_synthesize_with_retry
        
        # Create a phrase with an ellipsis
        phrase = Phrase(
            text="Wait for it...",
            translation="Maghintay ka...",
            language=Language.ENGLISH,
            section_type=SectionType.KEY_PHRASES,
            speaker="Person A"
        )
        
        # Process the phrase with a Path object
        output_dir = Path("output_dir")
        output_dir.mkdir(parents=True, exist_ok=True)
        await ssml_lesson_processor._process_phrase(phrase, output_dir)
        
        # Check that _synthesize_speech_with_retry was called with the processed text
        assert mock_synthesize_with_retry.call_count == 1, "_synthesize_speech_with_retry should be called once"
        
        # Get the arguments from the call
        call_args = mock_synthesize_with_retry.call_args_list[0]
        
        # The text is passed as a keyword argument
        tts_text = call_args.kwargs['text']
        assert ";" in tts_text, f"Ellipsis should be converted to semicolon in SSML mode, got: {tts_text}"

    @pytest.mark.asyncio
    async def test_phrase_processing_with_non_ssml_ellipsis(self, non_ssml_lesson_processor, mock_tts_service, mock_audio_processor):
        """Test that phrase processing handles ellipsis with semicolons."""
        # Setup mock to return a dummy audio file
        mock_tts_service.synthesize_speech.return_value = (b"dummy_audio", False)
        
        # Create an AsyncMock for _synthesize_speech_with_retry
        mock_synthesize_with_retry = AsyncMock(return_value=(b"dummy_audio", False))
        non_ssml_lesson_processor._synthesize_speech_with_retry = mock_synthesize_with_retry
        
        # Create a phrase with an ellipsis
        phrase = Phrase(
            text="Tubig? ... Oo, tubig.",
            translation="Water? ... Yes, water.",
            language=Language.TAGALOG,
            section_type=SectionType.KEY_PHRASES,
            speaker="Person A"
        )
        
        # Process the phrase with a Path object
        output_dir = Path("output_dir")
        output_dir.mkdir(parents=True, exist_ok=True)
        await non_ssml_lesson_processor._process_phrase(phrase, output_dir)
        
        # Verify the correct number of calls were made
        assert mock_synthesize_with_retry.call_count == 1, \
               "Should make exactly one call to _synthesize_speech_with_retry"
        
        # Get the arguments from the call
        call_args = mock_synthesize_with_retry.call_args_list[0]
        
        # The text is passed as a keyword argument
        tts_text = call_args.kwargs['text']
        
        # Should have semicolons instead of ellipsis
        assert ";" in tts_text, f"Ellipsis should be converted to semicolon, got: {tts_text}"
        # Should not have the original ellipsis
        assert "..." not in tts_text, f"Original ellipsis should be replaced, got: {tts_text}"
        
        # Verify no calls to synthesize_speech_with_pauses
        assert not hasattr(mock_tts_service, 'synthesize_speech_with_pauses') or \
               mock_tts_service.synthesize_speech_with_pauses.call_count == 0, \
               "Should not use pause-aware synthesis"
    
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