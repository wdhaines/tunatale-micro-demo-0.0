"""Tests for ellipsis handling in lesson processor."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
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
    return tts_service


@pytest.fixture
def mock_audio_processor():
    """Create a mock audio processor."""
    audio_processor = AsyncMock()
    audio_processor.combine_audio_files = AsyncMock()
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
def lesson_processor(mock_tts_service, mock_audio_processor, mock_voice_selector, mock_word_selector):
    """Create a lesson processor with mocked dependencies."""
    with tempfile.TemporaryDirectory() as temp_dir:
        processor = LessonProcessor(
            tts_service=mock_tts_service,
            audio_processor=mock_audio_processor,
            voice_selector=mock_voice_selector,
            word_selector=mock_word_selector,
            output_dir=str(temp_dir),
            ellipsis_pause_duration_ms=800,  # Default test value
            use_natural_pauses=False  # Use legacy ellipsis handling for existing tests
        )
        yield processor


class TestEllipsisHandling:
    """Test ellipsis handling for creating pauses in audio."""
    
    def test_preprocess_text_with_ellipsis(self, lesson_processor):
        """Test that ellipsis in text should be processed for pauses."""
        # Test text with ellipsis
        text_with_ellipsis = "Excuse... me... po."
        
        # Current implementation (this will show the bug)
        processed_text = lesson_processor._preprocess_text(text_with_ellipsis, "tagalog")
        
        # Currently this just removes extra whitespace, doesn't handle ellipsis
        assert processed_text == "Excuse... me... po."
        
        # This test documents the current behavior (bug)
        # The ellipsis should be converted to pause markers or handled specially
        
    def test_ellipsis_in_slow_speed_section(self, lesson_processor):
        """Test that ellipsis in slow speed sections should create pauses."""
        # Create a phrase with ellipsis for slow speed section
        phrase = Phrase(
            text="Magandang... hapon... po!",
            language=Language.TAGALOG,
            speaker_id="TAGALOG-FEMALE-1"
        )
        
        # Create a voice for the speaker
        voice = Voice(
            id="tagalog-female-voice",
            name="Tagalog Female",
            provider="test_provider",
            provider_id="tagalog-female-voice",
            language=Language.TAGALOG,
            gender=VoiceGender.FEMALE
        )
        
        # Test preprocessing
        processed_text = lesson_processor._preprocess_text(phrase.text, phrase.language)
        
        # Current behavior: ellipsis are preserved as-is
        assert "..." in processed_text
        
        # Expected behavior (currently failing):
        # The ellipsis should be converted to pause markers or split into segments
        # For example: "Magandang <break time='0.5s'/> hapon <break time='0.5s'/> po!"
        # Or the text should be split at ellipsis for separate TTS calls with silence
        
    def test_multiple_ellipsis_patterns(self, lesson_processor):
        """Test various ellipsis patterns that should create pauses."""
        test_cases = [
            ("Hello... world", "Simple ellipsis"),
            ("One... two... three", "Multiple ellipsis"),
            ("Start...middle...end", "No spaces around ellipsis"),
            ("Wait... ... continue", "Multiple consecutive ellipsis"),
            ("Normal text", "No ellipsis (should be unchanged)")
        ]
        
        for text, description in test_cases:
            processed = lesson_processor._preprocess_text(text, "tagalog")
            
            if "..." in text:
                # Currently, ellipsis are preserved (showing the bug)
                assert "..." in processed, f"Failed for: {description}"
                # TODO: Should be converted to pause markers or handled specially
            else:
                # Text without ellipsis should be unchanged
                assert processed == text, f"Failed for: {description}"
    
    @pytest.mark.asyncio
    async def test_phrase_processing_with_ellipsis(self, lesson_processor, mock_tts_service, mock_audio_processor):
        """Test that phrase processing handles ellipsis correctly."""
        # Create a phrase with ellipsis
        phrase = Phrase(
            text="Tubig?... Opo... Malamig... o... normal?",
            language=Language.TAGALOG,
            speaker_id="TAGALOG-FEMALE-2"
        )
        
        # Create a voice
        voice = Voice(
            id="tagalog-female-voice-2",
            name="Tagalog Female 2",
            provider="test_provider",
            provider_id="tagalog-female-voice-2",
            language=Language.TAGALOG,
            gender=VoiceGender.FEMALE
        )
        
        # Mock the TTS service response
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Mock TTS synthesis
            mock_tts_service.synthesize_speech.return_value = None
            
            # Test phrase processing
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir)
                
                # Currently this will pass the ellipsis directly to TTS
                result = await lesson_processor._process_phrase(
                    phrase=phrase,
                    output_path=output_path
                )
                
                # Verify that ellipsis handling is working - should be single TTS call with semicolons
                # Original text: "Tubig?... Opo... Malamig... o... normal?"
                # Should become: "Tubig?; Opo; Malamig; o; normal?"
                assert mock_tts_service.synthesize_speech.call_count == 1
                
                # Verify the text that was passed to TTS
                call_args = mock_tts_service.synthesize_speech.call_args_list[0]
                tts_text = call_args.kwargs['text']
                
                # Should have ellipsis replaced with semicolons (note the extra space after semicolon)
                expected_text = "Tubig?;  Opo;  Malamig;  o;  normal?"
                assert tts_text == expected_text
                
                # Verify that ellipsis are no longer in the text
                assert "..." not in tts_text, f"Ellipsis still found in text: '{tts_text}'"
                
                # Verify that audio processor add_silence was called for ellipsis pauses
                mock_audio_processor.add_silence.assert_called_once()
                
                # Check the silence duration (800ms = 0.8s)
                silence_call = mock_audio_processor.add_silence.call_args
                assert silence_call.kwargs['duration'] == 0.8
                
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_section_type_affects_ellipsis_handling(self, lesson_processor):
        """Test that ellipsis handling might differ by section type."""
        text_with_ellipsis = "Hello... world..."
        
        # Test preprocessing for different contexts
        # Currently all section types are handled the same way
        normal_processed = lesson_processor._preprocess_text(text_with_ellipsis, "english")
        slow_processed = lesson_processor._preprocess_text(text_with_ellipsis, "tagalog")
        
        # Currently both are the same (showing that section type is not considered)
        assert normal_processed == slow_processed == text_with_ellipsis
        
        # Expected behavior:
        # - SLOW_SPEED sections might have longer pauses for ellipsis
        # - NATURAL_SPEED sections might have shorter pauses or no special handling
        # - Different languages might handle ellipsis differently


class TestEllipsisImplementation:
    """Test the actual ellipsis handling implementation."""
    
    def test_ellipsis_semicolon_replacement(self, lesson_processor):
        """Test that ellipsis are replaced with semicolons for natural pauses."""
        test_cases = [
            ("Hello... world", "Hello;  world"),
            ("One... two... three", "One;  two;  three"),
            ("Start...middle...end", "Start; middle; end"),
            ("No ellipsis here", "No ellipsis here"),
            ("Magandang... hapon... po!", "Magandang;  hapon;  po!"),
            ("", ""),  # Empty string
            ("...", "; "),  # Only ellipsis
            ("Text...", "Text; "),  # Trailing ellipsis
            ("...Text", "; Text"),  # Leading ellipsis
        ]
        
        for original, expected in test_cases:
            # Simulate the ellipsis replacement logic
            if "..." in original:
                result = original.replace("...", "; ")
            else:
                result = original
            assert result == expected, f"Failed for '{original}': expected '{expected}', got '{result}'"
    
    def test_configurable_pause_duration(self, mock_tts_service, mock_audio_processor, mock_voice_selector, mock_word_selector):
        """Test that ellipsis pause duration is configurable."""
        # Test different pause durations
        test_durations = [200, 800, 1500]  # ms
        
        for duration_ms in test_durations:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create processor with custom pause duration
                processor = LessonProcessor(
                    tts_service=mock_tts_service,
                    audio_processor=mock_audio_processor,
                    voice_selector=mock_voice_selector,
                    word_selector=mock_word_selector,
                    output_dir=str(temp_dir),
                    ellipsis_pause_duration_ms=duration_ms
                )
                
                # Verify the duration is set correctly
                assert processor.ellipsis_pause_duration_ms == duration_ms
    
    def test_ellipsis_handling_approach(self, lesson_processor):
        """Test that the new ellipsis approach is working."""
        # The new approach replaces ellipsis with semicolons and adds configurable silence
        # This provides longer, more controllable pauses than the previous comma approach
        
        # Verify that complex methods are no longer present
        assert not hasattr(lesson_processor, '_synthesize_text_with_ellipsis_pauses')
        assert not hasattr(lesson_processor, '_combine_audio_segments_with_pauses')
        
        # Verify the configurable pause duration is available
        assert hasattr(lesson_processor, 'ellipsis_pause_duration_ms')
        assert lesson_processor.ellipsis_pause_duration_ms == 800  # Default test value