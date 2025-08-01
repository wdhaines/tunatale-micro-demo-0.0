"""Test pause marker processing to ensure TTS services receive clean text."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from tunatale.core.services.lesson_processor import LessonProcessor
from tunatale.core.models.phrase import Phrase
from tunatale.core.models.enums import Language, VoiceGender


class MockTTSService:
    """Mock TTS service that captures the text sent to it."""
    
    def __init__(self):
        self.synthesized_texts = []
        self.call_count = 0
        
    async def synthesize_speech(self, text, voice_id, output_path, **kwargs):
        """Mock synthesis that captures the text."""
        self.synthesized_texts.append(text)
        self.call_count += 1
        # Create a dummy audio file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake audio data")
        
    async def synthesize_speech_with_pauses(self, text, voice_id, output_path, **kwargs):
        """Mock pause-aware synthesis that captures the text."""
        # This should split the text and call synthesize_speech for each segment
        from tunatale.core.utils.tts_preprocessor import split_text_with_pauses
        segments = split_text_with_pauses(text)
        
        for segment_text, pause_duration in segments:
            if segment_text.strip():
                await self.synthesize_speech(segment_text, voice_id, output_path, **kwargs)
        
    async def validate_voice(self, voice_id):
        """Mock voice validation."""
        return True


class MockAudioProcessor:
    """Mock audio processor."""
    
    async def process_audio(self, input_file, output_file, **kwargs):
        return Path(output_file)
    
    async def add_silence(self, input_file, output_file, **kwargs):
        return Path(output_file)


@pytest.mark.asyncio
async def test_pause_markers_not_sent_to_tts():
    """Test that pause markers are NOT sent as literal text to TTS services.
    
    This test should FAIL initially, demonstrating the current bug where
    pause markers like [PAUSE:1s] are being sent as literal text to TTS.
    """
    # Create mock services
    mock_tts = MockTTSService()
    mock_audio = MockAudioProcessor()
    
    # Create lesson processor
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir)
        
        # Create mock voice_selector and word_selector
        mock_voice_selector = MagicMock()
        mock_word_selector = MagicMock()
        
        processor = LessonProcessor(
            tts_service=mock_tts,
            audio_processor=mock_audio,
            voice_selector=mock_voice_selector,
            word_selector=mock_word_selector,
            output_dir=str(output_path)
        )
        
        # Create a phrase with ellipses that should generate pause markers
        phrase = Phrase(
            text="Magkano... po?",
            translation="How much is it?",
            language=Language.TAGALOG,
            voice_id="TAGALOG-FEMALE-1"
        )
        
        # Process the phrase
        output_path = output_path / "phrases"
        output_path.mkdir(exist_ok=True)
        await processor._process_phrase(phrase, output_path)
        
        # Check what text was actually sent to TTS
        assert len(mock_tts.synthesized_texts) > 0, "TTS should have been called"
        
        # The critical test: TTS should NOT receive pause markers as literal text
        for text in mock_tts.synthesized_texts:
            assert "[PAUSE:" not in text, f"TTS received literal pause marker: '{text}'"
            assert "PAUSE" not in text, f"TTS received text containing 'PAUSE': '{text}'"
        
        # TTS should receive clean text segments
        expected_segments = ["Magkano", "po?"]
        actual_texts = [text.strip() for text in mock_tts.synthesized_texts if text.strip()]
        
        # This assertion will help us see what's actually being sent
        print(f"Expected clean segments: {expected_segments}")
        print(f"Actual texts sent to TTS: {actual_texts}")
        
        # Check that we got clean text segments
        assert any("Magkano" in text for text in actual_texts), "Should contain 'Magkano' segment"
        assert any("po?" in text for text in actual_texts), "Should contain 'po?' segment"


@pytest.mark.asyncio 
async def test_direct_ssml_markers_not_sent_to_tts():
    """Test that direct SSML markers are converted and not sent as literal text."""
    # Create mock services
    mock_tts = MockTTSService()
    mock_audio = MockAudioProcessor()
    
    # Create lesson processor
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir)
        
        # Create mock voice_selector and word_selector
        mock_voice_selector = MagicMock()
        mock_word_selector = MagicMock()
        
        processor = LessonProcessor(
            tts_service=mock_tts,
            audio_processor=mock_audio,
            voice_selector=mock_voice_selector,
            word_selector=mock_word_selector,
            output_dir=str(output_path)
        )
        
        # Create a phrase with direct SSML that should be converted to pause markers
        phrase = Phrase(
            text='Magkano <break time="1s"/> po?',
            translation="How much is it?",
            language=Language.TAGALOG,
            voice_id="TAGALOG-FEMALE-1"
        )
        
        # Process the phrase
        output_path = output_path / "phrases"
        output_path.mkdir(exist_ok=True)
        await processor._process_phrase(phrase, output_path)
        
        # Check what text was actually sent to TTS
        assert len(mock_tts.synthesized_texts) > 0, "TTS should have been called"
        
        # The critical test: TTS should NOT receive SSML or pause markers
        for text in mock_tts.synthesized_texts:
            assert "<break" not in text, f"TTS received literal SSML: '{text}'"
            assert "[PAUSE:" not in text, f"TTS received literal pause marker: '{text}'"
            assert "PAUSE" not in text, f"TTS received text containing 'PAUSE': '{text}'"
        
        print(f"Texts sent to TTS: {mock_tts.synthesized_texts}")


def test_text_processing_pipeline():
    """Test the complete text processing pipeline to ensure clean output."""
    from tunatale.core.utils.tts_preprocessor import enhanced_preprocess_text_for_tts, split_text_with_pauses
    
    print("=== Testing Text Processing Pipeline ===")
    
    # Test 1: Ellipses conversion
    original_text = "Magkano... po?"
    print(f"Original text: '{original_text}'")
    
    # Enhanced preprocessing
    processed_text, ssml_result = enhanced_preprocess_text_for_tts(
        text=original_text,
        language_code='tl-PH',
        provider_name='edge_tts',
        supports_ssml=False
    )
    print(f"After preprocessing: '{processed_text}'")
    
    # Lesson processor logic
    has_ellipsis = "..." in original_text
    has_pause_markers = "[PAUSE:" in processed_text
    
    if has_pause_markers:
        tts_text = processed_text
    elif has_ellipsis:
        tts_text = processed_text.replace("...", "; ")
    else:
        tts_text = processed_text
    
    print(f"Final TTS text: '{tts_text}'")
    
    # Pause-aware synthesis
    if "[PAUSE:" in tts_text:
        segments = split_text_with_pauses(tts_text)
        clean_segments = [seg[0] for seg in segments if seg[0].strip()]
        print(f"Clean segments for TTS: {clean_segments}")
        
        # Verify no pause markers in clean segments
        for segment in clean_segments:
            assert "[PAUSE:" not in segment, f"Clean segment contains pause marker: '{segment}'"
            assert "PAUSE" not in segment, f"Clean segment contains 'PAUSE': '{segment}'"
        
        print("✅ SUCCESS: All segments are clean!")
        return True
    else:
        print("❌ No pause markers detected")
        return False


if __name__ == "__main__":
    print("Running text processing pipeline test...")
    success = test_text_processing_pipeline()
    
    if success:
        print("\n🎉 FIXED: Pause markers are properly processed!")
    else:
        print("\n❌ ISSUE: Pause markers not being processed correctly")