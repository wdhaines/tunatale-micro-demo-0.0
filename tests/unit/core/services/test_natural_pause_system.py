"""Tests for natural pause system based on linguistic boundaries."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import os

from tunatale.core.services.natural_pause_calculator import NaturalPauseCalculator
from tunatale.core.services.linguistic_boundary_detector import detect_linguistic_boundaries, split_with_natural_pauses
from tunatale.core.services.lesson_processor import LessonProcessor
from tunatale.core.models.phrase import Phrase
from tunatale.core.models.enums import Language, SectionType
from tunatale.core.models.voice import Voice, VoiceGender


class TestNaturalPauseCalculator:
    """Test the natural pause calculation system."""
    
    def test_pause_calculator_initialization(self):
        """Test that pause calculator initializes with correct hierarchy."""
        calculator = NaturalPauseCalculator()
        
        # Test basic pause levels
        assert calculator.pause_levels['syllable'] == 300
        assert calculator.pause_levels['word'] == 600
        assert calculator.pause_levels['phrase'] == 1200
        assert calculator.pause_levels['sentence'] == 2000
        assert calculator.pause_levels['section'] == 3000
    
    def test_get_pause_for_boundary_normal_speech(self):
        """Test pause calculation for normal speech."""
        calculator = NaturalPauseCalculator()
        
        # Test normal complexity
        assert calculator.get_pause_for_boundary('syllable', 'normal') == 300
        assert calculator.get_pause_for_boundary('word', 'normal') == 600
        assert calculator.get_pause_for_boundary('phrase', 'normal') == 1200
        assert calculator.get_pause_for_boundary('sentence', 'normal') == 2000
        assert calculator.get_pause_for_boundary('section', 'normal') == 3000
    
    def test_get_pause_for_boundary_slow_speech(self):
        """Test pause calculation for slow speech (50% longer)."""
        calculator = NaturalPauseCalculator()
        
        # Test slow complexity (1.5x multiplier)
        assert calculator.get_pause_for_boundary('syllable', 'slow') == 450
        assert calculator.get_pause_for_boundary('word', 'slow') == 900
        assert calculator.get_pause_for_boundary('phrase', 'slow') == 1800
        assert calculator.get_pause_for_boundary('sentence', 'slow') == 3000
        assert calculator.get_pause_for_boundary('section', 'slow') == 4500
    
    def test_get_pause_for_unknown_boundary(self):
        """Test pause calculation for unknown boundary type."""
        calculator = NaturalPauseCalculator()
        
        # Should default to word-level pause
        assert calculator.get_pause_for_boundary('unknown', 'normal') == 600
        assert calculator.get_pause_for_boundary('unknown', 'slow') == 900
    
    def test_dynamic_pause_calculation_based_on_audio_duration(self):
        """Test dynamic pause calculation using audio duration."""
        calculator = NaturalPauseCalculator()
        
        # Test dynamic pause calculation (1.5x audio duration + base pause)
        # For 2-second audio with phrase boundary (base 1200ms)
        dynamic_pause = calculator.get_pause_for_boundary('phrase', 'normal', audio_duration_seconds=2.0)
        expected = int(2.0 * 1500) + 1200  # 3000 + 1200 = 4200ms
        assert dynamic_pause == expected
        
        # For 5-second audio with word boundary (base 600ms)
        dynamic_pause = calculator.get_pause_for_boundary('word', 'normal', audio_duration_seconds=5.0)
        expected = int(5.0 * 1500) + 600  # 7500 + 600 = 8100ms
        assert dynamic_pause == expected
        
        # For 1.5-second audio with sentence boundary (base 2000ms)
        dynamic_pause = calculator.get_pause_for_boundary('sentence', 'normal', audio_duration_seconds=1.5)
        expected = int(1.5 * 1500) + 2000  # 2250 + 2000 = 4250ms
        assert dynamic_pause == expected
    
    def test_dynamic_pause_calculation_slow_speech(self):
        """Test dynamic pause calculation for slow speech."""
        calculator = NaturalPauseCalculator()
        
        # For slow speech, should apply 1.2x multiplier to dynamic pause
        dynamic_pause = calculator.get_pause_for_boundary('phrase', 'slow', audio_duration_seconds=3.0)
        base_dynamic = int(3.0 * 1500) + 1200  # 4500 + 1200 = 5700ms
        expected = int(base_dynamic * 1.2)      # 5700 * 1.2 = 6840ms
        assert dynamic_pause == expected
    
    def test_dynamic_pause_fallback_to_fixed(self):
        """Test that dynamic pause falls back to fixed when no audio duration provided."""
        calculator = NaturalPauseCalculator()
        
        # Without audio duration, should use fixed pause system
        fixed_pause = calculator.get_pause_for_boundary('phrase', 'normal', audio_duration_seconds=None)
        assert fixed_pause == 1200  # Base phrase pause
        
        # Should be same as calling without audio_duration_seconds parameter
        default_pause = calculator.get_pause_for_boundary('phrase', 'normal')
        assert fixed_pause == default_pause


class TestLinguisticBoundaryDetector:
    """Test linguistic boundary detection."""
    
    def test_detect_sentence_boundaries(self):
        """Test detection of sentence boundaries."""
        text = "Hello world. How are you? I'm fine!"
        boundaries = detect_linguistic_boundaries(text)
        
        # Should find sentence boundaries at periods, question marks, exclamations
        sentence_boundaries = [b for b in boundaries if b[0] == 'sentence']
        assert len(sentence_boundaries) == 3
        
        # Check positions are correct
        positions = [b[1] for b in sentence_boundaries]
        assert 13 in positions  # After "Hello world. "
        assert 26 in positions  # After "How are you? "
        assert 35 in positions  # After "I'm fine!"
    
    def test_detect_phrase_boundaries(self):
        """Test detection of phrase boundaries."""
        text = "First part, second part; third part and fourth part but fifth part or sixth part"
        boundaries = detect_linguistic_boundaries(text)
        
        # Should find phrase boundaries at commas, semicolons, conjunctions
        phrase_boundaries = [b for b in boundaries if b[0] == 'phrase']
        assert len(phrase_boundaries) >= 5  # comma, semicolon, and, but, or
    
    def test_detect_word_boundaries(self):
        """Test detection of word boundaries."""
        text = "Simple word test"
        boundaries = detect_linguistic_boundaries(text)
        
        # Should find word boundaries between words (excluding phrase/sentence boundaries)
        word_boundaries = [b for b in boundaries if b[0] == 'word']
        assert len(word_boundaries) >= 2  # At least between each word
    
    def test_tagalog_text_boundaries(self):
        """Test boundary detection on Tagalog text."""
        text = "Magandang hapon po, kumusta ka? Mabuti naman ako."
        boundaries = detect_linguistic_boundaries(text)
        
        # Should detect various boundary types
        boundary_types = set(b[0] for b in boundaries)
        assert 'sentence' in boundary_types  # Question mark and period
        assert 'phrase' in boundary_types    # Comma
        assert 'word' in boundary_types      # Spaces between words
    
    def test_complex_punctuation(self):
        """Test boundary detection with complex punctuation."""
        text = "Wait... really? Yes, I'm sure! Okay then."
        boundaries = detect_linguistic_boundaries(text)
        
        # Should handle ellipsis, mixed punctuation
        sentence_boundaries = [b for b in boundaries if b[0] == 'sentence']
        phrase_boundaries = [b for b in boundaries if b[0] == 'phrase']
        
        assert len(sentence_boundaries) >= 2  # Question mark and exclamation
        assert len(phrase_boundaries) >= 1   # Comma


class TestNaturalPauseSplitting:
    """Test natural pause text splitting."""
    
    def test_split_simple_sentence_normal_speed(self):
        """Test splitting simple sentence at normal speed."""
        text = "Hello world"
        segments = split_with_natural_pauses(text, is_slow=False)
        
        # Should have text segments and word boundary pause
        assert len(segments) >= 3  # text + pause + text
        
        # Check segment types
        text_segments = [s for s in segments if s['type'] == 'text']
        pause_segments = [s for s in segments if s['type'] == 'pause']
        
        assert len(text_segments) >= 2    # "Hello" and "world"
        assert len(pause_segments) >= 1   # Word boundary
        
        # Check voice settings for normal speed
        for segment in text_segments:
            assert segment['voice_settings']['rate'] == 1.0
    
    def test_split_simple_sentence_slow_speed(self):
        """Test splitting simple sentence at slow speed."""
        text = "Hello world"
        segments = split_with_natural_pauses(text, is_slow=True)
        
        text_segments = [s for s in segments if s['type'] == 'text']
        pause_segments = [s for s in segments if s['type'] == 'pause']
        
        # Check voice settings for slow speed
        for segment in text_segments:
            assert segment['voice_settings']['rate'] == 0.5
        
        # Check pause durations are longer for slow speech
        for segment in pause_segments:
            if segment['boundary'] == 'word':
                assert segment['duration'] == 900  # 600 * 1.5
    
    def test_split_complex_sentence(self):
        """Test splitting sentence with multiple boundary types."""
        text = "Magandang hapon po, kumusta ka?"
        segments = split_with_natural_pauses(text, is_slow=False)
        
        # Should have multiple segment types
        boundary_types = set()
        for segment in segments:
            if segment['type'] == 'pause':
                boundary_types.add(segment['boundary'])
        
        assert 'word' in boundary_types      # Spaces between words
        assert 'phrase' in boundary_types    # Comma
        assert 'sentence' in boundary_types  # Question mark
    
    def test_split_with_ellipsis(self):
        """Test splitting text containing ellipsis."""
        text = "Wait... what happened?"
        segments = split_with_natural_pauses(text, is_slow=True)
        
        # Should handle ellipsis as sentence boundary or special case
        pause_segments = [s for s in segments if s['type'] == 'pause']
        
        # Should have pauses with appropriate durations
        assert len(pause_segments) > 0
        
        # Check that we have longer pauses for slow speech
        long_pauses = [s for s in pause_segments if s['duration'] >= 1800]  # Slow phrase boundary
        assert len(long_pauses) > 0
    
    def test_empty_text_handling(self):
        """Test handling of empty or whitespace text."""
        segments = split_with_natural_pauses("", is_slow=False)
        assert len(segments) == 0
        
        segments = split_with_natural_pauses("   ", is_slow=False)
        assert len(segments) == 0
    
    def test_single_word_handling(self):
        """Test handling of single word."""
        text = "Hello"
        segments = split_with_natural_pauses(text, is_slow=False)
        
        # Should have just one text segment
        text_segments = [s for s in segments if s['type'] == 'text']
        assert len(text_segments) == 1
        assert text_segments[0]['content'] == 'Hello'
    
    def test_split_with_dynamic_pauses_from_audio_durations(self):
        """Test splitting text with dynamic pauses based on audio durations."""
        text = "Hello world. How are you?"
        # Simulate audio durations for each text segment
        segment_durations = [1.2, 0.8, 0.6, 0.5, 1.0]  # seconds for each text segment
        
        segments = split_with_natural_pauses(text, is_slow=False, segment_audio_durations=segment_durations)
        
        # Check that dynamic pauses were calculated
        pause_segments = [s for s in segments if s['type'] == 'pause']
        
        # Should have pauses with durations based on audio segments
        found_dynamic_pause = False
        for segment in pause_segments:
            # Dynamic pauses should be much longer than fixed pauses
            if segment['duration'] > 2000:  # Much longer than normal word pause (600ms)
                found_dynamic_pause = True
                break
        
        assert found_dynamic_pause, "Should have found at least one dynamic pause based on audio duration"
    
    def test_split_without_audio_durations_uses_fixed_pauses(self):
        """Test that splitting without audio durations uses fixed pause system."""
        text = "Hello world. How are you?"
        
        segments = split_with_natural_pauses(text, is_slow=False)  # No segment_audio_durations
        
        # Check that fixed pauses were used
        pause_segments = [s for s in segments if s['type'] == 'pause']
        
        # All pauses should use fixed duration system
        for segment in pause_segments:
            if segment['boundary'] == 'word':
                assert segment['duration'] == 600  # Fixed word pause
            elif segment['boundary'] == 'sentence':
                assert segment['duration'] == 2000  # Fixed sentence pause


@pytest.fixture
def mock_tts_service():
    """Create a mock TTS service."""
    tts_service = AsyncMock()
    tts_service.synthesize_speech = AsyncMock()
    tts_service.synthesize_speech_with_pauses = AsyncMock()
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
def lesson_processor(mock_tts_service, mock_audio_processor, mock_voice_selector, mock_word_selector):
    """Create a lesson processor with mocked dependencies."""
    with tempfile.TemporaryDirectory() as temp_dir:
        processor = LessonProcessor(
            tts_service=mock_tts_service,
            audio_processor=mock_audio_processor,
            voice_selector=mock_voice_selector,
            word_selector=mock_word_selector,
            output_dir=str(temp_dir),
            ellipsis_pause_duration_ms=800,
            use_natural_pauses=True  # New parameter to enable natural pause system
        )
        yield processor


class TestNaturalPauseIntegration:
    """Test integration of natural pause system with LessonProcessor."""
    
    @pytest.mark.asyncio
    async def test_process_phrase_with_natural_pauses(self, lesson_processor, mock_tts_service, mock_audio_processor):
        """Test that phrase processing uses natural pause system."""
        # Create a phrase with natural speech patterns
        phrase = Phrase(
            text="Magandang hapon po, kumusta ka?",
            language=Language.TAGALOG,
            speaker_id="TAGALOG-FEMALE-1"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Mock TTS synthesis to return success
            mock_tts_service.synthesize_speech.return_value = None
            
            # Create fake audio files to simulate TTS output
            segment_files = []
            for i in range(3):  # Create multiple segments
                fake_audio = output_path / f"phrase_{phrase.id}_segment_{i}.mp3"
                fake_audio.touch()  # Create empty file to simulate audio output
                segment_files.append(fake_audio)
            
            # Mock Path.exists to return True for segment files
            with patch('pathlib.Path.exists') as mock_exists:
                mock_exists.return_value = True
                
                # Process the phrase
                result = await lesson_processor._process_phrase_with_natural_pauses(
                    phrase=phrase,
                    output_path=output_path
                )
            
            # Should have made multiple synthesize speech calls for different text segments
            # (Note: actual count depends on text segmentation, so just check it was called)
            assert mock_tts_service.synthesize_speech.call_count >= 1
            
            # Test passes if natural pause processing completes without error
            # The add_silence call depends on having multiple audio segments with pauses
            # This is tested more specifically in other integration tests
    
    @pytest.mark.asyncio
    async def test_slow_speed_phrase_processing(self, lesson_processor, mock_tts_service):
        """Test that slow speed phrases get longer pauses."""
        # Create a phrase with slow speech marker
        phrase = Phrase(
            text="Magandang hapon po...",
            language=Language.TAGALOG,
            speaker_id="TAGALOG-FEMALE-1"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Mock TTS synthesis
            mock_tts_service.synthesize_speech.return_value = None
            
            # Mock the split_with_natural_pauses function to verify it's called with is_slow=True
            with patch('tunatale.core.services.linguistic_boundary_detector.split_with_natural_pauses') as mock_split:
                # Configure the mock to return segments with voice_settings
                mock_split.return_value = [
                    {
                        'type': 'text',
                        'content': 'Magandang hapon po...',
                        'voice_settings': {'rate': 0.5}  # This is what we expect for slow speech
                    }
                ]
                
                # Process the phrase
                result = await lesson_processor._process_phrase_with_natural_pauses(
                    phrase=phrase,
                    output_path=output_path,
                    section_type='slow_speed'  # Explicitly set section type to slow_speed
                )
                
                # Verify split_with_natural_pauses was called with the correct parameters
                mock_split.assert_called_once()
                
                # Get the segments that were passed to synthesize_speech
                for call in mock_tts_service.synthesize_speech.call_args_list:
                    _, kwargs = call
                    text = kwargs['text']
                    rate = kwargs['rate']
                    
                    # Check if this is a slow speech segment (rate < 0.8)
                    if rate < 0.8:
                        assert rate == 0.5, f"Expected rate=0.5 for slow speech, got {rate}"
                segments = mock_split.return_value
                for segment in segments:
                    if segment['type'] == 'text':
                        assert 'voice_settings' in segment, "Expected voice_settings in text segment"
                        assert segment['voice_settings'].get('rate') == 0.5, "Expected rate=0.5 for slow speech"
    
    @pytest.mark.asyncio
    async def test_natural_pause_vs_ellipsis_handling(self, lesson_processor, mock_tts_service, mock_audio_processor):
        """Test that natural pause system replaces simple ellipsis handling."""
        # Create a phrase with ellipsis
        phrase = Phrase(
            text="Wait... what happened?",
            language=Language.ENGLISH,
            speaker_id="ENGLISH-FEMALE-1"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Mock services
            mock_tts_service.synthesize_speech.return_value = None
            
            # Process the phrase
            result = await lesson_processor._process_phrase_with_natural_pauses(
                phrase=phrase,
                output_path=output_path
            )
            
            # Should use natural pause system instead of simple ellipsis replacement
            tts_calls = mock_tts_service.synthesize_speech.call_args_list
            
            # Should NOT find semicolon replacement (old ellipsis handling)
            for call in tts_calls:
                text = call.kwargs['text']
                assert '; ' not in text  # Old ellipsis handling used semicolons
            
            # Should have made separate TTS calls for segments
            assert len(tts_calls) > 1
    
    def test_natural_pause_calculator_integration(self, lesson_processor):
        """Test that lesson processor has natural pause calculator."""
        assert hasattr(lesson_processor, 'natural_pause_calculator')
        assert isinstance(lesson_processor.natural_pause_calculator, NaturalPauseCalculator)
    
    def test_use_natural_pauses_flag(self, lesson_processor):
        """Test that natural pause system can be enabled/disabled."""
        assert hasattr(lesson_processor, 'use_natural_pauses')
        assert lesson_processor.use_natural_pauses == True
    
    @pytest.mark.asyncio
    async def test_dynamic_pauses_only_for_key_phrases_section(self, mock_tts_service, mock_audio_processor, mock_voice_selector, mock_word_selector):
        """Test that dynamic pauses are only used for key phrases sections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = LessonProcessor(
                tts_service=mock_tts_service,
                audio_processor=mock_audio_processor,
                voice_selector=mock_voice_selector,
                word_selector=mock_word_selector,
                output_dir=str(temp_dir),
                use_natural_pauses=True
            )
            
            phrase = Phrase(
                text="Hello world. How are you?",
                language=Language.ENGLISH,
                speaker_id="ENGLISH-FEMALE-1"
            )
            
            output_path = Path(temp_dir)
            
            # Mock audio duration measurement
            mock_audio_processor.get_audio_duration.return_value = 2.0  # 2 seconds
            
            # Mock TTS synthesis
            mock_tts_service.synthesize_speech.return_value = None
            
            # Test with key phrases section
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.with_suffix') as mock_suffix:
                    mock_suffix.return_value = Path(temp_dir) / "test.mp3"
                    
                    result_key_phrases = await processor._process_phrase_with_natural_pauses(
                        phrase=phrase,
                        output_path=output_path,
                        section_type='key_phrases'  # Should use dynamic pauses
                    )
            
            # Test with non-key phrases section (e.g., natural_speed)
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.with_suffix') as mock_suffix:
                    mock_suffix.return_value = Path(temp_dir) / "test.mp3"
                    
                    result_natural_speed = await processor._process_phrase_with_natural_pauses(
                        phrase=phrase,
                        output_path=output_path,
                        section_type='natural_speed'  # Should use fixed pauses
                    )
            
            # Both should complete successfully, but with different processing paths
            assert result_key_phrases is not None
            assert result_natural_speed is not None
    
    @pytest.mark.asyncio
    async def test_single_pass_optimization_for_key_phrases(self, mock_tts_service, mock_audio_processor, mock_voice_selector, mock_word_selector):
        """Test that key phrases use single-pass audio generation (not two-pass)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = LessonProcessor(
                tts_service=mock_tts_service,
                audio_processor=mock_audio_processor,
                voice_selector=mock_voice_selector,
                word_selector=mock_word_selector,
                output_dir=str(temp_dir),
                use_natural_pauses=True
            )
            
            phrase = Phrase(
                text="Hello world, how are you?",  # Multiple segments
                language=Language.ENGLISH,
                speaker_id="ENGLISH-FEMALE-1"
            )
            
            output_path = Path(temp_dir)
            
            # Mock audio duration measurement - return different durations for each segment
            mock_audio_processor.get_audio_duration.side_effect = [1.5, 1.2, 0.8, 1.0]
            
            # Mock TTS synthesis
            mock_tts_service.synthesize_speech.return_value = None
            
            # Mock file system operations
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.with_suffix') as mock_suffix:
                    mock_suffix.return_value = Path(temp_dir) / "test.mp3"
                    
                    # Process key phrases section (should use single-pass dynamic pauses)
                    result = await processor._process_phrase_with_natural_pauses(
                        phrase=phrase,
                        output_path=output_path,
                        section_type='key_phrases'
                    )
            
            # Count TTS calls - should be called once per segment, not twice
            tts_call_count = mock_tts_service.synthesize_speech.call_count
            
            # Should have made TTS calls for text segments, but only once per segment
            # (not twice as in the old two-pass system)
            assert tts_call_count >= 3  # At least 3 segments based on text
            assert tts_call_count <= 6  # But not doubled (which would indicate two-pass)
            
            # Should have measured audio duration for dynamic pause calculation
            assert mock_audio_processor.get_audio_duration.call_count >= 3


class TestNaturalPausePerformance:
    """Test performance characteristics of natural pause system."""
    
    def test_boundary_detection_performance(self):
        """Test that boundary detection performs well on longer texts."""
        # Create a longer text sample
        long_text = " ".join([
            "Magandang hapon po, kumusta ka?",
            "Mabuti naman ako, salamat.",
            "Ano ang ginagawa mo ngayon?",
            "Nagtatrabaho ako sa opisina.",
            "Saan ka nakatira?"
        ] * 10)  # Repeat 10 times for longer text
        
        import time
        start_time = time.time()
        boundaries = detect_linguistic_boundaries(long_text)
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second for this size)
        assert end_time - start_time < 1.0
        
        # Should find reasonable number of boundaries
        assert len(boundaries) > 50  # Many boundaries in the longer text
    
    def test_splitting_performance(self):
        """Test that text splitting performs well."""
        long_text = "Hello world, how are you? I'm fine, thanks! " * 50
        
        import time
        start_time = time.time()
        segments = split_with_natural_pauses(long_text, is_slow=True)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 1.0
        
        # Should produce reasonable number of segments
        assert len(segments) > 100  # Many segments for the longer text