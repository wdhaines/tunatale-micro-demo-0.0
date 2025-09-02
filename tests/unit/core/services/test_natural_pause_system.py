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
        assert calculator.pause_levels['syllable'] == 300
        assert calculator.pause_levels['word'] == 600
        assert calculator.pause_levels['phrase'] == 1200
        assert calculator.pause_levels['sentence'] == 2000
        assert calculator.pause_levels['section'] == 3000

    def test_get_pause_for_boundary_normal_speech(self):
        """Test pause calculation for normal speech."""
        calculator = NaturalPauseCalculator()
        assert calculator.get_pause_for_boundary('syllable', 'normal') == 300
        assert calculator.get_pause_for_boundary('word', 'normal') == 600
        assert calculator.get_pause_for_boundary('phrase', 'normal') == 1200
        assert calculator.get_pause_for_boundary('sentence', 'normal') == 2000
        assert calculator.get_pause_for_boundary('section', 'normal') == 3000

    def test_get_pause_for_boundary_slow_speech(self):
        """Test pause calculation for slow speech (50% longer)."""
        calculator = NaturalPauseCalculator()
        assert calculator.get_pause_for_boundary('syllable', 'slow') == 450
        assert calculator.get_pause_for_boundary('word', 'slow') == 900
        assert calculator.get_pause_for_boundary('phrase', 'slow') == 1800
        assert calculator.get_pause_for_boundary('sentence', 'slow') == 3000
        assert calculator.get_pause_for_boundary('section', 'slow') == 4500

    def test_get_pause_for_unknown_boundary(self):
        """Test pause calculation for unknown boundary type."""
        calculator = NaturalPauseCalculator()
        assert calculator.get_pause_for_boundary('unknown', 'normal') == 600
        assert calculator.get_pause_for_boundary('unknown', 'slow') == 900

    def test_dynamic_pause_calculation_based_on_audio_duration(self):
        """Test dynamic pause calculation using audio duration."""
        calculator = NaturalPauseCalculator()
        dynamic_pause = calculator.get_pause_for_boundary('phrase', 'normal', audio_duration_seconds=2.0)
        expected = max(0, int(2.0 * 1000) - 500)
        assert dynamic_pause == expected

    def test_dynamic_pause_calculation_slow_speech(self):
        """Test dynamic pause calculation for slow speech."""
        calculator = NaturalPauseCalculator()
        dynamic_pause = calculator.get_pause_for_boundary('phrase', 'slow', audio_duration_seconds=3.0)
        base_dynamic = max(0, int(3.0 * 1000) - 500)
        expected = int(base_dynamic * 1.2)
        assert dynamic_pause == expected

    def test_dynamic_pause_fallback_to_fixed(self):
        """Test that dynamic pause falls back to fixed when no audio duration provided."""
        calculator = NaturalPauseCalculator()
        fixed_pause = calculator.get_pause_for_boundary('phrase', 'normal', audio_duration_seconds=None)
        assert fixed_pause == 1200
        default_pause = calculator.get_pause_for_boundary('phrase', 'normal')
        assert fixed_pause == default_pause

    def test_section_specific_behavior_for_dynamic_pauses(self):
        """Test that dynamic pauses are only used for key phrases sections."""
        test_cases = [
            ('key_phrases', True),
            ('natural_speed', False),
            ('slow_speed', False),
            ('translated', False),
            (None, False),
        ]
        for section_type, expected_dynamic in test_cases:
            use_dynamic_pauses = (section_type == 'key_phrases')
            assert use_dynamic_pauses == expected_dynamic


class TestLinguisticBoundaryDetector:
    """Test linguistic boundary detection."""

    def test_detect_sentence_boundaries(self):
        """Test detection of sentence boundaries."""
        text = "Hello world. How are you? I'm fine!"
        boundaries = detect_linguistic_boundaries(text)
        sentence_boundaries = [b for b in boundaries if b[0] == 'sentence']
        assert len(sentence_boundaries) == 3
        positions = [b[1] for b in sentence_boundaries]
        assert 13 in positions
        assert 26 in positions
        assert 35 in positions

    def test_detect_phrase_boundaries(self):
        """Test detection of phrase boundaries."""
        text = "First part, second part; third part and fourth part but fifth part or sixth part"
        boundaries = detect_linguistic_boundaries(text)
        phrase_boundaries = [b for b in boundaries if b[0] == 'phrase']
        assert len(phrase_boundaries) >= 5

    def test_detect_word_boundaries(self):
        """Test detection of word boundaries."""
        text = "Simple word test"
        boundaries = detect_linguistic_boundaries(text)
        word_boundaries = [b for b in boundaries if b[0] == 'word']
        assert len(word_boundaries) >= 2

    def test_tagalog_text_boundaries(self):
        """Test boundary detection on Tagalog text."""
        text = "Magandang hapon po, kumusta ka? Mabuti naman ako."
        boundaries = detect_linguistic_boundaries(text)
        boundary_types = set(b[0] for b in boundaries)
        assert 'sentence' in boundary_types
        assert 'phrase' in boundary_types
        assert 'word' in boundary_types

    def test_complex_punctuation(self):
        """Test boundary detection with complex punctuation."""
        text = "Wait... really? Yes, I'm sure! Okay then."
        boundaries = detect_linguistic_boundaries(text)
        sentence_boundaries = [b for b in boundaries if b[0] == 'sentence']
        phrase_boundaries = [b for b in boundaries if b[0] == 'phrase']
        assert len(sentence_boundaries) >= 2
        assert len(phrase_boundaries) >= 1


class TestNaturalPauseSplitting:
    """Test natural pause text splitting."""

    def test_split_simple_sentence_normal_speed(self):
        """Test splitting simple sentence at normal speed."""
        text = "Hello world"
        segments = split_with_natural_pauses(text, is_slow=False)
        assert len(segments) >= 3
        text_segments = [s for s in segments if s['type'] == 'text']
        pause_segments = [s for s in segments if s['type'] == 'pause']
        assert len(text_segments) >= 2
        assert len(pause_segments) >= 1
        for segment in text_segments:
            assert segment['voice_settings']['rate'] == 1.0

    def test_split_simple_sentence_slow_speed(self):
        """Test splitting simple sentence at slow speed."""
        text = "Hello world"
        segments = split_with_natural_pauses(text, is_slow=True)
        text_segments = [s for s in segments if s['type'] == 'text']
        pause_segments = [s for s in segments if s['type'] == 'pause']
        for segment in text_segments:
            assert segment['voice_settings']['rate'] == 0.5
        for segment in pause_segments:
            if segment['boundary'] == 'word':
                assert segment['duration'] == 900

    def test_split_with_dynamic_pauses_from_audio_durations(self):
        """Test splitting text with dynamic pauses based on audio durations."""
        text = "Hello world. How are you?"
        segment_durations = [1.2, 0.8, 0.6, 0.5, 1.0]
        segments = split_with_natural_pauses(text, is_slow=False, segment_audio_durations=segment_durations)
        pause_segments = [s for s in segments if s['type'] == 'pause']
        found_dynamic_pause = any(s['duration'] >= 500 for s in pause_segments)
        assert found_dynamic_pause

    def test_split_without_audio_durations_uses_fixed_pauses(self):
        """Test that splitting without audio durations uses fixed pause system."""
        text = "Hello world. How are you?"
        segments = split_with_natural_pauses(text, is_slow=False)
        pause_segments = [s for s in segments if s['type'] == 'pause']
        for segment in pause_segments:
            if segment['boundary'] == 'word':
                assert segment['duration'] == 600
            elif segment['boundary'] == 'sentence':
                assert segment['duration'] == 2000
