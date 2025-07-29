"""Tests for TTS text preprocessing utilities."""
import pytest

from tunatale.core.utils.tts_preprocessor import (
    fix_abbreviation_pronunciation,
    preprocess_tagalog_for_tts,
    preprocess_text_for_tts
)

class TestTTSPreprocessor:
    """Tests for TTS text preprocessing functions."""

    def test_fix_abbreviation_pronunciation(self):
        """Test that abbreviations are properly formatted for TTS."""
        # Test CR abbreviation
        assert fix_abbreviation_pronunciation("Go to the CR") == "Go to the see are"
        assert fix_abbreviation_pronunciation("Show your ID") == "Show your eye dee"
        
        # Test case insensitivity
        assert fix_abbreviation_pronunciation("go to cr") == "go to see are"
        assert fix_abbreviation_pronunciation("show id please") == "show eye dee please"
        
        # Test partial matches are not replaced
        assert fix_abbreviation_pronunciation("credit") == "credit"
        assert fix_abbreviation_pronunciation("idea") == "idea"

    def test_preprocess_tagalog_for_tts(self):
        """Test that Tagalog text is properly preprocessed for TTS."""
        # Test common Tagalog fixes
        assert preprocess_tagalog_for_tts("ito") == "ito"
        assert preprocess_tagalog_for_tts("Ito ay") == "Ito ay"
        
        # Test case insensitivity
        assert preprocess_tagalog_for_tts("ITO AY") == "ITO AY"
        
        # Test partial matches are not replaced
        assert preprocess_tagalog_for_tts("pito") == "pito"
        assert preprocess_tagalog_for_tts("bakit") == "bakit"

    def test_preprocess_text_for_tts(self):
        """Test the combined preprocessing function."""
        # Test with English text
        assert preprocess_text_for_tts("Show your ID", "en-US") == "Show your eye dee"
        
        # Test with Tagalog text
        assert preprocess_text_for_tts("Ito ay test", "fil-PH") == "Ito ay test"
        
        # Test with both abbreviations and Tagalog fixes
        assert preprocess_text_for_tts("Ito ay CR", "fil-PH") == "Ito ay see are"
        
        # Test with unsupported language (should only apply abbreviations)
        assert preprocess_text_for_tts("Show ID and ito", "es-ES") == "Show eye dee and ito"

    @pytest.mark.parametrize("input_text,expected,lang", [
        # Abbreviation tests
        ("CR", "see are", "en-US"),
        ("ID", "eye dee", "en-US"),
        ("My ID is 123", "My eye dee is 123", "en-US"),
        
        # Tagalog tests - no more ito -> ee-toh conversion
        ("ito", "ito", "fil-PH"),
        ("Ito ay CR", "Ito ay see are", "fil-PH"),
        
        # Mixed language tests
        ("Show ID and ito", "Show eye dee and ito", "en-US"),  # Only English fixes
        ("Show ID and ito", "Show eye dee and ito", "fil-PH"),  # Only English fixes
        
        # Edge cases
        ("", "", "en-US"),  # Empty string
        (" ", " ", "en-US"),  # Whitespace only
        ("123", "123", "en-US"),  # Numbers only
        ("CREDIT", "CREDIT", "en-US"),  # Partial match
    ])
    def test_preprocess_edge_cases(self, input_text, expected, lang):
        """Test various edge cases for text preprocessing."""
        assert preprocess_text_for_tts(input_text, lang) == expected
