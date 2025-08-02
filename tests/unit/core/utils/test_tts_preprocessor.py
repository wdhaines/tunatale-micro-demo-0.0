"""Tests for TTS text preprocessing utilities."""
import pytest

from tunatale.core.utils.tts_preprocessor import (
    fix_abbreviation_pronunciation,
    preprocess_tagalog_for_tts,
    preprocess_text_for_tts,
    _is_likely_abbreviation,
    LETTER_PHONETICS,
    PROTECTED_WORDS,
    COMMON_WORDS
)

class TestUniversalAbbreviationHandler:
    """Tests for the universal abbreviation handler."""

    def test_letter_phonetics_mapping(self):
        """Test that all letters A-Z have phonetic mappings."""
        expected_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        actual_letters = set(LETTER_PHONETICS.keys())
        assert actual_letters == expected_letters, "Missing or extra letters in phonetic mapping"
        
        # Test specific phonetic mappings
        assert LETTER_PHONETICS['A'] == 'eh'
        assert LETTER_PHONETICS['B'] == 'bee'
        assert LETTER_PHONETICS['C'] == 'see'
        assert LETTER_PHONETICS['Z'] == 'zee'

    def test_basic_abbreviation_conversion(self):
        """Test basic abbreviation to phonetic conversion."""
        # Common travel abbreviations (must be all caps)
        assert fix_abbreviation_pronunciation("CR") == "see are"
        assert fix_abbreviation_pronunciation("ID") == "eye dee"
        assert fix_abbreviation_pronunciation("ATM") == "eh tee em"
        assert fix_abbreviation_pronunciation("GPS") == "gee pee ess"
        assert fix_abbreviation_pronunciation("USB") == "you ess bee"
        assert fix_abbreviation_pronunciation("WIFI") == "double-you eye eff eye"  # All caps

    def test_time_abbreviations(self):
        """Test time-related abbreviations AM and PM."""
        assert fix_abbreviation_pronunciation("9 AM") == "9 eh em"
        assert fix_abbreviation_pronunciation("3 PM") == "3 pee em"
        assert fix_abbreviation_pronunciation("11 AM to 2 PM") == "11 eh em to 2 pee em"

    def test_currency_abbreviations(self):
        """Test currency abbreviations."""
        assert fix_abbreviation_pronunciation("USD") == "you ess dee"
        assert fix_abbreviation_pronunciation("PHP") == "pee aych pee"
        assert fix_abbreviation_pronunciation("100 USD") == "100 you ess dee"

    def test_technology_abbreviations(self):
        """Test technology-related abbreviations."""
        assert fix_abbreviation_pronunciation("HTML") == "aych tee em ell"
        assert fix_abbreviation_pronunciation("HTTP") == "aych tee tee pee"
        assert fix_abbreviation_pronunciation("API") == "eh pee eye"
        assert fix_abbreviation_pronunciation("JSON") == "jay ess oh en"

    def test_organization_abbreviations(self):
        """Test organization abbreviations."""
        assert fix_abbreviation_pronunciation("FBI") == "eff bee eye"
        assert fix_abbreviation_pronunciation("CIA") == "see eye eh"
        assert fix_abbreviation_pronunciation("NASA") == "en eh ess eh"

    def test_protected_words_not_converted(self):
        """Test that protected words are not converted."""
        # Common English words
        assert fix_abbreviation_pronunciation("TO") == "TO"
        assert fix_abbreviation_pronunciation("IN") == "IN"
        assert fix_abbreviation_pronunciation("ON") == "ON"
        assert fix_abbreviation_pronunciation("ARE") == "ARE"
        assert fix_abbreviation_pronunciation("HE") == "HE"
        
        # Common Tagalog words
        assert fix_abbreviation_pronunciation("PO") == "PO"
        assert fix_abbreviation_pronunciation("ANG") == "ANG"
        assert fix_abbreviation_pronunciation("NG") == "NG"
        assert fix_abbreviation_pronunciation("SA") == "SA"

    def test_common_words_not_converted(self):
        """Test that common English words are not converted."""
        assert fix_abbreviation_pronunciation("CODE") == "CODE"
        assert fix_abbreviation_pronunciation("HELP") == "HELP"
        assert fix_abbreviation_pronunciation("GOOD") == "GOOD"
        assert fix_abbreviation_pronunciation("WORK") == "WORK"
        assert fix_abbreviation_pronunciation("HOME") == "HOME"

    def test_context_preservation(self):
        """Test that surrounding text is preserved."""
        assert fix_abbreviation_pronunciation("Go to the CR now") == "Go to the see are now"
        assert fix_abbreviation_pronunciation("Show me your ID please") == "Show me your eye dee please"
        assert fix_abbreviation_pronunciation("The FBI and CIA work together") == "The eff bee eye and see eye eh work together"
        assert fix_abbreviation_pronunciation("Meet at 9 AM sharp") == "Meet at 9 eh em sharp"

    def test_multiple_abbreviations(self):
        """Test handling multiple abbreviations in one text."""
        assert fix_abbreviation_pronunciation("Need CR, ATM, and WIFI") == "Need see are, eh tee em, and double-you eye eff eye"
        assert fix_abbreviation_pronunciation("FBI, CIA, and NASA") == "eff bee eye, see eye eh, and en eh ess eh"
        assert fix_abbreviation_pronunciation("9 AM to 5 PM") == "9 eh em to 5 pee em"

    def test_case_sensitivity(self):
        """Test that conversion only works with all capital letters."""
        # Only all caps are converted (by design)
        assert fix_abbreviation_pronunciation("CR") == "see are"
        assert fix_abbreviation_pronunciation("ID") == "eye dee"
        
        # Mixed case and lowercase are NOT converted (by design)
        assert fix_abbreviation_pronunciation("cr") == "cr"
        assert fix_abbreviation_pronunciation("Cr") == "Cr"
        assert fix_abbreviation_pronunciation("cR") == "cR"
        assert fix_abbreviation_pronunciation("id") == "id"

    def test_word_boundaries(self):
        """Test that only whole words are converted, not parts of words."""
        assert fix_abbreviation_pronunciation("credit") == "credit"  # CR inside word
        assert fix_abbreviation_pronunciation("idea") == "idea"      # ID inside word
        assert fix_abbreviation_pronunciation("CRazy") == "CRazy"    # CR at start
        assert fix_abbreviation_pronunciation("IDeal") == "IDeal"    # ID at start

    def test_single_letters(self):
        """Test single letter handling."""
        # Single letters that should be converted (except protected A, I)
        assert fix_abbreviation_pronunciation("B") == "bee"
        assert fix_abbreviation_pronunciation("C") == "see"
        assert fix_abbreviation_pronunciation("Z") == "zee"
        
        # Protected single letters
        assert fix_abbreviation_pronunciation("A") == "A"
        assert fix_abbreviation_pronunciation("I") == "I"

    def test_real_demo_content(self):
        """Test with real content from demo files."""
        # Examples from actual demo files
        assert fix_abbreviation_pronunciation("Excuse me po. CR break muna.") == "Excuse me po. see are break muna."
        assert fix_abbreviation_pronunciation("Excuse me po. CR?") == "Excuse me po. see are?"
        assert fix_abbreviation_pronunciation("ID po") == "eye dee po"
        assert fix_abbreviation_pronunciation("Tatlong gabi po. Okay po. ID po?") == "Tatlong gabi po. Okay po. eye dee po?"

    def test_mixed_real_scenarios(self):
        """Test realistic mixed scenarios."""
        assert fix_abbreviation_pronunciation("I need to find the CR, ATM, and WIFI.") == "I need to find the see are, eh tee em, and double-you eye eff eye."
        assert fix_abbreviation_pronunciation("Tour starts at 9 AM? Need to use ATM first.") == "Tour starts at 9 eh em? Need to use eh tee em first."
        assert fix_abbreviation_pronunciation("Do you accept USD or need PHP?") == "Do you accept you ess dee or need pee aych pee?"

    def test_edge_cases(self):
        """Test edge cases and special scenarios."""
        # Empty and whitespace
        assert fix_abbreviation_pronunciation("") == ""
        assert fix_abbreviation_pronunciation(" ") == " "
        assert fix_abbreviation_pronunciation("   ") == "   "
        
        # Numbers and punctuation
        assert fix_abbreviation_pronunciation("123") == "123"
        assert fix_abbreviation_pronunciation("!@#") == "!@#"
        
        # Mixed with punctuation
        assert fix_abbreviation_pronunciation("CR!") == "see are!"
        assert fix_abbreviation_pronunciation("(ID)") == "(eye dee)"
        assert fix_abbreviation_pronunciation("ID,") == "eye dee,"

    def test_is_likely_abbreviation_helper(self):
        """Test the abbreviation detection helper function."""
        # Should be detected as abbreviations
        assert _is_likely_abbreviation("CR") == True
        assert _is_likely_abbreviation("ID") == True
        assert _is_likely_abbreviation("ATM") == True
        assert _is_likely_abbreviation("GPS") == True
        assert _is_likely_abbreviation("NASA") == True
        assert _is_likely_abbreviation("FBI") == True
        
        # Should NOT be detected as abbreviations
        assert _is_likely_abbreviation("TO") == False  # Protected word
        assert _is_likely_abbreviation("PO") == False  # Protected Tagalog word
        assert _is_likely_abbreviation("CODE") == False  # Common word
        assert _is_likely_abbreviation("HELP") == False  # Common word
        assert _is_likely_abbreviation("A") == False    # Protected single letter
        assert _is_likely_abbreviation("I") == False    # Protected single letter

    @pytest.mark.parametrize("abbreviation,expected_phonetic", [
        ("CR", "see are"),
        ("ID", "eye dee"),
        ("ATM", "eh tee em"),
        ("GPS", "gee pee ess"),
        ("USB", "you ess bee"),
        ("WIFI", "double-you eye eff eye"),  # All caps
        ("HTML", "aych tee em ell"),
        ("HTTP", "aych tee tee pee"),
        ("NASA", "en eh ess eh"),
        ("FBI", "eff bee eye"),
        ("CIA", "see eye eh"),
        ("USD", "you ess dee"),
        ("PHP", "pee aych pee"),
        ("AM", "eh em"),
        ("PM", "pee em"),
    ])
    def test_abbreviation_phonetic_conversion(self, abbreviation, expected_phonetic):
        """Parametrized test for abbreviation to phonetic conversion."""
        assert fix_abbreviation_pronunciation(abbreviation) == expected_phonetic

    @pytest.mark.parametrize("protected_word", [
        "TO", "IN", "ON", "AT", "BY", "FOR", "WITH", "FROM", "UP", "OUT", "OFF",
        "PO", "ANG", "NG", "SA", "NA", "KO", "MO", "SYA", "KAMI", "KAYO", "SILA",
        "THE", "AND", "OR", "BUT", "SO", "IF", "AS", "IS", "IT", "BE", "DO", "GO",
        "CODE", "HELP", "GOOD", "WORK", "HOME", "NEED", "MAKE", "TIME", "LIFE"
    ])
    def test_protected_words_unchanged(self, protected_word):
        """Parametrized test to ensure protected words are not converted."""
        assert fix_abbreviation_pronunciation(protected_word) == protected_word
        assert fix_abbreviation_pronunciation(protected_word.lower()) == protected_word.lower()


class TestTTSPreprocessor:
    """Tests for TTS text preprocessing functions."""

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
        # Universal abbreviation tests
        ("CR", "see are", "en-US"),
        ("ID", "eye dee", "en-US"),
        ("My ID is 123", "My eye dee is 123", "en-US"),
        ("9 AM meeting", "9 eh em meeting", "en-US"),
        ("USD accepted", "you ess dee accepted", "en-US"),
        
        # Tagalog tests with abbreviations
        ("ito", "ito", "fil-PH"),
        ("Ito ay CR", "Ito ay see are", "fil-PH"),
        ("PO ang pangalan", "PO ang pangalan", "fil-PH"),  # PO protected
        
        # Mixed language tests
        ("Show ID and ito", "Show eye dee and ito", "en-US"),
        ("Show ID and ito", "Show eye dee and ito", "fil-PH"),
        
        # Edge cases
        ("", "", "en-US"),  # Empty string
        (" ", " ", "en-US"),  # Whitespace only
        ("123", "123", "en-US"),  # Numbers only
        ("CREDIT", "CREDIT", "en-US"),  # Partial match - should not convert
        ("CODE", "CODE", "en-US"),  # Common word - should not convert
    ])
    def test_preprocess_edge_cases(self, input_text, expected, lang):
        """Test various edge cases for text preprocessing."""
        assert preprocess_text_for_tts(input_text, lang) == expected
