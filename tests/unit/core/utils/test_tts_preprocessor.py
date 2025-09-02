"""Tests for TTS text preprocessing utilities."""
import pytest

from tunatale.core.utils.tts_preprocessor import (
    fix_abbreviation_pronunciation,
    preprocess_tagalog_for_tts,
    preprocess_text_for_tts,
    _is_likely_abbreviation,
    LETTER_PHONETICS,
    PROTECTED_WORDS,
    COMMON_WORDS,
    process_number_clarification,
    convert_time_to_spanish,
    convert_digits_to_tagalog,
    clarify_number,
    SPANISH_NUMBERS,
    TAGALOG_DIGITS,
    enhanced_preprocess_text_for_tts,
    fix_tagalog_syllables_for_key_phrases,
    ALL_SYLLABLE_PATTERNS
)


class TestTTSPreprocessor:
    """Tests for TTS text preprocessing functions."""

    def test_preprocess_tagalog_for_tts(self):
        """Test that Tagalog text is properly preprocessed for TTS."""
        assert preprocess_tagalog_for_tts("ito") == "ito"
        assert preprocess_tagalog_for_tts("Ito ay") == "Ito ay"
        assert preprocess_tagalog_for_tts("ITO AY") == "ITO AY"
        assert preprocess_tagalog_for_tts("pito") == "pito"
        assert preprocess_tagalog_for_tts("bakit") == "bakit"

    def test_preprocess_text_for_tts(self):
        """Test the combined preprocessing function."""
        assert preprocess_text_for_tts("Show your ID", "en-US") == "Show your ID"
        assert preprocess_text_for_tts("Ito ay test", "fil-PH") == "Ito ay test"
        assert preprocess_text_for_tts("Ito ay CR", "fil-PH") == "Ito ay see are"
        assert preprocess_text_for_tts("Show ID and ito", "es-ES") == "Show ID and ito"

    def test_abbreviation_language_filtering(self):
        """Test that abbreviation handling only applies to Tagalog language codes."""
        assert "eh em" not in preprocess_text_for_tts("Meeting at 9 AM", 'en-US', 'natural_speed')
        assert "eh em" in preprocess_text_for_tts("Meeting at 9 AM", 'fil-PH', 'natural_speed')

    def test_narrator_abbreviations(self):
        """Test that narrator text (English) doesn't get abbreviation processing."""
        text = "Good morning! Let's practice from 9 AM to 12 PM"
        english_result, _ = enhanced_preprocess_text_for_tts(text=text, language_code='en-US', provider_name='edge_tts', supports_ssml=False, section_type='natural_speed')
        tagalog_result, _ = enhanced_preprocess_text_for_tts(text=text, language_code='fil-PH', provider_name='edge_tts', supports_ssml=False, section_type='natural_speed')
        assert "eh em" not in english_result
        assert "pee em" not in english_result
        assert "eh em" in tagalog_result
        assert "pee em" in tagalog_result

    def test_syllable_preprocessing_context_awareness(self):
        """Test that syllable fixes only apply in Key Phrases sections for Tagalog."""
        test_text = "po to si bi"
        assert preprocess_text_for_tts(test_text, 'fil-PH', 'key_phrases') == "po towe si beey"
        assert "poh towe see beey" not in preprocess_text_for_tts(test_text, 'en-US', 'key_phrases')
        assert "poh towe see beey" not in preprocess_text_for_tts(test_text, 'fil-PH', 'natural_speed')

    def test_updated_vowel_patterns(self):
        """Test the updated vowel patterns based on user testing feedback."""
        assert preprocess_text_for_tts("i", "fil-PH", "key_phrases") == "eey"
        assert preprocess_text_for_tts("o", "fil-PH", "key_phrases") == "o"
        assert preprocess_text_for_tts("u", "fil-PH", "key_phrases").startswith("ooh")
        assert preprocess_text_for_tts("bi", "fil-PH", "key_phrases") == "beey"
        assert preprocess_text_for_tts("to", "en-US", "key_phrases") == "to"


class TestUniversalAbbreviationHandler:
    """Tests for the universal abbreviation handler."""

    def test_letter_phonetics_mapping(self):
        """Test that all letters A-Z have phonetic mappings."""
        assert set(LETTER_PHONETICS.keys()) == set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    @pytest.mark.parametrize("abbreviation,expected_phonetic", [
        ("CR", "see are"),
        ("ID", "eye dee"),
        ("ATM", "eh tee em"),
        ("GPS", "gee pee ess"),
        ("USB", "you ess bee"),
        ("WIFI", "double-you eye eff eye"),
        ("AM", "eh em"),
        ("PM", "pee em"),
    ])
    def test_abbreviation_phonetic_conversion(self, abbreviation, expected_phonetic):
        """Parametrized test for abbreviation to phonetic conversion."""
        assert fix_abbreviation_pronunciation(abbreviation) == expected_phonetic

    @pytest.mark.parametrize("protected_word", [
        "TO", "IN", "ON", "PO", "ANG", "NG", "SA", "CODE", "HELP", "A", "I"
    ])
    def test_protected_words_unchanged(self, protected_word):
        """Parametrized test to ensure protected words are not converted."""
        assert fix_abbreviation_pronunciation(protected_word) == protected_word

    def test_is_likely_abbreviation_helper(self):
        """Test the abbreviation detection helper function."""
        assert _is_likely_abbreviation("CR") == True
        assert _is_likely_abbreviation("TO") == False
        assert _is_likely_abbreviation("CODE") == False


class TestFilipinoNumberClarification:
    """Tests for the Filipino number clarification system."""

    def test_spanish_numbers_mapping(self):
        """Test that Spanish numbers are correctly mapped."""
        assert SPANISH_NUMBERS[1] == 'una'
        assert SPANISH_NUMBERS[8] == 'otso'

    def test_tagalog_digits_mapping(self):
        """Test that Tagalog digits are correctly mapped."""
        assert TAGALOG_DIGITS['1'] == 'isa'
        assert TAGALOG_DIGITS['9'] == 'siyam'

    def test_convert_time_to_spanish_basic(self):
        """Test basic time conversion to Spanish."""
        assert convert_time_to_spanish(8, 30) == "... alas... otso... y medya."

    def test_convert_digits_to_tagalog(self):
        """Test conversion of digits to Tagalog words."""
        assert convert_digits_to_tagalog("150") == "isa... lima... sero."

    def test_clarify_number_time_patterns(self):
        """Test clarification of time patterns."""
        assert "... alas... otso... y medya." in clarify_number("8:30")

    def test_process_number_clarification_with_clarify_tags(self):
        """Test number clarification with <clarify> tags."""
        text = "Alis sa <clarify>8:30</clarify> ng umaga."
        expected = "Alis sa 8:30. ... alas... otso... y medya. ng umaga."
        assert process_number_clarification(text) == expected
