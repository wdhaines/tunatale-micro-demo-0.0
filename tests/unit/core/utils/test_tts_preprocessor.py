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
    TAGALOG_DIGITS
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
        """Test basic abbreviation to phonetic conversion for Tagalog."""
        # Common travel abbreviations (converted to phonetic spellings for Tagalog)
        assert preprocess_text_for_tts("CR", "fil-PH") == "see are"
        assert preprocess_text_for_tts("ID", "fil-PH") == "eye dee"
        assert preprocess_text_for_tts("ATM", "fil-PH") == "eh tee em"  # 'A' is pronounced as 'eh' in Tagalog
        assert preprocess_text_for_tts("GPS", "fil-PH") == "gee pee ess"  # 'G' is pronounced as 'gee'
        assert preprocess_text_for_tts("USB", "fil-PH") == "you ess bee"
        assert preprocess_text_for_tts("WIFI", "fil-PH") == "double-you eye eff eye"  # Spelled out letter by letter

    def test_time_abbreviations(self):
        """Test time-related abbreviations AM and PM for Tagalog."""
        assert preprocess_text_for_tts("9 AM", "fil-PH") == "9 eh em"
        assert preprocess_text_for_tts("3 PM", "fil-PH") == "3 pee em"
        assert preprocess_text_for_tts("11 AM to 2 PM", "fil-PH") == "11 eh em to 2 pee em"

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
        # Test with English text - abbreviations should not be converted
        assert preprocess_text_for_tts("Show your ID", "en-US") == "Show your ID"
        
        # Test with Tagalog text - no changes expected for regular text
        assert preprocess_text_for_tts("Ito ay test", "fil-PH") == "Ito ay test"
        
        # Test with abbreviations in Tagalog - should be converted
        assert preprocess_text_for_tts("Ito ay CR", "fil-PH") == "Ito ay see are"
        
        # Test with unsupported language - abbreviations should not be converted
        assert preprocess_text_for_tts("Show ID and ito", "es-ES") == "Show ID and ito"

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
        ("Show ID and ito", "Show ID and ito", "en-US"),
        
        # Edge cases
        ("", "", "en-US"),  # Empty string
        (" ", " ", "en-US"),  # Whitespace only
        ("123", "123", "en-US"),  # Numbers only
        ("CREDIT", "CREDIT", "en-US"),  # Partial match - should not convert
        ("CODE", "CODE", "en-US"),  # Common word - should not convert
        # Test with abbreviations (should be left as-is now)
        ("Show your ID", "Show your ID", "en-US"),
        ("CR 123", "CR 123", "en-US"),
        ("My ID is 123", "My ID is 123", "en-US"),
        # Test with time
        ("9 AM meeting", "9 AM meeting", "en-US"),
        # Test with currency
        ("USD accepted", "USD accepted", "en-US"),
    ])
    def test_preprocess_edge_cases(self, input_text, expected, lang):
        """Test various edge cases for text preprocessing."""
        if input_text is None:
            result = preprocess_text_for_tts(None, lang)
        else:
            result = preprocess_text_for_tts(input_text, lang)
        assert result == expected, f"Failed for input: {input_text}"

    @pytest.mark.parametrize("input_text,expected,lang", [
        # Test with empty string
        ("", "", "en-US"),
        # Test with None (should return empty string)
        (None, "", "en-US"),
        # Test with whitespace
        ("   ", "   ", "en-US"),
        # Test with numbers
        ("123", "123", "en-US"),
        # Test with special characters
        ("Hello, world!", "Hello, world!", "en-US"),
        # Test with newlines
        ("Line 1\nLine 2", "Line 1\nLine 2", "en-US"),
    ])
    def test_preprocess_edge_cases(self, input_text, expected, lang):
        """Test various edge cases for text preprocessing."""
        result = preprocess_text_for_tts(input_text, lang)
        assert result == expected, f"Failed for input: {input_text}"


class TestFilipinoNumberClarification:
    """Tests for the Filipino number clarification system."""

    def test_spanish_numbers_mapping(self):
        """Test that Spanish numbers are correctly mapped."""
        assert SPANISH_NUMBERS[1] == 'una'
        assert SPANISH_NUMBERS[8] == 'otso'
        assert SPANISH_NUMBERS[12] == 'dose'

    def test_tagalog_digits_mapping(self):
        """Test that Tagalog digits are correctly mapped."""
        assert TAGALOG_DIGITS['0'] == 'zero'
        assert TAGALOG_DIGITS['1'] == 'isa'
        assert TAGALOG_DIGITS['5'] == 'lima'
        assert TAGALOG_DIGITS['9'] == 'siyam'

    def test_convert_time_to_spanish_basic(self):
        """Test basic time conversion to Spanish."""
        assert convert_time_to_spanish(8, 30) == "Alas otso y medya."
        assert convert_time_to_spanish(12, 0) == "Alas dose."
        assert convert_time_to_spanish(3, 15) == "Alas tres kinse."
        assert convert_time_to_spanish(6, 45) == "Kinse para alas syete."

    def test_convert_time_to_spanish_24_hour(self):
        """Test 24-hour to 12-hour conversion."""
        assert convert_time_to_spanish(14, 30) == "Alas dos y medya."  # 2:30 PM
        assert convert_time_to_spanish(20, 0) == "Alas otso."         # 8:00 PM
        assert convert_time_to_spanish(0, 30) == "Alas dose y medya." # 12:30 AM

    def test_convert_time_to_spanish_irregular_minutes(self):
        """Test time conversion with irregular minutes."""
        assert convert_time_to_spanish(9, 20) == "Alas nuwebe, 20."
        assert convert_time_to_spanish(11, 55) == "Alas onse, 55."

    def test_convert_digits_to_tagalog(self):
        """Test digit breakdown to Tagalog."""
        assert convert_digits_to_tagalog("150") == "isa lima zero."
        assert convert_digits_to_tagalog("203") == "dalawa zero tatlo."
        assert convert_digits_to_tagalog("1205") == "isa dalawa zero lima."
        assert convert_digits_to_tagalog("9") == "siyam."

    def test_convert_digits_to_tagalog_with_non_digits(self):
        """Test digit conversion ignoring non-digit characters."""
        assert convert_digits_to_tagalog("Room 203") == "dalawa zero tatlo."
        assert convert_digits_to_tagalog("555-1234") == "lima lima lima isa dalawa tatlo apat."

    def test_clarify_number_time_patterns(self):
        """Test clarification of time patterns."""
        assert clarify_number("8:30") == "8:30. Alas otso y medya."
        assert clarify_number("12:45") == "12:45. Kinse para alas una."
        assert clarify_number("3:15") == "3:15. Alas tres kinse."
        assert clarify_number("6:00") == "6:00. Alas seys."

    def test_clarify_number_phone_patterns(self):
        """Test clarification of phone number patterns."""
        assert clarify_number("555-1234") == "555-1234. lima lima lima, isa dalawa tatlo apat."
        assert clarify_number("123-456-7890") == "123-456-7890. isa dalawa tatlo, apat lima anim, pito walo siyam zero."

    def test_clarify_number_large_numbers(self):
        """Test clarification of large numbers."""
        assert clarify_number("150") == "150. isa lima zero."
        assert clarify_number("1205") == "1205. isa dalawa zero lima."
        assert clarify_number("500") == "500. lima zero zero."

    def test_clarify_number_small_numbers_no_clarification(self):
        """Test that small numbers don't get clarified by default."""
        assert clarify_number("5") == "5"
        assert clarify_number("25") == "25"
        assert clarify_number("99") == "99"

    def test_clarify_number_room_context(self):
        """Test clarification with room context."""
        assert clarify_number("203", "Room 203") == "203. dalawa zero tatlo."
        assert clarify_number("15", "kwarto 15") == "15. isa lima."

    def test_process_number_clarification_slow_speed(self):
        """Test number clarification in slow_speed sections."""
        # Time should be clarified
        result = process_number_clarification("Alis tayo ng 8:30 para sa 150 pesos.", "slow_speed")
        assert "8:30. Alas otso y medya." in result
        assert "150. isa lima zero." in result

    def test_process_number_clarification_natural_speed_no_tags(self):
        """Test that natural speed doesn't clarify without tags."""
        result = process_number_clarification("Alis tayo ng 8:30 para sa 150 pesos.", "natural_speed")
        # Should not have clarification
        assert "Alas otso y medya" not in result
        assert "isa lima zero" not in result

    def test_process_number_clarification_with_clarify_tags(self):
        """Test clarification with clarify tags."""
        result = process_number_clarification("Alis tayo ng <clarify>8:30</clarify> para sa 150 pesos.", "natural_speed")
        assert "8:30. Alas otso y medya." in result
        assert "isa lima zero" not in result  # 150 not in tags
        assert "<clarify>" not in result  # Tags should be removed

    def test_process_number_clarification_small_numbers_in_tags(self):
        """Test that small numbers in clarify tags get clarified."""
        result = process_number_clarification("Kwarto <clarify>5</clarify> po.", "natural_speed")
        assert "5. lima." in result
        assert "<clarify>" not in result

    def test_process_number_clarification_multiple_patterns(self):
        """Test multiple number patterns in one text."""
        text = "Meeting at <clarify>9:15</clarify> in Room 203, call 555-1234."
        result = process_number_clarification(text, "slow_speed")
        assert "9:15. Alas nuwebe kinse." in result
        assert "203. dalawa zero tatlo." in result
        assert "555-1234. lima lima lima, isa dalawa tatlo apat." in result

    def test_process_number_clarification_tag_removal(self):
        """Test that all clarify tags are removed from output."""
        result = process_number_clarification("Go to <clarify>Room 5</clarify> at <clarify>8:30</clarify>.", "natural_speed")
        assert "<clarify>" not in result
        assert "</clarify>" not in result
        assert "Room 5. lima. at 8:30. Alas otso y medya." in result

    def test_integration_with_preprocess_text_for_tts_slow_speed(self):
        """Test integration with main preprocessing function in slow speed."""
        result = preprocess_text_for_tts("Alis tayo ng 8:30 para sa 150 pesos.", "fil-PH", "slow_speed")
        assert "8:30. Alas otso y medya." in result
        assert "150. isa lima zero." in result

    def test_integration_with_preprocess_text_for_tts_clarify_tags(self):
        """Test integration with main preprocessing function using clarify tags."""
        result = preprocess_text_for_tts("Alis sa <clarify>8:30</clarify> ng umaga.", "fil-PH", "natural_speed")
        assert "8:30. Alas otso y medya." in result
        assert "<clarify>" not in result

    def test_integration_preserves_other_preprocessing(self):
        """Test that number clarification works with other preprocessing."""
        result = preprocess_text_for_tts("CR break at <clarify>8:30</clarify> po.", "fil-PH", "natural_speed")
        assert "see are break at 8:30. Alas otso y medya. po." in result

    @pytest.mark.parametrize("time_input,expected_spanish", [
        ("8:30", "Alas otso y medya."),
        ("12:00", "Alas dose."),
        ("3:15", "Alas tres kinse."),
        ("6:45", "Kinse para alas syete."),
        ("1:30", "Alas una y medya."),
        ("11:15", "Alas onse kinse."),
    ])
    def test_time_clarification_parametrized(self, time_input, expected_spanish):
        """Parametrized test for time clarification."""
        result = clarify_number(time_input)
        assert expected_spanish in result

    @pytest.mark.parametrize("number_input,expected_tagalog", [
        ("150", "isa lima zero."),
        ("203", "dalawa zero tatlo."),
        ("1205", "isa dalawa zero lima."),
        ("999", "siyam siyam siyam."),
        ("100", "isa zero zero."),
    ])
    def test_large_number_clarification_parametrized(self, number_input, expected_tagalog):
        """Parametrized test for large number clarification."""
        result = clarify_number(number_input)
        assert expected_tagalog in result

    def test_edge_cases_empty_and_invalid(self):
        """Test edge cases with empty and invalid inputs."""
        assert process_number_clarification("", "slow_speed") == ""
        assert process_number_clarification(None, "slow_speed") is None
        assert clarify_number("abc") == "abc"
        assert convert_digits_to_tagalog("") == ""

    def test_complex_real_world_examples(self):
        """Test complex real-world examples."""
        # Hotel check-in scenario
        text = "Kwarto <clarify>203</clarify>, check-in at 3:00 PM, checkout 11:00 AM."
        result = process_number_clarification(text, "natural_speed")
        assert "203. dalawa zero tatlo." in result
        assert "clarify" not in result

        # Travel scenario  
        text = "Flight departs 7:45, gate 150, call 555-HELP."
        result = process_number_clarification(text, "slow_speed")
        assert "7:45. Kinse para alas otso." in result
        assert "150. isa lima zero." in result
