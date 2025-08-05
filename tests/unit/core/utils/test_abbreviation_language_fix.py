#!/usr/bin/env python3
"""Test that abbreviation handler only applies to Tagalog TTS."""

from tunatale.core.utils.tts_preprocessor import preprocess_text_for_tts

def test_abbreviation_language_filtering():
    """Test that abbreviation handling only applies to Tagalog language codes."""
    
    print("=== Testing Abbreviation Language Filtering ===\n")
    
    # Test text with abbreviation
    test_text = "Meeting at 9 AM"
    
    # Test 1: English language code (should NOT apply abbreviation fixes)
    print("Test 1: English language code")
    english_result = preprocess_text_for_tts(test_text, 'en-US', 'natural_speed')
    print(f"Input:  '{test_text}'")
    print(f"Output: '{english_result}' (en-US)")
    
    if "eh em" in english_result:
        print("❌ PROBLEM: Abbreviation handler applied to English text")
    else:
        print("✅ SUCCESS: Abbreviation handler NOT applied to English text")
    print()
    
    # Test 2: Tagalog language code (should apply abbreviation fixes)  
    print("Test 2: Tagalog language code")
    tagalog_result = preprocess_text_for_tts(test_text, 'fil-PH', 'natural_speed')
    print(f"Input:  '{test_text}'")
    print(f"Output: '{tagalog_result}' (fil-PH)")
    
    if "eh em" in tagalog_result:
        print("✅ SUCCESS: Abbreviation handler applied to Tagalog text")
    else:
        print("❌ PROBLEM: Abbreviation handler NOT applied to Tagalog text")
    print()
    
    # Test 3: Another English variant (should NOT apply)
    print("Test 3: Another English variant")
    english_gb_result = preprocess_text_for_tts(test_text, 'en-GB', 'natural_speed')
    print(f"Input:  '{test_text}'")
    print(f"Output: '{english_gb_result}' (en-GB)")
    
    if "eh em" in english_gb_result:
        print("❌ PROBLEM: Abbreviation handler applied to English GB text")
    else:
        print("✅ SUCCESS: Abbreviation handler NOT applied to English GB text")
    print()
    
    # Test 4: Test common abbreviations in different languages
    print("Test 4: Common abbreviations")
    abbrev_text = "Show me your ID"
    
    english_abbrev = preprocess_text_for_tts(abbrev_text, 'en-US', 'natural_speed')
    tagalog_abbrev = preprocess_text_for_tts(abbrev_text, 'fil-PH', 'natural_speed')
    
    print(f"Input: '{abbrev_text}'")
    print(f"English (en-US): '{english_abbrev}'")
    print(f"Tagalog (fil-PH): '{tagalog_abbrev}'")
    
    if "eye dee" not in english_abbrev and "eye dee" in tagalog_abbrev:
        print("✅ SUCCESS: Abbreviations only processed for Tagalog")
    elif "eye dee" in english_abbrev:
        print("❌ PROBLEM: Abbreviations processed for English")
    elif "eye dee" not in tagalog_abbrev:
        print("❌ PROBLEM: Abbreviations NOT processed for Tagalog")
    else:
        print("⚠️  UNEXPECTED: Both or neither processed abbreviations")
    
    print()
    print("=== Summary ===")
    print("Abbreviation handler should:")
    print("✓ Apply to Tagalog/Filipino language codes (fil-*)")  
    print("✗ NOT apply to English language codes (en-*)")
    print("✗ NOT apply to other language codes")

if __name__ == "__main__":
    test_abbreviation_language_filtering()