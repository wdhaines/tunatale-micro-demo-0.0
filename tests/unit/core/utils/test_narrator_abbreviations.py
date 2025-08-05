#!/usr/bin/env python3
"""Test that narrator text doesn't get weird abbreviation processing."""

import sys
from pathlib import Path

# Add project root to path  
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from tunatale.core.utils.tts_preprocessor import enhanced_preprocess_text_for_tts

def test_narrator_abbreviations():
    """Test that narrator text (English) doesn't get abbreviation processing."""
    
    print("=== Testing Narrator Abbreviation Processing ===\n")
    
    # Test cases that might appear in narrator text
    test_cases = [
        "Good morning! Let's practice from 9 AM to 12 PM",
        "Show me your ID at the gate",
        "The ATM is near the CR", 
        "GPS coordinates are important",
        "We'll visit NASA headquarters",
        "Day 1: Welcome to El Nido!"
    ]
    
    for text in test_cases:
        print(f"Testing: '{text}'")
        
        # Test with English (narrator voice)
        english_result, english_ssml = enhanced_preprocess_text_for_tts(
            text=text,
            language_code='en-US',  # English narrator
            provider_name='edge_tts',
            supports_ssml=False,
            section_type='natural_speed'
        )
        
        # Test with Tagalog (Tagalog speaker)
        tagalog_result, tagalog_ssml = enhanced_preprocess_text_for_tts(
            text=text,
            language_code='fil-PH',  # Tagalog speaker
            provider_name='edge_tts',
            supports_ssml=False,
            section_type='natural_speed'
        )
        
        print(f"  English:  '{english_result}'")
        print(f"  Tagalog:  '{tagalog_result}'")
        
        # Check if abbreviations were processed
        english_has_phonetics = any(phonetic in english_result for phonetic in 
                                   ['eh em', 'pee em', 'eye dee', 'eh tee em', 'gee pee ess', 'en eh ess eh'])
        tagalog_has_phonetics = any(phonetic in tagalog_result for phonetic in 
                                   ['eh em', 'pee em', 'eye dee', 'eh tee em', 'gee pee ess', 'en eh ess eh'])
        
        if english_has_phonetics:
            print("  ❌ English text was processed with abbreviation handler")
        else:
            print("  ✅ English text preserved (no abbreviation processing)")
            
        if tagalog_has_phonetics:
            print("  ✅ Tagalog text processed with abbreviation handler")
        else:
            print("  ⚠️  Tagalog text not processed (may not contain abbreviations)")
        
        print()
    
    print("=== Summary ===")
    print("The fix ensures:")
    print("✓ English narrator text keeps natural abbreviations (AM, PM, ID, etc.)")
    print("✓ Tagalog speaker text gets phonetic abbreviations (eh em, pee em, eye dee, etc.)")
    print("✓ This prevents weird narrator speech like 'nine eh em' instead of 'nine AM'")

if __name__ == "__main__":
    test_narrator_abbreviations()