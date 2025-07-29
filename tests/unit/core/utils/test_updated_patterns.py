#!/usr/bin/env python3
"""Test the updated syllable patterns based on user feedback."""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from tunatale.core.utils.tts_preprocessor import preprocess_text_for_tts

def test_updated_vowel_patterns():
    """Test the updated vowel patterns based on user testing feedback.
    
    Final selections:
    - i → eey (not "ee", sounds more natural)
    - o → unchanged (natural "o" was best)  
    - a → ah (keeps existing good pronunciation)
    - e → eh (keeps existing good pronunciation)
    - u → ooh (not "oo", sounds more natural)
    """
    print("🔤 Updated Vowel Pattern Test")
    print("=" * 60)
    
    # Test cases based on user selections
    test_cases = [
        # Single vowels
        ("i", "fil-PH", "key_phrases", "eey"),
        ("o", "fil-PH", "key_phrases", "o"),    # unchanged 
        ("a", "fil-PH", "key_phrases", "ah"),
        ("e", "fil-PH", "key_phrases", "eh"),
        ("u", "fil-PH", "key_phrases", "ooh"),
        
        # Consonant + vowel combinations
        ("bi", "fil-PH", "key_phrases", "beey"),
        ("si", "fil-PH", "key_phrases", "seey"),
        ("mi", "fil-PH", "key_phrases", "meey"),
        ("ma", "fil-PH", "key_phrases", "mah"),
        ("su", "fil-PH", "key_phrases", "sooh"),
        ("mu", "fil-PH", "key_phrases", "mooh"),
        ("me", "fil-PH", "key_phrases", "meh"),
        
        # Should not change (other sections)
        ("i", "fil-PH", "natural_speed", "i"),
        ("bi", "fil-PH", "slow_speed", "bi"),
        
        # Should not change (English)
        ("i", "en-US", "key_phrases", "i"),
        ("to", "en-US", "key_phrases", "to"),
    ]
    
    print("Testing updated patterns:")
    print("Input | Language | Section | Expected → Actual | Status")
    print("-" * 65)
    
    for input_text, language, section, expected in test_cases:
        actual = preprocess_text_for_tts(input_text, language, section)
        status = "✅" if actual == expected else "❌"
        print(f"{input_text:5} | {language:6} | {section:12} | {expected:8} → {actual:8} | {status}")

def test_real_world_examples():
    """Test with realistic examples from lessons."""
    print(f"\n🎯 Real-World Example Test")
    print("=" * 60)
    
    examples = [
        # Example breakdowns from actual lessons
        ("gabi po", "fil-PH", "key_phrases", "gabi poh"),
        ("susi po", "fil-PH", "key_phrases", "susi poh"),
        ("ito po", "fil-PH", "key_phrases", "ee-toh poh"),
        
        # Syllable breakdowns  
        ("bi", "fil-PH", "key_phrases", "beey"),
        ("ga", "fil-PH", "key_phrases", "gah"),
        ("si", "fil-PH", "key_phrases", "seey"),
        ("su", "fil-PH", "key_phrases", "sooh"),
        ("po", "fil-PH", "key_phrases", "poh"),
        
        # Mixed context
        ("ID po", "fil-PH", "key_phrases", "eye dee poh"),
        ("salamat po", "fil-PH", "key_phrases", "salamat poh"),
    ]
    
    print("Testing real-world examples:")
    for input_text, language, section, expected in examples:
        actual = preprocess_text_for_tts(input_text, language, section)
        status = "✅" if actual == expected else "❌"
        print(f"'{input_text}' → '{actual}' {status}")
        if actual != expected:
            print(f"  Expected: '{expected}'")

if __name__ == "__main__":
    test_updated_vowel_patterns()
    test_real_world_examples()