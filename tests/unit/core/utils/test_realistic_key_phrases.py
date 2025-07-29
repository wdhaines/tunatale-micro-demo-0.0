#!/usr/bin/env python3
"""Test syllable preprocessing with realistic Key Phrases section content."""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from tunatale.core.utils.tts_preprocessor import preprocess_text_for_tts

def test_realistic_key_phrases_section():
    """Test with actual Key Phrases section content from lessons."""
    print("🎯 Realistic Key Phrases Section Test")
    print("=" * 70)
    
    # Simulate actual Key Phrases section content
    key_phrases_content = [
        # Full phrase (Tagalog)
        ("[TAGALOG-FEMALE-1]", "may kwarto po ba", "fil-PH"),
        
        # English translation (Narrator)
        ("[NARRATOR]", "do you have a room?", "en-US"),
        
        # Repeated phrase (Tagalog)
        ("[TAGALOG-FEMALE-1]", "may kwarto po ba", "fil-PH"),
        
        # Syllable breakdown (all Tagalog)
        ("[TAGALOG-FEMALE-1]", "ba", "fil-PH"),
        ("[TAGALOG-FEMALE-1]", "po ba", "fil-PH"),
        ("[TAGALOG-FEMALE-1]", "po", "fil-PH"),
        ("[TAGALOG-FEMALE-1]", "to", "fil-PH"),
        ("[TAGALOG-FEMALE-1]", "ar", "fil-PH"),
        ("[TAGALOG-FEMALE-1]", "warto", "fil-PH"),
        ("[TAGALOG-FEMALE-1]", "kwar", "fil-PH"),
        ("[TAGALOG-FEMALE-1]", "kwarto", "fil-PH"),
        ("[TAGALOG-FEMALE-1]", "kwarto po", "fil-PH"),
        ("[TAGALOG-FEMALE-1]", "kwarto po ba", "fil-PH"),
        ("[TAGALOG-FEMALE-1]", "may", "fil-PH"),
        ("[TAGALOG-FEMALE-1]", "may kwarto", "fil-PH"),
        ("[TAGALOG-FEMALE-1]", "may kwarto po ba", "fil-PH"),
        
        # Another phrase
        ("[TAGALOG-FEMALE-1]", "ID po", "fil-PH"),
        ("[NARRATOR]", "ID please", "en-US"),
        ("[TAGALOG-FEMALE-1]", "ID po", "fil-PH"),
        ("[TAGALOG-FEMALE-1]", "po", "fil-PH"),
        ("[TAGALOG-FEMALE-1]", "ID", "fil-PH"),
        ("[TAGALOG-FEMALE-1]", "ID po", "fil-PH"),
    ]
    
    print("Processing Key Phrases section content:")
    print("=" * 50)
    
    for speaker, text, language in key_phrases_content:
        processed = preprocess_text_for_tts(text, language, 'key_phrases')
        
        # Show what changed
        if processed != text:
            status = "✅ FIXED" if language.startswith('fil') else "❌ SHOULD NOT CHANGE"
            print(f"{speaker:20} | {language:6} | '{text}' → '{processed}' {status}")
        else:
            status = "✅ PRESERVED" if language.startswith('en') else "⚪ NO CHANGE"
            print(f"{speaker:20} | {language:6} | '{text}' → '{processed}' {status}")

def test_language_detection_accuracy():
    """Test that language detection correctly identifies when to apply fixes."""
    print(f"\n🔍 Language Detection Accuracy")
    print("=" * 70)
    
    test_cases = [
        # Should be fixed (Tagalog syllables in Key Phrases)
        ("po", "fil-PH", "key_phrases", True),
        ("to", "fil-PH", "key_phrases", True),
        ("bi", "fil-PH", "key_phrases", True),
        ("si", "fil-PH", "key_phrases", True),
        
        # Should NOT be fixed (English in Key Phrases)
        ("to", "en-US", "key_phrases", False),
        ("so", "en-US", "key_phrases", False),
        ("ID please", "en-US", "key_phrases", False),
        ("do you have a room?", "en-US", "key_phrases", False),
        
        # Should NOT be fixed (Tagalog in other sections)
        ("po", "fil-PH", "natural_speed", False),
        ("to", "fil-PH", "slow_speed", False),
        ("bi", "fil-PH", "translated", False),
        
        # Should NOT be fixed (No section context)
        ("po", "fil-PH", None, False),
        ("to", "fil-PH", None, False),
    ]
    
    for text, language, section, should_change in test_cases:
        processed = preprocess_text_for_tts(text, language, section)
        actually_changed = processed != text
        
        status = ""
        if should_change and actually_changed:
            status = "✅ CORRECT (Fixed as expected)"
        elif not should_change and not actually_changed:
            status = "✅ CORRECT (Preserved as expected)"
        elif should_change and not actually_changed:
            status = "❌ ERROR (Should have been fixed)"
        else:  # not should_change and actually_changed
            status = "❌ ERROR (Should have been preserved)"
        
        section_str = section or "None"
        print(f"  '{text}' | {language} | {section_str:12} → '{processed}' {status}")

if __name__ == "__main__":
    test_realistic_key_phrases_section()
    test_language_detection_accuracy()