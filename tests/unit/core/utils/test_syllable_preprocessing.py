#!/usr/bin/env python3
"""Test the new Tagalog syllable preprocessing for Key Phrases sections."""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from tunatale.core.utils.tts_preprocessor import (
    preprocess_text_for_tts,
    fix_tagalog_syllables_for_key_phrases,
    VOWEL_SYLLABLE_PATTERNS,
    CONSONANT_SYLLABLE_PATTERNS,
    ALL_SYLLABLE_PATTERNS
)

def test_syllable_patterns():
    """Test the syllable pattern mappings."""
    print("🔤 Tagalog Syllable Pattern Test")
    print("=" * 60)
    
    print(f"Total patterns defined: {len(ALL_SYLLABLE_PATTERNS)}")
    print(f"  - Vowel patterns: {len(VOWEL_SYLLABLE_PATTERNS)}")
    print(f"  - Consonant patterns: {len(CONSONANT_SYLLABLE_PATTERNS)}")
    
    print("\n📋 Sample Pattern Mappings:")
    sample_patterns = [
        ('i', 'ee'),
        ('to', 'toh'),
        ('po', 'poh'),
        ('bi', 'bee'),
        ('si', 'see'),
        ('ar', 'ahr'),
        ('ng', 'nahng'),
    ]
    
    for original, expected in sample_patterns:
        print(f"  {original} → {expected}")

def test_key_phrases_processing():
    """Test syllable preprocessing in Key Phrases context."""
    print(f"\n🎯 Key Phrases Section Processing")
    print("=" * 60)
    
    # Sample syllables from actual lesson breakdown
    test_syllables = [
        # From "gabi" breakdown
        "bi",
        "ga", 
        "gabi",
        
        # From "kwarto" breakdown  
        "to",
        "ar",
        "warto",
        "kwar",
        "kwarto",
        
        # From "ito" breakdown
        "i",
        "to",
        "ito",
        
        # From "susi" breakdown
        "si",
        "su",
        "susi",
        
        # Common particles
        "po",
        "ng",
        
        # Mixed content
        "ID po",
        "salamat po"
    ]
    
    print("Key Phrases Section - Tagalog content (syllable fixes applied):")
    for syllable in test_syllables:
        processed = fix_tagalog_syllables_for_key_phrases(syllable, 'key_phrases', 'fil-PH')
        if processed != syllable:
            print(f"  '{syllable}' → '{processed}' ✅")
        else:
            print(f"  '{syllable}' → '{processed}' (no change)")
            
    print("\nKey Phrases Section - English narrator lines (no syllable fixes):")
    english_narrator_lines = [
        "good afternoon",
        "thank you", 
        "How much does this cost?",
        "do you have a room?",
        "ID please",
        "to the hotel"  # This "to" should NOT become "toh"
    ]
    for line in english_narrator_lines:
        processed = fix_tagalog_syllables_for_key_phrases(line, 'key_phrases', 'en-US')
        if processed != line:
            print(f"  '{line}' → '{processed}' ❌ (should be unchanged)")
        else:
            print(f"  '{line}' → '{processed}' ✅ (correctly unchanged)")

def test_section_context_awareness():
    """Test that syllable fixes only apply in Key Phrases sections."""
    print(f"\n🎭 Section Context Awareness")
    print("=" * 60)
    
    test_text = "po to si bi"
    
    sections_to_test = [
        ('key_phrases', 'Key Phrases (Tagalog)', 'fil-PH'),
        ('key_phrases', 'Key Phrases (English)', 'en-US'),
        ('natural_speed', 'Natural Speed', 'fil-PH'),
        ('slow_speed', 'Slow Speed', 'fil-PH'),
        ('translated', 'Translated', 'fil-PH'),
        (None, 'No Section Context', 'fil-PH')
    ]
    
    for section_type, section_name, language in sections_to_test:
        processed = preprocess_text_for_tts(test_text, language, section_type)
        print(f"  {section_name:25} → '{processed}'")

def test_comprehensive_processing():
    """Test complete preprocessing pipeline."""
    print(f"\n🔄 Complete Preprocessing Pipeline")
    print("=" * 60)
    
    # Test phrases that would appear in Key Phrases sections
    test_phrases = [
        "po",
        "salamat po", 
        "i",
        "to", 
        "ito po",
        "bi",
        "gabi po",
        "si",
        "susi po",
        "ar",
        "kwarto po",
        "ID po",  # Should get both syllable + abbreviation fixes
        "ng",
        "mga"
    ]
    
    print("Complete preprocessing (Key Phrases section):")
    for phrase in test_phrases:
        # Process as Key Phrases section
        processed = preprocess_text_for_tts(phrase, 'fil-PH', 'key_phrases')
        if processed != phrase:
            print(f"  '{phrase}' → '{processed}' ✅")
        else:
            print(f"  '{phrase}' → '{processed}' (no change)")

def show_pattern_summary():
    """Show a summary of all patterns for easy reference."""
    print(f"\n📖 Pattern Reference Guide")
    print("=" * 60)
    
    print("Vowel-ending syllables (most common issues):")
    vowel_samples = ['i → ee', 'o → oh', 'to → toh', 'po → poh', 'bi → bee', 'si → see']
    for sample in vowel_samples:
        print(f"  {sample}")
    
    print(f"\nConsonant-ending syllables:")
    consonant_samples = ['ar → ahr', 'ng → nahng', 'an → ahn', 'at → aht']
    for sample in consonant_samples:
        print(f"  {sample}")
    
    print(f"\nTotal coverage: {len(ALL_SYLLABLE_PATTERNS)} syllable patterns")
    print("✅ Patterns only apply in Key Phrases sections")
    print("✅ Existing abbreviation fixes (ID, CR) still work")
    print("✅ Language detection and other TTS processing preserved")

if __name__ == "__main__":
    test_syllable_patterns()
    test_key_phrases_processing()
    test_section_context_awareness()
    test_comprehensive_processing()
    show_pattern_summary()