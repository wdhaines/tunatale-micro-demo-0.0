#!/usr/bin/env python3
"""Test regex patterns for day extraction."""

import re

def test_day_extraction_patterns():
    """Test various day extraction patterns."""
    print("🔍 Testing Day Extraction Patterns")
    print("=" * 50)
    
    test_cases = [
        # From narrator lines
        "[NARRATOR]: Day 1: Welcome to El Nido!",
        "[NARRATOR]: Day 2: At the Hotel",
        "[NARRATOR]: Day 3: Breakfast and Basic Needs",
        
        # From titles  
        "Demo 0.0.3 Day 1",
        "Day 12: Advanced Lesson",
        
        # From filenames
        "demo-0.0.3-day-1.txt",
        "demo-0.0.3-day-15.txt",
        
        # Edge cases
        "This is not a day lesson",
        "Day: Invalid format",
        ""
    ]
    
    for test_case in test_cases:
        # Test the pattern from our implementation
        day_match = re.search(r'day\s*(\d+)', test_case.lower())
        day_number = day_match.group(1) if day_match else "0"
        
        print(f"'{test_case}' -> Day {day_number}")

def test_section_classification():
    """Test section classification patterns."""
    print(f"\n🔍 Testing Section Classification")
    print("=" * 50)
    
    test_sections = [
        "Key Phrases:",
        "Key Phrases",
        "Vocabulary", 
        "Natural Speed Dialogue",
        "Natural Conversation",
        "Slow Speed Practice",
        "Slow Speed",
        "English Translation",
        "Translated Version",
        "Random Section",
        ""
    ]
    
    for section_title in test_sections:
        section_title_lower = section_title.lower() if section_title else ""
        
        if 'key' in section_title_lower or 'phrase' in section_title_lower or 'vocabulary' in section_title_lower:
            suffix = 'a'
        elif 'natural' in section_title_lower or 'normal' in section_title_lower:
            suffix = 'b'
        elif 'slow' in section_title_lower:
            suffix = 'c'
        elif 'translat' in section_title_lower or 'english' in section_title_lower:
            suffix = 'd'
        else:
            suffix = 'a'  # Default
            
        suffix_to_name = {
            'a': 'key_phrases',
            'b': 'natural_speed',
            'c': 'slow_speed',
            'd': 'translated'
        }
        type_name = suffix_to_name.get(suffix, 'key_phrases')
        
        print(f"'{section_title}' -> {suffix} ({type_name})")

if __name__ == "__main__":
    test_day_extraction_patterns()
    test_section_classification()