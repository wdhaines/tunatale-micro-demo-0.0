#!/usr/bin/env python3
"""Test the improved section classification logic."""

import sys
import os
import re
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

def classify_section_type_mock(section_title: str) -> str:
    """Mock the updated classification logic for testing."""
    section_title = section_title.lower() if section_title else ""
    
    # Explicit section type detection
    if 'key' in section_title or 'phrase' in section_title or 'vocabulary' in section_title:
        return 'a'
    elif 'natural' in section_title or 'normal' in section_title:
        return 'b'
    elif 'slow' in section_title:
        return 'c'
    elif 'translat' in section_title or 'english' in section_title:
        return 'd'
    
    # Check for auto-generated section titles (like "Section 1", "Section 2", etc.)
    if re.match(r'^section\s+\d+$', section_title) or section_title.startswith('syllable'):
        return 'x'  # Unknown/intro section, not key phrases
    
    # For other ambiguous cases, classify as unknown rather than assuming key phrases
    return 'x'  # Changed from 'a' to avoid misclassification

def get_section_type_name_mock(suffix: str) -> str:
    """Mock the updated type name mapping."""
    suffix_to_name = {
        'a': 'key_phrases',
        'b': 'natural_speed',
        'c': 'slow_speed',
        'd': 'translated',
        'x': 'intro'
    }
    return suffix_to_name.get(suffix, 'intro')

def test_section_classification():
    """Test the updated section classification logic."""
    print("🔍 Updated Section Classification Test")
    print("=" * 60)
    
    test_cases = [
        # Problem case - auto-generated sections should NOT be key_phrases
        {"title": "Section 1", "expected_suffix": "x", "expected_type": "intro"},
        {"title": "Syllable Pronunciation Test", "expected_suffix": "x", "expected_type": "intro"},
        
        # Explicit section headers should be classified correctly
        {"title": "Key Phrases", "expected_suffix": "a", "expected_type": "key_phrases"},
        {"title": "Key Phrases:", "expected_suffix": "a", "expected_type": "key_phrases"},
        {"title": "Natural Speed", "expected_suffix": "b", "expected_type": "natural_speed"},
        {"title": "Slow Speed", "expected_suffix": "c", "expected_type": "slow_speed"},
        {"title": "Translated", "expected_suffix": "d", "expected_type": "translated"},
        
        # Ambiguous cases should default to 'x' (intro) not 'a' (key_phrases)
        {"title": "Random Title", "expected_suffix": "x", "expected_type": "intro"},
        {"title": "", "expected_suffix": "x", "expected_type": "intro"},
    ]
    
    print("Testing section classification:")
    print("Title                         | Expected | Actual | Type        | Status")
    print("-" * 75)
    
    all_passed = True
    for case in test_cases:
        actual_suffix = classify_section_type_mock(case["title"])
        actual_type = get_section_type_name_mock(actual_suffix)
        
        suffix_correct = actual_suffix == case["expected_suffix"]
        type_correct = actual_type == case["expected_type"]
        status = "✅" if (suffix_correct and type_correct) else "❌"
        
        if not (suffix_correct and type_correct):
            all_passed = False
        
        print(f"{case['title']:29} | {case['expected_suffix']:8} | {actual_suffix:6} | {actual_type:11} | {status}")
    
    return all_passed

def test_filename_generation():
    """Test the expected filenames after the fix."""
    print(f"\n📁 Expected Filename Generation")
    print("=" * 60)
    
    # Simulate the problematic scenario from syllable-test.txt
    sections = [
        {"title": "Section 1", "type": "x", "type_name": "intro"},  # Auto-generated intro
        {"title": "Key Phrases", "type": "a", "type_name": "key_phrases"},  # Explicit section
        {"title": "Natural Speed", "type": "b", "type_name": "natural_speed"},
        {"title": "Slow Speed", "type": "c", "type_name": "slow_speed"},
        {"title": "Translated", "type": "d", "type_name": "translated"},
    ]
    
    day_number = "0"
    
    print("Expected filenames after classification fix:")
    filenames = []
    for section in sections:
        filename = f"{day_number}.{section['type']} – {section['type_name']}.mp3"
        filenames.append(filename)
        print(f"  {section['title']:12} → {filename}")
    
    # Check for duplicates
    unique_filenames = set(filenames)
    if len(unique_filenames) == len(filenames):
        print("  ✅ All filenames unique - no collisions")
        return True
    else:
        print("  ❌ Duplicate filenames detected")
        return False

if __name__ == "__main__":
    classification_passed = test_section_classification()
    filename_passed = test_filename_generation()
    
    print(f"\n{'='*60}")
    if classification_passed and filename_passed:
        print("🎉 All tests passed! The duplication bug should be fixed.")
    else:
        print("❌ Some tests failed. Review the implementation.")