#!/usr/bin/env python3
"""Test the new file naming scheme."""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from tunatale.core.parsers.lesson_parser import LessonParser
from tunatale.core.services.lesson_processor import LessonProcessor

def test_day_extraction():
    """Test day number extraction from lesson files."""
    print("🔍 Testing Day Number Extraction")
    print("=" * 50)
    
    # Create a dummy lesson processor to test methods
    processor = LessonProcessor(
        tts_service=None,  # We won't actually process TTS
        audio_processor=None,
        voice_selector=None,
        word_selector=None
    )
    
    # Parse an actual lesson file
    lesson_file = Path("tagalog/demo-0.0.3-day-1.txt")
    if lesson_file.exists():
        parser = LessonParser()
        lesson = parser.parse_file(lesson_file)
        
        print(f"Lesson title: {lesson.title}")
        print(f"Lesson description: {lesson.description}")
        
        # Test day number extraction
        day_number = processor._extract_day_number(lesson)
        print(f"Extracted day number: {day_number}")
        
        # Test section classification
        print(f"\nSection Classifications:")
        for i, section in enumerate(lesson.sections):
            section_suffix = processor._classify_section_type(section)
            section_type = processor._get_section_type_name(section_suffix)
            print(f"  Section {i+1}: '{section.title}' -> {section_suffix} ({section_type})")
            
        # Show what the file names would be
        print(f"\nExpected File Names:")
        print(f"  Main lesson: {day_number} - lesson.mp3")
        
        for i, section in enumerate(lesson.sections):
            section_suffix = processor._classify_section_type(section)
            section_type = processor._get_section_type_name(section_suffix)
            print(f"  Section {i+1}: {day_number}.{section_suffix} – {section_type}.mp3")
            
    else:
        print(f"❌ Lesson file not found: {lesson_file}")

def test_other_days():
    """Test with other day files."""
    print(f"\n🔍 Testing Other Day Files")
    print("=" * 50)
    
    processor = LessonProcessor(
        tts_service=None,
        audio_processor=None,
        voice_selector=None,
        word_selector=None
    )
    
    # Test other day files
    day_files = [
        "tagalog/demo-0.0.3-day-2.txt",
        "tagalog/demo-0.0.3-day-3.txt"
    ]
    
    for day_file in day_files:
        day_path = Path(day_file)
        if day_path.exists():
            parser = LessonParser()
            lesson = parser.parse_file(day_path)
            day_number = processor._extract_day_number(lesson)
            print(f"{day_file}: Day {day_number}")
        else:
            print(f"❌ {day_file}: File not found")

if __name__ == "__main__":
    test_day_extraction()
    test_other_days()