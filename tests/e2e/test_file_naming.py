#!/usr/bin/env python3
"""Test the new file naming scheme."""

import sys
import os
from pathlib import Path
import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tunatale.core.parsers.lesson_parser import LessonParser
from tunatale.core.services.lesson_processor import LessonProcessor

@pytest.mark.asyncio
async def test_day_extraction():
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
        lesson = await parser.parse_file(lesson_file)
        
        print(f"Lesson title: {lesson.title}")
        print(f"Sections found: {len(lesson.sections)}")
        
        # Test the day number extraction
        day_number = processor._extract_day_number(lesson)
        print(f"\nExtracted day number: {day_number}")
        
        # Test the file naming
        print(f"\nExpected File Names:")
        print(f"  Main lesson: {day_number} - lesson.mp3")
        
        for i, section in enumerate(lesson.sections):
            section_suffix = processor._classify_section_type(section)
            section_type = processor._get_section_type_name(section_suffix)
            print(f"  Section {i+1}: {day_number}.{section_suffix} – {section_type}.mp3")
            
    else:
        print(f"❌ Lesson file not found: {lesson_file}")
        pytest.fail(f"Lesson file not found: {lesson_file}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_day_extraction())
