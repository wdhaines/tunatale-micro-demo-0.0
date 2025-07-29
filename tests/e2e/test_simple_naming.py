#!/usr/bin/env python3
"""Simple test for file naming logic without full project dependencies."""

import re
from pathlib import Path

def extract_day_number_from_file(filepath):
    """Extract day number from lesson file content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for narrator day announcements
        lines = content.split('\n')
        for line in lines:
            if '[NARRATOR]' in line and 'Day' in line:
                day_match = re.search(r'day\s*(\d+)', line.lower())
                if day_match:
                    return day_match.group(1)
        
        # Fallback to filename
        filename = Path(filepath).name.lower()
        if 'day-' in filename:
            day_match = re.search(r'day-(\d+)', filename)
            if day_match:
                return day_match.group(1)
                
        return "0"
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return "0"

def classify_section_type_simple(section_title):
    """Simple section classification."""
    section_title = section_title.lower() if section_title else ""
    
    if 'key' in section_title or 'phrase' in section_title or 'vocabulary' in section_title:
        return 'a'
    elif 'natural' in section_title or 'normal' in section_title:
        return 'b'
    elif 'slow' in section_title:
        return 'c'
    elif 'translat' in section_title or 'english' in section_title:
        return 'd'
    
    return 'a'  # Default

def get_section_type_name(suffix):
    """Get readable name for section type."""
    suffix_to_name = {
        'a': 'key_phrases',
        'b': 'natural_speed',
        'c': 'slow_speed',
        'd': 'translated'
    }
    return suffix_to_name.get(suffix, 'key_phrases')

def test_day_files():
    """Test with actual day files."""
    print("🔍 Testing File Naming Logic")
    print("=" * 50)
    
    day_files = [
        "tagalog/demo-0.0.3-day-1.txt",
        "tagalog/demo-0.0.3-day-2.txt",
        "tagalog/demo-0.0.3-day-3.txt"
    ]
    
    for day_file in day_files:
        day_path = Path(day_file)
        if day_path.exists():
            day_number = extract_day_number_from_file(day_path)
            print(f"\n📁 {day_file}")
            print(f"   Day number: {day_number}")
            print(f"   Main lesson file: {day_number} - lesson.mp3")
            
            # Read file and look for sections
            try:
                with open(day_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple section detection (look for common patterns)
                sections = []
                lines = content.split('\n')
                current_section = None
                
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('[') and ':' in line and len(line) < 50:
                        # Likely a section header
                        current_section = line
                        sections.append(current_section)
                
                # Show expected section filenames
                for i, section_title in enumerate(sections[:4]):  # Limit to first 4
                    suffix = classify_section_type_simple(section_title)
                    type_name = get_section_type_name(suffix)
                    print(f"   Section {i+1}: {day_number}.{suffix} – {type_name}.mp3")
                    
            except Exception as e:
                print(f"   Error reading content: {e}")
        else:
            print(f"❌ {day_file}: File not found")

if __name__ == "__main__":
    test_day_files()