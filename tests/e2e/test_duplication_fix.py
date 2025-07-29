#!/usr/bin/env python3
"""Test script to demonstrate the Key Phrases duplication fix."""

def test_filename_uniqueness():
    """Test that the new naming scheme prevents file collisions."""
    print("🔧 Key Phrases Duplication Fix Test")
    print("=" * 60)
    
    # Simulate the scenario that was causing duplication
    scenarios = [
        {
            "description": "Before fix - both sections get same filename",
            "sections": [
                {"id": "abc123", "title": "Section 1", "type": "a", "type_name": "key_phrases"},
                {"id": "def456", "title": "Key Phrases", "type": "a", "type_name": "key_phrases"}
            ],
            "old_naming": True
        },
        {
            "description": "After fix - sections get unique filenames", 
            "sections": [
                {"id": "abc123", "title": "Section 1", "type": "a", "type_name": "key_phrases"},
                {"id": "def456", "title": "Key Phrases", "type": "a", "type_name": "key_phrases"}
            ],
            "old_naming": False
        }
    ]
    
    day_number = "0"  # From syllable-test.txt
    
    for scenario in scenarios:
        print(f"\n{scenario['description']}:")
        print("-" * 50)
        
        filenames = []
        for section in scenario['sections']:
            if scenario['old_naming']:
                # Old naming scheme (caused collisions)
                filename = f"{day_number}.{section['type']} – {section['type_name']}.mp3"
            else:
                # New naming scheme (prevents collisions)
                section_id_short = section['id'][:8]
                filename = f"{day_number}.{section['type']}_{section_id_short} – {section['type_name']}.mp3"
            
            filenames.append(filename)
            print(f"  {section['title']:12} → {filename}")
        
        # Check for duplicates
        unique_filenames = set(filenames)
        if len(unique_filenames) == len(filenames):
            print("  ✅ All filenames unique - no collisions")
        else:
            print("  ❌ Duplicate filenames detected - will cause overwriting")

def test_other_section_types():
    """Test that different section types still work correctly."""
    print(f"\n🎯 Other Section Types Test")
    print("=" * 60)
    
    sections = [
        {"id": "abc123", "title": "Key Phrases", "type": "a", "type_name": "key_phrases"},
        {"id": "def456", "title": "Natural Speed", "type": "b", "type_name": "natural_speed"},
        {"id": "ghi789", "title": "Slow Speed", "type": "c", "type_name": "slow_speed"},
        {"id": "jkl012", "title": "Translated", "type": "d", "type_name": "translated"}
    ]
    
    day_number = "1"
    
    print("Expected filenames for different section types:")
    for section in sections:
        section_id_short = section['id'][:8]
        filename = f"{day_number}.{section['type']}_{section_id_short} – {section['type_name']}.mp3"
        print(f"  {section['title']:12} → {filename}")

if __name__ == "__main__":
    test_filename_uniqueness()
    test_other_section_types()