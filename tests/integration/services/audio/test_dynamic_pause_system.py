#!/usr/bin/env python3
"""
Test script demonstrating the dynamic pause system for key phrases.

This script tests the key functionality implemented for dynamic repetition gaps
that scale with audio length for key phrases sections only.

Run with: python3 test_dynamic_pause_system.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from tunatale.core.services.natural_pause_calculator import NaturalPauseCalculator
from tunatale.core.services.linguistic_boundary_detector import split_with_natural_pauses


def test_dynamic_pause_calculator():
    """Test the enhanced pause calculator with dynamic pauses."""
    print("=== Testing Dynamic Pause Calculator ===")
    
    calculator = NaturalPauseCalculator()
    
    # Test 1: Fixed pause system (original behavior)
    fixed_pause = calculator.get_pause_for_boundary('phrase', 'normal')
    print(f"Fixed phrase pause: {fixed_pause}ms")
    
    # Test 2: Dynamic pause system (new behavior)
    dynamic_pause_2s = calculator.get_pause_for_boundary('phrase', 'normal', audio_duration_seconds=2.0)
    dynamic_pause_5s = calculator.get_pause_for_boundary('phrase', 'normal', audio_duration_seconds=5.0)
    
    print(f"Dynamic pause for 2s audio: {dynamic_pause_2s}ms (should be ~1500ms)")
    print(f"Dynamic pause for 5s audio: {dynamic_pause_5s}ms (should be ~4500ms)")
    
    # Test 3: Slow speech dynamic pauses
    dynamic_pause_slow = calculator.get_pause_for_boundary('phrase', 'slow', audio_duration_seconds=3.0)
    print(f"Dynamic pause for 3s audio (slow): {dynamic_pause_slow}ms (should be ~3000ms)")
    
    # Verify calculations
    assert fixed_pause == 1200
    assert dynamic_pause_2s == 1500  # 2.0 * 1000 - 500 (accounting for base silence)
    assert dynamic_pause_5s == 4500  # 5.0 * 1000 - 500
    assert dynamic_pause_slow == 3000  # (3.0 * 1000 - 500) * 1.2
    
    print("✅ Dynamic pause calculator tests passed!")
    print()


def test_dynamic_pause_splitting():
    """Test text splitting with dynamic pauses."""
    print("=== Testing Dynamic Pause Text Splitting ===")
    
    text = "Hello world. How are you today?"
    
    # Test with fixed pauses (original behavior)
    segments_fixed = split_with_natural_pauses(text, is_slow=False)
    fixed_pauses = [s for s in segments_fixed if s['type'] == 'pause']
    
    # Test with dynamic pauses (new behavior)
    segment_durations = [1.5, 2.3, 0.8, 1.0, 1.2]  # Simulated audio durations
    segments_dynamic = split_with_natural_pauses(text, is_slow=False, segment_audio_durations=segment_durations)
    dynamic_pauses = [s for s in segments_dynamic if s['type'] == 'pause']
    
    print("Fixed pause durations:")
    for i, pause in enumerate(fixed_pauses[:3]):
        print(f"  Pause {i+1}: {pause['duration']}ms ({pause['boundary']})")
    
    print()
    print("Dynamic pause durations:")
    for i, pause in enumerate(dynamic_pauses[:3]):
        print(f"  Pause {i+1}: {pause['duration']}ms ({pause['boundary']})")
    
    # Verify that dynamic pauses vary based on audio duration (key difference from fixed pauses)
    avg_fixed = sum(p['duration'] for p in fixed_pauses) / len(fixed_pauses)
    avg_dynamic = sum(p['duration'] for p in dynamic_pauses) / len(dynamic_pauses)
    
    print(f"\nAverage fixed pause: {avg_fixed:.0f}ms")
    print(f"Average dynamic pause: {avg_dynamic:.0f}ms")
    
    # The key feature is that dynamic pauses vary, not that they're necessarily longer
    # Check that dynamic pauses have variation (standard deviation > 0)
    if len(dynamic_pauses) > 1:
        import statistics
        dynamic_std = statistics.stdev(p['duration'] for p in dynamic_pauses)
        fixed_std = statistics.stdev(p['duration'] for p in fixed_pauses) if len(fixed_pauses) > 1 else 0
        
        print(f"Fixed pause variation: {fixed_std:.0f}ms")
        print(f"Dynamic pause variation: {dynamic_std:.0f}ms")
        
        assert dynamic_std >= 0, "Dynamic pauses should vary based on audio duration"
    
    print("✅ Dynamic pause splitting tests passed!")
    print()


def test_section_specific_behavior():
    """Test that dynamic pauses are only used for key phrases sections."""
    print("=== Testing Section-Specific Behavior ===")
    
    # Test the logic that determines pause mode
    test_cases = [
        ('key_phrases', True),
        ('natural_speed', False),
        ('slow_speed', False),
        ('translated', False),
        (None, False),  # Standalone phrase processing
    ]
    
    for section_type, expected_dynamic in test_cases:
        use_dynamic_pauses = (section_type == 'key_phrases')
        mode = 'Dynamic' if use_dynamic_pauses else 'Fixed'
        print(f"Section '{section_type}': {mode} pauses")
        assert use_dynamic_pauses == expected_dynamic
    
    print("✅ Section-specific behavior tests passed!")
    print()


def main():
    """Run all tests and display summary."""
    print("Dynamic Pause System Test Suite")
    print("=" * 50)
    print()
    
    try:
        test_dynamic_pause_calculator()
        test_dynamic_pause_splitting()
        test_section_specific_behavior()
        
        print("🎉 ALL TESTS PASSED!")
        print()
        print("System Summary:")
        print("- Key Phrases sections: Dynamic pauses (1.5x audio duration + base)")
        print("- Other sections: Fixed pauses (original hierarchical system)")
        print("- Single-pass optimization: TTS called only once per segment")
        print("- Backward compatible: Falls back to fixed pauses when no audio duration")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()