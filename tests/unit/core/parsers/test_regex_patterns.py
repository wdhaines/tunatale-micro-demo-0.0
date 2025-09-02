"""Unit tests for various regex patterns used in parsing."""
import re
import pytest


@pytest.mark.parametrize("test_case, expected_day", [
    ("[NARRATOR]: Day 1: Welcome to El Nido!", "1"),
    ("[NARRATOR]: Day 2: At the Hotel", "2"),
    ("Demo 0.0.3 Day 1", "1"),
    ("Day 12: Advanced Lesson", "12"),
    ("demo-0.0.3-day-1.txt", "1"),
    ("demo-0.0.3-day-15.txt", "15"),
    ("This is not a day lesson", None),
    ("Day: Invalid format", None),
    ("", None),
])
def test_day_extraction_patterns(test_case, expected_day):
    """Test various day extraction patterns."""
    day_match = re.search(r'day[- ]?(\d+)', test_case.lower())
    if expected_day:
        assert day_match is not None
        assert day_match.group(1) == expected_day
    else:
        assert day_match is None
