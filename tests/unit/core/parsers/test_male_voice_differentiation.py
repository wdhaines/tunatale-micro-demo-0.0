"""Tests for male voice differentiation in the lesson parser."""
import os
import tempfile
import asyncio
import pytest
from pathlib import Path

from tunatale.core.parsers.lesson_parser import LessonParser, parse_lesson_file
from tunatale.core.models.voice import Voice, VoiceGender


@pytest.fixture
def parser():
    """Create a parser instance with test voices."""
    parser = LessonParser()
    
    # Add test voices
    tagalog_male_voice = Voice(
        name="Test Tagalog Male",
        provider="test",
        provider_id="fil-PH-AngeloNeural",
        language="fil",
        gender=VoiceGender.MALE
    )
    
    english_voice = Voice(
        name="Test English Female",
        provider="test",
        provider_id="en-test-1",
        language="en",
        gender=VoiceGender.FEMALE
    )
    
    # Clear and register test voices
    parser.voices = {}
    parser.register_voice(tagalog_male_voice)
    parser.register_voice(english_voice)
    
    return parser


def test_tagalog_male_voice_differentiation(parser):
    """Test that TAGALOG-MALE-1 and TAGALOG-MALE-2 use the same voice but with different pitch/rate settings."""
    # Both speakers should use the same voice (Angelo)
    voice_id_1 = parser._get_voice_for_speaker("TAGALOG-MALE-1")
    voice_id_2 = parser._get_voice_for_speaker("TAGALOG-MALE-2")
    
    # Both should use the same voice
    assert voice_id_1 == "fil-PH-AngeloNeural", \
        f"TAGALOG-MALE-1 should use fil-PH-AngeloNeural, got {voice_id_1}"
    assert voice_id_2 == "fil-PH-AngeloNeural", \
        f"TAGALOG-MALE-2 should use fil-PH-AngeloNeural, got {voice_id_2}"
    
    # Test with a dialogue that includes both male speakers
    test_dialogue = """[DIALOGUE]
[TAGALOG-MALE-1]: Magandang umaga!
[TAGALOG-MALE-2]: Magandang umaga din po!
"""
    
    # Create a temporary file with the test dialogue
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_dialogue)
        temp_path = f.name
    
    try:
        # Parse the test dialogue
        lesson = asyncio.run(parse_lesson_file(temp_path))
        
        # Verify we have one section with two phrases
        assert len(lesson.sections) == 1
        section = lesson.sections[0]
        assert len(section.phrases) == 2
        
        # Verify each phrase has the correct voice, speaker ID, and TTS settings in metadata
        phrase1 = section.phrases[0]
        assert phrase1.voice_id == "fil-PH-AngeloNeural"
        assert phrase1.metadata.get('speaker') == "TAGALOG-MALE-1"
        
        # TAGALOG-MALE-1 should have default pitch/rate settings
        assert phrase1.metadata.get('tts_pitch') is None or float(phrase1.metadata.get('tts_pitch', 0)) == 0.0
        assert phrase1.metadata.get('tts_rate') is None or float(phrase1.metadata.get('tts_rate', 1.0)) == 1.0
        
        phrase2 = section.phrases[1]
        assert phrase2.voice_id == "fil-PH-AngeloNeural"
        assert phrase2.metadata.get('speaker') == "TAGALOG-MALE-2"
        
        # TAGALOG-MALE-2 should have custom pitch/rate settings
        # These values should match the implementation in the voice selector
        assert float(phrase2.metadata.get('tts_pitch', 0)) == 10.0  # Slightly higher pitch
        assert float(phrase2.metadata.get('tts_rate', 1.0)) == 0.9   # Slightly slower rate
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
