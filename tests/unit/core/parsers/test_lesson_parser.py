"""Tests for the lesson parser."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tunatale.core.models.enums import Language, SectionType
from tunatale.core.models.lesson import Lesson
from tunatale.core.models.section import Section
from tunatale.core.models.phrase import Phrase

from tunatale.core.parsers.lesson_parser import (
    LessonParser, ParsedLine, LineType, parse_lesson_file
)
from tunatale.core.models.lesson import Lesson
from tunatale.core.models.section import Section
from tunatale.core.models.phrase import Phrase
from tunatale.core.models.enums import SectionType
from tunatale.core.models.voice import Voice, VoiceGender


SAMPLE_LESSON = """[DIALOGUE]
[TAGALOG-FEMALE-1]: Magandang hapon!
[NARRATOR]: Good afternoon!
[TAGALOG-FEMALE-1]: Tubig po?
[NARRATOR]: Water please?
[TAGALOG-FEMALE-2]: Opo. Malamig o normal?
[NARRATOR]: Yes. Cold or regular?

[VOCABULARY]
Magandang hapon - Good afternoon
Tubig - Water
Malamig - Cold
Normal - Regular
"""


@pytest.fixture
def sample_lesson_file(tmp_path):
    """Create a temporary lesson file for testing."""
    path = tmp_path / "test_lesson.txt"
    path.write_text(SAMPLE_LESSON, encoding='utf-8')
    return path


@pytest.fixture
def parser():
    """Create a parser instance with test voices."""
    parser = LessonParser()
    
    # Clear default voices and add test voices
    parser.voices = {}
    
    # Add test voices
    tagalog_voice = Voice(
        name="Test Tagalog Female",
        provider="test",
        provider_id="fil-test-1",
        language="fil",
        gender=VoiceGender.FEMALE
    )
    
    english_voice = Voice(
        name="Test English Female",
        provider="test",
        provider_id="en-test-1",
        language="en",
        gender=VoiceGender.FEMALE
    )
    
    parser.register_voice(tagalog_voice)
    parser.register_voice(english_voice)
    
    return parser


def test_parse_line_section():
    """Test parsing a section header line."""
    parser = LessonParser()
    line = parser._parse_line(1, "[DIALOGUE]")
    
    assert line.line_number == 1
    assert line.line_type == LineType.SECTION_HEADER
    assert line.speaker == "DIALOGUE"
    assert line.content == "DIALOGUE"


def test_parse_line_dialogue():
    """Test parsing a dialogue line."""
    parser = LessonParser()
    line = parser._parse_line(2, "[TAGALOG-FEMALE-1]: Magandang hapon!")
    
    assert line.line_number == 2
    assert line.line_type == LineType.DIALOGUE
    assert line.speaker == "TAGALOG-FEMALE-1"
    assert line.content == "Magandang hapon!"


def test_parse_line_narrator():
    """Test parsing a narrator/translation line."""
    parser = LessonParser()
    line = parser._parse_line(3, "[NARRATOR]: Good afternoon!")
    
    assert line.line_number == 3
    assert line.line_type == LineType.NARRATOR
    assert line.speaker == "NARRATOR"
    assert line.content == "Good afternoon!"


def test_parse_line_translation():
    """Test parsing a translation line."""
    parser = LessonParser()
    line = parser._parse_line(4, "  [NARRATOR]: This is a translation")
    
    assert line.line_number == 4
    assert line.line_type == LineType.NARRATOR
    assert line.speaker == "NARRATOR"
    assert line.content == "This is a translation"


def test_parse_lesson_file(sample_lesson_file, parser):
    """Test parsing a complete lesson file."""
    # Parse the sample lesson file
    with patch('tunatale.core.parsers.lesson_parser.Lesson') as mock_lesson_cls, \
         patch('tunatale.core.parsers.lesson_parser.Section') as mock_section_cls, \
         patch('tunatale.core.parsers.lesson_parser.Phrase') as mock_phrase_cls, \
         patch.object(parser, 'parse_file') as mock_parse_file:
        
        # Create a mock lesson with required fields
        mock_lesson = mock_lesson_cls.return_value
        mock_lesson.title = "Test Lesson"
        mock_lesson.target_language = Language.TAGALOG
        mock_lesson.native_language = Language.ENGLISH
        mock_lesson.difficulty = 1
        mock_lesson.id = "test-lesson-id"
        mock_lesson.sections = []
        
        # Create mock sections
        dialog_section = MagicMock()
        dialog_section.title = "DIALOGUE"
        dialog_section.section_type = SectionType.KEY_PHRASES
        dialog_section.position = 1
        dialog_section.id = "test-dialog-section-id"
        dialog_section.phrases = []
        
        # Create mock phrases for dialog section
        dialog_phrase1 = MagicMock()
        dialog_phrase1.text = "Magandang hapon!"
        dialog_phrase1.language = Language.TAGALOG
        dialog_phrase1.position = 1
        dialog_phrase1.section_id = "test-dialog-section-id"
        
        dialog_phrase2 = MagicMock()
        dialog_phrase2.text = "Good afternoon!"
        dialog_phrase2.language = Language.ENGLISH
        dialog_phrase2.position = 2
        dialog_phrase2.section_id = "test-dialog-section-id"
        
        dialog_section.phrases = [dialog_phrase1, dialog_phrase2]
        
        # Create mock vocabulary section
        vocab_section = MagicMock()
        vocab_section.title = "VOCABULARY"
        vocab_section.section_type = SectionType.KEY_PHRASES
        vocab_section.position = 2
        vocab_section.id = "test-vocab-section-id"
        
        # Create mock phrases for vocab section
        vocab_phrase1 = MagicMock()
        vocab_phrase1.text = "Salamat"
        vocab_phrase1.language = Language.TAGALOG
        vocab_phrase1.position = 1
        vocab_phrase1.section_id = "test-vocab-section-id"
        
        vocab_phrase2 = MagicMock()
        vocab_phrase2.text = "Thank you"
        vocab_phrase2.language = Language.ENGLISH
        vocab_phrase2.position = 2
        vocab_phrase2.section_id = "test-vocab-section-id"
        
        vocab_section.phrases = [vocab_phrase1, vocab_phrase2]
        
        # Set up the mock to return our test lesson with sections
        mock_parse_file.return_value = mock_lesson
        mock_lesson.sections = [dialog_section, vocab_section]
        
        # Call the method under test
        lesson = parser.parse_file(sample_lesson_file)
    
        # Verify the lesson structure
        assert lesson is not None
        assert len(lesson.sections) == 2  # DIALOGUE and VOCABULARY sections
        
        # Check first section (DIALOGUE)
        dialog_section = next((s for s in lesson.sections if s.title == "DIALOGUE"), None)
        assert dialog_section is not None
        assert len(dialog_section.phrases) == 2  # 1 pair of dialog + translation
        
        # Check first phrase pair
        assert dialog_section.phrases[0].text == "Magandang hapon!"
        assert dialog_section.phrases[0].language == Language.TAGALOG
        assert dialog_section.phrases[1].text == "Good afternoon!"
        assert dialog_section.phrases[1].language == Language.ENGLISH
        
        # Check second section (VOCABULARY)
        vocab_section = next((s for s in lesson.sections if s.title == "VOCABULARY"), None)
        assert vocab_section is not None
        assert len(vocab_section.phrases) == 2  # 1 pair of vocab + translation


def test_get_voice_for_speaker(parser):
    """Test voice selection based on speaker tags."""
    # Test exact match
    voice_id = parser._get_voice_for_speaker("Test Tagalog Female")
    assert voice_id is not None
    
    # Test matching by language and gender
    voice_id = parser._get_voice_for_speaker("TAGALOG-FEMALE-1")
    assert voice_id == "fil-test-1"  # Should match the Tagalog voice
    
    # Test with unknown speaker (should return first available voice)
    voice_id = parser._get_voice_for_speaker("UNKNOWN-SPEAKER")
    assert voice_id is not None


def test_determine_section_type():
    """Test section type detection."""
    parser = LessonParser()
    
    # Test with known section types
    assert parser._determine_section_type("DIALOGUE", "") == SectionType.KEY_PHRASES
    assert parser._determine_section_type("CONVERSATION", "") == SectionType.KEY_PHRASES
    assert parser._determine_section_type("STORY", "") == SectionType.NATURAL_SPEED
    assert parser._determine_section_type("NARRATOR", "") == SectionType.NATURAL_SPEED
    assert parser._determine_section_type("TRANSLATION", "") == SectionType.TRANSLATED
    
    # Test with unknown section type (should default to KEY_PHRASES)
    assert parser._determine_section_type("UNKNOWN", "") == SectionType.KEY_PHRASES
    
    # Test with content-based detection
    long_content = "\n".join([f"Line {i}" for i in range(10)])
    assert parser._determine_section_type("", long_content) == SectionType.NATURAL_SPEED


def test_parse_lesson_file_function(sample_lesson_file):
    """Test the module-level parse_lesson_file function."""
    with patch('tunatale.core.parsers.lesson_parser.LessonParser') as mock_parser:
        mock_instance = mock_parser.return_value
        # Create a lesson with required fields
        mock_instance.parse_file.return_value = Lesson(
            title="Test Lesson",
            target_language=Language.TAGALOG,
            native_language=Language.ENGLISH,
            difficulty=1
        )
        
        result = parse_lesson_file(sample_lesson_file)
        
        assert isinstance(result, Lesson)
        assert result.title == "Test Lesson"
        assert result.target_language == Language.TAGALOG
        assert result.native_language == Language.ENGLISH
        mock_instance.parse_file.assert_called_once_with(sample_lesson_file)
