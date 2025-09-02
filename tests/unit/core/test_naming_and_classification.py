"""Unit tests for file naming and section classification logic."""
import pytest
from unittest.mock import MagicMock

from tunatale.core.parsers.lesson_parser import LessonParser
from tunatale.core.services.lesson_processor import LessonProcessor
from tunatale.core.models.enums import Language, SectionType
import pytest
from unittest.mock import MagicMock

from tunatale.core.parsers.lesson_parser import LessonParser
from tunatale.core.services.lesson_processor import LessonProcessor
from tunatale.core.models.lesson import Lesson
from tunatale.core.models.section import Section
from tunatale.core.models.enums import Language, SectionType
from tunatale.core.models.phrase import Phrase

@pytest.fixture
def lesson_processor():
    """Returns a LessonProcessor instance with dummy services."""
    return LessonProcessor(
        tts_service=None,
        audio_processor=None,
        voice_selector=None,
        word_selector=None
    )


class TestSectionClassification:
    """Tests for the section classification logic in LessonProcessor."""

    @pytest.mark.parametrize("title, expected_suffix", [
        ("Section 1", 'x'),
        ("Syllable Pronunciation Test", 'x'),
        ("Key Phrases", 'a'),
        ("Key Phrases:", 'a'),
        ("Natural Speed", 'b'),
        ("Slow Speed", 'c'),
        ("Translated", 'd'),
        ("Random Title", 'x'),
        ("Intro", 'x'),
    ])
    def test_section_classification_suffix(self, lesson_processor: LessonProcessor, title, expected_suffix):
        section = Section(title=title, phrases=[], section_type=SectionType.KEY_PHRASES, lesson_id="dummy_id")
        if title:
            assert lesson_processor._classify_section_type(section) == expected_suffix

    @pytest.mark.parametrize("suffix, expected_type_name", [
        ('a', 'key_phrases'),
        ('b', 'natural_speed'),
        ('c', 'slow_speed'),
        ('d', 'translated'),
        ('x', 'intro'),
    ])
    def test_section_type_name_mapping(self, lesson_processor: LessonProcessor, suffix, expected_type_name):
        assert lesson_processor._get_section_type_name(suffix) == expected_type_name


class TestDayNumberExtraction:
    """Tests for day number extraction logic."""

    def test_extract_day_number_from_title(self, lesson_processor: LessonProcessor):
        lesson = Lesson(title="Lesson for Day 15", sections=[], target_language=Language.ENGLISH)
        assert lesson_processor._extract_day_number(lesson) == "15"

    def test_extract_day_number_from_narrator(self, lesson_processor: LessonProcessor):
        mock_phrase = Phrase(text="Welcome to Day 3 of your journey.", language=Language.ENGLISH, voice_id="dummy_voice", metadata={"speaker": "NARRATOR"})
        lesson = Lesson(title="My Lesson", sections=[
            Section(title="Intro", phrases=[
                mock_phrase
            ], section_type=SectionType.KEY_PHRASES, lesson_id="dummy_id")
        ], target_language=Language.ENGLISH)
        assert lesson_processor._extract_day_number(lesson) == "3"

    def test_extract_day_number_from_filename(self, lesson_processor: LessonProcessor):
        lesson = Lesson(title="My Lesson", sections=[], source_filename="/path/to/demo-day-7.txt", target_language=Language.ENGLISH)
        assert lesson_processor._extract_day_number(lesson) == "0"

    def test_day_number_extraction_fallback(self, lesson_processor: LessonProcessor):
        lesson = Lesson(title="My Lesson", sections=[], target_language=Language.ENGLISH)
        assert lesson_processor._extract_day_number(lesson) == "0"
