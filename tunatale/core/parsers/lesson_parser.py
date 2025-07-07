"""Parser for TunaTale lesson files."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Union, TYPE_CHECKING

# Import models with conditional imports to avoid circular imports
if TYPE_CHECKING:
    from tunatale.core.models.lesson import Lesson
    from tunatale.core.models.section import Section
    from tunatale.core.models.phrase import Phrase
    from tunatale.core.models.voice import Voice

# Import models for runtime
from tunatale.core.models.lesson import Lesson
from tunatale.core.models.section import Section
from tunatale.core.models.phrase import Phrase
from tunatale.core.models.voice import Voice
from tunatale.core.models.enums import VoiceGender, SectionType, Language

if TYPE_CHECKING:
    from ..models.lesson import Lesson, Section
    from ..models.phrase import Phrase


class LineType(Enum):
    """Types of lines in a lesson file."""
    BLANK = auto()
    COMMENT = auto()
    SECTION_HEADER = auto()
    DIALOGUE = auto()
    NARRATOR = auto()
    TRANSLATION = auto()


@dataclass
class ParsedLine:
    """A parsed line from a lesson file."""
    line_number: int
    line_type: LineType
    content: str
    speaker: Optional[str] = None
    language: Optional[str] = None
    voice_id: Optional[str] = None
    
    def __str__(self) -> str:
        return f"{self.line_number}: {self.line_type.name} - {self.content}"


class LessonParser:
    """Parser for TunaTale lesson files."""
    
    # Regex patterns for different line types
    # Section headers must be at the start of the line, no leading content, and have valid section names
    SECTION_PATTERN = re.compile(r'^\s*\[(?P<type>DIALOGUE|VOCABULARY|GRAMMAR|NOTES|EXERCISES)\]\s*$')
    # Dialogue lines have a speaker in brackets followed by a colon and content
    DIALOGUE_PATTERN = re.compile(r'^\s*\[(?P<speaker>[^\]]+)\]\s*:\s*(?P<content>.*)$')
    # Translation lines are specifically marked with [NARRATOR]:
    TRANSLATION_PATTERN = re.compile(r'^\s*\[NARRATOR\]\s*:\s*(?P<translation>.*)$')
    # Comments start with #
    COMMENT_PATTERN = re.compile(r'^\s*#.*$')
    
    def __init__(self):
        self.current_section: Optional[Section] = None
        self.current_phrase: Optional[Phrase] = None
        self.voices: Dict[str, Voice] = {}
        self._register_default_voices()
    
    def _register_default_voices(self) -> None:
        """Register default voices for common speaker patterns."""
        # Default voices for common speaker patterns
        self.register_voice(
            Voice(
                name="Tagalog Female 1",
                provider="edge_tts",
                provider_id="fil-PH-BlessicaNeural",
                language=Language.TAGALOG,
                gender=VoiceGender.FEMALE,
                is_active=True
            )
        )
        self.register_voice(
            Voice(
                name="Tagalog Male 1",
                provider="edge_tts",
                provider_id="fil-PH-JasminNeural",
                language="fil",
                gender=VoiceGender.MALE,
                is_active=True
            )
        )
        self.register_voice(
            Voice(
                name="English Female 1",
                provider="edge_tts",
                provider_id="en-US-AriaNeural",
                language="en",
                gender=VoiceGender.FEMALE,
                is_active=True
            )
        )
    
    def register_voice(self, voice: Voice) -> None:
        """Register a voice with the parser."""
        self.voices[voice.name] = voice
    
    def parse_file(self, file_path: Union[str, Path]) -> Lesson:
        """Parse a lesson file and return a Lesson object."""
        path = Path(file_path)
        lines = path.read_text(encoding='utf-8').splitlines()
        
        # Create a new lesson
        lesson = Lesson(
            title=path.stem.replace('_', ' ').title(),
            target_language=Language.TAGALOG,  # Default to Tagalog
            native_language=Language.ENGLISH,  # Default to English
            description=f"Generated from {path.name}"
        )
        
        parsed_lines = self._parse_lines(lines)
        self._build_lesson(lesson, parsed_lines)
        
        return lesson
    
    def _parse_lines(self, lines: List[str]) -> List[ParsedLine]:
        """Parse all lines in the file."""
        parsed_lines = []
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            parsed_line = self._parse_line(i, line)
            if parsed_line:
                parsed_lines.append(parsed_line)
        
        return parsed_lines
    
    def _parse_line(self, line_number: int, line: str) -> Optional[ParsedLine]:
        """Parse a single line and return a ParsedLine object.
        
        Args:
            line_number: The line number in the source file
            line: The line content to parse
            
        Returns:
            A ParsedLine object representing the parsed line, or None if the line should be skipped
        """
        # Strip whitespace from the line
        line = line.strip()
        
        # Handle empty lines and comments
        if not line or self.COMMENT_PATTERN.match(line):
            return ParsedLine(
                line_number=line_number,
                line_type=LineType.BLANK,
                content=line
            )
        
        # Check for section headers (e.g., [DIALOGUE] or [VOCABULARY])
        section_match = self.SECTION_PATTERN.match(line)
        if section_match:
            return ParsedLine(
                line_number=line_number,
                line_type=LineType.SECTION_HEADER,
                content=section_match.group('type'),
                speaker=section_match.group('type')
            )
        
        # Check for translation lines (e.g., [NARRATOR]: Some text)
        translation_match = self.TRANSLATION_PATTERN.match(line)
        if translation_match:
            # For test compatibility, use NARRATOR type if it exists, otherwise TRANSLATION
            line_type = LineType.NARRATOR if hasattr(LineType, 'NARRATOR') else LineType.TRANSLATION
            return ParsedLine(
                line_number=line_number,
                line_type=line_type,
                content=translation_match.group('translation').strip(),
                speaker='NARRATOR'
            )
        
        # Check for dialogue lines (e.g., [TAGALOG-FEMALE-1]: Some text)
        dialogue_match = self.DIALOGUE_PATTERN.match(line)
        if dialogue_match:
            speaker = dialogue_match.group('speaker').strip()
            content = dialogue_match.group('content').strip()
            
            # Skip empty dialogue lines
            if not content:
                return ParsedLine(
                    line_number=line_number,
                    line_type=LineType.BLANK,
                    content=line
                )
                
            return ParsedLine(
                line_number=line_number,
                line_type=LineType.DIALOGUE,
                content=content,
                speaker=speaker
            )
        
        # If we get here, it's an unrecognized line type - treat as plain text
        return ParsedLine(
            line_number=line_number,
            line_type=LineType.DIALOGUE,
            content=line,
            speaker=None
        )
    
    def _build_lesson(self, lesson: Lesson, parsed_lines: List[ParsedLine]) -> None:
        """Build a Lesson object from parsed lines."""
        current_section = None
        current_phrase = None
        
        for i, parsed in enumerate(parsed_lines):
            if parsed.line_type == LineType.SECTION_HEADER:
                # Create a new section
                section_type = self._determine_section_type(parsed.speaker, parsed.content)
                current_section = Section(
                    title=parsed.content or f"Section {len(lesson.sections) + 1}",
                    section_type=section_type,
                    lesson_id=lesson.id,
                    position=len(lesson.sections) + 1
                )
                lesson.add_section(current_section)
                current_phrase = None
                
            elif parsed.line_type == LineType.DIALOGUE and current_section:
                # Create a new phrase
                voice_id = self._get_voice_for_speaker(parsed.speaker)
                current_phrase = Phrase(
                    text=parsed.content,
                    language=lesson.target_language,  # Default to target language
                    voice_id=voice_id,
                    position=len(current_section.phrases) + 1
                )
                current_section.add_phrase(current_phrase)
                
                # Check if next line is a translation
                if i + 1 < len(parsed_lines) and parsed_lines[i+1].line_type == LineType.TRANSLATION:
                    translation = parsed_lines[i+1]
                    # Add translation as a separate phrase in the same section
                    translation_phrase = Phrase(
                        text=translation.content,
                        language=lesson.native_language,
                        voice_id=self._get_voice_for_speaker("NARRATOR"),
                        position=len(current_section.phrases) + 1,
                        metadata={"is_translation": True, "original_text": parsed.content}
                    )
                    current_section.add_phrase(translation_phrase)
    
    def _determine_section_type(self, section_header: str, content: str) -> SectionType:
        """Determine the section type from the header and content."""
        section_header = section_header.upper()
        
        # Try to determine section type from header
        if 'DIALOG' in section_header or 'CONVERSATION' in section_header:
            return SectionType.KEY_PHRASES
        elif 'NARRATOR' in section_header or 'STORY' in section_header:
            return SectionType.NATURAL_SPEED
        elif 'TRANSLAT' in section_header:
            return SectionType.TRANSLATED
        
        # Try to determine from content if header is not clear
        if not content:
            return SectionType.KEY_PHRASES  # Default
            
        # Count lines to guess section type
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) > 3:
            return SectionType.NATURAL_SPEED
        return SectionType.KEY_PHRASES
    
    def _get_voice_for_speaker(self, speaker: Optional[str]) -> Optional[str]:
        """Get the appropriate voice ID for a speaker.
        
        Args:
            speaker: The speaker tag or name to find a voice for
            
        Returns:
            The provider_id of the matching voice as a string, or None if no match found
        """
        if not speaker:
            return None
            
        # Try exact match first
        for voice in self.voices.values():
            if voice.name.lower() == speaker.lower():
                return str(voice.provider_id)  # Return provider_id as string
                
        # Try to match by language and gender from speaker tag
        # Format: LANGUAGE-GENDER-NUMBER (e.g., TAGALOG-FEMALE-1)
        parts = speaker.upper().split('-')
        if len(parts) >= 2:
            language = parts[0]
            gender = parts[1] if len(parts) > 1 else None
            
            # Map language codes
            if language == 'TAGALOG':
                language = 'fil'
            elif language == 'ENGLISH':
                language = 'en'
                
            # Map gender
            gender_map = {
                'MALE': VoiceGender.MALE,
                'FEMALE': VoiceGender.FEMALE,
                'NEUTRAL': VoiceGender.NEUTRAL
            }
            voice_gender = gender_map.get(gender) if gender else None
            
            # Find matching voice
            for voice in self.voices.values():
                if (voice.language == language and 
                    (not voice_gender or voice.gender == voice_gender)):
                    return str(voice.provider_id)  # Return provider_id as string
                    
        # Return first available voice if no match found
        if self.voices:
            return str(next(iter(self.voices.values())).provider_id)  # Return provider_id as string
            
        return None


def parse_lesson_file(file_path: Union[str, Path]) -> Lesson:
    """Parse a lesson file and return a Lesson object."""
    parser = LessonParser()
    return parser.parse_file(file_path)
