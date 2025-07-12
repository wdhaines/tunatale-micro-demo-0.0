"""Parser for TunaTale lesson files."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union, TYPE_CHECKING

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
    # Section headers are lines with just [NARRATOR]: Section Name
    SECTION_PATTERN = re.compile(r'^\s*\[NARRATOR\]\s*:\s*(?P<content>.*?)\s*$')
    # Dialogue lines have a speaker in brackets followed by a colon and content
    DIALOGUE_PATTERN = re.compile(r'^\s*\[(?P<speaker>[^\]]+)\]\s*:\s*(?P<content>.*)$')
    # Comments start with #
    COMMENT_PATTERN = re.compile(r'^\s*#.*$')
    # Speaker pattern for voice mapping (e.g., TAGALOG-FEMALE-1)
    SPEAKER_PATTERN = re.compile(r'^([A-Z]+)-([A-Z]+)-(\d+)$')
    
    def __init__(self):
        self.current_section: Optional[Section] = None
        self.current_phrase: Optional[Phrase] = None
        self.voices: Dict[str, Voice] = {}
        self.last_voice_id: Optional[str] = None  # Track the last used voice ID
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
                provider_id="fil-PH-BlessicaNeural",
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
    
    async def parse_file(
        self, 
        file_path: Union[str, Path], 
        progress_callback: Optional[Callable[[int, int, str, Dict[str, Any]], Awaitable[None]]] = None
    ) -> Lesson:
        """Parse a lesson file and return a Lesson object.
        
        Args:
            file_path: Path to the lesson file
            progress_callback: Optional callback for progress updates
        """
        path = Path(file_path)
        lines = path.read_text(encoding='utf-8').splitlines()
        total_lines = len(lines)
        
        # Create a new lesson
        lesson = Lesson(
            title=path.stem.replace('_', ' ').title(),
            target_language=Language.TAGALOG,  # Default to Tagalog
            native_language=Language.ENGLISH,  # Default to English
            description=f"Generated from {path.name}"
        )
        
        # Parse lines with progress reporting
        if progress_callback:
            await progress_callback(0, total_lines, "Parsing lesson file...", {"phase": "parsing"})
        
        parsed_lines = []
        for i, line in enumerate(lines, 1):
            parsed_line = self._parse_line(i, line)
            if parsed_line:
                parsed_lines.append(parsed_line)
            if progress_callback and i % 5 == 0:  # Update progress every 5 lines
                await progress_callback(i, total_lines, f"Parsing line {i}/{total_lines}...", {"phase": "parsing"})
        
        if progress_callback:
            await progress_callback(total_lines, total_lines, "Parsing complete", {"phase": "parsing", "completed": True})
        
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
        try:
            # Strip whitespace from the line
            line = line.strip()
            
            # Handle empty lines and comments
            if not line or self.COMMENT_PATTERN.match(line):
                return ParsedLine(
                    line_number=line_number,
                    line_type=LineType.BLANK,
                    content=line
                )
            
            # Check for section headers (e.g., [NARRATOR]: DIALOGUE:)
            section_match = self.SECTION_PATTERN.match(line)
            if section_match:
                content = section_match.group('content').strip()
                # Check if this is a section header (ends with colon or is all caps)
                if content.endswith(':') or (content.isupper() and ' ' in content):
                    section_name = content.strip(' :')
                    if section_name:  # Only return section header if we have a name
                        return ParsedLine(
                            line_number=line_number,
                            line_type=LineType.SECTION_HEADER,
                            content=section_name,
                            speaker='NARRATOR',
                            voice_id=self._get_voice_for_speaker('NARRATOR')
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
                
                # Determine line type based on speaker
                if speaker.upper() == 'NARRATOR':
                    line_type = LineType.NARRATOR
                else:
                    line_type = LineType.DIALOGUE
                
                # Get the voice for this speaker and update last_voice_id
                voice_id = self._get_voice_for_speaker(speaker)
                
                # Ensure last_voice_id is updated for dialogue lines
                self.last_voice_id = voice_id
                
                return ParsedLine(
                    line_number=line_number,
                    line_type=line_type,
                    content=content,
                    speaker=speaker,
                    voice_id=voice_id,
                    language="tagalog" if speaker and ('TAGALOG' in speaker.upper() or 'FILIPINO' in speaker.upper()) else "english"
                )
            
            # If we get here, it's an unrecognized line type - treat as plain text
            # Check if it looks like a phrase (no brackets, not empty after strip)
            if line and '[' not in line and ']' not in line:
                # For plain text lines (breakdown lines), always use the Tagalog voice
                # This ensures all breakdown lines are in Tagalog
                voice_id = "fil-PH-BlessicaNeural"
                
                # For plain text lines, use DIALOGUE type as per test expectations
                return ParsedLine(
                    line_number=line_number,
                    line_type=LineType.DIALOGUE,
                    content=line,
                    speaker=None,
                    voice_id=voice_id,
                    language="tagalog"  # Always use Tagalog for breakdown lines
                )
            
            # If we can't parse it, log a warning and skip the line
            print(f"Warning: Could not parse line {line_number}: {line}")
            return None
            
        except Exception as e:
            import traceback
            error_msg = f"Error parsing line {line_number}: {line}\n"
            error_msg += f"Error type: {type(e).__name__}\n"
            error_msg += f"Error details: {str(e)}\n"
            error_msg += "Traceback:\n"
            error_msg += "".join(traceback.format_exc())
            print(error_msg)
            return None
    
    def _build_lesson(self, lesson: Lesson, parsed_lines: List[ParsedLine]) -> None:
        """Build a Lesson object from parsed lines."""
        # Filter out None values from parsed_lines
        parsed_lines = [line for line in parsed_lines if line is not None]
        
        # If no valid lines, add a default section
        if not parsed_lines:
            self._add_default_section(lesson)
            return
            
        # First, group lines into sections
        sections = []
        current_section_lines = []
        
        for line in parsed_lines:
            if line.line_type == LineType.SECTION_HEADER:
                if current_section_lines:
                    sections.append(current_section_lines)
                current_section_lines = [line]
            else:
                current_section_lines.append(line)
        
        if current_section_lines:
            sections.append(current_section_lines)
        
        # If no sections were created, add all lines to a default section
        if not sections and parsed_lines:
            sections = [parsed_lines]
        
        # Process each section
        for section_lines in sections:
            if not section_lines:
                continue
                
            # First line is the section header if available, otherwise create a default one
            if section_lines[0].line_type == LineType.SECTION_HEADER:
                section_header = section_lines[0]
                section_content = section_lines[1:]  # Skip the header line
            else:
                section_header = ParsedLine(
                    line_number=section_lines[0].line_number,
                    line_type=LineType.SECTION_HEADER,
                    content=f"Section {len(lesson.sections) + 1}",
                    speaker='NARRATOR'
                )
                section_content = section_lines
            
            # Create a new section
            section_type = self._determine_section_type(section_header.speaker, section_header.content)
            
            current_section = Section(
                title=section_header.content or f"Section {len(lesson.sections) + 1}",
                section_type=section_type,
                lesson_id=str(lesson.id) if hasattr(lesson, 'id') else None,
                position=len(lesson.sections) + 1
            )
            lesson.add_section(current_section)
            
            # Process the content lines in the section
            i = 0
            while i < len(section_content):
                line = section_content[i]
                
                # Skip section headers that might have been included in content
                if line.line_type == LineType.SECTION_HEADER:
                    i += 1
                    continue
                
                # Handle dialogue and narrator lines
                if line.line_type in (LineType.DIALOGUE, LineType.NARRATOR):
                    # Create a new phrase
                    language = self._determine_language(line.speaker or "NARRATOR", lesson)
                    voice_id = self._get_voice_for_speaker(line.speaker or "NARRATOR")
                    
                    # Skip empty content
                    if not line.content.strip():
                        i += 1
                        continue
                    
                    try:
                        current_phrase = Phrase(
                            text=line.content,
                            language=language,
                            voice_id=voice_id,
                            position=len(current_section.phrases) + 1,
                            section_id=str(current_section.id) if hasattr(current_section, 'id') else None,
                            metadata={"speaker": line.speaker or "NARRATOR"}
                        )
                        current_section.add_phrase(current_phrase)
                        # Update the last used voice to ensure consistency
                        self.last_voice_id = voice_id
                        
                        # Check if next line is a translation (NARRATOR line that's not a section header)
                        if (i + 1 < len(section_content) and 
                            section_content[i+1].line_type == LineType.NARRATOR and 
                            not section_content[i+1].content.strip().endswith(':')):
                            
                            translation = section_content[i+1]
                            # Add translation as a separate phrase in the same section
                            translation_phrase = Phrase(
                                text=translation.content,
                                language=lesson.native_language,
                                voice_id=self._get_voice_for_speaker("NARRATOR"),
                                position=len(current_section.phrases) + 1,
                                section_id=str(current_section.id) if hasattr(current_section, 'id') else None,
                                metadata={"is_translation": True, "original_text": line.content}
                            )
                            current_section.add_phrase(translation_phrase)
                            i += 1  # Skip the translation line in the next iteration
                            
                            # The next lines are breakdown lines - they should use the same voice as the original phrase
                            # Look ahead to find all consecutive non-empty lines that are not section headers or dialogue starters
                            j = i + 1
                            while j < len(section_content):
                                next_line = section_content[j]
                                if (not next_line.content.strip() or 
                                    next_line.line_type == LineType.SECTION_HEADER or 
                                    (next_line.line_type == LineType.DIALOGUE and next_line.speaker)):
                                    break
                                    
                                # Create a new phrase with the same voice and language as the original phrase
                                breakdown_phrase = Phrase(
                                    text=next_line.content,
                                    language=language,  # Same language as the original phrase
                                    voice_id=voice_id,  # Same voice as the original phrase
                                    position=len(current_section.phrases) + 1,
                                    section_id=str(current_section.id) if hasattr(current_section, 'id') else None,
                                    metadata={"is_breakdown": True, "original_text": line.content}
                                )
                                current_section.add_phrase(breakdown_phrase)
                                
                                # Update the last used voice to ensure consistency
                                self.last_voice_id = voice_id
                                j += 1
                            
                            # Skip the breakdown lines we just processed
                            if j > i + 1:
                                i = j - 1  # -1 because we'll increment i at the end of the loop
                            
                    except Exception as e:
                        print(f"Error creating phrase from line {line.line_number}: {line.content}")
                        print(f"Error details: {str(e)}")
                
                i += 1
        
        # If no sections were created, add a default section with all content
        if not lesson.sections:
            self._add_default_section(lesson)
            
    def _add_default_section(self, lesson: 'Lesson') -> None:
        """Add a default section to the lesson."""
        default_section = Section(
            title="Default Section",
            section_type=SectionType.KEY_PHRASES,
            lesson_id=str(lesson.id) if hasattr(lesson, 'id') else None,
            position=1
        )
        lesson.add_section(default_section)
    
    def _determine_section_type(self, section_header: Optional[str], content: str) -> SectionType:
        """Determine the section type from the header and content.
        
        Args:
            section_header: The section header text, or None if not provided
            content: The section content
            
        Returns:
            The determined section type
        """
        # Check for content-based detection first (for empty headers)
        if content:
            # Count non-empty lines
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if len(lines) > 3:
                return SectionType.NATURAL_SPEED
        
        # If we have a section header, use it to determine the type
        if section_header:
            section_header = section_header.upper()
            
            # Try to determine section type from header
            if 'KEY PHRASE' in section_header or 'VOCAB' in section_header:
                return SectionType.KEY_PHRASES
            elif 'NATURAL' in section_header or 'CONVERSATION' in section_header:
                return SectionType.NATURAL_SPEED
            elif 'TRANSLAT' in section_header or 'ENGLISH' in section_header:
                return SectionType.TRANSLATED
            elif 'DIALOG' in section_header or 'DIALOGUE' in section_header:
                return SectionType.KEY_PHRASES
        
        # Default to KEY_PHRASES if we can't determine the type
        return SectionType.KEY_PHRASES
    
    def _get_voice_for_speaker(self, speaker: str) -> str:
        """Get the voice ID for a speaker.
        
        Args:
            speaker: The speaker name or identifier. If empty or None, returns the last used voice.
            
        Returns:
            The voice ID to use for the speaker, or the last used voice if no speaker is specified.
        """
        # If no speaker is specified, return the last used voice or default to Tagalog
        if not speaker or speaker.upper() == 'NARRATOR':
            # For narrator lines, use the default English voice
            if speaker and speaker.upper() == 'NARRATOR':
                voice_id = "en-US-AriaNeural"
            else:
                # For empty speaker (breakdown lines), always use Tagalog voice
                voice_id = "fil-PH-BlessicaNeural"
        else:
            # Default to English voice
            default_voice = "en-US-AriaNeural"
            voice_id = None
            
            # Check if we have a registered voice for this speaker
            if speaker in self.voices:
                voice_id = self.voices[speaker].provider_id
            else:
                # Try to determine voice from speaker name pattern (e.g., TAGALOG-FEMALE-1)
                speaker_match = self.SPEAKER_PATTERN.match(speaker.upper())
                if speaker_match:
                    language, gender, _ = speaker_match.groups()
                    if language == 'TAGALOG':
                        voice_id = "fil-PH-BlessicaNeural" if gender == 'FEMALE' else "fil-PH-AngeloNeural"
                else:
                    # Fallback to language detection from speaker name
                    speaker_upper = speaker.upper()
                    if 'TAGALOG' in speaker_upper or 'FILIPINO' in speaker_upper:
                        voice_id = "fil-PH-BlessicaNeural" if 'FEMALE' in speaker_upper else "fil-PH-AngeloNeural"
                    elif 'NARRATOR' in speaker_upper or 'ENGLISH' in speaker_upper:
                        voice_id = default_voice
            
            # If we couldn't determine a voice, use the last used voice or default
            if not voice_id:
                voice_id = self.last_voice_id or default_voice
        
        # Update the last used voice
        self.last_voice_id = voice_id
        return voice_id
    
    def _determine_language(self, speaker: str, lesson: Lesson) -> Language:
        """Determine the language for a speaker."""
        if not speaker:
            return lesson.target_language
            
        # Try to determine language from speaker name pattern (e.g., TAGALOG-FEMALE-1)
        speaker_match = self.SPEAKER_PATTERN.match(speaker.upper())
        if speaker_match:
            language, _, _ = speaker_match.groups()
            if language == 'TAGALOG':
                return Language.TAGALOG
            elif language == 'ENGLISH':
                return Language.ENGLISH
        
        # Fallback to language detection from speaker name
        speaker_upper = speaker.upper()
        if 'TAGALOG' in speaker_upper or 'FILIPINO' in speaker_upper:
            return Language.TAGALOG
        elif 'NARRATOR' in speaker_upper or 'ENGLISH' in speaker_upper:
            return Language.ENGLISH
        
        return lesson.target_language

async def parse_lesson_file(
    file_path: Union[str, Path], 
    progress_callback: Optional[Callable[[int, int, str, Dict[str, Any]], Awaitable[None]]] = None
) -> Lesson:
    """Parse a lesson file and return a Lesson object.
    
    Args:
        file_path: Path to the lesson file
        progress_callback: Optional callback for progress updates
            - current: Current progress (number of lines processed)
            - total: Total number of lines to process
            - status: Current status message
            - **kwargs: Additional progress data
    """
    parser = LessonParser()
    return await parser.parse_file(file_path, progress_callback=progress_callback)
