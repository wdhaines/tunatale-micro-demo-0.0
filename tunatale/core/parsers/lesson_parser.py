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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.line_number}: {self.line_type.name} - {self.content}"


class LessonParser:
    """Parser for TunaTale lesson files."""
    
    # Regex patterns for different line types
    # Section headers are lines containing "Key Phrases", "Natural Speed", "Slow Speed", or "Translated" with optional colon
    # or standalone lines that exactly match these phrases (case insensitive)
    SECTION_PATTERN = re.compile(r'(?P<prefix>.*?)(?P<type>Key Phrases|Natural Speed|Slow Speed|Translated)(?::|\s|$)(?P<suffix>.*)$', re.IGNORECASE)
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
        # Tagalog voices
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
                provider_id="fil-PH-AngeloNeural",
                language=Language.TAGALOG,
                gender=VoiceGender.MALE,
                is_active=True
            )
        )
        # English voices
        self.register_voice(
            Voice(
                name="English Female 1",
                provider="edge_tts",
                provider_id="en-US-JennyNeural",
                language=Language.ENGLISH,
                gender=VoiceGender.FEMALE,
                is_active=True
            )
        )
        self.register_voice(
            Voice(
                name="English Male 1",
                provider="edge_tts",
                provider_id="en-US-GuyNeural",
                language=Language.ENGLISH,
                gender=VoiceGender.MALE,
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
            # Debug logging
            import logging
            logger = logging.getLogger(__name__)
            
            # Strip whitespace from the line
            original_line = line
            line = line.strip()
            
            # Log the line being processed
            logger.debug(f"\nProcessing line {line_number}: '{original_line}'")
            logger.debug(f"Stripped line: '{line}'")
            
            # Handle empty lines and comments
            if not line or self.COMMENT_PATTERN.match(line):
                line_type = LineType.BLANK if not line else LineType.COMMENT
                logger.debug(f"Matched {line_type.name} line")
                return ParsedLine(
                    line_number=line_number,
                    line_type=line_type,
                    content=line
                )
            
            # Check for section headers (lines containing section keywords)
            logger.debug(f"Checking for section header in line: '{line}'")
            section_match = self.SECTION_PATTERN.search(line)
            if section_match:
                # Extract the section type and surrounding content
                prefix = section_match.group('prefix').strip()
                section_type = section_match.group('type').strip()  # Don't strip the colon here
                suffix = section_match.group('suffix').strip()
                
                logger.debug(f"Section match found - prefix: '{prefix}', type: '{section_type}', suffix: '{suffix}'")
                
                # Handle narrator prefix (e.g., [NARRATOR]: Key Phrases: ...)
                if prefix.startswith('[') and ']' in prefix:
                    speaker = prefix[1:prefix.index(']')].strip()
                    prefix = prefix[prefix.index(']') + 1:].lstrip(':').strip()
                    
                    # Reconstruct the full content with proper spacing around the section type
                    content = f"{prefix} {section_type} {suffix}".strip()
                    
                    logger.debug(f"Narrator prefix detected - speaker: '{speaker}', reconstructed content: '{content}'")
                    
                    return ParsedLine(
                        line_number=line_number,
                        line_type=LineType.SECTION_HEADER,
                        content=content,
                        speaker=speaker,
                        voice_id=self._get_voice_for_speaker(speaker),
                        language='english' if speaker.upper() == 'NARRATOR' else 'tagalog',
                        metadata={"original_line": original_line}  # Store original line for debugging
                    )
                else:
                    # It's a standalone section header, include any prefix text with proper spacing
                    content = f"{prefix} {section_type} {suffix}".strip()
                    
                    logger.debug(f"Standalone section header - content: '{content}'")
                    
                    return ParsedLine(
                        line_number=line_number,
                        line_type=LineType.SECTION_HEADER,
                        content=content,
                        speaker=None,
                        language=None,
                        voice_id=None,
                        metadata={"original_line": original_line}  # Store original line for debugging
                    )
            else:
                logger.debug("No section header match")
            
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
                
                # Create a new metadata dictionary for this phrase
                metadata = {"speaker": speaker}
                
                # Store the current phrase metadata for _get_voice_for_speaker to use
                temp_phrase = type('TempPhrase', (), {'metadata': metadata})
                self.current_phrase = temp_phrase
                
                # Get the voice for this speaker and update last_voice_id
                voice_id = self._get_voice_for_speaker(speaker)
                
                # Extract any TTS settings that were set by _get_voice_for_speaker
                tts_metadata = {}
                if hasattr(self.current_phrase, 'metadata'):
                    tts_metadata = {
                        'tts_pitch': self.current_phrase.metadata.get('tts_pitch'),
                        'tts_rate': self.current_phrase.metadata.get('tts_rate'),
                        'speaker_id': self.current_phrase.metadata.get('speaker_id')
                    }
                
                # Clear the current_phrase to prevent metadata leakage
                self.current_phrase = None
                
                # Determine language based on speaker
                if speaker and ('TAGALOG' in speaker.upper() or 'FILIPINO' in speaker.upper()):
                    language = "tagalog"
                else:
                    language = "english"
                
                # Update last_voice_id and last_language
                self.last_voice_id = voice_id
                self.last_language = language
                
                # Store the speaker's language for inheritance
                self._speaker_language = language
                
                # Create the parsed line with all metadata
                parsed_line = ParsedLine(
                    line_number=line_number,
                    line_type=line_type,
                    content=content,
                    speaker=speaker,
                    language=language,
                    voice_id=voice_id,
                    metadata=metadata.copy()  # Include the metadata in the parsed line
                )
                
                # Update the metadata with TTS settings
                parsed_line.metadata.update(tts_metadata)
                
                return parsed_line
            
            # If we get here, it's an unrecognized line type - treat as plain text
            # Check if it looks like a phrase (no brackets, not empty after strip)
            if line and '[' not in line and ']' not in line:
                # For plain text lines, use the last speaker's voice and language
                if self.last_voice_id:
                    voice_id = self.last_voice_id
                    
                    # Determine language based on the speaker's language if available,
                    # otherwise use the last language, or determine from voice ID
                    if hasattr(self, '_speaker_language'):
                        language = self._speaker_language
                    elif hasattr(self, 'last_language'):
                        language = self.last_language
                    else:
                        language = "tagalog" if 'fil-' in voice_id.lower() else "english"
                else:
                    # Default to Tagalog if no previous voice
                    voice_id = "fil-PH-BlessicaNeural"
                    language = "tagalog"
                
                # For plain text lines, use DIALOGUE type as per test expectations
                return ParsedLine(
                    line_number=line_number,
                    line_type=LineType.DIALOGUE,
                    content=line,
                    speaker=None,
                    voice_id=voice_id,
                    language=language
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
                section_content = section_lines  # Include explicit section headers as phrases
            else:
                section_header = ParsedLine(
                    line_number=section_lines[0].line_number,
                    line_type=LineType.SECTION_HEADER,
                    content=f"Section {len(lesson.sections) + 1}",
                    speaker='NARRATOR'
                )
                section_content = section_lines  # Don't include auto-generated headers as phrases
            
            # Create a new section
            section_content_text = '\n'.join([line.content for line in section_content])
            section_type = self._determine_section_type(section_header.content, section_content_text)
            
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
                
                # Handle section headers as narrator phrases
                if line.line_type == LineType.SECTION_HEADER:
                    # Create a phrase for the section header
                    header_phrase = Phrase(
                        text=line.content,
                        language=line.language or 'english',  # Default to English for section headers
                        voice_id=line.voice_id or self._get_voice_for_speaker('NARRATOR'),
                        position=len(current_section.phrases) + 1,
                        section_id=str(current_section.id) if hasattr(current_section, 'id') else None,
                        metadata={"is_section_header": True, "speaker": "NARRATOR"}
                    )
                    current_section.add_phrase(header_phrase)
                    i += 1
                    continue
                
                # Handle dialogue and narrator lines
                if line.line_type in (LineType.DIALOGUE, LineType.NARRATOR):
                    # Create a new phrase
                    # Use the language from the ParsedLine if available, otherwise determine it
                    language = line.language if line.language else self._determine_language(line.speaker or "NARRATOR", lesson)
                    voice_id = line.voice_id if line.voice_id else self._get_voice_for_speaker(line.speaker or "NARRATOR")
                    
                    # Skip empty content
                    if not line.content.strip():
                        i += 1
                        continue
                    
                    try:
                        # Create metadata with speaker and any TTS settings from the line
                        metadata = {"speaker": line.speaker or "NARRATOR"}
                        
                        # Include any TTS settings from the parsed line
                        if hasattr(line, 'metadata'):
                            if line.metadata and 'tts_pitch' in line.metadata and line.metadata['tts_pitch'] is not None:
                                metadata['tts_pitch'] = line.metadata['tts_pitch']
                            if line.metadata and 'tts_rate' in line.metadata and line.metadata['tts_rate'] is not None:
                                metadata['tts_rate'] = line.metadata['tts_rate']
                            if line.metadata and 'speaker_id' in line.metadata and line.metadata['speaker_id'] is not None:
                                metadata['speaker_id'] = line.metadata['speaker_id']
                        
                        current_phrase = Phrase(
                            text=line.content,
                            language=language,
                            voice_id=voice_id,
                            position=len(current_section.phrases) + 1,
                            section_id=str(current_section.id) if hasattr(current_section, 'id') else None,
                            metadata=metadata
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
        """Determine the type of a section based on its header and content.

        Args:
            section_header: The section header text, if any.
            content: The section content.

        Returns:
            The determined SectionType.
        """
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Determining section type for header: '{section_header}' with content: {content[:50]}...")
    
        # If we have a section header, use that to determine the type
        if section_header:
            # Remove [NARRATOR]: prefix if present
            if section_header.upper().startswith('[NARRATOR]:'):
                section_header = section_header[len('[NARRATOR]:'):].strip()
                
            # Remove any trailing colon and strip whitespace
            section_header = section_header.rstrip(':').strip()
            section_header_upper = section_header.upper()
            logger.debug(f"Processing section header: '{section_header}' (normalized: '{section_header_upper}')")
            
            # Check for slow speed section
            if 'SLOW SPEED' in section_header_upper or 'SLOW' in section_header_upper:
                logger.debug(f"Matched SLOW_SPEED section: '{section_header}'")
                return SectionType.SLOW_SPEED
                
            # Check for natural speed section - more flexible matching
            natural_keywords = ['NATURAL', 'CONVERSATION', 'DIALOG', 'DIALOGUE', 'SPEED', 'CONV']
            if any(x in section_header_upper for x in natural_keywords):
                matched_keywords = [k for k in natural_keywords if k in section_header_upper]
                logger.debug(f"Matched NATURAL_SPEED section with keywords: {matched_keywords}")
                return SectionType.NATURAL_SPEED
            
            # Check for exact matches first
            if section_header_upper in ['NATURAL SPEED', 'NATURAL', 'SPEED']:
                logger.debug(f"Matched exact NATURAL_SPEED section: '{section_header}'")
                return SectionType.NATURAL_SPEED
                
            # Check for key phrases section
            key_phrases_keywords = ['KEY PHRASE', 'VOCAB', 'PHRASES', 'VOCABULARY', 'KEY', 'PHRASE']
            if any(x in section_header_upper for x in key_phrases_keywords):
                matched_keywords = [k for k in key_phrases_keywords if k in section_header_upper]
                logger.debug(f"Matched KEY_PHRASES section with keywords: {matched_keywords}")
                return SectionType.KEY_PHRASES
                
            # Check for exact matches for key phrases
            if section_header_upper in ['KEY PHRASES', 'KEY PHRASE', 'PHRASES', 'VOCAB']:
                logger.debug(f"Matched exact KEY_PHRASES section: '{section_header}'")
                return SectionType.KEY_PHRASES
                
            # Check for translated section
            translated_keywords = ['TRANSLAT', 'ENGLISH', 'TRANSLATION', 'TRANSLATE']
            if any(x in section_header_upper for x in translated_keywords):
                matched_keywords = [k for k in translated_keywords if k in section_header_upper]
                logger.debug(f"Matched TRANSLATED section with keywords: {matched_keywords}")
                return SectionType.TRANSLATED
    
        # Fall back to content-based detection
        if content:
            # Count non-empty lines
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            logger.debug(f"Content-based detection: Found {len(lines)} non-empty lines")
            if len(lines) > 3:
                logger.debug("Content-based detection: Returning NATURAL_SPEED due to line count > 3")
                return SectionType.NATURAL_SPEED
        
        # Default to KEY_PHRASES if we can't determine the type
        logger.debug("Could not determine section type, defaulting to KEY_PHRASES")
        return SectionType.KEY_PHRASES
    
    def _get_voice_for_speaker(self, speaker: Optional[str]) -> str:
        """Get the appropriate voice ID for a speaker.
        
        Args:
            speaker: The speaker name or identifier. If empty or None, returns the last used voice.
            
        Returns:
            The voice ID to use for the speaker, or the last used voice if no speaker is specified.
            The returned voice ID is guaranteed to be one of our registered voices.
        """
        # Define our valid voice IDs
        valid_voice_ids = {
            "fil-PH-BlessicaNeural",  # Tagalog Female 1
            "fil-PH-RosaNeural",      # Tagalog Female 2
            "fil-PH-AngeloNeural",    # Tagalog Male
            "en-US-GuyNeural",        # English Male (Guy)
            "en-US-AriaNeural"        # English Female (Aria) - kept for backward compatibility
        }
        
        # Default voices by type
        DEFAULT_ENGLISH_VOICE = "en-US-GuyNeural"  # Changed to male voice
        DEFAULT_TAGALOG_VOICE = "fil-PH-BlessicaNeural"
        
        # If no speaker is specified, return the last used voice or default to Tagalog
        if not speaker or speaker.upper() == 'NARRATOR':
            # For narrator lines, use the male English voice
            if speaker and speaker.upper() == 'NARRATOR':
                voice_id = DEFAULT_ENGLISH_VOICE
            else:
                # For empty speaker (breakdown lines), always use Tagalog voice
                voice_id = DEFAULT_TAGALOG_VOICE
        else:
            voice_id = None
            speaker_upper = speaker.upper()
            
            # Check if we have a registered voice for this speaker
            if speaker in self.voices:
                voice_id = self.voices[speaker].provider_id
            # Try to determine voice from speaker name pattern (e.g., TAGALOG-FEMALE-1)
            elif (speaker_match := self.SPEAKER_PATTERN.match(speaker_upper)):
                language, gender, number = speaker_match.groups()
                speaker_id = f"{language.lower()}-{gender.lower()}-{number}"  # e.g., 'tagalog-female-2'
                
                if language == 'TAGALOG':
                    if gender == 'FEMALE':
                        # Both TAGALOG-FEMALE-1 and TAGALOG-FEMALE-2 use the same voice
                        voice_id = "fil-PH-BlessicaNeural"
                        # Set speaker_id and TTS settings in the phrase metadata
                        if hasattr(self, 'current_phrase') and self.current_phrase:
                            self.current_phrase.metadata['speaker_id'] = speaker_id
                            
                            # Set default pitch/rate for TAGALOG-FEMALE-1
                            if number == '1':
                                self.current_phrase.metadata['tts_pitch'] = '0.0'
                                self.current_phrase.metadata['tts_rate'] = '1.0'
                            # Set custom pitch/rate for TAGALOG-FEMALE-2
                            elif number == '2':
                                self.current_phrase.metadata['tts_pitch'] = '-15.0'
                                self.current_phrase.metadata['tts_rate'] = '0.6'
                            else:
                                # Default pitch/rate for TAGALOG-FEMALE-1
                                self.current_phrase.metadata['tts_pitch'] = '0.0'
                                self.current_phrase.metadata['tts_rate'] = '1.0'
                    else:
                        voice_id = "fil-PH-AngeloNeural"  # Male Tagalog voice
                else:  # Default to English for any other language
                    voice_id = DEFAULT_ENGLISH_VOICE
            elif 'NARRATOR' in speaker_upper or 'ENGLISH' in speaker_upper:
                voice_id = "en-US-GuyNeural"  # Always use Guy for English
            elif 'TAGALOG' in speaker_upper or 'FILIPINO' in speaker_upper:
                voice_id = "fil-PH-BlessicaNeural" if 'FEMALE' in speaker_upper else "fil-PH-AngeloNeural"
                # Set default pitch/rate for non-pattern matched Tagalog female speakers
                if hasattr(self, 'current_phrase') and self.current_phrase and 'FEMALE' in speaker_upper:
                    self.current_phrase.metadata['tts_pitch'] = '0.0'
                    self.current_phrase.metadata['tts_rate'] = '1.0'
            
            # If we couldn't determine a valid voice, use the last used voice or default
            if not voice_id or voice_id not in valid_voice_ids:
                voice_id = self.last_voice_id if self.last_voice_id in valid_voice_ids else DEFAULT_TAGALOG_VOICE
        
        # Ensure the selected voice ID is one of our valid voices
        if voice_id not in valid_voice_ids:
            # Fall back to default Tagalog voice if somehow we still have an invalid voice
            voice_id = DEFAULT_TAGALOG_VOICE
        
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
