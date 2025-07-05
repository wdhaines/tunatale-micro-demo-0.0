"""Core TTS processing functionality."""

import asyncio
import re
import os
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, BinaryIO, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import numpy as np
import soundfile as sf
import io
import time
from pydub import AudioSegment

import edge_tts

# Set up logging
logger = logging.getLogger(__name__)

# Audio settings
EDGE_TTS_SAMPLE_RATE = 24000  # Edge TTS generates audio at 24kHz
OUTPUT_SAMPLE_RATE = 44100    # Output sample rate for final files
CHANNELS = 1
SUBTYPE = 'PCM_16'

class SectionType(Enum):
    """Types of sections in the script."""
    KEY_PHRASES = "Key Phrases"
    NATURAL_SPEED = "Natural Speed"
    SLOW_SPEED = "Slow Speed"
    TRANSLATED = "Translated"

@dataclass
class VoiceLine:
    """Represents a single line of text with voice attributes."""
    voice_tag: str
    text: str
    section: SectionType
    speed: float = 1.0
    pitch_shift: int = 0  # semitones

@dataclass
class AudioSegmentInfo:
    """Container for audio segment and its metadata."""
    audio_data: np.ndarray
    sample_rate: int
    voice_tag: str
    section: SectionType
    text: str
    
    @property
    def duration_ms(self) -> int:
        """Get duration in milliseconds."""
        return int((len(self.audio_data) / self.sample_rate) * 1000)

class MultiVoiceTTS:
    """Processes multi-voice TTS generation from scripts."""
    
    def __init__(self, output_dir: str = "output", output_format: str = "mp3", bitrate: str = "192k", max_concurrent: int = 3):
        """Initialize the TTS processor.
        
        Args:
            output_dir: Directory to save output files
            output_format: Output audio format ('wav' or 'mp3')
            bitrate: Bitrate for output files (e.g., '128k', '192k', '256k')
            max_concurrent: Maximum number of concurrent TTS requests
        """
        self.voices = {
            'NARRATOR': 'en-US-AriaNeural',
            'TAGALOG-FEMALE-1': 'fil-PH-BlessicaNeural',
            'TAGALOG-FEMALE-2': 'fil-PH-BlessicaNeural',
            'TAGALOG-MALE-1': 'fil-PH-AngeloNeural',
            'TAGALOG-MALE-2': 'fil-PH-AngeloNeural'
        }
        
        # Concurrency control
        self.max_concurrent = max(1, min(max_concurrent, 10))  # Limit to 10 concurrent requests
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Voice-specific adjustments
        self.voice_adjustments = {
            'TAGALOG-FEMALE-2': {'pitch_shift': 2, 'speed': 0.95},
            'TAGALOG-MALE-2': {'pitch_shift': -1, 'speed': 0.95}
        }
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.section_markers = []
        self.voice_lines: List[VoiceLine] = []
        
        # Audio settings
        self.silence_between_lines = 300  # ms
        self.silence_between_sections = 1000  # ms
        self.practice_syllable_pause = 500  # ms
        self.practice_phrase_pause = 1000  # ms
        
        # Audio format settings
        self.audio_format = output_format.lower()
        self.bitrate = bitrate
        
        # Validate output format
        if self.audio_format not in ['wav', 'mp3']:
            logger.warning(f"Unsupported audio format: {self.audio_format}. Defaulting to 'wav'.")
            self.audio_format = "wav"
        
    def _add_voice_line(self, voice_tag: str, text: str, section: SectionType, line_number: int) -> None:
        """Add a voice line to the list of voice lines."""
        # Skip empty lines
        if not text.strip():
            logger.warning(f"Skipping empty voice line at line {line_number}")
            return
            
        # Clean up the text (remove extra whitespace, etc.)
        text = ' '.join(text.split())
        
        # Create a new VoiceLine object
        voice_line = VoiceLine(
            voice_tag=voice_tag,
            text=text,
            section=section,
            speed=1.0,  # Default speed, can be adjusted later
            pitch_shift=0  # Default pitch shift, can be adjusted later
        )
        
        # Apply any voice-specific adjustments
        if voice_tag in self.voice_adjustments:
            adjustments = self.voice_adjustments[voice_tag]
            voice_line.speed = adjustments.get('speed', 1.0)
            voice_line.pitch_shift = adjustments.get('pitch_shift', 0)
        
        # Add to the list of voice lines
        self.voice_lines.append(voice_line)
        
        # Log the voice line for debugging
        logger.debug(f"Added voice line - Section: {section.value}, Voice: {voice_tag}, "
                   f"Text: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    def parse_script(self, script_path: Union[str, Path]) -> None:
        """Parse the input script file into voice lines and sections.
        
        Args:
            script_path: Path to the script file to parse
            
        Raises:
            FileNotFoundError: If the script file doesn't exist
            ValueError: If the script file is empty or invalid
        """
        # Convert to Path object if needed
        script_path = Path(script_path) if isinstance(script_path, str) else script_path
            
        if not script_path.exists():
            raise FileNotFoundError(f"Script file not found: {script_path}")
                
        if script_path.stat().st_size == 0:
            raise ValueError(f"Script file is empty: {script_path}")
            
        logger.info(f"Parsing script: {script_path}")
            
        # Reset state
        self.voice_lines = []
        self.section_markers = []
        
        # Read the script file
        with open(script_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip() for line in f]  # Keep empty lines for section detection
            
        if not any(line.strip() for line in lines):
            raise ValueError(f"No valid content found in script: {script_path}")
            
        # Extract day number from filename if possible
        day_match = re.search(r'(?:day|d)[\s_]*(\d+)', script_path.name.lower())
        self.day = int(day_match.group(1)) if day_match else 1
        
        logger.debug(f"\n{'='*80}")
        logger.debug("SCRIPT PARSING DEBUG")
        logger.debug(f"Script has {len(lines)} lines")
        logger.debug("First 10 lines of script:")
        for i, line in enumerate(lines[:10], 1):
            logger.debug(f"{i}: {line}")
        
        # Initialize with default section if none found
        current_section = None
        
        # First, scan the file to find section headers and process content
        section_headers = []
        current_section_start = 1  # Start from line 1
        
        logger.debug("\nScanning for section headers...")
        
        for i, line in enumerate(lines, 1):
            clean_line = line.strip()
            section_type = None
            
            logger.debug(f"\nProcessing line {i}: '{clean_line}'")
            
            # Check if this is a section header in narrator format: [NARRATOR]: Section Name
            narrator_match = re.match(r'^\[NARRATOR\]\s*:\s*(.+)$', clean_line, re.IGNORECASE)
            if narrator_match:
                section_text = narrator_match.group(1).strip()
                logger.debug(f"Found narrator line with text: '{section_text}'")
                section_type = self._determine_section_type(section_text)
                
                if section_type is not None:
                    logger.debug(f"Detected section type from narrator: {section_type}")
                    # Save the previous section if it exists
                    if current_section is not None:
                        section_headers.append({
                            'type': current_section,
                            'name': current_section.value,
                            'start_line': current_section_start,
                            'end_line': i - 1
                        })
                        logger.debug(f"Ended section {current_section} at line {i-1}")
                    
                    # Start new section
                    current_section = section_type
                    current_section_start = i
                    logger.debug(f"Started new section {section_type} at line {i}")
                    continue
            
            # Skip empty lines when looking for standalone section headers
            if not clean_line:
                logger.debug("Skipping empty line")
                continue
                
            # Check for standalone section headers (e.g., "Natural Speed" or "Key Phrases")
            potential_section_type = self._determine_section_type(clean_line)
            if potential_section_type is not None:
                logger.debug(f"Potential section type detected: {potential_section_type}")
                
                # Only start a new section if it's different from the current one
                if potential_section_type != current_section:
                    # Save the previous section if it exists
                    if current_section is not None:
                        section_headers.append({
                            'type': current_section,
                            'name': current_section.value,
                            'start_line': current_section_start,
                            'end_line': i - 1
                        })
                        logger.debug(f"Ended section {current_section} at line {i-1}")
                    
                    # Start new section
                    current_section = potential_section_type
                    current_section_start = i
                    logger.debug(f"Started new section {potential_section_type} at line {i}")
                    continue
            
            # If we haven't found a section yet, default to KEY_PHRASES
            if current_section is None:
                current_section = SectionType.KEY_PHRASES
                logger.debug(f"No section detected yet, defaulting to {current_section}")
        
        # Add the final section
        if current_section is not None:
            section_headers.append({
                'type': current_section,
                'name': current_section.value,
                'start_line': current_section_start,
                'end_line': len(lines)
            })
            logger.debug(f"\nFinal section {current_section} from line {current_section_start} to {len(lines)}")
        
        # Log all detected sections
        logger.debug("\nDetected sections:")
        for i, header in enumerate(section_headers, 1):
            logger.debug(f"{i}. {header['type'].value}: lines {header['start_line']}-{header['end_line']}")
        
        # Verify we found all required sections
        required_sections = set([SectionType.KEY_PHRASES, SectionType.NATURAL_SPEED, 
                               SectionType.SLOW_SPEED, SectionType.TRANSLATED])
        found_sections = set(header['type'] for header in section_headers)
        missing_sections = required_sections - found_sections
        
        if missing_sections:
            missing_names = ', '.join(s.value for s in missing_sections)
            logger.warning(f"Missing required section headers: {missing_names}")
            # Instead of raising an error, log a warning and continue
            # This allows us to see the debug output even if sections are missing
        
        # Sort section headers by start line
        section_headers.sort(key=lambda x: x['start_line'])
        
        # Create section markers based on the found headers
        for i, header in enumerate(section_headers):
            start_line = header['start_line']
            # End line is the line before the next section starts, or end of file
            end_line = section_headers[i+1]['start_line'] - 1 if i+1 < len(section_headers) else len(lines)
            
            self.section_markers.append({
                'type': header['type'],
                'name': header['name'],
                'start_line': start_line,
                'end_line': end_line
            })
            
            logger.debug(f"Section {i+1}: {header['type'].value} - lines {start_line}-{end_line}")
        
        # Process each section's content using _parse_section_content
        for i, marker in enumerate(self.section_markers):
            start_line = marker['start_line']
            end_line = marker['end_line']
            section_type = marker['type']
            
            # Skip processing if this is the last section marker and it's empty
            if i == len(self.section_markers) - 1 and start_line > len(lines):
                continue
                
            # Get the section content (from start_line to end_line, 1-based to 0-based conversion)
            section_lines = []
            for idx in range(start_line, min(end_line + 1, len(lines) + 1)):
                if idx - 1 >= len(lines):  # Skip if we're past the end of the file
                    continue
                    
                line = lines[idx - 1].strip()
                
                # Skip empty lines at the start of the section
                if not section_lines and not line:
                    continue
                    
                # Don't skip any lines - section headers are now processed in _parse_section_content
                pass
                    
                section_lines.append(line)
            
            # Join lines while preserving empty lines within the section
            section_content = '\n'.join(section_lines)
            
            logger.debug(f"\n{'='*80}")
            logger.debug(f"Processing section: {section_type.value} (lines {start_line}-{end_line})")
            logger.debug(f"Section content:\n{section_content}")
            
            # Parse the section content
            if section_content.strip():
                self._parse_section_content(section_content, section_type)
            else:
                logger.debug(f"  Section {section_type.value} is empty or contains only whitespace")
        
        # Process any content before the first section header
        if self.section_markers and self.section_markers[0]['start_line'] > 1:
            first_section_start = self.section_markers[0]['start_line']
            content_before = '\n'.join(lines[:first_section_start-1])
            if content_before.strip():
                logger.debug(f"\n{'='*80}")
                logger.debug(f"Processing content before first section (lines 1-{first_section_start-1})")
                logger.debug(f"Content before first section:\n{content_before}")
                self._parse_section_content(content_before, SectionType.KEY_PHRASES)
        
        # If no section markers were found, parse the entire file as KEY_PHRASES
        elif not self.section_markers and lines:
            logger.debug("No section markers found, parsing entire file as KEY_PHRASES")
            self._parse_section_content('\n'.join(lines), SectionType.KEY_PHRASES)
        
        # Log section statistics
        section_counts = {}
        for line in self.voice_lines:
            section_counts[line.section] = section_counts.get(line.section, 0) + 1
        
        logger.info(f"Parsed {len(self.voice_lines)} voice lines in {len(section_counts)} sections:")
        for section, count in section_counts.items():
            logger.info(f"  - {section.value}: {count} lines")
    
    def _determine_section_type(self, line: str) -> Optional[SectionType]:
        """Determine the section type from a line of text.
        
        Args:
            line: The line of text to analyze for section type
            
        Returns:
            SectionType if a section type is detected, None otherwise
        """
        logger.debug(f"\n{'='*80}")
        logger.debug(f"_determine_section_type called with: '{line}'")
        
        # Skip empty lines or lines that are just whitespace
        if not line or line.isspace():
            logger.debug("Skipping empty line")
            return None
            
        # Convert to lowercase for case-insensitive matching
        line_lower = line.lower()
            
        # First check for standalone section headers (e.g., "Key Phrases:")
        section_headers = {
            "key phrases": SectionType.KEY_PHRASES,
            "natural speed": SectionType.NATURAL_SPEED,
            "slow speed": SectionType.SLOW_SPEED,
            "translated": SectionType.TRANSLATED
        }
        
        # Check for standalone section headers (e.g., "Key Phrases:")
        for header, section_type in section_headers.items():
            # Match at start of line, with optional colon and whitespace
            if re.match(f'^\\s*{re.escape(header)}[:\\s]*$', line_lower):
                logger.debug(f"✅ Standalone section header: '{header}' -> {section_type.value}")
                self.current_section = section_type
                return section_type
        
        # Then check for narrator-prefixed section headers (e.g., "[NARRATOR]: Key Phrases")
        narrator_match = re.match(r'^\s*\[narrator\]\s*:', line_lower, re.IGNORECASE)
        if narrator_match:
            # Extract the text after the narrator tag
            narrator_text = line[narrator_match.end():].strip().lower()
            for header, section_type in section_headers.items():
                # Match the header exactly in the narrator text
                if re.match(f'^\\s*{re.escape(header)}\\s*$', narrator_text):
                    logger.debug(f"✅ Narrator line with section header: '{header}' -> {section_type.value}")
                    self.current_section = section_type
                    return section_type
                    
        # No section type detected in this line
        return None
            
    def _parse_section_content(self, content: str, section: SectionType) -> None:
        """Parse the content of a section into voice lines.
        
        Handles the following formats:
        1. [VOICE_TAG]: Text
        2. Text without a tag (uses current voice or defaults to NARRATOR)
        """
        logger.debug(f"\n{'='*80}")
        logger.debug(f"=== PARSING SECTION: {section.value} ===")
        logger.debug(f"Content:\n{content}")
        
        # Split into lines and process each line
        lines = [line.rstrip() for line in content.split('\n')]
        logger.debug(f"Found {len(lines)} lines in section")
        
        # Default voice for untagged lines
        current_voice = 'NARRATOR'
        
        for i, line in enumerate(lines):
            original_line = line  # Keep original line for logging
            line = line.strip()
            
            # Handle empty lines - these should be silent pauses
            if not line:
                logger.debug(f"  Line {i+1}: Empty line - adding silent pause")
                # Add a silent pause for empty lines in any section
                self.voice_lines.append(VoiceLine(
                    voice_tag='SILENCE',
                    text='',  # Empty text for silent pause
                    section=section,
                    speed=1.0,
                    pitch_shift=0
                ))
                logger.debug(f"  Added silent pause in {section.value} section")
                continue
                
            logger.debug(f"Processing line: '{original_line}'")
            
            # Check for section header lines
            section_type = self._determine_section_type(line)
            if section_type is not None:
                logger.debug(f"  Processing section header: {line}")
                
                # Extract just the section name without the [NARRATOR]: prefix
                section_name = section_type.value
                if ']:' in line:
                    # Extract text after the last colon to get just the section name
                    section_name = line.split(']:', 1)[1].strip()
                
                # Add section header as NARRATOR voice with just the section name
                self.voice_lines.append(VoiceLine(
                    voice_tag='NARRATOR',
                    text=section_name,
                    section=section_type,
                    speed=1.0,
                    pitch_shift=0
                ))
                # Add a pause after the section header
                self.voice_lines.append(VoiceLine(
                    voice_tag='SILENCE',
                    text='[PAUSE]',
                    section=section_type,
                    speed=1.0,
                    pitch_shift=0
                ))
                continue
            
            # Check for voice tag at the start of the line (case-insensitive)
            voice_match = re.match(r'^\s*\[([^\]]+)\]\s*:(.*)', line, re.IGNORECASE)
            
            if voice_match:
                # Line starts with a voice tag
                voice_tag = voice_match.group(1).strip().upper()  # Normalize to uppercase
                text = voice_match.group(2).strip()
                
                # Only update current_voice if the voice tag is recognized
                if voice_tag in self.voices:
                    current_voice = voice_tag
                    logger.debug(f"  Found voice tag: {voice_tag}")
                else:
                    logger.warning(f"  Unrecognized voice tag: {voice_tag}. Using current voice: {current_voice}")
                
                # Add the voice line if there's text
                if text:
                    self.voice_lines.append(VoiceLine(
                        voice_tag=current_voice,
                        text=text,
                        section=section,
                        speed=1.0,
                        pitch_shift=0
                    ))
                    logger.debug(f"  Added voice line to {section.value}: [{current_voice}]: {text}")
            else:
                # Line without a voice tag - handle based on content and section type
                if line.startswith('---'):
                    logger.debug("  Skipping divider line")
                    continue
                
                # Handle special markers
                if line.upper() == '[PAUSE]':
                    self.voice_lines.append(VoiceLine(
                        voice_tag='SILENCE',
                        text=line,
                        section=section,
                        speed=1.0,
                        pitch_shift=0
                    ))
                    logger.debug(f"  Added explicit pause to {section.value}")
                    continue
                    
                # For all sections, create a new voice line for each non-empty line
                if line.strip():
                    # For natural/slow speed sections, preserve the exact line including whitespace
                    if section in [SectionType.NATURAL_SPEED, SectionType.SLOW_SPEED, SectionType.TRANSLATED]:
                        text = original_line  # Preserve original formatting
                    else:
                        text = line.strip()
                    
                    # Add the voice line
                    self.voice_lines.append(VoiceLine(
                        voice_tag=current_voice,
                        text=text,
                        section=section,
                        speed=1.0,
                        pitch_shift=0
                    ))
                    logger.debug(f"  Added text to {section.value} (voice: {current_voice}): {text}")
                    
                    # Add a small pause after each line in Key Phrases section
                    if section == SectionType.KEY_PHRASES and line.strip() != '[PAUSE]':
                        self.voice_lines.append(VoiceLine(
                            voice_tag='SILENCE',
                            text='[PAUSE]',
                            section=section,
                            speed=1.0,
                            pitch_shift=0
                        ))
                        logger.debug("  Added pause after key phrase")
    
    async def generate_tts(self, text: str, voice: str, rate: str = '+0%', pitch: str = '+0Hz') -> Tuple[np.ndarray, int]:
        """Generate TTS audio with parallel processing.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use for TTS
            rate: Speech rate adjustment (e.g., '+0%', '+10%', '-5%')
            pitch: Pitch adjustment (e.g., '+0Hz', '+2st', '-1st')
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not text.strip():
            return self._create_silence(100), EDGE_TTS_SAMPLE_RATE
            
        # Validate rate and pitch parameters
        if not rate or rate in ('+-0%', '-+0%'):
            rate = '+0%'
        if not pitch or pitch in ('+-0Hz', '-+0Hz'):
            pitch = '+0Hz'  # Default to no pitch shift
            
        # Split text into sentences for parallel processing
        sentences = self._split_into_sentences(text)
        
        # If there's only one sentence, process it directly
        if len(sentences) <= 1:
            return await self._generate_tts_single(text, voice, rate, pitch)
            
        # Process sentences in parallel with limited concurrency
        async def process_sentence(sentence: str):
            async with self._semaphore:
                try:
                    return await self._generate_tts_single(sentence, voice, rate, pitch)
                except Exception as e:
                    logger.error(f"Error processing sentence: {sentence[:50]}... - {e}")
                    # Return silence on error to allow processing to continue
                    silence_duration = 1000  # 1 second of silence
                    silence_samples = int(silence_duration * EDGE_TTS_SAMPLE_RATE / 1000)
                    return np.zeros(silence_samples, dtype='float32'), EDGE_TTS_SAMPLE_RATE
        
        # Process all sentences in parallel
        tasks = [process_sentence(s) for s in sentences if s.strip()]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Combine results
        audio_segments = []
        sample_rate = EDGE_TTS_SAMPLE_RATE
        
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                audio_data, sr = result
                if audio_data is not None and len(audio_data) > 0:
                    if sr != sample_rate:
                        logger.warning(f"Sample rate mismatch: expected {sample_rate}, got {sr}")
                    audio_segments.append(audio_data)
        
        if not audio_segments:
            raise ValueError("No valid audio segments were generated")
            
        # Concatenate all audio segments
        combined_audio = np.concatenate(audio_segments)
        return combined_audio, sample_rate
        
    async def _generate_tts_single(self, text: str, voice: str, rate: str, pitch: str) -> Tuple[np.ndarray, int]:
        """Generate TTS audio for a single text chunk with enhanced error handling.
        
        Args:
            text: The text to convert to speech
            voice: The voice to use
            rate: The speech rate (e.g., '+0%', '+10%', '-5%')
            pitch: The pitch adjustment (e.g., '+0Hz', '+2Hz', '-1Hz')
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            ValueError: If TTS generation fails or returns no audio
            Exception: For other TTS generation errors
        """
        # Clean and validate input text
        text = text.strip()
        if not text or not self._is_valid_text(text):
            logger.debug("Skipping TTS for invalid/empty text, returning silence")
            return self._create_silence(500), EDGE_TTS_SAMPLE_RATE
            
        # Log the TTS generation attempt with more context
        logger.debug(
            f"Generating TTS with parameters:\n"
            f"  Voice: {voice}\n"
            f"  Rate: {rate}, Pitch: {pitch}\n"
            f"  Text length: {len(text)} chars\n"
            f"  Preview: '{text[:50]}{'...' if len(text) > 50 else ''}'"
        )
        
        # Create a temporary file for the TTS output
        temp_wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            
            # Generate TTS using edge-tts with a timeout
            try:
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=voice,
                    rate=rate,
                    pitch=pitch,
                    volume='+0%'  # Keep volume at normal level
                )
                
                # Save with a timeout to prevent hanging
                try:
                    await asyncio.wait_for(communicate.save(temp_wav_path), timeout=30.0)
                except asyncio.TimeoutError:
                    raise ValueError("TTS generation timed out after 30 seconds")
                    
            except Exception as e:
                error_msg = f"TTS generation failed: {str(e)}"
                if "No audio was received" in str(e):
                    error_msg = "TTS service returned no audio - likely due to invalid input parameters or service issues"
                raise ValueError(error_msg) from e
            
            # Verify the file was created and has content
            if not os.path.exists(temp_wav_path):
                raise ValueError("TTS generation failed: No output file was created")
                
            file_size = os.path.getsize(temp_wav_path)
            if file_size < 100:  # Very small file likely contains an error
                with open(temp_wav_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(1000)
                    if 'error' in content.lower() or 'exception' in content.lower():
                        raise ValueError(f"TTS service returned an error: {content}")
                raise ValueError(f"TTS generation failed: Output file is too small ({file_size} bytes)")
            
            # Try to load and validate the generated audio
            try:
                audio_data, sample_rate = sf.read(temp_wav_path, dtype='float32')
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                    audio_data = np.mean(audio_data, axis=1)  # Convert to mono by averaging channels
                
                # Check for silent or invalid audio
                if len(audio_data) == 0:
                    raise ValueError("TTS generation failed: No audio data in file")
                
                rms = np.sqrt(np.mean(np.square(audio_data)))
                if rms < 0.001:  # Very quiet audio
                    raise ValueError(f"TTS generation failed: Audio is too quiet (RMS: {rms:.6f})")
                
                duration = len(audio_data) / sample_rate
                logger.debug(f"Generated {duration:.2f}s of audio at {sample_rate}Hz")
                return audio_data, sample_rate
                
            except Exception as e:
                raise ValueError(f"Failed to process generated audio: {str(e)}") from e
            
        except Exception as e:
            # Log the error with all relevant context for debugging
            logger.error(
                f"TTS generation error - Voice: {voice}, Rate: {rate}, Pitch: {pitch}, "
                f"Text length: {len(text)} chars, Error: {str(e)}"
            )
            # Return silence for non-critical errors to allow processing to continue
            if isinstance(e, ValueError) and any(msg in str(e).lower() for msg in ["no audio", "too small", "too quiet"]):
                return self._create_silence(500), EDGE_TTS_SAMPLE_RATE
            raise  # Re-raise the exception for the retry logic to handle
            
        finally:
            # Clean up the temporary file if it was created
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_wav_path}: {e}")
    
    def _is_valid_text(self, text: str) -> bool:
        """Check if text is valid for TTS generation.
        
        Args:
            text: The text to validate
            
        Returns:
            bool: True if the text is valid for TTS, False otherwise
        """
        if not text or not text.strip():
            return False
            
        # Remove all whitespace and check if anything remains
        stripped = text.strip()
        if not stripped:
            return False
            
        # Common minimal texts that shouldn't be sent to TTS
        minimal_texts = {'.', '..', '...', ',', ';', ':', '!', '?', '-', '--', '---'}
        if stripped in minimal_texts:
            return False
            
        # Check if text contains any letters, numbers, or meaningful characters
        # This will exclude things like just punctuation or whitespace
        return any(c.isalnum() or c in "'" for c in stripped)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for parallel processing.
        
        Args:
            text: Input text to split into sentences
            
        Returns:
            List of sentence strings
        """
        if not text.strip():
            return []
            
        # Simple sentence splitting - can be enhanced with more sophisticated NLP if needed
        sentences = []
        for sentence in re.split(r'(?<=[.!?])\s+', text):
            if sentence.strip():
                sentences.append(sentence.strip())
        return sentences

    async def _generate_tts_single(self, text: str, voice: str, rate: str, pitch: str) -> Tuple[np.ndarray, int]:
        """Generate TTS audio for a single text chunk with enhanced error handling.
        
        Args:
            text: The text to convert to speech
            voice: The voice to use
            rate: The speech rate (e.g., '+0%', '+10%', '-5%')
            pitch: The pitch adjustment (e.g., '+0Hz', '+2Hz', '-1Hz')
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            Exception: If all retry attempts fail
        """
        max_retries = 3
        last_error = None
        
        # Clean and validate input
        sentence = text.strip()
        
        # Early return for empty or invalid text
        if not sentence or not self._is_valid_text(sentence):
            logger.debug(f"Skipping TTS for invalid/empty text: '{sentence}'. Returning silence.")
            return self._create_silence(500), EDGE_TTS_SAMPLE_RATE
            
        # Log the start of TTS generation with more context
        logger.debug(
            f"Starting TTS generation (attempt 1/{max_retries + 1}):\n"
            f"  Voice: {voice}\n"
            f"  Rate: {rate}, Pitch: {pitch}\n"
            f"  Text length: {len(sentence)} chars\n"
            f"  Preview: '{sentence[:50]}{'...' if len(sentence) > 50 else ''}'"
        )
        
        for attempt in range(max_retries + 1):
            try:
                # Additional validation before making the TTS call
                if not self._is_valid_text(sentence):
                    logger.warning(f"Text failed validation on attempt {attempt + 1}: '{sentence}'. Returning silence.")
                    return self._create_silence(500), EDGE_TTS_SAMPLE_RATE
                
                # Generate TTS using edge-tts
                communicate = edge_tts.Communicate(text=sentence, voice=voice, rate=rate, pitch=pitch)
                
                # Create a temporary file to store the output
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    temp_path = temp_file.name
                
                # Generate the TTS audio
                await communicate.save(temp_path)
                
                # Read the generated audio file
                audio_data, sample_rate = sf.read(temp_path)
                
                # Clean up the temporary file
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_path}: {e}")
                
                # Verify we got valid audio data
                if audio_data is None or len(audio_data) == 0:
                    raise ValueError("No audio data received from TTS service")
                
                logger.debug(f"Successfully generated TTS on attempt {attempt + 1} (samples: {len(audio_data)}, rate: {sample_rate})")
                return audio_data, sample_rate
                
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt  # Exponential backoff
                
                # Log the specific error with more context
                error_msg = str(e).replace('\n', ' ').strip()
                log_level = logging.WARNING if attempt < max_retries else logging.ERROR
                logger.log(
                    log_level,
                    f"Attempt {attempt + 1}/{max_retries + 1} failed:\n"
                    f"  Voice: {voice}\n"
                    f"  Rate: {rate}, Pitch: {pitch}\n"
                    f"  Text length: {len(sentence)} chars\n"
                    f"  Preview: '{sentence[:50]}{'...' if len(sentence) > 50 else ''}'\n"
                    f"  Error: {error_msg}"
                )
                
                # If we have retries left, wait and try again
                if attempt < max_retries:
                    logger.info(f"Retrying in {wait_time}s... (attempt {attempt + 2}/{max_retries + 1})")
                    await asyncio.sleep(wait_time)
                
                # On the last attempt, log the failure and return silence
                if attempt == max_retries:
                    logger.error(
                        f"Failed to generate TTS after {max_retries + 1} attempts. "
                        f"Voice: {voice}, Rate: {rate}, Pitch: {pitch}, "
                        f"Text: '{sentence[:100]}{'...' if len(sentence) > 100 else ''}'. "
                        f"Last error: {error_msg}"
                    )
                    # Return silence instead of failing completely
                    silence_duration = 1000  # 1 second of silence
                    logger.warning(f"Returning {silence_duration}ms of silence as fallback")
                    return self._create_silence(silence_duration), EDGE_TTS_SAMPLE_RATE
        
        # This should never be reached due to the return in the loop
        raise Exception("Unexpected error in _generate_tts_single")
    
    async def generate_audio(self) -> None:
        """Generate audio for all voice lines using parallel processing."""
        if not self.voice_lines:
            logger.warning("No voice lines to process")
            return
        
        logger.info(f"Generating audio for {len(self.voice_lines)} voice lines")
        logger.info(f"Using parallel processing with up to {self.max_concurrent} concurrent TTS requests")
    
        # Process voice lines in batches
        batch_size = self.max_concurrent * 2
        audio_segments: List[AudioSegmentInfo] = []
        
        for i in range(0, len(self.voice_lines), batch_size):
            batch = self.voice_lines[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(self.voice_lines)-1)//batch_size + 1} "
                       f"({len(batch)} lines)")
            
            # Process batch in parallel
            batch_tasks = []
            for line in batch:
                # Apply voice-specific adjustments
                if line.voice_tag in self.voice_adjustments:
                    adjustments = self.voice_adjustments[line.voice_tag]
                    if 'pitch_shift' in adjustments:
                        line.pitch_shift = adjustments['pitch_shift']
                    if 'speed' in adjustments:
                        line.speed = adjustments['speed']
                
                # Create task for this line
                task = self._process_voice_line(line)
                batch_tasks.append(task)
            
            # Wait for all tasks in the batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in batch processing: {result}")
                elif result is not None:
                    audio_segments.append(result)
        
        # Combine all segments
        if audio_segments:
            await self._combine_audio_segments(audio_segments)
        else:
            logger.warning("No audio segments were generated")
    
    async def _process_voice_line(self, line: VoiceLine) -> Optional[AudioSegmentInfo]:
        """Process a single voice line and return its audio segment."""
        logger.debug(f"Processing line: {line.voice_tag}: {line.text}")
        
        try:
            # Handle silent pauses for empty lines
            if line.voice_tag == 'SILENCE':
                silence_duration = 500  # ms
                audio_data = self._create_silence(silence_duration)
                logger.debug(f"Generated {silence_duration}ms of silence")
                
                return AudioSegmentInfo(
                    audio_data=audio_data,
                    sample_rate=EDGE_TTS_SAMPLE_RATE,
                    voice_tag=line.voice_tag,
                    section=line.section,
                    text=''  # Empty text for silent pause
                )
            
            # Generate TTS for non-silent lines
            voice = self.voices.get(line.voice_tag, self.voices['NARRATOR'])
            
            # Calculate rate with proper sign handling
            if line.speed > 1.0:
                rate = f'+{int((line.speed - 1.0) * 100)}%'
            elif line.speed < 1.0:
                rate = f'-{int((1.0 - line.speed) * 100)}%'
            else:
                rate = '+0%'  # Default to no rate change
                
            # Convert pitch shift to Hz for Edge TTS
            # 1 semitone ≈ 6% change in frequency
            # We'll use a base of 1Hz per semitone for simplicity
            if line.pitch_shift > 0:
                pitch = f'+{abs(line.pitch_shift)}Hz'  # Higher pitch
            elif line.pitch_shift < 0:
                pitch = f'-{abs(line.pitch_shift)}Hz'  # Lower pitch
            else:
                pitch = '+0Hz'  # No pitch shift
            
            # Generate TTS using parallel processing
            audio_data, sample_rate = await self.generate_tts(line.text, voice, rate, pitch)
            
            return AudioSegmentInfo(
                audio_data=audio_data,
                sample_rate=sample_rate,
                voice_tag=line.voice_tag,
                section=line.section,
                text=line.text
            )
            
        except Exception as e:
            logger.error(f"Error processing voice line '{line.text}': {e}")
            # Fall back to silence if TTS generation fails
            audio_data = self._create_silence(1000)  # 1 second of silence as fallback
            return AudioSegmentInfo(
                audio_data=audio_data,
                sample_rate=EDGE_TTS_SAMPLE_RATE,
                voice_tag=line.voice_tag,
                section=line.section,
                text=line.text
            )
        
    def _create_silence(self, duration_ms: int, sample_rate: int = EDGE_TTS_SAMPLE_RATE) -> np.ndarray:
        """Create a silent audio segment of the specified duration.
        
        Args:
            duration_ms: Duration of silence in milliseconds
            sample_rate: Sample rate in Hz (defaults to Edge TTS sample rate)
            
        Returns:
            NumPy array of zeros with the specified duration
        """
        num_samples = int((duration_ms / 1000.0) * sample_rate)
        logger.debug(f"Creating silence: {duration_ms}ms -> {num_samples} samples at {sample_rate}Hz")
        return np.zeros(num_samples, dtype=np.float32)
        
    def _calculate_pause_duration(self, current_line: VoiceLine, next_line: Optional[VoiceLine]) -> int:
        """Calculate the appropriate pause duration between voice lines.
        
        Args:
            current_line: The current voice line
            next_line: The next voice line, or None if this is the last line
            
        Returns:
            int: The pause duration in milliseconds
        """
        if not next_line:
            return 0  # No pause after the last line
                
        # Longer pause between sections
        if current_line.section != next_line.section:
            return self.silence_between_sections
                
        # Special handling for practice sections
        if current_line.section in [SectionType.KEY_PHRASES, SectionType.NATURAL_SPEED, 
                                  SectionType.SLOW_SPEED, SectionType.TRANSLATED]:
            # Check if this is a practice phrase or syllable
            if len(current_line.text.split()) > 1:
                return self.practice_phrase_pause
            return self.practice_syllable_pause
                
        # Default pause between lines
        return self.silence_between_lines

    def _save_audio(self, audio_data: np.ndarray, sample_rate: int, output_file: Path) -> None:
        """Save audio data to a file.
        
        Args:
            audio_data: Audio data as a numpy array
            sample_rate: Sample rate of the audio
            output_file: Path to save the audio file to
        """
        try:
            if output_file.suffix.lower() == '.wav':
                sf.write(
                    str(output_file),
                    audio_data,
                    sample_rate,
                    format='WAV',
                    subtype='PCM_16'
                )
            else:  # MP3
                # Convert float32 to int16 for MP3 export
                int16_audio = (audio_data * 32767).astype(np.int16)
                
                # Create a temporary WAV file in memory
                with io.BytesIO() as wav_io:
                    sf.write(
                        wav_io,
                        int16_audio,
                        sample_rate,
                        format='WAV',
                        subtype='PCM_16'
                    )
                    wav_io.seek(0)
                    
                    # Convert to MP3 using pydub
                    audio = AudioSegment.from_wav(wav_io)
                    audio.export(str(output_file), format='mp3', bitrate=self.bitrate)
                    
        except Exception as e:
            logger.error(f"Error saving audio to {output_file}: {str(e)}")
            raise
            
    async def _combine_audio_segments(self, segments: List[AudioSegmentInfo]) -> None:
        """Combine multiple audio segments into a single audio file.
        
        Args:
            segments: List of AudioSegmentInfo objects to combine
        """
        if not segments:
            logger.warning("No audio segments to combine")
            return
            
        logger.info(f"Combining {len(segments)} audio segments...")
        
        # Group segments by section
        sections = {}
        for segment in segments:
            if segment.section not in sections:
                sections[segment.section] = []
            sections[segment.section].append(segment)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Combine all segments for the complete audio
            all_audio = np.array([], dtype=np.float32)
            
            for i, segment in enumerate(segments):
                logger.debug(f"Processing segment {i+1}/{len(segments)}: "
                           f"{segment.voice_tag}: {segment.text[:50]}{'...' if len(segment.text) > 50 else ''}")
                
                # Add silence between segments if needed
                if i > 0:
                    if segment.section != segments[i-1].section:
                        silence = self._create_silence(self.silence_between_sections, sample_rate=segment.sample_rate)
                        logger.debug(f"  Adding section silence: {self.silence_between_sections}ms")
                    else:
                        silence = self._create_silence(self.silence_between_lines, sample_rate=segment.sample_rate)
                        logger.debug(f"  Adding line silence: {self.silence_between_lines}ms")
                    
                    all_audio = np.concatenate([all_audio, silence])
                
                # Add the audio data
                all_audio = np.concatenate([all_audio, segment.audio_data])
            
            # Resample to output sample rate if needed
            if len(all_audio) > 0:
                logger.debug(f"Resampling combined audio from {EDGE_TTS_SAMPLE_RATE}Hz to {OUTPUT_SAMPLE_RATE}Hz")
                import librosa
                all_audio = librosa.resample(
                    all_audio,
                    orig_sr=EDGE_TTS_SAMPLE_RATE,
                    target_sr=OUTPUT_SAMPLE_RATE
                )
            
            # Save the combined audio
            output_file = self.output_dir / f"Combined.{self.audio_format}"
            self._save_audio(all_audio, OUTPUT_SAMPLE_RATE, output_file)
            logger.info(f"Saved combined audio to {output_file}")
            
            # Save individual section files
            for section, section_segments in sections.items():
                if not section_segments:
                    continue
                    
                logger.info(f"Processing section: {section.value}")
                section_audio = np.array([], dtype=np.float32)
                
                for i, segment in enumerate(section_segments):
                    # Add silence between segments if needed
                    if i > 0:
                        silence = self._create_silence(self.silence_between_lines, sample_rate=segment.sample_rate)
                        section_audio = np.concatenate([section_audio, silence])
                    
                    # Add the audio data
                    section_audio = np.concatenate([section_audio, segment.audio_data])
                
                # Resample to output sample rate if needed
                if len(section_audio) > 0:
                    section_audio = librosa.resample(
                        section_audio,
                        orig_sr=EDGE_TTS_SAMPLE_RATE,
                        target_sr=OUTPUT_SAMPLE_RATE
                    )
                
                # Save the section audio
                section_file = self.output_dir / f"{section.value.replace(' ', '')}.{self.audio_format}"
                self._save_audio(section_audio, OUTPUT_SAMPLE_RATE, section_file)
                logger.info(f"  Saved section audio to {section_file}")
            
        except Exception as e:
            logger.error(f"Error creating combined audio: {e}", exc_info=True)
            raise
