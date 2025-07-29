"""Service for processing lessons and generating audio."""

import asyncio
import json
import logging
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from uuid import UUID, uuid4

import aiofiles
import aiofiles.os
from pydantic import BaseModel

from tunatale.core.models.audio_config import AudioConfig
from tunatale.core.models.phrase import Phrase
from tunatale.core.models.section import Section
from tunatale.core.models.lesson import Lesson
from tunatale.core.models import ProcessedPhrase, ProcessedSection, ProcessedLesson
from tunatale.core.ports.audio_processor import AudioProcessor
from tunatale.core.ports.lesson_processor import (
    LessonProcessor as LessonProcessorInterface,
    ProcessedPhrase as ProcessedPhraseInterface,
    ProcessedSection as ProcessedSectionInterface,
    ProcessedLesson as ProcessedLessonInterface,
)
from tunatale.core.ports.tts_service import TTSService, TTSTransientError, TTSRateLimitError, TTSValidationError
from tunatale.core.ports.voice_selector import VoiceSelector
from tunatale.core.ports.word_selector import WordSelector
from tunatale.core.exceptions import TTSValidationError, TTSServiceError

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Class to track performance metrics for the lesson processor."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.phrases_processed: int = 0
        self.sections_processed: int = 0
        self.audio_generated: int = 0
        self.phase_timings: Dict[str, float] = {}
        self.phase_start_times: Dict[str, float] = {}

    def start_phase(self, phase_name: str) -> None:
        """Record the start of a processing phase."""
        self.phase_start_times[phase_name] = time.time()

    def end_phase(self, phase_name: str) -> None:
        """Record the end of a processing phase and calculate metrics."""
        if phase_name in self.phase_start_times:
            duration = time.time() - self.phase_start_times[phase_name]
            self.phase_timings[phase_name] = duration
            logger.debug(
                "Phase '%s' completed in %.2f seconds", phase_name, duration
            )

    def get_phase_metrics(self, phase_name: str) -> Dict[str, float]:
        """Get metrics for a specific phase."""
        duration = self.phase_timings.get(phase_name, 0.0)
        return {"duration": duration}

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        total_time = (
            self.end_time - self.start_time if self.start_time and self.end_time else 0.0
        )
        return {
            "total_time_seconds": total_time,
            "phrases_processed": self.phrases_processed,
            "sections_processed": self.sections_processed,
            "audio_files_generated": self.audio_generated,
            "phase_timings": self.phase_timings,
        }


class LessonProcessor(LessonProcessorInterface):
    """Service for processing lessons and generating audio."""

    def __init__(
        self,
        tts_service: TTSService,
        audio_processor: AudioProcessor,
        voice_selector: VoiceSelector,
        word_selector: WordSelector,
        max_workers: int = 4,
        output_dir: str = "output",
        ellipsis_pause_duration_ms: int = 800,
        use_natural_pauses: bool = True,
    ):
        """Initialize the lesson processor with performance monitoring.

        Args:
            tts_service: The TTS service to use for generating speech.
            audio_processor: The audio processor to use for processing audio files.
            voice_selector: The voice selector to use for selecting voices.
            word_selector: The word selector to use for selecting words.
            max_workers: The maximum number of worker threads to use for parallel processing.
            output_dir: The base directory where output files will be saved.
            ellipsis_pause_duration_ms: Duration in milliseconds for pauses created by ellipsis (...).
            use_natural_pauses: Whether to use the natural pause system based on linguistic boundaries.
        """
        self.tts_service = tts_service
        self.audio_processor = audio_processor
        self.voice_selector = voice_selector
        self.word_selector = word_selector
        self.max_workers = max_workers
        self.output_dir = Path(output_dir)
        self.ellipsis_pause_duration_ms = ellipsis_pause_duration_ms
        self.use_natural_pauses = use_natural_pauses
        
        # Initialize natural pause system if enabled
        if self.use_natural_pauses:
            from .natural_pause_calculator import NaturalPauseCalculator
            self.natural_pause_calculator = NaturalPauseCalculator()
        
        self.metrics = PerformanceMetrics()

    async def process_lesson(
        self, lesson: Lesson, output_dir: Optional[str] = None, progress: Optional[Any] = None
    ) -> ProcessedLesson:
        """Process a lesson and generate audio files.

        Args:
            lesson: The lesson to process.
            output_dir: The directory where output files will be saved. If not provided,
                a timestamped subdirectory under the configured output directory will be used.
            progress: Optional progress reporter for tracking progress.

        Returns:
            ProcessedLesson: The processed lesson with audio file paths.
        """
        self.metrics.start_time = time.time()
        self.metrics.start_phase("total")
        
        # Initialize progress tracking
        task_id = f"lesson_{lesson.id or 'default'}"
        if progress:
            await progress.update(task_id=task_id, completed=0, total=100, status="Starting lesson processing")

        try:
            # Create output directory
            if output_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = str(self.output_dir / f"lesson_{timestamp}")
            else:
                output_dir = str(Path(output_dir).resolve())

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Calculate total steps for progress tracking
            total_steps = 4  # Processing sections, generating section audio, generating lesson audio, saving metadata
            current_step = 0
            
            if progress:
                await progress.update(task_id=task_id, completed=current_step, total=total_steps, status="Processing sections...")

            # Process sections in parallel with progress reporting
            processed_sections = await self._process_sections(lesson, output_path, progress=progress)
            current_step += 1
            
            if progress:
                await progress.update(task_id=task_id, completed=current_step, total=total_steps, status="Generating section audio...")

            # Generate section audio files
            section_audio_files = await self._generate_section_audio(processed_sections, output_path)
            current_step += 1
            
            if progress:
                await progress.update(task_id=task_id, completed=current_step, total=total_steps, status="Generating lesson audio...")

            # Generate lesson audio file with progress reporting
            lesson_audio_file = await self._generate_lesson_audio(
                lesson,
                section_audio_files, 
                output_path,
                progress=progress  # Pass progress reporter to lesson audio generation
            )
            current_step += 1
            
            if progress:
                await progress.update(task_id=task_id, completed=current_step, total=total_steps, status="Saving metadata...")

            # Prepare and save metadata
            metadata = self._prepare_metadata(lesson, processed_sections, lesson_audio_file, output_path)
            metadata_file = output_path / "metadata.json"
            self._save_metadata(metadata, metadata_file)
            current_step += 1

            # Log metrics
            self.metrics.end_phase("total")
            self._log_metrics()
            
            if progress:
                await progress.update(task_id=task_id, completed=total_steps, total=total_steps, status="Completed!")
                await progress.complete(task_id=task_id)
            self.metrics.end_phase("total")
            self.metrics.end_time = time.time()

            # Log performance metrics
            self._log_metrics()

            # Create the ProcessedLesson object
            processed_lesson = ProcessedLesson(
                id=lesson.id,
                title=lesson.title,
                language=lesson.target_language,
                level=str(lesson.difficulty),  # Convert to string to match the expected type
                sections=processed_sections,
                audio_file=lesson_audio_file,
                metadata_file=metadata_file,
                output_dir=output_path,
            )
            
            # Convert to dictionary for CLI compatibility
            result = {
                'id': str(processed_lesson.id) if processed_lesson.id else None,
                'title': processed_lesson.title,
                'language': processed_lesson.language,
                'level': processed_lesson.level,
                'sections': [],
                'audio_file': str(processed_lesson.audio_file) if processed_lesson.audio_file else None,
                'metadata_file': str(processed_lesson.metadata_file) if processed_lesson.metadata_file else None,
                'output_dir': str(processed_lesson.output_dir) if processed_lesson.output_dir else None,
                'success': True
            }
            
            # Process sections
            for section in processed_lesson.sections:
                section_dict = {
                    'id': str(section.id) if section.id else None,
                    'title': section.title,
                    'phrases': [],
                    'audio_file': str(section.audio_file) if section.audio_file else None
                }
                
                # Process phrases in the section
                for phrase in section.phrases:
                    phrase_dict = {
                        'id': str(phrase.id) if phrase.id else None,
                        'text': phrase.text,
                        'translation': phrase.translation,
                        'language': phrase.language,
                        'audio_file': str(phrase.audio_file) if phrase.audio_file else None
                    }
                    
                    # Add metadata if it exists
                    if hasattr(phrase, 'metadata') and phrase.metadata:
                        phrase_dict['metadata'] = phrase.metadata
                    
                    section_dict['phrases'].append(phrase_dict)
                
                result['sections'].append(section_dict)
            
            return result

        except Exception as e:
            logger.error("Error processing lesson: %s", str(e))
            raise

    async def _process_sections(
        self, lesson: Lesson, output_path: Path, progress: Optional[Any] = None
    ) -> List[ProcessedSection]:
        """Process all sections in parallel.

        Args:
            lesson: The lesson containing sections to process.
            output_path: Path to the output directory.
            progress: Optional progress reporter for tracking progress.

        Returns:
            List of processed sections.
        """
        section_tasks = []
        for section in lesson.sections:
            section_output_path = output_path / f"section_{section.id}"
            section_output_path.mkdir(exist_ok=True)
            task = self._process_section(lesson, section, section_output_path, progress=progress)
            section_tasks.append(task)

        return await asyncio.gather(*section_tasks)

    async def _process_section(
        self, lesson: Optional[Lesson], section: Section, output_path: Path, progress: Optional[Any] = None
    ) -> ProcessedSection:
        """Process a single section.

        Args:
            lesson: The lesson containing this section (needed for day number extraction).
            section: The section to process.
            output_path: Path to the output directory for this section.
            progress: Optional progress reporter for tracking progress.

        Returns:
            Processed section with audio file paths.
        """
        self.metrics.sections_processed += 1
        logger.info("Processing section: %s", section.title)

        # Update progress if available
        if progress:
            await progress.update(
                task_id=f"section_{section.id}",
                completed=0,
                total=len(section.phrases) + 1,  # +1 for the concatenation step
                status=f"Processing section: {section.title}"
            )

        # Process phrases in parallel
        phrase_tasks = []
        for i, phrase in enumerate(section.phrases):
            # Update progress for each phrase
            if progress:
                await progress.update(
                    task_id=f"section_{section.id}",
                    completed=i + 1,
                    total=len(section.phrases) + 1,  # +1 for the concatenation step
                    status=f"Processing phrase {i+1}/{len(section.phrases)}"
                )
            
            # Get section type for TTS preprocessing
            section_type = self._get_section_type_name(self._classify_section_type(section)) if lesson else None
            task = self._process_phrase(phrase, output_path, section_type)
            phrase_tasks.append(task)

        processed_phrases = await asyncio.gather(*phrase_tasks)

        # Generate section audio with new naming pattern when lesson context is available
        if lesson:
            # New naming pattern: [Number of day].[suffix] – [type].mp3
            day_number = self._extract_day_number(lesson)
            section_suffix = self._classify_section_type(section)
            section_type = self._get_section_type_name(section_suffix)
            section_filename = f"{day_number}.{section_suffix} – {section_type}"
        else:
            # Fallback to old naming for standalone section processing
            section_filename = self._sanitize_filename(section.title)
        
        temp_section_audio_file = output_path / f"{section_filename}.mp3"
        
        # Update progress before concatenation
        if progress:
            await progress.update(
                task_id=f"section_{section.id}",
                completed=len(processed_phrases),
                total=len(section.phrases) + 1,  # +1 for the concatenation step
                status=f"Concatenating {len(processed_phrases)} phrases..."
            )
        
        # Get list of valid audio files
        audio_files = [p.audio_file for p in processed_phrases if p.audio_file and p.audio_file.exists()]
        
        if not audio_files:
            logger.warning(f"No valid audio files found for section {section.id}")
        else:
            await self._concatenate_audio_files(
                audio_files,
                temp_section_audio_file,
                progress=progress  # Pass progress reporter to concatenation
            )
        
        # Place the section audio file in the main output directory
        # Since CLI now passes main directory directly, we move to parent to get to main level
        top_level_dir = output_path.parent
        final_section_audio_file = top_level_dir / f"{section_filename}.mp3"
        
        if temp_section_audio_file.exists():
            # Move the file to the main output directory
            shutil.move(str(temp_section_audio_file), str(final_section_audio_file))
            logger.debug(f"Moved section audio file from {temp_section_audio_file} to {final_section_audio_file}")
            section_audio_file = final_section_audio_file
        else:
            section_audio_file = temp_section_audio_file
        
        # Update progress after concatenation
        if progress:
            await progress.update(
                task_id=f"section_{section.id}",
                completed=len(processed_phrases) + 1,
                total=len(section.phrases) + 1,  # +1 for the concatenation step
                status="Section processing complete"
            )

        return ProcessedSection(
            id=section.id,
            title=section.title,
            phrases=processed_phrases,
            audio_file=section_audio_file,
        )

    async def process_section(
        self,
        section: Section,
        output_dir: Path,
        **options: Any
    ) -> Dict[str, Any]:
        """Process a section and generate audio.

        Args:
            section: The section to process.
            output_dir: Directory to save output files.
            **options: Additional processing options.

        Returns:
            Dictionary with processing results.
        """
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a section-specific subdirectory for processing
            section_output_path = output_dir / f"section_{section.id}"
            section_output_path.mkdir(exist_ok=True)
            
            # Process the section using the internal implementation
            processed_section = await self._process_section(None, section, section_output_path, None)
            
            # Move the final section audio file to the requested output directory
            if processed_section.audio_file and processed_section.audio_file.exists():
                section_filename = self._sanitize_filename(section.title)
                final_audio_path = output_dir / f"{section_filename}.mp3"
                
                if processed_section.audio_file != final_audio_path:
                    shutil.move(str(processed_section.audio_file), str(final_audio_path))
                    logger.debug(f"Moved section audio file from {processed_section.audio_file} to {final_audio_path}")
                    processed_section.audio_file = final_audio_path
            
            # Convert to the expected dictionary format
            result = {
                'id': str(processed_section.id) if processed_section.id else None,
                'title': processed_section.title,
                'phrases': [
                    {
                        'id': str(p.id) if p.id else None,
                        'text': p.text,
                        'language': p.language,
                        'audio_file': str(p.audio_file) if p.audio_file else None,
                        'metadata': getattr(p, 'metadata', {}) or {},
                        'success': True
                    } for p in processed_section.phrases
                ],
                'audio_file': str(processed_section.audio_file) if processed_section.audio_file else None,
                'success': True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing section '{section.title}': {e}")
            return {
                'id': str(section.id) if section.id else None,
                'title': section.title,
                'phrases': [],
                'audio_file': None,
                'metadata': {},
                'success': False,
                'error': {
                    'error_code': 'PROCESSING_ERROR',
                    'error_message': str(e)
                }
            }

    async def process_phrase(
        self,
        phrase: Phrase,
        output_dir: Path,
        **options: Any
    ) -> Dict[str, Any]:
        """Process a single phrase and generate audio.

        Args:
            phrase: The phrase to process.
            output_dir: Directory to save output files.
            **options: Additional processing options.

        Returns:
            Dictionary with processing results or error information.
        """
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process the phrase using natural pauses if enabled, otherwise use standard processing
            # Note: section_type is None for standalone phrase processing
            if self.use_natural_pauses:
                processed_phrase = await self._process_phrase_with_natural_pauses(phrase, output_dir, None)
            else:
                processed_phrase = await self._process_phrase(phrase, output_dir, None)
            
            # Convert to the expected dictionary format
            result = {
                'id': str(processed_phrase.id) if processed_phrase.id else None,
                'text': processed_phrase.text,
                'language': processed_phrase.language,
                'audio_file': str(processed_phrase.audio_file) if processed_phrase.audio_file else None,
                'metadata': processed_phrase.metadata or {},
                'success': True
            }
            
            # Add translation if it exists
            if hasattr(processed_phrase, 'translation') and processed_phrase.translation is not None:
                result['translation'] = processed_phrase.translation
                
            return result

        except TTSServiceError as e:
            logger.error(f"TTS service error processing phrase: {e}")
            return {
                'success': False,
                'error': {
                    'error_code': e.error_code or 'TTS_SERVICE_ERROR',
                    'error_message': str(e)
                }
            }
            
        except TTSValidationError as e:
            logger.error(f"TTS validation error processing phrase: {e}")
            return {
                'success': False,
                'error': {
                    'error_code': 'TTS_VALIDATION_ERROR',
                    'error_message': str(e)
                }
            }
            
        except Exception as e:
            logger.error(f"Unexpected error processing phrase: {e}", exc_info=True)
            return {
                'success': False,
                'error': {
                    'error_code': 'UNKNOWN_ERROR',
                    'error_message': str(e)
                }
            }

    async def _process_phrase(
        self, phrase: Phrase, output_path: Path, section_type: Optional[str] = None
    ) -> ProcessedPhrase:
        """Process a single phrase (internal implementation).

        Args:
            phrase: The phrase to process.
            output_path: Path to the output directory for this phrase.
            section_type: Type of section for context-aware processing.

        Returns:
            Processed phrase with audio file path.
        """
        self.metrics.phrases_processed += 1
        logger.debug("Processing phrase: %s", phrase.text)

        try:
            # Preprocess text
            preprocessed_text = self._preprocess_text(phrase.text, phrase.language, section_type)

            # Use the voice ID from the phrase if available, otherwise get it from voice selector
            if hasattr(phrase, 'voice_id') and phrase.voice_id:
                voice_id = phrase.voice_id
            else:
                # Get voice ID - extract gender from metadata or use None
                voice_id = await self.voice_selector.get_voice_id(
                    language=phrase.language,
                    gender=phrase.metadata.get("gender") if phrase.metadata else None,
                    speaker_id=phrase.metadata.get("speaker_id") if phrase.metadata else None,
                )

            # Generate audio file with appropriate extension
            audio_file = output_path / f"phrase_{phrase.id}"
            
            # First validate the voice before attempting to synthesize speech
            if hasattr(self.tts_service, 'validate_voice'):
                await self.tts_service.validate_voice(voice_id)
                
            # Handle ellipsis with configurable pause duration
            has_ellipsis = "..." in preprocessed_text
            if has_ellipsis:
                # Replace ellipsis with semicolons for better natural TTS pauses than commas
                tts_text = preprocessed_text.replace("...", "; ")
            else:
                tts_text = preprocessed_text
                
            # Retry TTS synthesis with exponential backoff
            await self._synthesize_speech_with_retry(
                text=tts_text,
                voice_id=voice_id,
                output_path=str(audio_file),
                rate=phrase.metadata.get("rate", 1.0) if phrase.metadata else 1.0,
                pitch=phrase.metadata.get("pitch", 0.0) if phrase.metadata else 0.0,
                volume=1.0,
                speaker_id=phrase.metadata.get("speaker_id") if phrase.metadata else None,
                phrase_text=phrase.text
            )
            
            # If the original text had ellipsis, add configurable silence for longer pauses
            if has_ellipsis and self.ellipsis_pause_duration_ms > 0:
                original_file = audio_file.with_suffix('.mp3')
                temp_file = output_path / f"temp_{original_file.name}"
                
                # Add silence to create longer pauses
                silence_duration_s = self.ellipsis_pause_duration_ms / 1000.0
                await self.audio_processor.add_silence(
                    input_file=original_file,
                    output_file=temp_file,
                    duration=silence_duration_s,
                    position="end"
                )
                
                # Replace original file with processed version
                if temp_file.exists():
                    original_file.unlink()
                    temp_file.rename(original_file)
            
            # Update audio_file path to include the correct extension
            audio_file = audio_file.with_suffix('.mp3')  # Edge TTS always outputs MP3

            # Process audio if audio_config is provided in metadata
            audio_config = phrase.metadata.get("audio_config") if phrase.metadata else None
            if audio_config:
                await self._process_audio(audio_file, audio_config)

            # Create a dictionary with the required fields
            processed_data = {
                "id": phrase.id,
                "text": phrase.text,
                "language": phrase.language,
                "audio_file": audio_file,
                "metadata": phrase.metadata or {},
            }
            
            # Only add translation if it exists in the phrase
            if hasattr(phrase, 'translation') and phrase.translation is not None:
                processed_data["translation"] = phrase.translation
                
            return ProcessedPhrase(**processed_data)

        except Exception as e:
            logger.error("Error processing phrase '%s': %s", phrase.text, str(e))
            raise

    async def _process_phrase_with_natural_pauses(
        self, phrase: Phrase, output_path: Path, section_type: Optional[str] = None
    ) -> ProcessedPhrase:
        """Process a phrase using natural pause system based on linguistic boundaries.
        
        Args:
            phrase: The phrase to process.
            output_path: Path to the output directory for this phrase.
            section_type: Type of section for context-aware processing.
            
        Returns:
            Processed phrase with audio file path.
        """
        from .linguistic_boundary_detector import split_with_natural_pauses
        
        self.metrics.phrases_processed += 1
        logger.debug("Processing phrase with natural pauses: %s", phrase.text)

        try:
            # Preprocess text
            preprocessed_text = self._preprocess_text(phrase.text, phrase.language, section_type)

            # Get voice ID
            if hasattr(phrase, 'voice_id') and phrase.voice_id:
                voice_id = phrase.voice_id
            else:
                voice_id = await self.voice_selector.get_voice_id(
                    language=phrase.language,
                    gender=phrase.metadata.get("gender") if phrase.metadata else None,
                    speaker_id=phrase.metadata.get("speaker_id") if phrase.metadata else None,
                )

            # Validate voice
            if hasattr(self.tts_service, 'validate_voice'):
                await self.tts_service.validate_voice(voice_id)

            # Determine if this should be slow speech
            is_slow = (
                "..." in preprocessed_text or  # Contains ellipsis
                (phrase.metadata and phrase.metadata.get("rate", 1.0) < 0.8) or  # Slow rate setting
                "[SLOW]" in preprocessed_text.upper()  # Explicit slow marker
            )

            # Check if this is a key phrases section to use dynamic pauses
            use_dynamic_pauses = (section_type == 'key_phrases')
            logger.debug(f"Processing phrase '{phrase.text}' in section_type='{section_type}', using {'dynamic' if use_dynamic_pauses else 'fixed'} pauses")
            
            if use_dynamic_pauses:
                # Dynamic pause mode for key phrases: Single-pass with duration measurement
                initial_segments = split_with_natural_pauses(preprocessed_text, is_slow=is_slow)
                
                if not initial_segments:
                    # Fallback to original processing if no segments
                    return await self._process_phrase(phrase, output_path)

                # Generate audio for text segments and collect their durations
                audio_parts = []
                segment_durations = []
                
                for i, segment in enumerate(initial_segments):
                    if segment['type'] == 'text':
                        # Create audio file for this segment
                        segment_file = output_path / f"phrase_{phrase.id}_segment_{i}"
                        
                        # Get voice settings from segment
                        voice_settings = segment.get('voice_settings', {})
                        rate = voice_settings.get('rate', 1.0)
                        
                        # Synthesize speech for this text segment (only once!)
                        await self._synthesize_speech_with_retry(
                            text=segment['content'],
                            voice_id=voice_id,
                            output_path=str(segment_file),
                            rate=rate,
                            pitch=phrase.metadata.get("pitch", 0.0) if phrase.metadata else 0.0,
                            volume=1.0,
                            speaker_id=phrase.metadata.get("speaker_id") if phrase.metadata else None,
                            phrase_text=phrase.text
                        )
                        
                        segment_file = segment_file.with_suffix('.mp3')
                        if segment_file.exists():
                            audio_parts.append(segment_file)
                            # Get duration of this audio segment
                            try:
                                duration = await self.audio_processor.get_audio_duration(segment_file)
                                segment_durations.append(duration)
                            except Exception as e:
                                logger.warning(f"Failed to get duration for segment {i}: {e}")
                                segment_durations.append(0.0)  # Fallback to 0 duration
                        
                # Calculate dynamic pauses and add them between existing audio segments
                from .natural_pause_calculator import NaturalPauseCalculator
                from .linguistic_boundary_detector import detect_linguistic_boundaries
                
                calculator = NaturalPauseCalculator()
                complexity = 'slow' if is_slow else 'normal'
                boundaries = detect_linguistic_boundaries(preprocessed_text)
                
                # Build final audio with dynamic pauses inserted
                final_audio_parts = []
                segment_index = 0
                
                for boundary_type, pos in boundaries:
                    # Add the audio segment if we have one
                    if segment_index < len(audio_parts):
                        final_audio_parts.append(audio_parts[segment_index])
                        
                        # Calculate dynamic pause for this segment
                        audio_duration = segment_durations[segment_index] if segment_index < len(segment_durations) else None
                        pause_duration = calculator.get_pause_for_boundary(
                            boundary_type, complexity, audio_duration_seconds=audio_duration
                        )
                        
                        # Add pause after this segment
                        if segment_index < len(audio_parts) - 1:  # Don't add pause after last segment
                            pause_file = output_path / f"phrase_{phrase.id}_pause_{segment_index}.mp3"
                            silence_duration_s = pause_duration / 1000.0
                            
                            await self.audio_processor.add_silence(
                                input_file=audio_parts[segment_index],
                                output_file=pause_file,
                                duration=silence_duration_s,
                                position="end"
                            )
                            
                            if pause_file.exists():
                                final_audio_parts[-1] = pause_file  # Replace with version that has silence
                        
                        segment_index += 1
                
                # If there are remaining audio segments without boundaries, add them
                while segment_index < len(audio_parts):
                    final_audio_parts.append(audio_parts[segment_index])
                    segment_index += 1
                
                audio_parts = final_audio_parts
            
            else:
                # Fixed pause mode for non-key phrases: Use original natural pause system
                segments = split_with_natural_pauses(preprocessed_text, is_slow=is_slow)  # No audio durations = fixed pauses
                
                if not segments:
                    # Fallback to original processing if no segments
                    return await self._process_phrase(phrase, output_path)

                # Generate audio with fixed pauses
                audio_parts = []
                for i, segment in enumerate(segments):
                    if segment['type'] == 'text':
                        # Create temporary audio file for this segment
                        segment_file = output_path / f"phrase_{phrase.id}_segment_{i}"
                        
                        # Get voice settings from segment
                        voice_settings = segment.get('voice_settings', {})
                        rate = voice_settings.get('rate', 1.0)
                        
                        # Synthesize speech for this text segment
                        await self._synthesize_speech_with_retry(
                            text=segment['content'],
                            voice_id=voice_id,
                            output_path=str(segment_file),
                            rate=rate,
                            pitch=phrase.metadata.get("pitch", 0.0) if phrase.metadata else 0.0,
                            volume=1.0,
                            speaker_id=phrase.metadata.get("speaker_id") if phrase.metadata else None,
                            phrase_text=phrase.text
                        )
                        
                        segment_file = segment_file.with_suffix('.mp3')
                        if segment_file.exists():
                            audio_parts.append(segment_file)
                    
                    elif segment['type'] == 'pause':
                        # Create silence for pause
                        pause_file = output_path / f"phrase_{phrase.id}_pause_{i}.mp3"
                        silence_duration_s = segment['duration'] / 1000.0
                        
                        # Create a minimal audio file and add silence
                        if audio_parts:  # Only add silence if we have previous audio
                            await self.audio_processor.add_silence(
                                input_file=audio_parts[-1],  # Use last audio file as base
                                output_file=pause_file,
                                duration=silence_duration_s,
                                position="end"
                            )
                            # Replace the last audio file with the one with silence
                            if pause_file.exists():
                                audio_parts[-1] = pause_file

            # Combine all audio parts into final file
            final_audio_file = output_path / f"phrase_{phrase.id}.mp3"
            
            if len(audio_parts) > 1:
                await self.audio_processor.concatenate_audio(
                    [str(f) for f in audio_parts],
                    str(final_audio_file)
                )
                
                # Clean up temporary segment files
                for part in audio_parts:
                    if part != final_audio_file and part.exists():
                        try:
                            part.unlink()
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to clean up temporary file {part}: {cleanup_error}")
            elif len(audio_parts) == 1:
                # Just one part, rename it to final name
                audio_parts[0].rename(final_audio_file)
            
            # Process audio if audio_config is provided
            audio_config = phrase.metadata.get("audio_config") if phrase.metadata else None
            if audio_config:
                await self._process_audio(final_audio_file, audio_config)

            # Create processed phrase data
            processed_data = {
                "id": phrase.id,
                "text": phrase.text,
                "language": phrase.language,
                "audio_file": final_audio_file,
                "metadata": phrase.metadata or {},
            }

            # Only add translation if it exists
            if hasattr(phrase, 'translation') and phrase.translation is not None:
                processed_data["translation"] = phrase.translation
                
            return ProcessedPhrase(**processed_data)

        except Exception as e:
            logger.error("Error processing phrase with natural pauses '%s': %s", phrase.text, str(e))
            raise

    def _sanitize_filename(self, title: str) -> str:
        """Convert a section title to a safe filename.
        
        Args:
            title: The section title to convert.
            
        Returns:
            A safe filename string with only alphanumeric characters and underscores.
        """
        import re
        import unicodedata
        
        # Normalize unicode characters
        filename = unicodedata.normalize('NFKD', title)
        
        # Replace spaces with underscores and convert to lowercase
        filename = filename.strip().lower().replace(' ', '_')
        
        # Remove any characters that are not alphanumeric or underscores
        filename = re.sub(r'[^a-z0-9_]', '', filename)
        
        # Remove multiple consecutive underscores
        filename = re.sub(r'_+', '_', filename)
        
        # Remove leading/trailing underscores
        filename = filename.strip('_')
        
        # If the filename is empty or too short, use a default
        if not filename or len(filename) < 2:
            filename = 'section'
            
        return filename

    def _extract_day_number(self, lesson: Lesson) -> str:
        """Extract day number from lesson title or content.
        
        Args:
            lesson: The lesson object to extract day number from.
            
        Returns:
            Day number as string, or "0" if not found.
        """
        # First check the lesson title
        if lesson.title:
            day_match = re.search(r'day\s*(\d+)', lesson.title.lower())
            if day_match:
                return day_match.group(1)
        
        # Then check the first section for narrator day announcements
        for section in lesson.sections:
            for phrase in section.phrases:
                # Check if this phrase is from the narrator (stored in metadata)
                speaker_info = phrase.metadata.get('speaker', '')
                if speaker_info and 'narrator' in speaker_info.lower():
                    day_match = re.search(r'day\s*(\d+)', phrase.text.lower())
                    if day_match:
                        return day_match.group(1)
        
        # Fallback: try to extract from filename in the lesson description
        if lesson.description and 'day-' in lesson.description.lower():
            day_match = re.search(r'day-(\d+)', lesson.description.lower())
            if day_match:
                return day_match.group(1)
        
        return "0"
    
    def _classify_section_type(self, section: Section) -> str:
        """Classify section type to determine suffix (a/b/c/d).
        
        Args:
            section: The section to classify.
            
        Returns:
            Single letter suffix: 'a' for key_phrases, 'b' for natural_speed, 
            'c' for slow_speed, 'd' for translated, 'x' for unknown/intro.
        """
        section_title = section.title.lower() if section.title else ""
        
        # Explicit section type detection
        if 'key' in section_title or 'phrase' in section_title or 'vocabulary' in section_title:
            return 'a'
        elif 'natural' in section_title or 'normal' in section_title:
            return 'b'
        elif 'slow' in section_title:
            return 'c'
        elif 'translat' in section_title or 'english' in section_title:
            return 'd'
        
        # Check for auto-generated section titles (like "Section 1", "Section 2", etc.)
        if re.match(r'^section\s+\d+$', section_title) or section_title.startswith('syllable'):
            return 'x'  # Unknown/intro section, not key phrases
        
        # For other ambiguous cases, classify as unknown rather than assuming key phrases
        return 'x'  # Changed from 'a' to avoid misclassification
    
    def _get_section_type_name(self, suffix: str) -> str:
        """Get the readable name for a section type suffix.
        
        Args:
            suffix: Single letter suffix (a/b/c/d/x).
            
        Returns:
            Readable section type name.
        """
        suffix_to_name = {
            'a': 'key_phrases',
            'b': 'natural_speed',
            'c': 'slow_speed',
            'd': 'translated',
            'x': 'intro'
        }
        return suffix_to_name.get(suffix, 'intro')
        
    def _preprocess_text(self, text: str, language: str, section_type: Optional[str] = None) -> str:
        """Preprocess text before TTS processing.

        Args:
            text: The text to preprocess.
            language: The language of the text (can be string or Language enum).
            section_type: Type of section for context-aware processing.

        Returns:
            Preprocessed text with abbreviations and language-specific fixes applied.
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Apply TTS preprocessing (abbreviations, language-specific fixes)
        try:
            from tunatale.core.utils.tts_preprocessor import preprocess_text_for_tts
            from tunatale.core.models.enums import Language
            
            # Handle both string and Language enum input
            if hasattr(language, 'code'):  # It's a Language enum
                lang_code = language.code
            else:  # It's a string
                lang_code = str(language).lower()
            
            # Map language to full language code for TTS
            language_code = {
                'tagalog': 'fil-PH',
                'fil': 'fil-PH',
                'tl': 'fil-PH',  # Alternative Tagalog code
                'english': 'en-US',
                'en': 'en-US',
                'spanish': 'es-ES',
                'es': 'es-ES'
            }.get(lang_code, 'en-US')  # Default to en-US if not found
            
            logger.debug(f"Preprocessing text with language code: {language_code}, section: {section_type}")
            text = preprocess_text_for_tts(text, language_code, section_type)
            logger.debug(f"Preprocessed text for TTS: '{text}'")
        except ImportError as e:
            logger.warning(f"Failed to import TTS preprocessor: {e}")
        except Exception as e:
            logger.warning(f"Error in TTS preprocessing: {e}")
            
        return text

    async def _synthesize_speech_with_retry(
        self,
        text: str,
        voice_id: str,
        output_path: str,
        rate: float,
        pitch: float,
        volume: float,
        speaker_id: Optional[str],
        phrase_text: str,
        max_retries: int = 3
    ) -> None:
        """Synthesize speech with retry logic for transient errors.
        
        Args:
            text: The text to synthesize.
            voice_id: The voice ID to use.
            output_path: Path to save the audio file.
            rate: Speech rate.
            pitch: Speech pitch.
            volume: Speech volume.
            speaker_id: Optional speaker ID for voice modifications.
            phrase_text: Original phrase text for logging.
            max_retries: Maximum number of retry attempts.
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                await self.tts_service.synthesize_speech(
                    text=text,
                    voice_id=voice_id,
                    output_path=output_path,
                    rate=rate,
                    pitch=pitch,
                    volume=volume,
                    speaker_id=speaker_id
                )
                return  # Success
                
            except TTSTransientError as e:
                last_error = e
                if attempt < max_retries:
                    # Exponential backoff: 1s, 2s, 4s
                    delay = 2 ** attempt
                    logger.warning(
                        "TTS transient error for phrase '%s' (attempt %d/%d): %s. Retrying in %d seconds...",
                        phrase_text, attempt + 1, max_retries + 1, str(e), delay
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "TTS failed for phrase '%s' after %d attempts: %s",
                        phrase_text, max_retries + 1, str(e)
                    )
                    raise TTSServiceError(
                        f"TTS failed after {max_retries + 1} attempts: {str(e)}"
                    ) from e
                    
            except TTSRateLimitError as e:
                last_error = e
                if attempt < max_retries:
                    # Use retry_after if provided, otherwise exponential backoff
                    delay = getattr(e, 'retry_after', 2 ** attempt)
                    logger.warning(
                        "TTS rate limit for phrase '%s' (attempt %d/%d): %s. Retrying in %d seconds...",
                        phrase_text, attempt + 1, max_retries + 1, str(e), delay
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "TTS rate limited for phrase '%s' after %d attempts: %s",
                        phrase_text, max_retries + 1, str(e)
                    )
                    raise TTSServiceError(
                        f"TTS rate limited after {max_retries + 1} attempts: {str(e)}"
                    ) from e
                    
            except TTSValidationError as e:
                # Don't retry validation errors
                logger.error("TTS validation error for phrase '%s': %s", phrase_text, str(e))
                raise
                
            except Exception as e:
                # For other errors, treat as non-retryable
                logger.error("TTS error for phrase '%s': %s", phrase_text, str(e))
                raise
        
        # This should never be reached, but just in case
        if last_error:
            raise TTSServiceError(f"TTS failed after {max_retries + 1} attempts") from last_error

    async def _process_audio(
        self, audio_file: Path, config: AudioConfig
    ) -> None:
        """Process audio file with the given configuration.

        Args:
            audio_file: Path to the audio file to process.
            config: Audio processing configuration.
        """
        try:
            # Apply audio processing
            if config.normalize:
                await self.audio_processor.normalize(str(audio_file))

            if config.trim_silence:
                await self.audio_processor.trim_silence(
                    str(audio_file),
                    silence_threshold=config.silence_threshold,
                    silence_duration=config.silence_duration,
                )

            if config.fade_in > 0 or config.fade_out > 0:
                await self.audio_processor.fade(
                    str(audio_file),
                    fade_in=config.fade_in,
                    fade_out=config.fade_out,
                )

        except Exception as e:
            logger.error("Error processing audio file %s: %s", audio_file, str(e))
            raise

    async def _concatenate_audio_files(
        self, input_files: List[Path], output_file: Path, progress: Optional[Any] = None
    ) -> None:
        """Concatenate multiple audio files into a single file.

        Args:
            input_files: List of input audio file paths.
            output_file: Output audio file path.
            progress: Optional progress reporter for tracking progress.
        """
        if not input_files:
            logger.warning("No input files provided for concatenation")
            return

        # Filter out any non-existent files
        valid_inputs = [f for f in input_files if f.exists()]
        if not valid_inputs:
            logger.error("No valid input files found for concatenation")
            return
            
        if len(valid_inputs) != len(input_files):
            logger.warning("Some input files were not found and will be skipped")
            
        input_files = valid_inputs

        # Create a task ID for progress reporting
        task_id = f"concat_{output_file.stem}"
        
        # Determine the output extension based on the first input file
        first_ext = input_files[0].suffix.lower()
        output_file = output_file.with_suffix(first_ext)
        
        if len(input_files) == 1:
            # If there's only one file, just copy it
            shutil.copy2(input_files[0], output_file)
            if progress:
                await progress.update(
                    task_id=task_id,
                    completed=1, 
                    total=1,
                    status="Copied single file",
                    phase="concatenate"
                )
            return

        total_steps = len(input_files) + 1  # +1 for the concatenation step
        
        # Initialize progress if available
        if progress:
            await progress.update(
                task_id=task_id,
                completed=0, 
                total=total_steps, 
                status="Preparing files...",
                phase="concatenate"
            )

        # Use the files directly if they're already in the same format
        if all(f.suffix.lower() == first_ext for f in input_files[1:]):
            audio_files = input_files
        else:
            # Convert all files to the same format as the first file
            audio_files = []
            for i, input_file in enumerate(input_files):
                if progress:
                    await progress.update(
                        task_id=task_id,
                        completed=i,
                        total=total_steps,
                        status=f"Converting file {i+1}/{len(input_files)} to {first_ext}",
                        phase="concatenate"
                    )
                
                if input_file.suffix.lower() != first_ext:
                    try:
                        # Convert to the target format
                        converted_file = input_file.with_suffix(first_ext)
                        await self.audio_processor.convert_format(
                            str(input_file), str(converted_file), first_ext[1:]
                        )
                        if converted_file.exists() and converted_file.stat().st_size > 0:
                            audio_files.append(converted_file)
                        else:
                            logger.warning(f"Failed to convert {input_file} - file is empty or not created")
                    except Exception as e:
                        logger.error(f"Error converting {input_file}: {str(e)}")
                else:
                    if input_file.stat().st_size > 0:
                        audio_files.append(input_file)
                    else:
                        logger.warning(f"Skipping empty file: {input_file}")

        if not audio_files:
            logger.error("No valid audio files to concatenate after filtering")
            if progress:
                await progress.update(
                    task_id=task_id,
                    status="Error: No valid audio files to concatenate",
                    phase="error"
                )
            return

        try:
            # Update progress before concatenation
            if progress:
                await progress.update(
                    task_id=task_id,
                    completed=len(audio_files),
                    total=total_steps,
                    status=f"Concatenating {len(audio_files)} files...",
                    phase="concatenate"
                )

            # Concatenate audio files using the audio processor
            await self.audio_processor.concatenate_audio(
                [str(f) for f in audio_files], 
                str(output_file)
            )

            # Verify output file was created and has content
            if not output_file.exists() or output_file.stat().st_size == 0:
                raise RuntimeError("Failed to create valid output file")

            # Update progress after successful concatenation
            if progress:
                await progress.update(
                    task_id=task_id,
                    completed=total_steps,
                    total=total_steps,
                    status="Concatenation complete",
                    phase="concatenate"
                )
                # Mark task as complete
                await progress.complete_task(task_id)
                
        except Exception as e:
            # Update progress on error
            error_msg = f"Error during concatenation: {str(e)}"
            logger.error(error_msg)
            if progress:
                await progress.update(
                    task_id=task_id,
                    status=error_msg,
                    phase="error"
                )
            # Re-raise the exception to be handled by the caller
            raise
        
        self.metrics.audio_generated += 1

    async def _generate_section_audio(
        self, sections: List[ProcessedSection], output_path: Path
    ) -> Dict[str, Path]:
        """Generate audio files for sections.

        Args:
            sections: List of processed sections.
            output_path: Path to the output directory.

        Returns:
            Dictionary mapping section IDs to audio file paths.
        """
        section_audio_files = {}
        for section in sections:
            if section.audio_file and section.audio_file.exists():
                section_audio_files[str(section.id)] = section.audio_file

        return section_audio_files

    async def _generate_lesson_audio(
        self, 
        lesson: Lesson,
        section_audio_files: Dict[str, Path], 
        output_path: Path,
        progress: Optional[Any] = None
    ) -> Path:
        """Generate a single audio file for the entire lesson.

        Args:
            lesson: The lesson object to extract day number from.
            section_audio_files: Dictionary mapping section IDs to audio file paths.
            output_path: Path to the output directory.
            progress: Optional progress reporter for tracking progress.

        Returns:
            Path to the generated lesson audio file.
        """
        # Extract day number for new naming scheme
        day_number = self._extract_day_number(lesson)
        
        # Use new naming pattern: [Number of day] - lesson.mp3
        # Place the lesson file at the top level of the run directory
        lesson_audio_file = output_path / f"{day_number} - lesson.mp3"
        audio_files = list(section_audio_files.values())

        if audio_files:
            if progress:
                await progress.update(
                    task_id="lesson_audio",
                    completed=0,
                    total=1,  # Just one step for lesson audio generation
                    status=f"Generating lesson audio from {len(audio_files)} sections..."
                )
            
            await self._concatenate_audio_files(
                audio_files, 
                lesson_audio_file,
                progress=progress  # Pass progress reporter to concatenation
            )
        
        # Ensure the file has the correct extension
        lesson_audio_file = lesson_audio_file.with_suffix('.mp3')
        return lesson_audio_file

    @staticmethod
    def json_serializer(obj):
        """Custom JSON serializer for handling various types."""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            # Convert objects with __dict__ to dict, excluding private attributes
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        raise TypeError(f"Type {type(obj)} not serializable")
        
    def _prepare_metadata(
        self,
        lesson: Lesson,
        processed_sections: List[ProcessedSection],
        lesson_audio_file: Path,
        output_path: Path,
    ) -> Dict[str, Any]:
        """Prepare metadata for the processed lesson.

        Args:
            lesson: The original lesson.
            processed_sections: List of processed sections.
            lesson_audio_file: Path to the lesson audio file.
            output_path: Path to the output directory.

        Returns:
            Dictionary containing the metadata.
        """
        # Convert UUIDs to strings for JSON serialization
        def convert_uuids(obj):
            if isinstance(obj, dict):
                return {
                    str(k) if isinstance(k, UUID) else k: convert_uuids(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, (list, tuple)):
                return [convert_uuids(x) for x in obj]
            elif isinstance(obj, UUID):
                return str(obj)
            return obj
        
        try:
            # Prepare sections metadata
            sections_metadata = []
            for section in processed_sections:
                section_data = {
                    "id": str(section.id),
                    "title": section.title,
                    "audio_file": section.audio_file.relative_to(output_path)
                    if section.audio_file
                    else None,
                    "phrases": [
                        {
                            "id": str(phrase.id),
                            "text": phrase.text,
                            "translation": phrase.translation,
                            "language": phrase.language,
                            "audio_file": phrase.audio_file.relative_to(output_path)
                            if phrase.audio_file
                            else None,
                            "metadata": phrase.metadata,
                        }
                        for phrase in section.phrases
                    ],
                }
                sections_metadata.append(section_data)

            # Prepare lesson metadata
            metadata = {
                "id": str(lesson.id),
                "title": lesson.title,
                "language": lesson.target_language,
                "level": lesson.difficulty,
                "created_at": datetime.now().isoformat(),
                "audio_file": lesson_audio_file.relative_to(output_path),
                "sections": sections_metadata,
                "metrics": self.metrics.get_summary(),
            }

            # Convert UUIDs to strings
            metadata = convert_uuids(metadata)

            # Add audio files index
            audio_files = {
                "sections": {
                    str(section.id): str(
                        section.audio_file.relative_to(output_path)
                        if section.audio_file
                        else ""
                    )
                    for section in processed_sections
                },
                "phrases": {}
            }
            
            # Add section audio files
            for section in metadata.get('sections', []):
                section_id = str(section.get('section_id', ''))
                if section_id and 'audio_file' in section:
                    audio_files['sections'][section_id] = str(section['audio_file'])
                    
                    # Add phrase audio files
                    for phrase in section.get('phrases', []):
                        phrase_id = str(phrase.get('phrase_id', ''))
                        if phrase_id and 'audio_file' in phrase:
                            if section_id not in audio_files['phrases']:
                                audio_files['phrases'][section_id] = {}
                            audio_files['phrases'][section_id][phrase_id] = str(phrase['audio_file'])
            
            metadata['audio_files'] = audio_files
            
            return metadata

        except Exception as e:
            logger.error("Error preparing metadata: %s", str(e))
            raise

    def _save_metadata(self, metadata: Dict[str, Any], output_path: Path) -> None:
        """Save metadata to a JSON file.

        Args:
            metadata: The metadata to save.
            output_path: Path to the output JSON file.
        """
        try:
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file with custom serializer
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=self.json_serializer)
                
        except Exception as e:
            logger.error("Failed to save metadata: %s", str(e))
            raise

    def _log_metrics(self) -> None:
        """Log performance metrics."""
        metrics = self.metrics.get_summary()
        logger.info(
            "Lesson processing completed in %.2f seconds", metrics["total_time_seconds"]
        )
        logger.info("Phrases processed: %d", metrics["phrases_processed"])
        logger.info("Sections processed: %d", metrics["sections_processed"])
        logger.info("Audio files generated: %d", metrics["audio_files_generated"])
        
        # Log phase timings
        logger.info("Phase timings:")
        for phase, duration in metrics["phase_timings"].items():
            logger.info("  %s: %.2f seconds", phase, duration)
