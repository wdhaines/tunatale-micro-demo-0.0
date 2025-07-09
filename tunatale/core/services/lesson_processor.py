"""Service for processing lessons and generating audio."""
from typing import Union, Dict, Any, Optional, List
import asyncio
import logging
import os
import uuid
from pathlib import Path
import shutil
import time
import psutil
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from tunatale.core.exceptions import AudioProcessingError, TTSServiceError
from tunatale.core.models.lesson import Lesson, SectionType
from tunatale.core.models.phrase import Phrase
from tunatale.core.models.section import Section
from tunatale.core.ports.audio_processor import AudioProcessor
from tunatale.core.ports.lesson_processor import (
    LessonProcessorBase,
    ProgressCallback,
)
from tunatale.core.ports.tts_service import TTSService

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Class to track performance metrics for the lesson processor."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    memory_usage_bytes: int = 0
    peak_memory_usage_bytes: int = 0
    phase_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def record_phase_start(self, phase_name: str) -> None:
        """Record the start of a processing phase."""
        self.phase_metrics[phase_name] = {
            'start_time': time.time(),
            'memory_start': psutil.Process().memory_info().rss
        }
    
    def record_phase_end(self, phase_name: str) -> None:
        """Record the end of a processing phase and calculate metrics."""
        if phase_name not in self.phase_metrics:
            return
            
        phase = self.phase_metrics[phase_name]
        phase['end_time'] = time.time()
        phase['duration'] = phase['end_time'] - phase['start_time']
        
        # Record memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        phase['memory_end'] = memory_info.rss
        phase['memory_usage'] = max(0, phase['memory_end'] - phase.get('memory_start', 0))
        
        # Update global metrics
        self.memory_usage_bytes = memory_info.rss
        self.peak_memory_usage_bytes = max(self.peak_memory_usage_bytes, memory_info.rss)
    
    def get_phase_metrics(self, phase_name: str) -> Dict[str, Any]:
        """Get metrics for a specific phase."""
        return self.phase_metrics.get(phase_name, {})
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        # Calculate phase durations
        phase_durations = {
            phase: metrics.get('duration', 0)
            for phase, metrics in self.phase_metrics.items()
            if 'duration' in metrics
        }
        
        return {
            'total_duration': total_duration,
            'memory_usage_mb': self.memory_usage_bytes / (1024 * 1024),
            'peak_memory_usage_mb': self.peak_memory_usage_bytes / (1024 * 1024),
            'phase_durations': phase_durations,
            'phases': self.phase_metrics,
        }

class LessonProcessor(LessonProcessorBase):
    """Service for processing lessons and generating audio."""
    
    def __init__(
        self,
        tts_service: TTSService,
        audio_processor: AudioProcessor,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the lesson processor with performance monitoring.
        
        Args:
            tts_service: Text-to-speech service to use for audio generation
            audio_processor: Audio processor for post-processing audio files
            config: Optional configuration dictionary
        """
        self.tts_service = tts_service
        self.audio_processor = audio_processor
        self.config = config or {}
        self.performance = PerformanceMetrics()
        
        # Initialize tracemalloc for memory tracking
        tracemalloc.start()
    
    async def process_lesson(
        self,
        lesson: Lesson,
        output_dir: Union[str, Path],
        progress_callback: Optional[ProgressCallback] = None,
        **options: Any
    ) -> Dict[str, Any]:
        """Process a lesson and generate audio files.
        
        Args:
            lesson: The lesson to process
            output_dir: Directory to save generated audio files
            progress_callback: Optional callback for progress updates
            **options: Additional processing options
            
        Returns:
            Dictionary with processing results and metadata
        """
        self.performance.record_phase_start('process_lesson')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize result dictionary
        result = {
            'lesson_id': str(lesson.id),
            'output_dir': str(output_dir.absolute()),
            'sections': [],
            'start_time': time.time(),
            'end_time': None,
            'success': True,
            'error': None,
            'performance': {},
            'processing_info': {}
        }
        
        try:
            # Process each section in the lesson
            for section in lesson.sections:
                section_result = await self.process_section(
                    section=section,
                    lesson=lesson,
                    output_dir=output_dir,
                    progress_callback=progress_callback,
                    **options
                )
                result['sections'].append(section_result)
                
                # Check for errors in section processing
                if not section_result.get('success', False):
                    result['success'] = False
                    result['error'] = section_result.get('error', 'Unknown error in section processing')
                    break
            
            # Combine section audio files if there are multiple sections
            section_audio_files = [s.get('audio_file') for s in result['sections'] if s.get('audio_file')]
            if section_audio_files:
                # Get output format from config or default to 'mp3'
                output_format = self.config.get('output_format', 'mp3')
                final_audio = output_dir / f'lesson_combined.{output_format}'
                await self.audio_processor.concatenate_audio(
                    input_files=section_audio_files,
                    output_file=final_audio
                )
                result['final_audio_file'] = str(final_audio)
            
            # Generate lesson metadata
            result['end_time'] = time.time()
            result['duration'] = result['end_time'] - result['start_time']
            
            # Record performance metrics
            self.performance.record_phase_end('process_lesson')
            result['performance'] = self.performance.get_summary()
            
            # Save metadata to a JSON file
            metadata_path = output_dir / 'metadata.json'
            # Include the lesson object in the metadata for saving
            result_with_lesson = result.copy()
            result_with_lesson['lesson'] = lesson
            self._save_metadata(result_with_lesson, metadata_path)
            
            return result
            
        except Exception as e:
            logger.exception("Error processing lesson")
            result['success'] = False
            result['error'] = str(e)
            result['end_time'] = time.time()
            result['duration'] = result['end_time'] - result['start_time']
            
            # Record performance metrics even in case of error
            self.performance.record_phase_end('process_lesson')
            result['performance'] = self.performance.get_summary()
            
            return result
    
    async def process_section(
        self,
        section: Section,
        lesson: Lesson,
        output_dir: Path,
        progress_callback: Optional[ProgressCallback] = None,
        **options: Any
    ) -> Dict[str, Any]:
        """Process a single section of a lesson.
        
        Args:
            section: The section to process
            lesson: The parent lesson
            output_dir: Base directory for output (will create sections and phrases subdirectories)
            progress_callback: Optional callback for progress updates
            **options: Additional processing options
            
        Returns:
            Dictionary with section processing results
        """
        # Generate a unique identifier for this processing run
        process_id = f"{int(time.time())}_{os.getpid()}"
        
        logger.info("=" * 100)
        logger.info(f"[START] process_section - Process ID: {process_id}")
        logger.info(f"- Section ID: {getattr(section, 'id', 'NO_ID')}")
        logger.info(f"- Section type: {getattr(section, 'section_type', 'NO_TYPE')}")
        logger.info(f"- Output dir: {output_dir}")
        logger.info(f"- Options: {options}")
        logger.info(f"- Process ID: {process_id}")
        logger.info("-" * 100)
        
        # Log section details
        logger.debug(f"Section details: {section}")
        if hasattr(section, '__dict__'):
            logger.debug(f"Section attributes: {section.__dict__}")
        
        # Log lesson details if available
        if lesson:
            logger.debug(f"Parent lesson ID: {getattr(lesson, 'id', 'NO_ID')}")
            logger.debug(f"Lesson target language: {getattr(lesson, 'target_language', 'NOT_SET')}")
            logger.debug(f"Lesson native language: {getattr(lesson, 'native_language', 'NOT_SET')}")
        
        # Record performance metrics
        try:
            self.performance.record_phase_start(f'section_{section.id}')
            logger.debug("Recorded phase start for section")
        except Exception as e:
            logger.error(f"Error recording phase start: {e}", exc_info=True)
            raise
        
        # Create section directory directly in the output directory
        try:
            # Create a unique section directory name without nesting
            section_dir = output_dir / f'section_{section.id}'
            logger.debug(f"Ensuring section directory exists: {section_dir}")
            section_dir.mkdir(parents=True, exist_ok=True)
            
            logger.debug("Using section directory for phrase files")
        except Exception as e:
            logger.error(f"Error creating directories: {e}", exc_info=True)
            raise
        
        result = {
            'section_id': str(section.id) if hasattr(section, 'id') else 'NO_ID',
            'section_type': section.section_type.value if hasattr(section, 'section_type') else 'NO_TYPE',
            'phrases': [],
            'start_time': time.time(),
            'end_time': None,
            'success': True,
            'error': None,
            'audio_file': None,
            'output_dirs': {
                'sections': str(section_dir.relative_to(output_dir)) if output_dir else 'NO_OUTPUT_DIR',
                'phrases': str(section_dir.relative_to(output_dir)) if output_dir else 'NO_OUTPUT_DIR'
            },
            'performance': {}
        }
        
        logger.debug(f"Initialized result dict: {result}")
        
        try:
            # Process each phrase in the section
            phrase_results = []
            if not hasattr(section, 'phrases') or not section.phrases:
                logger.warning(f"Section {getattr(section, 'id', 'NO_ID')} has no phrases to process")
            else:
                total_phrases = len(section.phrases)
                logger.info(f"Processing {total_phrases} phrases in section {getattr(section, 'id', 'NO_ID')}")
                
                for i, phrase in enumerate(section.phrases, 1):
                    phrase_id = getattr(phrase, 'id', f'phrase_{i}')
                    phrase_text = getattr(phrase, 'text', 'NO_TEXT')
                    
                    logger.info(f"[Phrase {i}/{total_phrases}] ID: {phrase_id}")
                    logger.info(f"[Phrase {i}/{total_phrases}] Text: {phrase_text[:100]}{'...' if len(phrase_text) > 100 else ''}")
                    
                    # Log detailed phrase information
                    logger.debug(f"Phrase type: {type(phrase)}")
                    if hasattr(phrase, '__dict__'):
                        # Filter out any large binary data from logs
                        safe_phrase_data = {
                            k: v for k, v in phrase.__dict__.items() 
                            if not k.startswith('_') and not callable(v)
                        }
                        logger.debug(f"Phrase attributes: {safe_phrase_data}")
                    
                    # Log voice settings if available
                    if hasattr(phrase, 'voice_settings'):
                        logger.debug(f"Phrase voice settings: {getattr(phrase, 'voice_settings', {})}")
                    
                    # Log language information
                    if hasattr(phrase, 'language'):
                        logger.debug(f"Phrase language: {getattr(phrase, 'language', 'NOT_SET')}")
                    
                    logger.info(f"[Phrase {i}/{total_phrases}] Starting processing...")
                    
                    try:
                        # Process the phrase with a semaphore to limit concurrency
                        logger.debug(f"[Phrase {i}/{total_phrases}] Acquiring semaphore...")
                        phrase_result = await self._process_phrase_with_semaphore(
                            phrase=phrase,
                            phrase_dir=section_dir,  # Use section_dir as the base output directory
                            semaphore=asyncio.Semaphore(5),  # Limit concurrency
                            **options
                        )
                        logger.debug(f"[Phrase {i}/{total_phrases}] Semaphore released")
                        
                        # Add the successful result to the list
                        phrase_results.append(phrase_result)
                        
                    except Exception as e:
                        logger.error(f"[Phrase {i}/{total_phrases}] Error in _process_phrase_with_semaphore: {str(e)}", exc_info=True)
                        # Create a failed result to continue processing other phrases
                        phrase_result = {
                            'phrase_id': phrase_id,
                            'text': phrase_text,
                            'success': False,
                            'error': str(e),
                            'start_time': time.time(),
                            'end_time': time.time(),
                            'duration': 0,
                            'audio_file': None
                        }
                        phrase_results.append(phrase_result)
                    
                    # Log phrase processing result
                    success = phrase_result.get('success', False)
                    duration = phrase_result.get('duration', 0)
                    error = phrase_result.get('error')
                    
                    if success:
                        logger.info(f"[Phrase {i}/{total_phrases}] Completed successfully in {duration:.2f}s")
                        if 'audio_file' in phrase_result:
                            logger.debug(f"[Phrase {i}/{total_phrases}] Audio file: {phrase_result['audio_file']}")
                    else:
                        logger.error(f"[Phrase {i}/{total_phrases}] Failed: {error}")
                    
                    # Log memory usage periodically
                    if i % 10 == 0 or i == total_phrases:
                        current, peak = tracemalloc.get_traced_memory()
                        logger.debug(f"[Memory] Current: {current / 10**6:.2f}MB, Peak: {peak / 10**6:.2f}MB - After {i}/{total_phrases} phrases")
                    
                    # Update progress
                    if progress_callback:
                        current = len(phrase_results)
                        total = len(section.phrases)
                        status = f'Processed {current} of {total} phrases'
                        logger.debug(f"Updating progress: {status}")
                        try:
                            await progress_callback(current, total, status, phrase_id=str(getattr(phrase, 'id', 'NO_ID')))
                        except Exception as e:
                            logger.error(f"Error in progress callback: {e}", exc_info=True)
            
            result['phrases'] = phrase_results
            logger.debug(f"Processed {len(phrase_results)} phrases")
            
            # Check for any phrase errors
            phrase_errors = [r for r in phrase_results if not r.get('success', True)]
            audio_files = [r['audio_file'] for r in phrase_results if r.get('audio_file')]
            
            if phrase_errors:
                error_msg = f"{len(phrase_errors)} phrases failed to process"
                logger.warning(error_msg)
                result['success'] = False
                result['error'] = error_msg
                # Include the first error message for debugging
                if phrase_errors and phrase_errors[0].get('error'):
                    result['error'] += f": {phrase_errors[0]['error']}"
                    logger.error(f"First error: {phrase_errors[0]['error']}")
            
            # Always try to set an audio file if available
            if audio_files:
                logger.debug(f"Found {len(audio_files)} audio files")
                if not phrase_errors:
                    # If no errors, try to concatenate all audio files
                    output_format = self.config.get('output_format', 'wav')
                    combined_audio = section_dir / f'section_combined.{output_format}'
                    logger.debug(f"Attempting to concatenate audio files to {combined_audio}")
                    try:
                        await self.audio_processor.concatenate_audio(
                            input_files=audio_files,
                            output_file=combined_audio
                        )
                        result['audio_file'] = str(combined_audio)
                        logger.debug(f"Successfully created combined audio file: {result['audio_file']}")
                    except Exception as e:
                        error_msg = f"Error concatenating audio files: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        # If concatenation fails, fail the section but still return the individual phrase results
                        result['success'] = False
                        result['error'] = error_msg
                        # Don't set audio_file since concatenation failed
                        logger.error(f"Section audio concatenation failed: {error_msg}")
                else:
                    # If there were phrase errors, just use the first available audio file
                    # but mark the section as failed
                    result['audio_file'] = audio_files[0]
                    result['success'] = False
                    logger.warning(f"Using first audio file due to phrase errors: {result['audio_file']}")
            else:
                error_msg = "No audio files were generated for this section"
                logger.warning(error_msg)
                result['success'] = False
                result['error'] = error_msg
            
            logger.debug(f"Section processing complete. Success: {result['success']}")
            return result
            
        except Exception as e:
            error_msg = f"Error processing section {getattr(section, 'id', 'NO_ID')}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result['success'] = False
            result['error'] = str(e)
            return result
            
        finally:
            # Record performance metrics
            try:
                self.performance.record_phase_end(f'section_{section.id}')
                result['performance'] = self.performance.get_phase_metrics(f'section_{section.id}')
                logger.debug(f"Recorded performance metrics for section {getattr(section, 'id', 'NO_ID')}")
            except Exception as e:
                logger.error(f"Error recording performance metrics: {e}", exc_info=True)
            
            result['end_time'] = time.time()
            result['duration'] = result['end_time'] - result['start_time']
            logger.debug(f"[END] process_section - Completed in {result['duration']:.2f}s")
            logger.debug("=" * 80)
            
            # If there were any errors, log them
            if not result.get('success', True):
                logger.error(f"Section processing failed: {result.get('error')}")
            
            return result
    
    async def _process_phrase_with_semaphore(
        self,
        phrase: Phrase,
        phrase_dir: Path,
        semaphore: asyncio.Semaphore,
        **options: Any
    ) -> Dict[str, Any]:
        """Process a phrase with a semaphore to limit concurrency.
        
        Args:
            phrase: The phrase to process
            phrase_dir: Directory to save phrase audio
            semaphore: Semaphore to limit concurrency
            **options: Additional processing options
            
        Returns:
            Dictionary with processing results
        """
        async with semaphore:
            return await self._process_phrase(phrase, phrase_dir, **options)
    
    async def process_phrase(
        self,
        phrase: Phrase,
        output_dir: Path,
        **options: Any
    ) -> Dict[str, Any]:
        """Process a single phrase and generate audio.
        
        This is the public interface for processing a single phrase.
        
        Args:
            phrase: The phrase to process
            output_dir: Directory to save the output audio file
            **options: Additional processing options
                - process_audio: Whether to apply audio processing (default: True)
                - tts_options: Additional options for TTS service
                - audio_processing_options: Additional options for audio processing
                
        Returns:
            Dictionary containing processing results with the following keys:
                - phrase_id: ID of the processed phrase
                - text: The phrase text
                - audio_file: Path to the generated audio file
                - success: Whether processing was successful
                - error: Error message if processing failed
                - start_time: Processing start timestamp
                - end_time: Processing end timestamp
                - duration: Processing duration in seconds
                - performance: Performance metrics for the operation
        """
        return await self._process_phrase(phrase, output_dir, **options)
    
    async def _process_phrase(
        self,
        phrase: Phrase,
        output_dir: Path,
        **options: Any
    ) -> Dict[str, Any]:
        """Process a single phrase (internal implementation).
        
        Args:
            phrase: The phrase to process
            output_dir: Base directory for output (will create phrases subdirectory)
            **options: Additional processing options
            
        Returns:
            Dictionary with processing results
        """
        logger.debug("=" * 80)
        logger.debug(f"[START] _process_phrase")
        logger.debug(f"- Phrase text: {getattr(phrase, 'text', 'NO_TEXT')}")
        logger.debug(f"- Phrase type: {type(phrase)}")
        logger.debug(f"- Output dir: {output_dir}")
        logger.debug(f"- Options: {options}")
        
        # Log phrase attributes for debugging
        try:
            logger.debug(f"- Phrase attributes: {dir(phrase)}")
            logger.debug(f"- Phrase ID: {getattr(phrase, 'id', 'NO_ID')}")
            logger.debug(f"- Phrase language: {getattr(phrase, 'language', 'NO_LANGUAGE')}")
            logger.debug(f"- Phrase voice_id: {getattr(phrase, 'voice_id', 'NO_VOICE_ID')}")
        except Exception as e:
            logger.error(f"Error inspecting phrase: {e}", exc_info=True)
        
        # Generate a unique phrase ID
        try:
            phrase_id = str(phrase.id) if hasattr(phrase, 'id') and phrase.id else str(uuid.uuid4())
            logger.debug(f"Using phrase_id: {phrase_id}")
        except Exception as e:
            logger.error(f"Error generating phrase ID: {e}", exc_info=True)
            phrase_id = str(uuid.uuid4())
            logger.debug(f"Generated fallback phrase_id: {phrase_id}")
        
        # Record performance metrics
        try:
            self.performance.record_phase_start(f'phrase_{phrase_id}')
        except Exception as e:
            logger.error(f"Error recording phase start: {e}", exc_info=True)
        
        # Create output directory if it doesn't exist
        try:
            logger.debug(f"Ensuring output directory exists: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Directory creation complete")
        except Exception as e:
            logger.error(f"Error creating output directory: {e}", exc_info=True)
            raise
        
        # Initialize result dictionary
        try:
            # Get the language from the phrase, defaulting to None if not set
            language = getattr(phrase, 'language', None)
            
            result = {
                'phrase_id': phrase_id,
                'text': getattr(phrase, 'text', 'NO_TEXT'),
                'language': language.value if language is not None else None,  # Use enum value
                'voice_id': str(getattr(phrase, 'voice_id', '')),
                'start_time': time.time(),
                'end_time': None,
                'success': True,
                'error': None,
                'audio_file': None,
                'output_dir': str(output_dir.relative_to(output_dir.parent)) if output_dir.parent else str(output_dir),
                'performance': {}
            }
            logger.debug(f"Initialized result dictionary: {result}")
        except Exception as e:
            logger.error(f"Error initializing result dictionary: {e}", exc_info=True)
            raise
            
        logger.debug("=" * 80)
        
        try:
            # Get output format from config or default to 'mp3' to match test expectations
            output_format = self.config.get('output_format', 'mp3')
            
            # Generate audio for the phrase
            audio_file = output_dir / f'phrase_{phrase_id}.{output_format}'
            
            # Get voice settings from phrase or use defaults
            voice_settings = getattr(phrase, 'voice_settings', {}) or {}
            # Get voice_id directly from the phrase object, not from voice_settings
            voice_id = getattr(phrase, 'voice_id', None)
            
            # Enhanced logging
            logger.debug(f"Processing phrase: {phrase.text}")
            logger.debug(f"Phrase ID: {phrase_id}")
            logger.debug(f"Voice settings: {voice_settings}")
            logger.debug(f"Voice ID from settings: {voice_id}")
            logger.debug(f"Phrase attributes: {dir(phrase)}")
            logger.debug(f"Phrase has voice_id attr: {hasattr(phrase, 'voice_id')}")
            logger.debug(f"Phrase voice_id: {getattr(phrase, 'voice_id', 'NOT SET')}")
            
            # Get TTS options and add language info
            tts_options = options.get('tts_options', {})
            if not voice_id and hasattr(phrase, 'language'):
                tts_options['language'] = phrase.language
                logger.debug(f"Added language to TTS options: {phrase.language}")
            
            logger.debug(f"TTS options: {tts_options}")
            
            try:
                logger.debug(f"Calling TTS service with parameters:")
                logger.debug(f"  text: {phrase.text}")
                logger.debug(f"  voice_id: {voice_id}")
                logger.debug(f"  output_path: {audio_file}")
                logger.debug(f"  rate: {voice_settings.get('rate', 1.0)}")
                logger.debug(f"  pitch: {voice_settings.get('pitch', 0.0)}")
                logger.debug(f"  volume: {voice_settings.get('volume', 1.0)}")
                logger.debug(f"  Additional TTS options: {tts_options}")
                
                tts_result = await self.tts_service.synthesize_speech(
                    text=phrase.text,
                    voice_id=voice_id,
                    output_path=audio_file,
                    rate=voice_settings.get('rate', 1.0),
                    pitch=voice_settings.get('pitch', 0.0),
                    volume=voice_settings.get('volume', 1.0),
                    **tts_options
                )
                logger.debug(f"TTS service returned: {tts_result}")
                
                # Always set the audio file path in the result
                result['audio_file'] = str(audio_file)
                logger.debug(f"Set result['audio_file'] to: {result['audio_file']}")
                
                # Apply audio processing if needed
                if options.get('process_audio', True):
                    # Use the same output format for processed audio files
                    processed_audio = output_dir / f'phrase_{phrase_id}_processed.{output_format}'
                    logger.debug(f"Processing audio with output file: {processed_audio}")
                    try:
                        # First normalize the audio
                        normalized_audio = output_dir / f'phrase_{phrase_id}_normalized.{output_format}'
                        await self.audio_processor.normalize_audio(
                            input_file=audio_file,
                            output_file=normalized_audio,
                            target_level=-16.0  # Standard LUFS level for speech
                        )
                        
                        # Then trim any leading/trailing silence
                        await self.audio_processor.trim_silence(
                            input_file=normalized_audio,
                            output_file=processed_audio,
                            threshold=-40.0  # dB threshold for silence
                        )
                        
                        # Clean up the intermediate normalized file
                        if normalized_audio.exists():
                            normalized_audio.unlink()
                            
                        # Update the audio file path to point to the processed file
                        result['audio_file'] = str(processed_audio)
                        logger.debug(f"Updated result['audio_file'] to processed file: {result['audio_file']}")
                        
                    except Exception as e:
                        error_msg = f"Audio processing failed: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        # Mark the phrase as failed and include the error
                        result['success'] = False
                        result['error'] = error_msg
                        # Keep the original audio file path if processing fails
                        logger.warning("Using unprocessed audio file due to processing error")
                        # Return the result with the error instead of re-raising
                        result['end_time'] = time.time()
                        result['duration'] = result['end_time'] - result['start_time']
                        self.performance.record_phase_end(f'phrase_{phrase_id}')
                        result['performance'] = self.performance.get_phase_metrics(f'phrase_{phrase_id}')
                        return result
            except Exception as e:
                logger.error(f"Error in TTS or audio processing: {str(e)}", exc_info=True)
                raise
            result['end_time'] = time.time()
            result['duration'] = result['end_time'] - result['start_time']
            
            # Record performance metrics
            self.performance.record_phase_end(f'phrase_{phrase_id}')
            result['performance'] = self.performance.get_phase_metrics(f'phrase_{phrase_id}')
            
            return result
            
        except Exception as e:
            logger.exception(f"Error processing phrase: {phrase.text}")
            result['success'] = False
            result['error'] = str(e)
            result['end_time'] = time.time()
            result['duration'] = result['end_time'] - result['start_time']
            
            # Record performance metrics even in case of error
            self.performance.record_phase_end(f'phrase_{phrase_id}')
            result['performance'] = self.performance.get_phase_metrics(f'phrase_{phrase_id}')
            
            return result
    
    def _save_metadata(self, metadata: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Save metadata to a JSON file.
        
        Args:
            metadata: Dictionary containing metadata to save
            output_path: Path where the metadata JSON file should be saved
        """
        import json
        from datetime import datetime, date
        from enum import Enum
        from uuid import UUID
        from pathlib import Path as PathType
        
        def json_serializer(obj):
            """Custom JSON serializer for handling various types."""
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, UUID):
                return str(obj)
            elif isinstance(obj, PathType):
                return str(obj)
            elif hasattr(obj, '__dict__'):
                # Convert objects with __dict__ to dict, excluding private attributes
                return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
            raise TypeError(f"Type {type(obj)} not serializable")
        
        try:
            # Create a copy of the metadata to use for saving
            metadata_to_save = {}
            
            # Extract lesson data first to use for top-level fields
            if 'lesson' in metadata and metadata['lesson'] is not None:
                lesson = metadata['lesson']
                
                # Extract lesson summary fields for top level
                metadata_to_save.update({
                    'title': getattr(lesson, 'title', 'Untitled Lesson'),
                    'description': getattr(lesson, 'description', ''),
                    'language': str(getattr(lesson, 'target_language', '')),
                    'lesson_id': str(getattr(lesson, 'id', '')),
                    'created_at': getattr(lesson, 'created_at', '').isoformat() if hasattr(lesson, 'created_at') else '',
                    'updated_at': getattr(lesson, 'updated_at', '').isoformat() if hasattr(lesson, 'updated_at') else ''
                })
                
                # Get the full lesson data as a dictionary
                lesson_data = json.loads(lesson.model_dump_json())
                
                # Add the full lesson data under a 'lesson' key
                metadata_to_save['lesson'] = lesson_data
            
            # Copy all other metadata fields
            for key, value in metadata.items():
                if key != 'lesson' and key not in metadata_to_save:
                    metadata_to_save[key] = value
            
            # Initialize processing_info if not present
            if 'processing_info' not in metadata_to_save:
                metadata_to_save['processing_info'] = {}
            
            # Add timestamp if not present
            if 'timestamp' not in metadata_to_save['processing_info']:
                metadata_to_save['processing_info']['timestamp'] = datetime.utcnow().isoformat() + 'Z'
            
            # Add duration to processing_info if start_time and end_time are available
            if 'start_time' in metadata and 'end_time' in metadata:
                duration = metadata['end_time'] - metadata['start_time']
                metadata_to_save['processing_info']['duration'] = duration
            
            # Add output_dir to processing_info
            metadata_to_save['processing_info']['output_dir'] = str(Path(output_path).parent)
            
            # Add audio_files section
            audio_files = {
                'final': metadata_to_save.get('final_audio_file', ''),
                'sections': {},
                'phrases': {}
            }
            
            # Add section audio files
            for section in metadata_to_save.get('sections', []):
                section_id = section.get('section_id')
                if section_id and 'audio_file' in section:
                    audio_files['sections'][section_id] = section['audio_file']
                    
                    # Add phrase audio files
                    for phrase in section.get('phrases', []):
                        phrase_id = phrase.get('phrase_id')
                        if phrase_id and 'audio_file' in phrase:
                            if section_id not in audio_files['phrases']:
                                audio_files['phrases'][section_id] = {}
                            audio_files['phrases'][section_id][phrase_id] = phrase['audio_file']
            
            metadata_to_save['audio_files'] = audio_files
            
            # Ensure directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file with custom serializer
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_to_save, f, indent=2, ensure_ascii=False, default=json_serializer)
                
        except Exception as e:
            logger.error("Failed to save metadata: %s", str(e))
            raise
                
        except Exception as e:
            logger.error("Failed to save metadata: %s", str(e))
