"""Service for processing lessons and generating audio."""
from typing import Union, Dict, Any, Optional, List, Tuple
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
import tempfile
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

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

def get_short_path(base_dir: Path, prefix: str = '', suffix: str = '', max_length: int = 64) -> Path:
    """
    Generate a short, unique path within the given directory with strict length limits.
    
    Args:
        base_dir: Base directory for the path
        prefix: Optional prefix for the filename
        suffix: Optional suffix (including extension)
        max_length: Maximum length for the filename (not including path)
        
    Returns:
        Path object with a guaranteed short filename
    """
    # Generate a very short unique ID (4 chars should be enough for our purposes)
    short_id = str(uuid.uuid4().hex)[:4]
    
    # Create a safe prefix (alphanumeric and underscores only)
    safe_prefix = ''.join(c if c.isalnum() else '_' for c in str(prefix))[:8]
    
    # Build the base filename
    if safe_prefix:
        base_name = f"{safe_prefix}_{short_id}"
    else:
        base_name = short_id
        
    # Add suffix if provided
    if suffix:
        # Clean and truncate suffix to reasonable length
        clean_suffix = ''.join(c if c.isalnum() or c in '._- ' else '_' for c in str(suffix))
        clean_suffix = clean_suffix[:16]  # Limit suffix length
        base_name = f"{base_name}_{clean_suffix}"
    
    # Ensure the filename isn't too long
    if len(base_name) > max_length:
        base_name = base_name[:max_length]
        
    # Clean up any double underscores
    base_name = base_name.replace('__', '_').strip('_')
    
    # Ensure we have at least some name
    if not base_name:
        base_name = f"f{short_id}"
        
    # Return the full path
    return base_dir / base_name

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
        
        def get_short_path(base_dir: Path, prefix: str = '', suffix: str = '', max_length: int = 64) -> Path:
            """
            Generate a short, unique path within the given directory with strict length limits.
            
            Args:
                base_dir: Base directory for the path
                prefix: Optional prefix for the filename
                suffix: Optional suffix (including extension)
                max_length: Maximum length for the filename (not including path)
                
            Returns:
                Path object with a guaranteed short filename
            """
            # Generate a very short unique ID (4 chars should be enough for our purposes)
            short_id = str(uuid.uuid4().hex)[:4]
            
            # Create a safe prefix (alphanumeric and underscores only)
            safe_prefix = ''.join(c if c.isalnum() else '_' for c in str(prefix))[:8]
            
            # Build the base filename
            if safe_prefix:
                base_name = f"{safe_prefix}_{short_id}"
            else:
                base_name = short_id
                
            # Add suffix if provided
            if suffix:
                # Clean and truncate suffix to reasonable length
                clean_suffix = ''.join(c if c.isalnum() or c in '._- ' else '_' for c in str(suffix))
                clean_suffix = clean_suffix[:16]  # Limit suffix length
                base_name = f"{base_name}_{clean_suffix}"
            
            # Ensure the filename isn't too long
            if len(base_name) > max_length:
                base_name = base_name[:max_length]
                
            # Clean up any double underscores
            base_name = base_name.replace('__', '_').strip('_')
            
            # Ensure we have at least some name
            if not base_name:
                base_name = f"f{short_id}"
                
            # Return the full path
            return base_dir / base_name
        
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
        output_dir: Union[str, Path],
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
        # Ensure output_dir is a Path object
        output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
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
        
        # Create section directory with a very short name
        section_short_id = str(section.id).split('-')[0][:4]  # First 4 chars of first part of UUID
        section_dir = get_short_path(output_dir, prefix='s', suffix=section_short_id, max_length=12)
        section_dir.mkdir(parents=True, exist_ok=True)
        
        # Create phrases subdirectory with a very short name
        phrase_dir = section_dir / 'p'  # Single character directory name
        phrase_dir.mkdir(parents=True, exist_ok=True)
        
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
            # Initialize phrase results list in the result dictionary
            result['phrases'] = []
            if not hasattr(section, 'phrases') or not section.phrases:
                logger.warning(f"Section {getattr(section, 'id', 'NO_ID')} has no phrases to process")
            else:
                total_phrases = len(section.phrases)
                logger.info(f"Processing {total_phrases} phrases in section {getattr(section, 'id', 'NO_ID')}")
                
                # Initialize semaphore for concurrency control
                max_concurrent = options.get('max_concurrent_phrases', 5)
                semaphore = asyncio.Semaphore(max_concurrent)
                
                # Process phrases one by one with concurrency control
                processing_tasks = []
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
                    
                    # Create task for processing this phrase with semaphore
                    task = asyncio.create_task(
                        self._process_phrase_with_semaphore(
                            phrase=phrase,
                            phrase_dir=phrase_dir,
                            semaphore=semaphore,
                            **options
                        )
                    )
                    processing_tasks.append((i, phrase, task))
                
                # Process results as they complete
                for i, phrase, task in processing_tasks:
                    try:
                        # Wait for the task to complete
                        phrase_result = await task
                        
                        # Add the processed phrase to the result
                        result['phrases'].append(phrase_result)
                        
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
                            # Update section success status if any phrase fails
                            result['success'] = False
                        
                        # Update progress after each phrase
                        if progress_callback:
                            current = len(result['phrases'])
                            total = len(section.phrases)
                            status = f'Processed {current} of {total} phrases'
                            logger.debug(f"Updating progress: {status}")
                            try:
                                await progress_callback(
                                    current, 
                                    total, 
                                    status, 
                                    phrase_id=str(getattr(phrase, 'id', 'NO_ID'))
                                )
                            except Exception as e:
                                logger.error(f"Error in progress callback: {e}", exc_info=True)
                        
                        # Log memory usage periodically
                        if i % 10 == 0 or i == total_phrases:
                            current_mem, peak_mem = tracemalloc.get_traced_memory()
                            logger.debug(f"[Memory] Current: {current_mem / 10**6:.2f}MB, Peak: {peak_mem / 10**6:.2f}MB - After {i}/{total_phrases} phrases")
                            
                    except Exception as e:
                        logger.error(f"Error processing phrase {i}: {str(e)}", exc_info=True)
                        error_result = {
                            'phrase_id': getattr(phrase, 'id', f'phrase_{i}'),
                            'text': getattr(phrase, 'text', 'NO_TEXT'),
                            'success': False,
                            'error': f'Processing failed: {str(e)}',
                            'start_time': time.time(),
                            'end_time': time.time(),
                            'duration': 0,
                            'audio_file': None
                        }
                        result['phrases'].append(error_result)
                        result['success'] = False
            
            # Get all audio files from successfully processed phrases
            audio_files = [r['audio_file'] for r in result['phrases'] if r.get('success', False) and r.get('audio_file')]
            
            # If we have any failed phrases, update the error message
            phrase_errors = [r for r in result['phrases'] if not r.get('success', True)]
            if phrase_errors:
                error_msg = f"{len(phrase_errors)} phrases failed to process"
                logger.warning(error_msg)
                result['success'] = False
                
                # Include the first error message for debugging if not already set
                if phrase_errors and phrase_errors[0].get('error'):
                    if not result.get('error'):
                        result['error'] = error_msg
                    result['error'] += f": {phrase_errors[0]['error']}"
                    logger.error(f"First error: {phrase_errors[0]['error']}")
            
            # Always try to create a combined audio file if we have any audio files
            if audio_files:
                logger.debug(f"Found {len(audio_files)} audio files for section {getattr(section, 'id', 'NO_ID')}")
                output_format = self.config.get('output_format', 'mp3')
                
                # Create a very short combined audio filename
                combined_audio = get_short_path(
                    section_dir,
                    prefix='c',  # Single character prefix
                    suffix=f'.{output_format}',
                    max_length=16  # Very short filename
                )
                
                logger.debug(f"Attempting to concatenate {len(audio_files)} audio files to {combined_audio}")
                
                # Verify all audio files exist before concatenation and normalize paths
                valid_audio_files = []
                missing_files = []
                for audio_file in audio_files:
                    try:
                        # Ensure path is absolute and resolve any symlinks
                        abs_path = Path(audio_file).resolve()
                        if abs_path.exists():
                            valid_audio_files.append(str(abs_path))
                        else:
                            missing_files.append(audio_file)
                    except (OSError, RuntimeError) as e:
                        logger.warning(f"Error resolving audio file path {audio_file}: {e}")
                        missing_files.append(audio_file)
                
                if missing_files:
                    logger.warning(f"Missing {len(missing_files)} audio files, skipping concatenation")
                    logger.debug(f"Missing files: {missing_files}")
                
                if valid_audio_files:
                    try:
                        # Ensure the combined_audio directory exists
                        combined_audio.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Log the first few audio files being concatenated
                        logger.debug(f"Concatenating audio files (first 3): {audio_files[:3]}")
                        
                        # Use only the valid audio files for concatenation
                        await self.audio_processor.concatenate_audio(
                            input_files=valid_audio_files,
                            output_file=combined_audio
                        )
                        
                        # Verify the combined file was created
                        if combined_audio.exists() and combined_audio.stat().st_size > 0:
                            result['audio_file'] = str(combined_audio.absolute())
                            logger.info(f"Successfully created combined audio file: {result['audio_file']} "
                                      f"({combined_audio.stat().st_size / 1024:.1f} KB)")
                            
                            # If there were phrase errors, still mark as failed but include the combined audio
                            if phrase_errors:
                                result['success'] = False
                                logger.warning("Section completed with errors but combined audio was created")
                            else:
                                result['success'] = True
                                result['error'] = None  # Clear any previous errors if we succeeded
                        else:
                            error_msg = f"Combined audio file was not created or is empty: {combined_audio}"
                            logger.error(error_msg)
                            raise FileNotFoundError(error_msg)
                            
                    except Exception as e:
                        error_msg = f"Error concatenating audio files: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        
                        # If concatenation fails, still include the first audio file if available
                        if audio_files and Path(audio_files[0]).exists():
                            result['audio_file'] = audio_files[0]
                            logger.warning(f"Using first audio file due to concatenation error: {result['audio_file']}")
                        
                        # Only set the error if we don't have a more specific one
                        if not result.get('error'):
                            result['error'] = error_msg
                        result['success'] = False
            else:
                # Get all error messages from failed phrases
                error_msgs = [r.get('error') for r in result['phrases'] if r.get('error')]
                if error_msgs:
                    error_msg = f"No audio files were generated for this section. Errors: {'; '.join(error_msgs[:3])}"
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
    
    async def _process_phrase_batch(
        self,
        phrases: List[Phrase],
        output_dir: Path,
        **options: Any
    ) -> List[Dict[str, Any]]:
        """Process a batch of phrases with the same voice settings.
        
        This method processes multiple phrases in a single TTS request to improve performance.
        It combines all phrases with appropriate pauses, generates a single audio file,
        then splits it back into individual phrase audio segments.
        
        Args:
            phrases: List of phrases to process as a batch
            output_dir: Directory to save audio files
            **options: Additional processing options
                - silence_between_phrases: Duration of silence between phrases in seconds (default: 1.0)
                - tts_options: Additional options for TTS service
                - audio_processing_options: Additional options for audio processing
                
        Returns:
            List of processing results for each phrase with the following keys:
                - phrase_id: ID of the phrase
                - text: The phrase text
                - audio_file: Path to the generated audio file (None if failed)
                - success: Boolean indicating if processing was successful
                - error: Error message if processing failed
                - start_time: Processing start timestamp
                - end_time: Processing end timestamp
                - duration: Processing duration in seconds
        """
        self.performance.record_phase_start('process_phrase_batch')
        start_time = time.time()
        
        # Initialize empty results list
        results = []
        empty_phrase_results = []
        
        # Handle empty phrase list
        if not phrases:
            logger.warning("No phrases provided for batch processing")
            return []
        
        # Get options with defaults
        silence_between_phrases = options.get('silence_between_phrases', 1.0)
        output_format = options.get('output_format', 'mp3')
        batch_id = str(uuid.uuid4())[:8]  # Short ID for temp files
        
        # Track metadata for each phrase
        phrase_metadatas = []
        valid_phrases = []
        valid_phrase_indices = []
        
        # First pass: validate phrases and collect metadata
        for i, phrase in enumerate(phrases):
            phrase_id = getattr(phrase, 'id', str(uuid.uuid4()))
            phrase_text = getattr(phrase, 'text', '').strip()
            
            # Skip empty phrases
            if not phrase_text:
                logger.warning(f"Skipping empty phrase with ID: {phrase_id}")
                empty_phrase_results.append({
                    'phrase_id': phrase_id,
                    'text': '',
                    'audio_file': None,
                    'success': False,
                    'error': 'Empty phrase text',
                    'start_time': time.time(),
                    'end_time': time.time(),
                    'duration': 0.0
                })
                continue
                
            # Create a unique filename for this phrase
            phrase_file = output_dir / f"phrase_{phrase_id}.{output_format}"
            
            # Store metadata for this phrase
            phrase_metadatas.append({
                'id': phrase_id,
                'text': phrase_text,
                'file': phrase_file,
                'start_time': time.time(),
                'duration': 0.0,
                'original_index': i  # Track original position for result ordering
            })
            valid_phrases.append(phrase)
            valid_phrase_indices.append(i)
        
        # Handle single phrase as a special case (no need for batching)
        if len(valid_phrases) == 1:
            try:
                phrase = valid_phrases[0]
                phrase_meta = phrase_metadatas[0]
                
                # Process the single phrase
                result = await self._process_phrase(
                    phrase=phrase,
                    output_dir=output_dir,
                    **options
                )
                
                # Update metadata
                end_time = time.time()
                phrase_meta['end_time'] = end_time
                phrase_meta['duration'] = end_time - phrase_meta['start_time']
                
                # Create the result dictionary
                single_result = {
                    'phrase_id': phrase_meta['id'],
                    'text': phrase_meta['text'],
                    'audio_file': result.get('audio_file'),
                    'success': result.get('success', False),
                    'error': result.get('error'),
                    'start_time': float(phrase_meta['start_time']),
                    'end_time': float(end_time),
                    'duration': float(phrase_meta['duration'])
                }
                
                # Merge with empty results if any
                all_results = empty_phrase_results.copy()
                all_results.insert(phrase_meta['original_index'], single_result)
                return all_results
                
            except Exception as e:
                logger.error(f"Error processing single phrase in batch: {e}", exc_info=True)
                error_result = {
                    'phrase_id': str(phrase_metadatas[0]['id']),
                    'text': str(phrase_metadatas[0]['text']) if phrase_metadatas[0]['text'] is not None else '',
                    'audio_file': None,
                    'success': False,
                    'error': str(e) if e else 'Unknown error',
                    'start_time': float(phrase_metadatas[0]['start_time']),
                    'end_time': float(time.time()),
                    'duration': float(time.time() - phrase_metadatas[0]['start_time'])
                }
                all_results = empty_phrase_results.copy()
                all_results.insert(phrase_metadatas[0]['original_index'], error_result)
                return all_results
        
        # For multiple phrases, process as a batch
        batch_audio_file = None
        temp_outputs = []
        temp_with_silence = []
        
        try:
            # Create a temporary directory for batch processing
            temp_dir = output_dir / 'temp'
            temp_dir.mkdir(parents=True, exist_ok=True)
            batch_audio_file = temp_dir / f'batch_{batch_id}.{output_format}'
            
            # Ensure the directory exists
            batch_audio_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Filter out empty or invalid phrases and collect text for TTS
            valid_phrases = []
            valid_metadatas = []
            batch_texts = []
            
            for phrase, meta in zip(phrases, phrase_metadatas):
                text = getattr(phrase, 'text', '').strip()
                if not text:
                    logger.warning(f"Skipping empty phrase with ID: {meta.get('id', 'unknown')}")
                    empty_phrase_results.append({
                        'phrase_id': meta.get('id', str(uuid.uuid4())),
                        'text': '',
                        'audio_file': None,
                        'success': False,
                        'error': 'Empty phrase text',
                        'start_time': meta.get('start_time', time.time()),
                        'end_time': time.time(),
                        'duration': time.time() - meta.get('start_time', time.time())
                    })
                else:
                    valid_phrases.append(phrase)

            # Process each phrase individually
            results = []
            for i, phrase in enumerate(valid_phrases):
                try:
                    # Process the phrase
                    result = await self._process_phrase(
                        phrase,
                        output_dir,
                        **options
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing phrase {i+1}/{len(valid_phrases)}: {e}", exc_info=True)
                    results.append({
                        'phrase_id': getattr(phrase, 'id', str(uuid.uuid4())),
                        'text': getattr(phrase, 'text', ''),
                        'audio_file': None,
                        'success': False,
                        'error': str(e),
                        'start_time': time.time(),
                        'end_time': time.time(),
                        'duration': 0.0
                    })
            
            return results
            # [Previous code continues...]
            
        except Exception as e:
            error_msg = f"Error in batch processing: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Log detailed error information
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'batch_size': len(phrases) if 'phrases' in locals() else 0,
                'batch_audio_file': batch_audio_file if 'batch_audio_file' in locals() else None,
                'batch_audio_exists': batch_audio_file.exists() if 'batch_audio_file' in locals() else False,
                'batch_audio_size': batch_audio_file.stat().st_size if 'batch_audio_file' in locals() and batch_audio_file.exists() else 0,
                'temp_outputs_count': len(temp_outputs) if 'temp_outputs' in locals() else 0,
                'temp_outputs_valid': sum(1 for f in temp_outputs if f is not None and f.exists()) if 'temp_outputs' in locals() else 0
            }
            logger.error(f"Batch processing error details: {error_details}")
            
            # Return error results for all phrases
            error_results = []
            for meta in phrase_metadatas:
                error_results.append({
                    'phrase_id': meta.get('id', str(uuid.uuid4())),
                    'text': meta.get('text', ''),
                    'audio_file': None,
                    'success': False,
                    'error': error_msg,
                    'error_details': error_details,
                    'start_time': meta.get('start_time', time.time()),
                    'end_time': time.time(),
                    'duration': time.time() - meta.get('start_time', time.time())
                })
            
            # Log the full error with context
            logger.error(
                "Batch processing failed for phrases: %s",
                [{"id": m.get('id'), "text": m.get('text')[:50] + '...' if m.get('text') else None} 
                 for m in phrase_metadatas]
            )
            
            return error_results + empty_phrase_results
        finally:
            # Clean up temporary files
            for temp_file in temp_outputs + temp_with_silence + ([batch_audio_file] if batch_audio_file and batch_audio_file.exists() else []):
                try:
                    if temp_file and temp_file.exists():
                        temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")
            
        # Combine results with empty phrase results
        all_results = empty_phrase_results.copy()
        for i, result in enumerate(results):
            if i < len(phrase_metadatas):
                all_results.insert(phrase_metadatas[i]['original_index'], result)
            else:
                all_results.append(result)
                
        # Record performance metrics
        self.performance.record_phase_end('process_phrase_batch')
        
        return all_results
        
        try:
            # Create a temporary file for the combined audio with a unique name
            temp_dir = output_dir / 'temp'
            temp_dir.mkdir(parents=True, exist_ok=True)
            batch_audio_file = temp_dir / f'batch_{batch_id}.{output_format}'
            
            # Ensure the directory exists
            batch_audio_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Calculate split ratios based on phrase durations
            phrase_durations = [meta.get('duration', 1.0) for meta in phrase_metadatas]
            total_duration = sum(phrase_durations)
            if total_duration <= 0:
                total_duration = 1.0  # Avoid division by zero
                
            # Calculate cumulative ratios for splitting
            split_ratios = []
            cumulative = 0.0
            for duration in phrase_durations[:-1]:  # All but the last one
                cumulative += duration / total_duration
                split_ratios.append(cumulative)
            
            # Calculate split points in milliseconds
            split_points = [int(r * total_duration * 1000) for r in split_ratios]
            logger.debug(f"Split points (ms): {split_points} for ratios: {split_ratios}")
            
        except Exception as e:
            error_msg = f"Error calculating split points: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Return results for all phrases with error
            error_results = []
            for meta in phrase_metadatas:
                error_results.append({
                    'phrase_id': meta.get('id', str(uuid.uuid4())),
                    'text': meta.get('text', ''),
                    'audio_file': None,
                    'success': False,
                    'error': error_msg,
                    'start_time': meta.get('start_time', time.time()),
                    'end_time': time.time(),
                    'duration': time.time() - meta.get('start_time', time.time())
                })
            return error_results + empty_phrase_results

        # Filter out empty or invalid phrases
        valid_phrases = []
        valid_metadatas = []
        for phrase, meta in zip(phrases, phrase_metadatas):
            text = getattr(phrase, 'text', '')
            if not text or not text.strip():
                logger.warning(f"Skipping empty phrase with ID: {meta.get('id', 'unknown')}")
                empty_phrase_results.append({
                    'phrase_id': meta.get('id', str(uuid.uuid4())),
                    'text': '',
                    'audio_file': None,
                    'success': False,
                    'error': 'Empty phrase text',
                    'start_time': meta.get('start_time', time.time()),
                    'end_time': time.time(),
                    'duration': time.time() - meta.get('start_time', time.time())
                })
            else:
                valid_phrases.append(phrase)
                valid_metadatas.append(meta)
        
        # If no valid phrases, return empty results
        if not valid_phrases:
            logger.warning("No valid phrases to process in batch")
            return empty_phrase_results

        # Create temporary files for each phrase
        try:
            logger.info(f"Processing batch of {len(valid_phrases)} valid phrases with {silence_between_phrases}s silence between phrases")

            # Create temporary directory for this batch
            temp_dir = output_dir / 'temp'
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Create temporary files for each valid phrase
            temp_outputs = []
            for i in range(len(valid_phrases)):
                temp_file = temp_dir / f'phrase_{i}_{batch_id}.{output_format}'
                try:
                    temp_file.touch()  # Create empty file
                    temp_outputs.append(temp_file)
                    logger.debug(f"Created temp file: {temp_file}")
                except Exception as e:
                    error_msg = f"Failed to create temp file {temp_file}: {str(e)}"
                    logger.error(error_msg)
                    # Clean up any created temp files
                    for f in temp_outputs:
                        try:
                            if f.exists():
                                f.unlink()
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to clean up temp file {f}: {cleanup_error}")
                    raise AudioProcessingError(error_msg) from e
                
                # Split the audio at calculated points
                try:
                    logger.debug(f"Splitting audio into {len(temp_outputs)} segments")
                    split_start = time.time()
                    
                    await self.audio_processor.split_audio(
                        input_file=batch_audio_file,
                        output_files=temp_outputs,
                        split_points=split_points,
                        **options.get('audio_processing_options', {})
                    )
                    
                    split_duration = time.time() - split_start
                    logger.debug(f"Audio splitting completed in {split_duration:.2f}s")
                    
                    # Verify all output files were created and are valid
                    for i, temp_file in enumerate(temp_outputs):
                        if not temp_file.exists() or temp_file.stat().st_size == 0:
                            raise AudioProcessingError(f"Split audio file not created or is empty: {temp_file}")
                            
                except Exception as e:
                    error_msg = f"Error splitting audio: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    # Clean up any created temp files
                    for f in temp_outputs:
                        try:
                            if f.exists():
                                f.unlink()
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to clean up temp file {f}: {cleanup_error}")
                    raise AudioProcessingError(error_msg) from e
                
                # Process each phrase with proper silence insertion
                results = []
                temp_with_silence = []
                
                # First pass: add silence to each phrase as needed
                for i, (temp_file, phrase_file) in enumerate(zip(temp_outputs, [m['file'] for m in phrase_metadatas])):
                    try:
                        logger.debug(f"Processing phrase {i}: {temp_file} -> {phrase_file}")
                        
                        # Create a temporary file for the phrase with silence
                        with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as tmp:
                            temp_with_silence_file = Path(tmp.name)
                        
                        # Add silence at the beginning (except for first phrase)
                        if i > 0 and silence_between_phrases > 0:
                            logger.debug(f"  - Adding {silence_between_phrases}s silence to start")
                            try:
                                await self.audio_processor.add_silence(
                                    input_file=temp_file,
                                    output_file=temp_with_silence_file,
                                    duration=silence_between_phrases,
                                    position='start',
                                    **options.get('audio_processing_options', {})
                                )
                                # Clean up the original temp file
                                if temp_file.exists():
                                    temp_file.unlink()
                                temp_file = temp_with_silence_file
                            except Exception as e:
                                logger.error(f"Error adding silence to start of phrase {i}: {e}")
                                # Continue with original file if silence addition fails
                        
                        # Add silence at the end (except for last phrase)
                        if i < len(valid_phrases) - 1 and silence_between_phrases > 0:
                            logger.debug(f"  - Adding {silence_between_phrases}s silence to end")
                            try:
                                final_output = phrase_file
                                await self.audio_processor.add_silence(
                                    input_file=temp_file,
                                    output_file=final_output,
                                    duration=silence_between_phrases,
                                    position='end',
                                    **options.get('audio_processing_options', {})
                                )
                                # Clean up the temp file if it's different from the final output
                                if temp_file != final_output and temp_file.exists():
                                    temp_file.unlink()
                            except Exception as e:
                                logger.error(f"Error adding silence to end of phrase {i}: {e}")
                                # Fall back to copying the file if silence addition fails
                                shutil.copy2(temp_file, phrase_file)
                        else:
                            # Just copy the file for the last phrase or if no silence needed
                            shutil.copy2(temp_file, phrase_file)
                        
                        # Verify the final output file
                        if not phrase_file.exists() or phrase_file.stat().st_size == 0:
                            raise AudioProcessingError(f"Failed to create valid output file: {phrase_file}")
                            
                        temp_with_silence.append(phrase_file)
                        logger.debug(f"  - Successfully processed phrase {i}: {phrase_file}")
                        
                    except Exception as e:
                        error_msg = f"Error processing phrase {i}: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        
                        # Create a failed result for this phrase
                        phrase_result = {
                            'phrase_id': phrase_metadatas[i].get('id', str(uuid.uuid4())),
                            'text': phrase_metadatas[i].get('text', ''),
                            'audio_file': None,
                            'success': False,
                            'error': error_msg,
                            'start_time': phrase_metadatas[i].get('start_time', time.time()),
                            'end_time': time.time(),
                            'duration': time.time() - phrase_metadatas[i].get('start_time', time.time())
                        }
                        results.append(phrase_result)
                        continue
                        
                        # Ensure the file exists and is accessible
                        if not phrase_file.exists():
                            error_msg = f"Expected audio file not found: {phrase_file}"
                            logger.warning(error_msg)
                            raise AudioProcessingError(error_msg)
                        
                        # Get the phrase text, preferring original_text if available
                        original_text = phrase_metadatas[i].get('original_text', 
                                                             phrase_metadatas[i].get('text', ''))
                        
                        # Verify the audio file is valid
                        file_size = phrase_file.stat().st_size
                        logger.debug(f"Generated phrase audio: {phrase_file.name} ({file_size} bytes)")
                        
                        if file_size == 0:
                            error_msg = f"Generated empty audio file: {phrase_file}"
                            logger.error(error_msg)
                            raise AudioProcessingError(error_msg)
                        
                        logger.debug(f"\n=== PROCESSING PHRASE {i} ===")
                        logger.debug(f"Phrase text from metadata: '{phrase_metadatas[i].get('text', 'NO_TEXT')}'")
                        logger.debug(f"Original text from metadata: '{phrase_metadatas[i].get('original_text', 'NO_ORIGINAL_TEXT')}'")
                        logger.debug(f"Using text: '{original_text}'")
                        
                        # Get the phrase ID from the phrase object if available, otherwise from metadata
                        phrase_id = phrase_metadatas[i].get('id', str(uuid.uuid4()))
                        
                        # Create the phrase result with timing information
                        phrase_result = {
                            'phrase_id': str(phrase_id),
                            'text': str(original_text),
                            'audio_file': str(phrase_file.absolute()),
                            'success': True,
                            'error': None,
                            'start_time': phrase_metadatas[i].get('start_time', time.time()),
                            'end_time': time.time(),
                            'duration': time.time() - phrase_metadatas[i].get('start_time', time.time())
                        }
                        results.append(phrase_result)
                        
                # Clean up temporary files
                for temp_file in temp_outputs + temp_with_silence:
                    try:
                        if temp_file and temp_file.exists() and temp_file.is_file():
                            temp_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")
                
                # Clean up the batch audio file
                try:
                    if batch_audio_file and batch_audio_file.exists():
                        batch_audio_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up batch audio file {batch_audio_file}: {e}")
                
                # Return combined results (successful phrases + empty phrase results)
                return results + empty_phrase_results
                
        except Exception as e:
            logger.error(f"Error in batch processing: {e}", exc_info=True)
            
            # Clean up any temporary files
            for temp_file in temp_outputs + temp_with_silence:
                try:
                    if temp_file and temp_file.exists() and temp_file.is_file():
                        temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")
            
            # Clean up the batch audio file if it exists
            try:
                if batch_audio_file and batch_audio_file.exists():
                    batch_audio_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up batch audio file {batch_audio_file}: {e}")
            
            # Create error results for all phrases in this batch
            error_results = []
            for meta in phrase_metadatas:
                error_results.append({
                    'phrase_id': meta.get('id', str(uuid.uuid4())),
                    'text': meta.get('text', ''),
                    'audio_file': None,
                    'success': False,
                    'error': f"Batch processing failed: {str(e)}",
                    'start_time': meta.get('start_time', time.time()),
                    'end_time': time.time(),
                    'duration': time.time() - meta.get('start_time', time.time())
                })
            
            return error_results + empty_phrase_results
            
            # Clean up the batch audio file with error handling
            if batch_audio_file and batch_audio_file.exists():
                try:
                    batch_audio_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up batch audio file {batch_audio_file}: {e}")
            
            return results
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e) or "No error message"
            logger.error(f"Error in batch processing (type: {error_type}): {error_msg}", exc_info=True)
            
            # Log the error details for debugging
            logger.debug(f"Batch processing failed with error: {error_msg}")
            logger.debug(f"Batch size: {len(phrases)} phrases")
            
            # Log the first few phrases for context
            sample_phrases = [getattr(p, 'text', 'NO_TEXT')[:50] + '...' for p in phrases[:3]]
            logger.debug(f"Sample phrases in batch: {sample_phrases}")
            
            # If batch processing fails, fall back to individual processing
            logger.warning("Falling back to individual phrase processing")
            results = []
            success_count = 0
            
            for phrase, metadata in zip(phrases, phrase_metadatas):
                try:
                    phrase_result = await self._process_phrase(phrase, output_dir, **options)
                    # Ensure audio_file is set in the result
                    if phrase_result.get('success', False) and 'audio_file' not in phrase_result:
                        phrase_id = str(metadata.get('id', str(uuid.uuid4())))
                        output_format = self.config.get('output_format', 'mp3')
                        expected_file = output_dir / f'phrase_{phrase_id}.{output_format}'
                        if expected_file.exists():
                            phrase_result['audio_file'] = str(expected_file.absolute())
                            logger.debug(f"Added missing audio_file to result: {phrase_result['audio_file']}")
                    results.append(phrase_result)
                    if phrase_result.get('success', False):
                        success_count += 1
                except Exception as inner_e:
                    error_type = type(inner_e).__name__
                    error_msg = str(inner_e) or "No error message"
                    logger.error(f"Error processing phrase individually (type: {error_type}): {error_msg}", 
                                exc_info=True)
                    error_result = {
                        'phrase_id': str(metadata['id']),  # Ensure ID is string
                        'text': str(metadata.get('text', '')),  # Ensure text is string
                        'audio_file': None,
                        'success': False,
                        'error': str(error_msg) if error_msg else 'Unknown error',  # Ensure error is string
                        'error_type': str(error_type) if error_type else 'Exception',
                        'start_time': float(metadata.get('start_time', time.time())),
                        'end_time': float(time.time()),
                        'duration': float(time.time() - metadata.get('start_time', time.time()))
                    }
                    results.append(error_result)
                    logger.warning(f"Error processing phrase: {error_result}")
            
            # Log summary of fallback results
            total_phrases = len(phrases)
            logger.info(f"Completed fallback processing: {success_count}/{total_phrases} phrases processed successfully")
            
            if success_count < total_phrases:
                logger.warning(f"{total_phrases - success_count} phrases failed during fallback processing")
            
            return results

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
            # Get the language from the phrase, defaulting to English if not set
            language = getattr(phrase, 'language', None)
            # Ensure we have a valid language (use English as default)
            language_value = language.value if language is not None else 'en'
            
            result = {
                'phrase_id': phrase_id,
                'text': getattr(phrase, 'text', 'NO_TEXT'),
                'language': language_value,  # Always include language, default to 'en' if not set
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
            
            # Check for empty or whitespace-only phrases
            phrase_text = getattr(phrase, 'text', '')
            if not phrase_text or not phrase_text.strip():
                logger.warning(f"Skipping empty or whitespace-only phrase with ID: {phrase_id}")
                result.update({
                    'success': False,
                    'error': 'Empty or whitespace-only phrase',
                    'end_time': time.time(),
                    'duration': time.time() - result['start_time']
                })
                logger.debug(f"Returning early for empty phrase: {result}")
                return result
                
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
            # Log detailed timing information
            process_start = time.time()
            logger.debug(f"[TIMING] Starting phrase processing at {process_start:.6f}")
            
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
                logger.debug(f"[TIMING] {time.time():.6f} - Starting TTS service call")
                logger.debug(f"Calling TTS service with parameters:")
                logger.debug(f"  text: {phrase.text}")
                logger.debug(f"  voice_id: {voice_id}")
                logger.debug(f"  output_path: {audio_file}")
                logger.debug(f"  rate: {voice_settings.get('rate', 1.0)}")
                logger.debug(f"  pitch: {voice_settings.get('pitch', 0.0)}")
                logger.debug(f"  volume: {voice_settings.get('volume', 1.0)}")
                logger.debug(f"  Additional TTS options: {tts_options}")
                
                tts_start = time.time()
                # Try TTS generation with retries
                max_retries = 3
                retry_delay = 0.5  # Initial delay in seconds
                last_error = None
                
                for attempt in range(max_retries):
                    try:
                        # Generate temporary file path for this attempt
                        temp_audio_file = audio_file.parent / f"{audio_file.stem}_attempt{attempt}{audio_file.suffix}"
                        
                        logger.debug(f"[TTS] Attempt {attempt + 1}/{max_retries} - Generating speech")
                        tts_result = await self.tts_service.synthesize_speech(
                            text=phrase.text,
                            voice_id=voice_id,
                            output_path=temp_audio_file,
                            rate=voice_settings.get('rate', 1.0),
                            pitch=voice_settings.get('pitch', 0.0),
                            volume=voice_settings.get('volume', 1.0),
                            **tts_options
                        )
                        tts_end = time.time()
                        tts_duration = tts_end - tts_start
                        
                        logger.debug(f"[TIMING] {tts_end:.6f} - TTS service call completed in {tts_duration:.6f}s")
                        logger.debug(f"TTS service returned: {tts_result}")
                        
                        # Validate the generated file
                        if not temp_audio_file.exists():
                            raise AudioProcessingError(f"TTS service did not generate an output file at {temp_audio_file}")
                            
                        file_size = temp_audio_file.stat().st_size
                        logger.debug(f"[AUDIO] Generated audio file: {temp_audio_file} ({file_size} bytes)")
                        
                        if file_size == 0:
                            raise AudioProcessingError(f"Generated audio file is empty (0 bytes): {temp_audio_file}")
                            
                        # Basic file validation by checking header
                        with open(temp_audio_file, 'rb') as f:
                            header = f.read(4)
                            if not header:
                                raise AudioProcessingError(f"Empty file or read error: {temp_audio_file}")
                        
                        # If we get here, the file is valid
                        # Move the temp file to the final location
                        if audio_file.exists():
                            audio_file.unlink()
                        temp_audio_file.rename(audio_file)
                        logger.debug(f"[AUDIO] Successfully validated and moved audio file to {audio_file}")
                        break
                        
                    except Exception as e:
                        last_error = e
                        logger.warning(f"[TTS] Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                        
                        # Clean up any temporary files
                        if 'temp_audio_file' in locals() and temp_audio_file.exists():
                            try:
                                temp_audio_file.unlink()
                            except Exception as cleanup_error:
                                logger.warning(f"Failed to clean up temporary file {temp_audio_file}: {cleanup_error}")
                        
                        if attempt < max_retries - 1:
                            # Exponential backoff
                            retry_delay *= 2
                            logger.debug(f"[TTS] Retrying in {retry_delay:.1f} seconds...")
                            await asyncio.sleep(retry_delay)
                        else:
                            # Last attempt failed
                            raise AudioProcessingError(
                                f"Failed to generate valid audio after {max_retries} attempts. "
                                f"Last error: {str(last_error)}"
                            ) from last_error
                else:
                    # This should never be reached due to the raise in the loop
                    raise AudioProcessingError("Unexpected error in TTS generation retry loop")
                
                # Always set the audio file path in the result
                result['audio_file'] = str(audio_file.absolute())
                logger.debug(f"Set result['audio_file'] to: {result['audio_file']}")
                
                # Apply audio processing if needed
                if options.get('process_audio', True):
                    # Use the same output format for processed audio files
                    processed_audio = output_dir / f'phrase_{phrase_id}_processed.{output_format}'
                    logger.debug(f"[AUDIO] Starting audio processing with output file: {processed_audio}")
                    audio_process_start = time.time()
                    
                    # Validate input audio file before processing
                    if not audio_file.exists():
                        raise AudioProcessingError(f"Input audio file does not exist: {audio_file}")
                        
                    file_size = audio_file.stat().st_size
                    if file_size == 0:
                        raise AudioProcessingError(f"Input audio file is empty (0 bytes): {audio_file}")
                    
                    logger.debug(f"[AUDIO] Input file validation passed: {audio_file} ({file_size} bytes)")
                    
                    try:
                        # First normalize the audio
                        normalized_audio = output_dir / f'phrase_{phrase_id}_normalized.{output_format}'
                        logger.debug(f"[AUDIO] Normalizing audio to {normalized_audio}")
                        normalize_start = time.time()
                        
                        # Add retry logic for audio processing
                        max_retries = 2
                        for attempt in range(max_retries):
                            try:
                                await self.audio_processor.normalize_audio(
                                    input_file=audio_file,
                                    output_file=normalized_audio,
                                    target_level=-16.0  # Standard LUFS level for speech
                                )
                                # Verify the normalized file
                                if not normalized_audio.exists() or normalized_audio.stat().st_size == 0:
                                    raise AudioProcessingError("Normalization failed to create valid output file")
                                break
                            except Exception as e:
                                if attempt == max_retries - 1:
                                    raise
                                logger.warning(f"[AUDIO] Normalization attempt {attempt + 1} failed, retrying...")
                                await asyncio.sleep(0.5)  # Short delay before retry
                        
                        normalize_duration = time.time() - normalize_start
                        logger.debug(f"[TIMING] Audio normalization completed in {normalize_duration:.6f}s")
                        
                        # Then trim any leading/trailing silence
                        logger.debug(f"[AUDIO] Trimming silence from {normalized_audio}")
                        trim_start = time.time()
                        
                        # Add retry logic for silence trimming
                        max_retries = 2
                        for attempt in range(max_retries):
                            try:
                                # Ensure the input file exists and is valid
                                if not normalized_audio.exists() or normalized_audio.stat().st_size == 0:
                                    raise AudioProcessingError("Normalized audio file is missing or empty")
                                
                                # Process the audio
                                await self.audio_processor.trim_silence(
                                    input_file=normalized_audio,
                                    output_file=processed_audio,
                                    threshold=-40.0  # dB threshold for silence
                                )
                                
                                # Verify the output file
                                if not processed_audio.exists() or processed_audio.stat().st_size == 0:
                                    raise AudioProcessingError("Silence trimming failed to create valid output file")
                                
                                # If we get here, processing was successful
                                break
                                
                            except Exception as e:
                                if attempt == max_retries - 1:
                                    # Last attempt failed, log and re-raise
                                    logger.error(f"[AUDIO] Final attempt ({attempt + 1}/{max_retries}) failed: {str(e)}")
                                    raise
                                    
                                # Log warning and retry
                                logger.warning(f"[AUDIO] Silence trimming attempt {attempt + 1} failed, retrying...")
                                await asyncio.sleep(0.5)  # Short delay before retry
                        
                        trim_duration = time.time() - trim_start
                        logger.debug(f"[TIMING] Silence trimming completed in {trim_duration:.6f}s")
                        
                        # Clean up the intermediate normalized file
                        if normalized_audio.exists():
                            normalized_audio.unlink()
                            logger.debug(f"[AUDIO] Cleaned up intermediate file: {normalized_audio}")
                        
                        # Verify the processed file
                        if processed_audio.exists():
                            file_size = processed_audio.stat().st_size
                            logger.debug(f"[AUDIO] Processed audio file: {processed_audio} ({file_size} bytes)")
                            if file_size == 0:
                                logger.error("ERROR: Processed audio file is empty!")
                            
                            # Update the audio file path to point to the processed file
                            result['audio_file'] = str(processed_audio.absolute())
                            logger.debug(f"[AUDIO] Updated result['audio_file'] to: {result['audio_file']}")
                            
                            # Verify the file exists and is not empty
                            if not processed_audio.exists() or processed_audio.stat().st_size == 0:
                                error_msg = "Processed audio file is missing or empty"
                                logger.error(f"ERROR: {error_msg}")
                                raise AudioProcessingError(error_msg)
                        else:
                            error_msg = "Audio processing did not generate output file"
                            logger.error(f"ERROR: {error_msg}")
                            raise AudioProcessingError(error_msg)
                            
                        # Log total processing time
                        audio_process_duration = time.time() - audio_process_start
                        logger.debug(f"[TIMING] Total audio processing completed in {audio_process_duration:.6f}s")
                        
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
        
        def convert_uuids(obj):
            """Recursively convert UUIDs to strings in dictionaries and lists."""
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
                
                # Get the full lesson data as a dictionary and convert any UUIDs to strings
                lesson_data = json.loads(lesson.model_dump_json())
                lesson_data = convert_uuids(lesson_data)
                
                # Add the full lesson data under a 'lesson' key
                metadata_to_save['lesson'] = lesson_data
            
            # Copy all other metadata fields, ensuring UUIDs are converted
            for key, value in metadata.items():
                if key != 'lesson' and key not in metadata_to_save:
                    metadata_to_save[key] = convert_uuids(value)
            
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
                'final': str(metadata_to_save.get('final_audio_file', '')),
                'sections': {},
                'phrases': {}
            }
            
            # Add section audio files
            for section in metadata_to_save.get('sections', []):
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
