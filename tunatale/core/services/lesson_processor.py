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
                # Get output format from config or default to 'wav'
                output_format = self.config.get('output_format', 'wav')
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
            output_dir: Directory to save generated audio files
            progress_callback: Optional callback for progress updates
            **options: Additional processing options
            
        Returns:
            Dictionary with section processing results
        """
        self.performance.record_phase_start(f'section_{section.id}')
        section_dir = output_dir / f'section_{section.id}'
        section_dir.mkdir(exist_ok=True)
        
        result = {
            'section_id': str(section.id),
            'section_type': section.section_type.value,
            'phrases': [],
            'start_time': time.time(),
            'end_time': None,
            'success': True,
            'error': None,
            'audio_file': None,
            'performance': {}
        }
        
        try:
            # Process each phrase in the section
            phrase_results = []
            for phrase in section.phrases:
                phrase_result = await self._process_phrase_with_semaphore(
                    phrase=phrase,
                    phrase_dir=section_dir,
                    semaphore=asyncio.Semaphore(5),  # Limit concurrency
                    **options
                )
                phrase_results.append(phrase_result)
                
                # Update progress
                if progress_callback:
                    current = len(phrase_results)
                    total = len(section.phrases)
                    progress = current / total
                    status = f'Processed {current} of {total} phrases'
                    await progress_callback(current, total, status, phrase_id=str(phrase.id))
            
            result['phrases'] = phrase_results
            
            # Check for any phrase errors
            phrase_errors = [r for r in phrase_results if not r.get('success', True)]
            audio_files = [r['audio_file'] for r in phrase_results if r.get('audio_file')]
            
            if phrase_errors:
                result['success'] = False
                result['error'] = f"{len(phrase_errors)} phrases failed to process"
                # Include the first error message for debugging
                if phrase_errors[0].get('error'):
                    result['error'] += f": {phrase_errors[0]['error']}"
            
            # Always try to set an audio file if available
            if audio_files:
                if not phrase_errors:
                    # If no errors, try to concatenate all audio files
                    output_format = self.config.get('output_format', 'wav')
                    combined_audio = section_dir / f'section_combined.{output_format}'
                    try:
                        await self.audio_processor.concatenate_audio(
                            input_files=audio_files,
                            output_file=combined_audio
                        )
                        result['audio_file'] = str(combined_audio)
                    except Exception as e:
                        logger.error(f"Error concatenating audio files: {str(e)}")
                        result['success'] = False
                        result['error'] = f"Audio concatenation failed: {str(e)}"
                        # Fall back to using the first audio file if concatenation fails
                        result['audio_file'] = audio_files[0]
                else:
                    # If there were phrase errors, use the first successful audio file
                    result['audio_file'] = audio_files[0]
            
            result['end_time'] = time.time()
            result['duration'] = result['end_time'] - result['start_time']
            
            # Record performance metrics
            self.performance.record_phase_end(f'section_{section.id}')
            result['performance'] = self.performance.get_phase_metrics(f'section_{section.id}')
            
            # If there were any errors, log them
            if not result.get('success', True):
                logger.error(f"Section processing failed: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.exception(f"Error processing section {section.id}")
            result['success'] = False
            result['error'] = str(e)
            result['end_time'] = time.time()
            result['duration'] = result['end_time'] - result['start_time']
            
            # Record performance metrics even in case of error
            self.performance.record_phase_end(f'section_{section.id}')
            result['performance'] = self.performance.get_phase_metrics(f'section_{section.id}')
            
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
            output_dir: Directory to save phrase audio
            **options: Additional processing options
            
        Returns:
            Dictionary with processing results
        """
        phrase_id = str(phrase.id) if hasattr(phrase, 'id') else str(uuid.uuid4())
        self.performance.record_phase_start(f'phrase_{phrase_id}')
        
        result = {
            'phrase_id': phrase_id,
            'text': phrase.text,
            'language': phrase.language if hasattr(phrase, 'language') else None,
            'voice_id': getattr(phrase, 'voice_id', None),
            'start_time': time.time(),
            'end_time': None,
            'success': True,
            'error': None,
            'audio_file': None,
            'performance': {}
        }
        
        try:
            # Get output format from config or default to 'wav'
            output_format = self.config.get('output_format', 'wav')
            
            # Generate audio for the phrase
            audio_file = output_dir / f'phrase_{phrase_id}.{output_format}'
            
            # Get voice settings from phrase or use defaults
            voice_settings = getattr(phrase, 'voice_settings', {}) or {}
            
            # Generate speech
            logger.debug(f"Calling TTS service with output_file: {audio_file}")
            logger.debug(f"Voice settings: {voice_settings}")
            logger.debug(f"TTS options: {options.get('tts_options', {})}")
            
            try:
                tts_result = await self.tts_service.synthesize_speech(
                    text=phrase.text,
                    output_file=audio_file,
                    **voice_settings,
                    **options.get('tts_options', {})
                )
                logger.debug(f"TTS service returned: {tts_result}")
                
                # Always set the audio file path in the result
                result['audio_file'] = str(audio_file)
                logger.debug(f"Set result['audio_file'] to: {result['audio_file']}")
                
                # Apply audio processing if needed
                if options.get('process_audio', True):
                    processed_audio = output_dir / f'phrase_{phrase_id}_processed.wav'
                    logger.debug(f"Processing audio with output file: {processed_audio}")
                    try:
                        await self.audio_processor.process_audio(
                            input_file=audio_file,
                            output_file=processed_audio,
                            **options.get('audio_processing_options', {})
                        )
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
                        # Re-raise the exception to be caught by the outer try/except
                        raise AudioProcessingError(error_msg) from e
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
