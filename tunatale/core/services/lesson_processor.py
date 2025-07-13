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
            
            # Initialize final_audio_file as None by default
            result['final_audio_file'] = None
            
            # Combine section audio files if there are multiple sections
            section_audio_files = [s.get('audio_file') for s in result['sections'] if s.get('audio_file')]
            
            if section_audio_files:
                try:
                    # Get output format from config or default to 'mp3'
                    output_format = self.config.get('output_format', 'mp3')
                    
                    # Get the top-level output directory (parent of the sections directory)
                    top_level_dir = output_dir.parent if output_dir.name == 'sections' else output_dir
                    
                    # Ensure the output directory exists
                    top_level_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save combined audio in the top-level output directory
                    combined_audio = top_level_dir / f'lesson_combined.{output_format}'
                    
                    try:
                        await self.audio_processor.concatenate_audio(
                            input_files=section_audio_files,
                            output_file=combined_audio
                        )
                        
                        # Apply final normalization to the combined audio
                        final_audio = top_level_dir / f'lesson_combined_normalized.{output_format}'
                        logger.info(f"Applying final normalization to combined audio: {combined_audio} -> {final_audio}")
                        
                        try:
                            await self.audio_processor.normalize_audio(
                                input_file=combined_audio,
                                output_file=final_audio,
                                target_level=-16.0  # Standard LUFS level for speech
                            )
                            
                            # Verify the normalized file was created
                            if final_audio.exists() and final_audio.stat().st_size > 0:
                                result['final_audio_file'] = str(final_audio)
                                logger.info(f"Final combined audio file saved to: {final_audio}")
                                
                                # Clean up the non-normalized combined file
                                try:
                                    combined_audio.unlink()
                                except Exception as e:
                                    logger.warning(f"Failed to clean up non-normalized combined audio: {e}")
                            else:
                                logger.warning("Normalized audio file was not created or is empty")
                                # Fall back to the non-normalized version if it exists
                                if combined_audio.exists() and combined_audio.stat().st_size > 0:
                                    result['final_audio_file'] = str(combined_audio)
                                    logger.info(f"Using non-normalized combined audio: {combined_audio}")
                                    
                        except Exception as e:
                            logger.error(f"Error normalizing combined audio: {e}")
                            # Fall back to the non-normalized version if it exists
                            if combined_audio.exists() and combined_audio.stat().st_size > 0:
                                result['final_audio_file'] = str(combined_audio)
                                logger.info(f"Using non-normalized combined audio after normalization error: {combined_audio}")
                    
                    except Exception as e:
                        logger.error(f"Error concatenating section audio files: {e}")
                        result['error'] = f"Failed to concatenate section audio files: {e}"
                
                except Exception as e:
                    logger.error(f"Error processing combined audio: {e}")
                    result['error'] = f"Error processing combined audio: {e}"
                    
                except Exception as e:
                    logger.error(f"Error normalizing combined audio: {e}")
                    # Fall back to the non-normalized version
                    final_audio = combined_audio
                    result['final_audio_file'] = str(final_audio)
                    result['warning'] = f"Failed to normalize combined audio: {str(e)}"
            
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
        
    async def _process_phrase(
        self,
        phrase: Phrase,
        output_dir: Path,
        skip_normalization: bool = True,
        **options: Any
    ) -> Dict[str, Any]:
        """Process a single phrase (internal implementation).
        
        Args:
            phrase: The phrase to process
            output_dir: Base directory for output (will create phrases subdirectory)
            skip_normalization: Whether to skip normalization (default: True)
            **options: Additional processing options
                
        Returns:
            Dictionary with processing results
        """
        # Initialize result dictionary
        result = {
            'phrase_id': str(phrase.id) if hasattr(phrase, 'id') else 'NO_ID',
            'text': phrase.text,
            'start_time': time.time(),
            'end_time': None,
            'duration': None,
            'success': True,
            'error': None,
            'audio_file': None,
            'processing_info': {}
        }
        
        # Get output format from options or use default
        output_format = options.get('output_format', 'mp3')
        
        # Generate a unique ID for this phrase if it doesn't have one
        phrase_id = getattr(phrase, 'id', str(uuid.uuid4()))
        
        try:
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            audio_file = output_dir / f"phrase_{phrase_id}.{output_format}"
            
            # Process the phrase text
            processed_text = self._preprocess_text(phrase.text)
            
            # Update the result with the processed text
            result['text'] = processed_text
            
            # Create output directory if it doesn't exist
            audio_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Get voice_id from phrase or use default
            voice_id = getattr(phrase, 'voice_id', 'en-US-JennyNeural')
            
            # Synthesize speech using EdgeTTS service
            # EdgeTTS service saves the file directly to the specified output path
            tts_result = await self.tts_service.synthesize_speech(
                text=processed_text,
                voice_id=voice_id,
                output_path=audio_file,
                output_format=output_format
            )
            
            # Verify the audio file was created
            if not audio_file.exists() or audio_file.stat().st_size == 0:
                raise AudioProcessingError(f"Failed to generate audio file: {audio_file}")
                
            # Update result with audio file and voice_id
            result['audio_file'] = str(audio_file)
            result['voice_id'] = voice_id
            
            try:
                # Process the audio (normalize, trim silence, etc.)
                await self.audio_processor.normalize_audio(
                    input_file=audio_file,
                    output_file=audio_file,
                    output_format=output_format,
                    phrase_id=phrase_id
                )
                
                # Verify the processed file exists and is not empty
                if not audio_file.exists() or audio_file.stat().st_size == 0:
                    raise AudioProcessingError("Processed audio file is empty")
                
                # Update the result with the audio file path
                result['audio_file'] = str(audio_file.absolute())
                
                # Update timing and performance metrics
                result['end_time'] = time.time()
                result['duration'] = result['end_time'] - result['start_time']
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing phrase: {str(e)}", exc_info=True)
                result.update({
                    'success': False,
                    'error': str(e),
                    'end_time': time.time(),
                    'duration': time.time() - result.get('start_time', 0)
                })
                return result
                
        except Exception as e:
            logger.error(f"Error in _process_phrase: {str(e)}", exc_info=True)
            result.update({
                'success': False,
                'error': str(e),
                'end_time': time.time(),
                'duration': time.time() - result.get('start_time', 0)
            })
            return result
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before TTS processing.
        
        Args:
            text: The text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return text
            
        # Basic text normalization
        text = text.strip()
        
        # Remove any extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    async def process_phrase(
        self,
        phrase: Phrase,
        output_dir: Path,
        **options: Any
    ) -> Dict[str, Any]:
        """Process a single phrase and generate audio.
        
        Args:
            phrase: The phrase to process
            output_dir: Directory to save output files
            **options: Additional processing options
            
        Returns:
            Dictionary with processing results
        """
        result = await self._process_phrase(phrase, output_dir, **options)
        
        # Ensure language is included in the result (using the enum value)
        if 'language' not in result and hasattr(phrase, 'language'):
            result['language'] = phrase.language.value.lower() if hasattr(phrase.language, 'value') else str(phrase.language)
            
        return result
        
    async def process_section(
        self,
        section: Section,
        output_dir: Path,
        **options: Any
    ) -> Dict[str, Any]:
        """Process a section and generate audio.
        
        Args:
            section: The section to process
            output_dir: Directory to save output files
            **options: Additional processing options
                - progress_callback: Optional callback for progress updates
                
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        result = {
            'success': True,
            'section_id': str(section.id),
            'section_type': section.section_type.value if hasattr(section, 'section_type') else 'default',
            'start_time': start_time,
            'phrases': [],
            'processing_info': {}
        }
        
        # Get progress callback if provided
        progress_callback = options.pop('progress_callback', None)
        
        try:
            # Create section output directory
            section_dir = output_dir / 'sections' / str(section.id)
            section_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each phrase in the section
            results = []
            total_phrases = len(section.phrases)
            
            for i, phrase in enumerate(section.phrases, 1):
                try:
                    # Call progress callback if provided
                    if progress_callback:
                        await progress_callback(
                            current=i,
                            total=total_phrases,
                            status=f"Processing phrase {i} of {total_phrases}",
                            phase='process_section',
                            section_id=str(section.id),
                            phrase_id=str(phrase.id) if hasattr(phrase, 'id') else 'unknown'
                        )
                    
                    # Process the phrase
                    phrase_result = await self.process_phrase(phrase, section_dir, **options)
                    results.append(phrase_result)
                    
                except Exception as e:
                    logger.error(f"Error processing phrase {getattr(phrase, 'id', 'unknown')}: {str(e)}")
                    results.append({
                        'success': False,
                        'error': str(e),
                        'phrase_id': str(phrase.id) if hasattr(phrase, 'id') else 'unknown',
                        'text': getattr(phrase, 'text', '')
                    })
            
            # Update result with phrase results
            result['phrases'] = results
            
            # Generate section audio by concatenating phrase audio files
            audio_files = [r['audio_file'] for r in results if r.get('success') and 'audio_file' in r and r['audio_file'] is not None]
            
            # Always include audio_file in result, set to None if not available
            result['audio_file'] = None
            
            if audio_files:
                output_file = section_dir / f"section_{section.id}.mp3"
                try:
                    output_file = await self.audio_processor.concatenate_audio(
                        audio_files,
                        output_file,
                        format='mp3'
                    )
                    
                    # Update result with success data
                    result.update({
                        'audio_file': str(output_file),
                        'success': True,
                        'num_phrases': total_phrases,
                        'successful_phrases': sum(1 for r in results if r.get('success')),
                        'phrase_results': results  # Keep for backward compatibility
                    })
                    
                except Exception as e:
                    error_msg = f"AudioProcessingError: Failed to generate section audio: {str(e)}"
                    logger.error(error_msg)
                    result.update({
                        'success': False,
                        'error': error_msg,
                        'audio_file': None,  # Explicitly set to None on error
                        'phrase_results': results  # Keep for backward compatibility
                    })
            else:
                error_msg = 'AudioProcessingError: No valid phrase audio files to concatenate'
                logger.warning(error_msg)
                result.update({
                    'success': False,
                    'error': error_msg,
                    'audio_file': None,  # Explicitly set to None when no audio files
                    'phrase_results': results  # Keep for backward compatibility
                })
            
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error processing section {section.id}: {str(e)}"
            logger.exception(error_msg)
            result.update({
                'success': False,
                'error': error_msg,
                'phrases': results if 'results' in locals() else []
            })
            return result
            
        finally:
            # Update timing information
            result['end_time'] = time.time()
            result['duration'] = result['end_time'] - start_time
    
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
