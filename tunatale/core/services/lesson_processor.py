"""Service for processing lessons and generating audio."""
from __future__ import annotations
from typing import Union, Dict, Any, Optional, List, Tuple, cast
import asyncio
import logging
from tunatale.core.exceptions import TTSValidationError, AudioProcessingError
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
                                
                                # Rename the audio files to standard names and update the final_audio_file path if it was renamed
                                try:
                                    renamed_files = await self._rename_audio_files(
                                        output_dir=top_level_dir,
                                        section_audio_files=[Path(f) for f in section_audio_files if f],
                                        sections=result.get('sections', [])
                                    )
                                    result['renamed_files'] = renamed_files
                                    
                                    # Update the final_audio_file path if it was renamed
                                    if '_new_combined_path' in renamed_files and renamed_files['_new_combined_path']:
                                        result['final_audio_file'] = str(renamed_files['_new_combined_path'])
                                        logger.info(f"Updated final_audio_file to {result['final_audio_file']} after renaming")
                                except Exception as e:
                                    logger.warning(f"Failed to rename audio files: {e}")
                                
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
            voice_id = await self._get_voice_for_phrase(phrase)
            
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
                
            except AudioProcessingError as e:
                # For AudioProcessingError, we want to preserve the error message
                error_msg = f"Audio processing error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                result.update({
                    'success': False,
                    'error': error_msg,
                    'end_time': time.time(),
                    'duration': time.time() - result.get('start_time', 0)
                })
                return result
                
            except Exception as e:
                # For other errors, we want to include the error type in the message
                error_msg = f"Error processing phrase: {str(e)}"
                logger.error(error_msg, exc_info=True)
                result.update({
                    'success': False,
                    'error': error_msg,
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
        
    async def _get_valid_voice_id(self, language: Union[str, 'Language']) -> str:
        """Get a valid voice ID for the given language.
        
        Args:
            language: The language code (e.g., 'en-US', 'fil-PH') or Language enum
            
        Returns:
            A valid voice ID for the language
            
        Raises:
            TTSValidationError: If no valid voice is found for the language
        """
        # Handle Language enum input
        from tunatale.core.models.enums import Language
        if isinstance(language, Language):
            # Map Language enum to language codes
            language_code_map = {
                Language.ENGLISH: 'en-US',
                Language.TAGALOG: 'fil-PH',
                Language.SPANISH: 'es-ES',
            }
            language = language_code_map.get(language, 'en-US')
        
        # Define our allowed voices
        ALLOWED_VOICES = {
            # English voices
            'en-US': 'en-US-AriaNeural',  # Primary English voice
            'en': 'en-US-AriaNeural',     # Fallback for generic English
            'english': 'en-US-AriaNeural', # Fallback for 'english' string
            
            # Tagalog voices
            'fil-PH': 'fil-PH-BlessicaNeural',  # Primary Tagalog voice
            'fil': 'fil-PH-BlessicaNeural',     # Fallback for generic Tagalog
            'tl': 'fil-PH-BlessicaNeural',      # Alternative Tagalog code
            'tagalog': 'fil-PH-BlessicaNeural', # Fallback for 'tagalog' string
            
            # Spanish voices
            'es-ES': 'es-ES-ElviraNeural',     # Primary Spanish voice
            'es': 'es-ES-ElviraNeural',        # Fallback for generic Spanish
            'spanish': 'es-ES-ElviraNeural',   # Fallback for 'spanish' string
        }
        
        # First try exact match
        if language in ALLOWED_VOICES:
            return ALLOWED_VOICES[language]
            
        # Try case-insensitive match
        language_lower = language.lower()
        for lang_code, voice_id in ALLOWED_VOICES.items():
            if lang_code.lower() == language_lower:
                return voice_id
                
        # Try matching language prefix (e.g., 'en-' matches 'en-US')
        for lang_code, voice_id in ALLOWED_VOICES.items():
            if language.lower().startswith(lang_code.lower() + '-'):
                return voice_id
                
        # Fall back to English if no match found
        logger.warning(f"No specific voice found for language '{language}', falling back to English (en-US-AriaNeural)")
        return 'en-US-AriaNeural'

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
        result = {
            'success': False,
            'error': None,
            'start_time': time.time(),
            'end_time': None,
            'duration': None,
            'phrase_id': str(phrase.id) if hasattr(phrase, 'id') else None,
            'voice_id': None,
            'language': None
        }

        try:
            # Get voice ID based on phrase language
            voice_id = None
            if phrase.language:
                try:
                    # Get a valid voice ID for the language
                    voice_id = await self._get_valid_voice_id(phrase.language.value)
                    
                    # Validate the voice ID with the TTS service
                    await self.tts_service.validate_voice(voice_id)
                    
                    result['voice_id'] = voice_id
                    result['language'] = phrase.language.value.lower()
                    logger.info(f"Selected voice {voice_id} for language {phrase.language}")
                    
                except TTSValidationError as e:
                    # If validation fails, log the error and try to recover with a default voice
                    error_msg = f"Invalid voice '{voice_id}' for language {phrase.language}: {str(e)}"
                    logger.error(error_msg)
                    
                    # Try to get a fallback voice
                    try:
                        voice_id = 'en-US-AriaNeural'  # Fallback to default English voice
                        await self.tts_service.validate_voice(voice_id)
                        logger.warning(f"Falling back to default voice: {voice_id}")
                        result['voice_id'] = voice_id
                        result['language'] = 'en-US'  # Force English as fallback
                    except TTSValidationError as fallback_error:
                        # If fallback fails, return the original error
                        error_msg = f"Failed to get valid voice for language {phrase.language}: {str(e)}. Fallback also failed: {str(fallback_error)}"
                        logger.error(error_msg)
                        result.update({
                            'success': False,
                            'error': {
                                'error_code': 'TTS_VALIDATION_ERROR',
                                'error_message': error_msg
                            },
                            'end_time': time.time(),
                            'duration': time.time() - result['start_time']
                        })
                        return result

            # Create output directory for phrase audio
            phrase_dir = output_dir / 'phrases'
            phrase_dir.mkdir(parents=True, exist_ok=True)

            # Generate a unique filename based on phrase ID and language
            phrase_id = getattr(phrase, 'id', 'unknown_phrase')
            output_file = phrase_dir / f"{phrase_id}_{phrase.language.value.lower()}.mp3"

            # Process the phrase text
            processed_text = self._preprocess_text(phrase.text)
            if not processed_text:
                raise ValueError("Processed text is empty")

            # Generate audio using TTS service
            await self.tts_service.synthesize_speech(
                text=processed_text,
                voice_id=voice_id,
                output_path=output_file,
                rate=options.get('rate', 1.0),
                pitch=options.get('pitch', 0.0),
                volume=options.get('volume', 1.0)
            )

            # Validate the generated audio file
            if not output_file.exists():
                error_msg = f"Audio file not generated: {output_file}"
                result.update({
                    'success': False,
                    'error': {
                        'error_code': 'AUDIO_PROCESSING_ERROR',
                        'error_message': f"Audio processing error: {error_msg}"
                    },
                    'end_time': time.time(),
                    'duration': time.time() - result['start_time'],
                    'audio_file': None
                })
                return result
            
            if not output_file.stat().st_size > 0:
                error_msg = f"Generated audio file is empty: {output_file}"
                result.update({
                    'success': False,
                    'error': {
                        'error_code': 'AUDIO_PROCESSING_ERROR',
                        'error_message': f"Audio processing error: {error_msg}"
                    },
                    'end_time': time.time(),
                    'duration': time.time() - result['start_time'],
                    'audio_file': None
                })
                return result

            # Apply audio normalization if needed
            if not options.get('skip_normalization', False):
                try:
                    normalized_file = output_file.with_stem(f"{output_file.stem}_normalized")
                    await self.audio_processor.normalize_audio(
                        input_file=output_file,
                        output_file=normalized_file,
                        **options.get('normalization_options', {})
                    )
                    # If normalization succeeded, use the normalized file
                    if normalized_file.exists() and normalized_file.stat().st_size > 0:
                        output_file = normalized_file
                except AudioProcessingError as e:
                    error_msg = f"Audio normalization failed: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    result.update({
                        'success': False,
                        'error': {
                            'error_code': 'AUDIO_PROCESSING_ERROR',
                            'error_message': error_msg
                        },
                        'end_time': time.time(),
                        'duration': time.time() - result['start_time'],
                        'audio_file': None
                    })
                    return result

            # Update result with success data
            result.update({
                'success': True,
                'error': None,
                'end_time': time.time(),
                'duration': time.time() - result['start_time'],
                'audio_file': str(output_file),
                'voice_id': voice_id,
                'language': phrase.language.value.lower(),
                'text': phrase.text  # Include the original phrase text in the result
            })

        except Exception as e:
            error_msg = f"Error processing phrase {phrase.id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Determine error code based on exception type
            error_code = 'UNKNOWN_ERROR'
            if isinstance(e, TTSValidationError):
                error_code = 'TTS_VALIDATION_ERROR'
            elif isinstance(e, AudioProcessingError):
                error_code = 'AUDIO_PROCESSING_ERROR'
            elif isinstance(e, ValueError):
                error_code = 'VALIDATION_ERROR'
                
            result.update({
                'success': False,
                'error': {
                    'error_code': error_code,
                    'error_message': error_msg
                },
                'end_time': time.time(),
                'duration': time.time() - result['start_time'],
                'audio_file': None
            })

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
        """
        result = {
            'success': False,
            'error': None,
            'start_time': time.time(),
            'end_time': None,
            'duration': None,
            'section_id': str(section.id) if hasattr(section, 'id') else 'unknown',
            'title': section.title,
            'type': section.section_type.value if section.section_type else None,  # For backward compatibility
            'section_type': section.section_type.value if section.section_type else None,  # New standard key
            'audio_file': None,
            'phrases': []
        }

        try:
            # Create section directory
            section_dir = output_dir / 'sections'
            section_dir.mkdir(parents=True, exist_ok=True)

            # Process each phrase in the section
            total_phrases = len(section.phrases)
            results = []
            success_count = 0
            has_errors = False
            
            for phrase in section.phrases:
                try:
                    # Process the phrase
                    phrase_result = await self.process_phrase(phrase, output_dir, **options)
                    
                    # Add the result to our collection
                    results.append(phrase_result)
                    
                    # Track success/failure
                    if phrase_result.get('success'):
                        success_count += 1
                    else:
                        has_errors = True
                    
                    # Update progress callback if provided
                    if options.get('progress_callback'):
                        current = len(results)
                        status = f"Processed phrase {getattr(phrase, 'id', 'unknown')}"
                        try:
                            # Call with (current, total, status) signature
                            if asyncio.iscoroutinefunction(options['progress_callback']):
                                await options['progress_callback'](current, total_phrases, status)
                            else:
                                options['progress_callback'](current, total_phrases, status)
                        except Exception as e:
                            logger.warning(f"Error in progress callback: {e}")
                
                except Exception as e:
                    has_errors = True
                    error_msg = f"Error processing phrase {getattr(phrase, 'id', 'unknown')}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    
                    # Create a detailed error result for this phrase
                    error_result = {
                        'success': False,
                        'error': {
                            'error_code': 'PHRASE_PROCESSING_ERROR',
                            'error_message': error_msg
                        },
                        'phrase_id': str(phrase.id) if hasattr(phrase, 'id') else 'unknown',
                        'text': getattr(phrase, 'text', ''),
                        'start_time': time.time(),
                        'end_time': time.time(),
                        'duration': 0,
                        'audio_file': None
                    }
                    results.append(error_result)
            
            # Update result with phrase results
            result['phrases'] = results
            result['successful_phrases'] = success_count
            result['num_phrases'] = total_phrases
            
            # If any phrase failed, set section-level error but still return all results
            if has_errors:
                error_msg = "One or more phrases failed to process"
                logger.warning(error_msg)
                result.update({
                    'success': False,
                    'error': {
                        'error_code': 'SECTION_PROCESSING_ERROR',
                        'error_message': error_msg
                    },
                    'audio_file': None
                })
                return result
            
            # Update result with phrase results and overall success status
            result['phrases'] = results
            result['successful_phrases'] = success_count
            
            # If no phrases succeeded, set section error
            if success_count == 0:
                error_msg = "No valid phrase audio files to concatenate"
                logger.warning(error_msg)
                result.update({
                    'success': False,
                    'error': {
                        'error_code': 'NO_VALID_AUDIO_FILES',
                        'error_message': error_msg
                    },
                    'audio_file': None
                })
                return result
            
            # Generate section audio by concatenating phrase audio files
            audio_files = [r['audio_file'] for r in results if r.get('success') and 'audio_file' in r and r['audio_file'] is not None]
            logger.debug(f"Audio files to concatenate: {audio_files}")
            
            # Always set audio_file in result, even if it's None
            result['audio_file'] = None
            logger.debug(f"Initial audio_file set to: {result['audio_file']}")
            
            # Update result with phrase results
            result['phrases'] = results
            result['successful_phrases'] = sum(1 for r in results if r.get('success'))
            result['num_phrases'] = total_phrases
            logger.debug(f"Phrase results: {results}")
            
            if audio_files:
                # Create a section audio file name based on section title
                section_title = section.title.lower().replace(' ', '_') if section.title else f'section_{len(results)}'
                output_file = section_dir / f"{section_title}.mp3"
                
                # Always include phrases in the result, even if concatenation fails
                result.update({
                    'phrases': results,
                    'successful_phrases': sum(1 for r in results if r.get('success')),
                    'num_phrases': total_phrases,
                    'phrase_results': results  # Keep for backward compatibility
                })
                
                try:
                    output_file = await self.audio_processor.concatenate_audio(
                        audio_files,
                        output_file,
                        format='mp3'
                    )
                    
                    # Update result with success data
                    result.update({
                        'audio_file': str(output_file),
                        'success': True
                    })
                    
                except Exception as e:
                    error_msg = f"Failed to generate section audio: {str(e)}"
                    logger.error(error_msg)
                    result.update({
                        'success': False,
                        'error': error_msg,
                        'audio_file': None  # Explicitly set to None on error
                    })
            else:
                error_msg = 'No valid phrase audio files to concatenate'
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
                'phrases': results if 'results' in locals() else [],
                'audio_file': None  # Ensure audio_file is always set
            })
            return result
            
        finally:
            # Update timing information
            result['end_time'] = time.time()
            result['duration'] = result['end_time'] - result['start_time']
    
    async def _rename_audio_files(
        self,
        output_dir: Path,
        section_audio_files: List[Path],
        sections: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Rename audio files to standard names based on section names.
        
        Args:
            output_dir: Directory containing the audio files
            section_audio_files: List of section audio files to process
            sections: List of section metadata containing names and audio file paths
            
        Returns:
            Dictionary mapping original filenames to new filenames
        """
        renamed = {}
        
        # 1. Rename lesson_combined_normalized.mp3 to full_lesson.mp3
        combined_file = output_dir / 'lesson_combined_normalized.mp3'
        new_combined_path = None
        if combined_file.exists():
            new_name = output_dir / 'full_lesson.mp3'
            combined_file.rename(new_name)
            renamed[str(combined_file)] = str(new_name)
            new_combined_path = new_name
            logger.info(f"Renamed {combined_file.name} to {new_name.name}")
        
        # Store the new combined file path to return later
        renamed['_new_combined_path'] = new_combined_path
        
        # 2. Create a mapping of section audio files to their metadata
        audio_file_map = {}
        for section in sections:
            if 'audio_file' in section and section['audio_file']:
                audio_file_map[section['audio_file']] = section
        
        # 3. Rename section files based on their names
        for file_path in section_audio_files:
            file_path_str = str(file_path)
            if file_path_str in audio_file_map:
                section = audio_file_map[file_path_str]
                section_name = section.get('name', '').lower()
                
                # Determine the new filename based on section name
                # First try special cases with flexible matching
                if any(keyword in section_name for keyword in ['key phrase', 'key_phrase', 'key-phrases']):
                    new_name = 'key_phrases.mp3'
                elif any(keyword in section_name for keyword in ['natural speed', 'natural_speed', 'natural-speed']):
                    new_name = 'natural_speed.mp3'
                elif any(keyword in section_name for keyword in ['translated', 'translation']):
                    new_name = 'translated.mp3'
                else:
                    # For other sections, use the section title as the filename
                    title = section.get('title', '')
                    if title:
                        # Sanitize the title for use as a filename
                        safe_name = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in title.lower())
                        safe_name = safe_name.strip('-_ ').replace(' ', '_')
                        if not safe_name:
                            safe_name = f"section_{len(renamed) + 1}"
                        new_name = f"{safe_name}.mp3"
                    else:
                        # Fallback if no title is available
                        new_name = f"section_{len(renamed) + 1}.mp3"
                
                # Rename the file
                new_path = output_dir / new_name
                if new_path != file_path:
                    try:
                        file_path.rename(new_path)
                        renamed[str(file_path)] = str(new_path)
                        logger.info(f"Renamed {file_path.name} to {new_name}")
                    except OSError as e:
                        logger.warning(f"Failed to rename {file_path} to {new_name}: {e}")
        
        return renamed

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
