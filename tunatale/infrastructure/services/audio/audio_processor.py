"""Audio processing service for TunaTale."""
import asyncio
import logging
import os
import shutil
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
from pydub import AudioSegment, effects
from pydub.silence import detect_nonsilent

from tunatale.core.exceptions import AudioProcessingError
from tunatale.core.ports.audio_processor import AudioProcessor

# Configure logging
logger = logging.getLogger(__name__)

# Audio format configurations
AUDIO_FORMATS = {
    'mp3': {'codec': 'libmp3lame', 'bitrate': '128k'},
    'wav': {'format': 'wav'},
    'ogg': {'codec': 'libvorbis', 'bitrate': '128k'},
    'm4a': {'codec': 'aac', 'bitrate': '128k'},
}

DEFAULT_FORMAT = 'mp3'


class AudioProcessorService(AudioProcessor):
    """Service for audio processing operations.
    
    This service provides various audio processing capabilities including:
    - Audio format conversion
    - Loudness normalization
    - Silence trimming and addition
    - Audio concatenation
    - Audio splitting at specific timestamps
    - Batch processing of audio files
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the audio processor.
        
        Args:
            config: Configuration dictionary with the following keys:
                - default_format: Default audio format (default: 'mp3')
                - silence_duration_ms: Default silence duration in ms (default: 500)
                - silence_threshold: Silence threshold in dB (default: -50)
                - target_lufs: Target loudness in LUFS (default: -16)
                - max_peak: Maximum peak level in dB (default: -1.0)
        """
        self.config = config or {}
        self.default_format = self.config.get('default_format', DEFAULT_FORMAT)
        
        # Audio processing parameters
        self.silence_duration_ms = self.config.get('silence_duration_ms', 500)
        self.silence_threshold = self.config.get('silence_threshold', -50)
        self.target_lufs = self.config.get('target_lufs', -16.0)
        self.max_peak = self.config.get('max_peak', -1.0)
        
        # Audio format configurations
        self.audio_formats = {
            'mp3': {'codec': 'libmp3lame', 'bitrate': '128k'},
            'wav': {'format': 'wav'},
            'ogg': {'codec': 'libvorbis', 'bitrate': '128k'},
            'm4a': {'codec': 'aac', 'bitrate': '128k'},
        }
    
    async def process_audio(
        self,
        input_file: Union[str, Path, BinaryIO],
        output_file: Union[str, Path],
        add_silence_ms: int = 0,
        **kwargs
    ) -> Path:
        """Process audio file with optional silence addition.
        
        Args:
            input_file: Input file path or file-like object
            output_file: Output file path
            add_silence_ms: Duration of silence to add in milliseconds
            **kwargs: Additional processing parameters
                - target_level: Target loudness in LUFS (default: -16.0)
                - silence_position: Where to add silence ('start', 'end', or 'both')
                
        Returns:
            Path to the processed output file
            
        Raises:
            AudioProcessingError: If there's an error processing the audio
        """
        try:
            # Get output format from output file extension or use default
            output_path = Path(output_file)
            output_format = output_path.suffix[1:] or self.default_format
            
            # Create a temporary file with the correct extension
            with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as tmp:
                temp_path = Path(tmp.name)
            
            # Load the input audio
            audio = await self._load_audio(input_file)
            
            # Apply audio processing
            if add_silence_ms > 0:
                silence_position = kwargs.get('silence_position', 'end')
                audio = await self._add_silence(audio, add_silence_ms, silence_position)
            
            # Normalize audio if target level is specified
            target_level = kwargs.get('target_level')
            if target_level is not None:
                audio = await self._normalize_audio(audio, target_level)
            
            # Export the processed audio
            await self._export_audio(audio, temp_path, output_format, **kwargs)
            
            # Move the temporary file to the final destination
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                output_path.unlink()
            temp_path.rename(output_path)
            
            return output_path
            
        except Exception as e:
            # Clean up any temporary files
            if 'temp_path' in locals() and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary file: {cleanup_error}")
            
            raise AudioProcessingError(f"Error processing audio: {e}") from e
    
    async def concatenate_audio(
        self,
        input_files: List[Union[str, Path, BinaryIO]],
        output_file: Union[str, Path],
        format: str = None,
        **kwargs
    ) -> Path:
        """Concatenate multiple audio files into a single file with robust error handling.
        
        Args:
            input_files: List of input file paths or file-like objects
            output_file: Output file path
            format: Output format (default: determined by output_file extension)
            **kwargs: Additional format-specific parameters
            
        Returns:
            Path to the output file
            
        Raises:
            AudioProcessingError: If there's an error processing the audio
        """
        if not input_files:
            raise AudioProcessingError("No input files provided for concatenation")

        output_path = Path(output_file)
        output_format = format or output_path.suffix[1:] or self.default_format
        temp_files = []
        
        try:
            # Ensure the output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load and validate all audio files first
            audio_segments = []
            for file_obj in input_files:
                try:
                    # Handle both file paths and file-like objects
                    if hasattr(file_obj, 'read'):
                        # It's a file-like object, read its content
                        file_obj.seek(0)  # Ensure we're at the start
                        content = file_obj.read()
                        if not content:
                            logger.warning("Skipping empty file-like object")
                            continue
                            
                        # Create a temporary file with the content
                        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                            tmp.write(content)
                            temp_file = Path(tmp.name)
                            temp_files.append(temp_file)
                        
                        audio_path = temp_file
                    else:
                        # It's a file path
                        file_path = Path(file_obj)
                        audio_path = file_path
                        logger.debug(f"Loading audio file: {file_path}")
                    
                    # Load the audio file directly as MP3
                    audio = await self._load_audio(str(audio_path))
                    if len(audio) > 0:  # Only add non-empty segments
                        audio_segments.append(audio)
                except Exception as load_error:
                    logger.warning(f"Failed to load audio file {audio_path}: {load_error}")
                    # If loading a file fails, we should raise an error to the caller
                    raise AudioProcessingError(f"Failed to load audio file: {audio_path}") from load_error
            
            if not audio_segments:
                raise AudioProcessingError("No valid audio segments to concatenate")
            
            # Concatenate all audio segments
            logger.debug(f"Concatenating {len(audio_segments)} audio segments")
            combined = audio_segments[0]
            for segment in audio_segments[1:]:
                combined += segment
            
            # Export the combined audio with format-specific parameters
            await self._export_audio(combined, output_path, output_format, **kwargs)
            
            # Verify the output file was created and is valid
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise AudioProcessingError("Failed to create valid output file")
                
            return output_path
            
        except Exception as e:
            # Clean up any partially created output file
            if 'output_path' in locals() and output_path.exists():
                try:
                    output_path.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up invalid output file: {cleanup_error}")
            raise AudioProcessingError(f"Error concatenating audio: {e}") from e
    
    async def add_silence(
        self,
        input_file: Union[str, Path, BinaryIO],
        output_file: Union[str, Path],
        duration: float,
        position: str = "end",
        **kwargs
    ) -> Path:
        """Add silence to an audio file.
        
        Args:
            input_file: Input file path or file-like object
            output_file: Output file path
            duration: Duration of silence in seconds
            position: Where to add silence ('start', 'end', or 'both')
            **kwargs: Additional format-specific parameters
            
        Returns:
            Path to the output file
            
        Raises:
            AudioProcessingError: If there's an error processing the audio
        """
        try:
            output_path = Path(output_file)
            output_format = output_path.suffix[1:] or self.default_format
            
            # Load the input audio
            audio = await self._load_audio(input_file)
            
            # Add silence
            silence_ms = int(duration * 1000)
            audio = await self._add_silence(audio, silence_ms, position)
            
            # Export the processed audio
            await self._export_audio(audio, output_path, output_format, **kwargs)
            
            return output_path
            
        except Exception as e:
            raise AudioProcessingError(f"Error adding silence to audio: {e}") from e
    
    async def normalize_audio(
        self,
        input_file: Union[str, Path, BinaryIO],
        output_file: Union[str, Path],
        target_level: float = None,
        **kwargs
    ) -> Path:
        """Normalize audio to a target loudness level.
        
        Args:
            input_file: Input file path or file-like object
            output_file: Output file path
            target_level: Target loudness in LUFS (default: -16.0)
            **kwargs: Additional format-specific parameters
            
        Returns:
            Path to the output file
            
        Raises:
            AudioProcessingError: If there's an error processing the audio
        """
        try:
            output_path = Path(output_file)
            output_format = output_path.suffix[1:] or self.default_format
            
            # Load the input audio
            audio = await self._load_audio(input_file)
            
            # Normalize audio
            target = target_level if target_level is not None else self.target_lufs
            audio = await self._normalize_audio(audio, target)
            
            # Export the processed audio
            await self._export_audio(audio, output_path, output_format, **kwargs)
            
            return output_path
            
        except Exception as e:
            raise AudioProcessingError(f"Error normalizing audio: {e}") from e
    
    async def trim_silence(
        self,
        input_file: Union[str, Path, BinaryIO],
        output_file: Union[str, Path],
        threshold: float = None,
        **kwargs
    ) -> Path:
        """Trim silence from the beginning and end of an audio file.
        
        Args:
            input_file: Input file path or file-like object
            output_file: Output file path
            threshold: Silence threshold in dB (default: -50)
            **kwargs: Additional format-specific parameters
            
        Returns:
            Path to the output file
            
        Raises:
            AudioProcessingError: If there's an error processing the audio
        """
        try:
            output_path = Path(output_file)
            output_format = output_path.suffix[1:] or self.default_format
            
            # Load the input audio
            audio = await self._load_audio(input_file)
            
            # Trim silence
            silence_threshold = threshold if threshold is not None else self.silence_threshold
            audio = await self._trim_silence(audio, silence_threshold)
            
            # Export the processed audio
            await self._export_audio(audio, output_path, output_format, **kwargs)
            
            return output_path
            
        except Exception as e:
            raise AudioProcessingError(f"Error trimming silence from audio: {e}") from e
    
    async def convert_format(
        self,
        input_file: Union[str, Path, BinaryIO],
        output_file: Union[str, Path],
        format: str = None,
        **kwargs
    ) -> Path:
        """Convert audio file to a different format.
        
        Args:
            input_file: Input file path or file-like object
            output_file: Output file path
            format: Target audio format (default: determined from output_file)
            **kwargs: Additional format-specific parameters
            
        Returns:
            Path to the output file
            
        Raises:
            AudioProcessingError: If there's an error converting the audio
        """
        try:
            output_path = Path(output_file)
            output_format = format or output_path.suffix[1:] or self.default_format
            
            # Load the input audio
            audio = await self._load_audio(input_file)
            
            # Export the audio in the target format
            await self._export_audio(audio, output_path, output_format, **kwargs)
            
            return output_path
            
        except Exception as e:
            raise AudioProcessingError(f"Error converting audio format: {e}") from e
    
    async def get_audio_duration(
        self,
        audio_file: Union[str, Path, BinaryIO]
    ) -> float:
        """Get the duration of an audio file in seconds.
        
        Args:
            audio_file: Path to audio file or file-like object
            
        Returns:
            Duration in seconds as a float
            
        Raises:
            AudioProcessingError: If there's an error reading the file
        """
        try:
            audio = await self._load_audio(audio_file)
            return len(audio) / 1000.0  # Convert ms to seconds
            
        except Exception as e:
            raise AudioProcessingError(f"Error getting audio duration: {e}") from e
    
    async def process_text_with_pauses(
        self,
        audio_file: Union[str, Path, BinaryIO],
        output_file: Union[str, Path],
        **kwargs
    ) -> Path:
        """Process audio that contains pause markers in the original text.
        
        This method handles audio generated from text that contained [PAUSE:Xs] markers
        by detecting these markers in the audio and replacing them with actual silence.
        
        Args:
            audio_file: Input audio file path or file-like object
            output_file: Output file path
            **kwargs: Additional processing parameters
            
        Returns:
            Path to the processed output file with actual pauses
            
        Raises:
            AudioProcessingError: If there's an error processing the audio
        """
        try:
            output_path = Path(output_file)
            output_format = output_path.suffix[1:] or self.default_format
            
            # Load the input audio
            audio = await self._load_audio(audio_file)
            
            # For now, return the audio as-is since pause marker detection 
            # in audio would be complex. The real solution is to handle
            # pause markers during TTS generation, not in post-processing.
            
            # Export the audio
            await self._export_audio(audio, output_path, output_format, **kwargs)
            
            return output_path
            
        except Exception as e:
            raise AudioProcessingError(f"Error processing text with pauses: {e}") from e
    
    async def _export_audio(
        self,
        audio: AudioSegment,
        output_path: Path,
        format: str = "mp3",
        **kwargs
    ) -> Path:
        """Export audio to the specified path with validation and retries.
        
        Args:
            audio: AudioSegment to export
            output_path: Destination path for the exported audio
            **kwargs: Additional arguments passed to AudioSegment.export()
            
        Returns:
            Path to the exported audio file
            
        Raises:
            AudioProcessingError: If export fails after all retries
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Always use MP3 format
        format = 'mp3'
        if not output_path.suffix.lower() == '.mp3':
            output_path = output_path.with_suffix('.mp3')
        
        # Set default export parameters for MP3
        export_params = {
            'format': 'mp3',
            'bitrate': self.config.get('bitrate', '128k'),
            'parameters': ['-ar', '44100', '-ac', '2'],
            **kwargs
        }
        
        # Try to export with retries
        max_retries = self.config.get('max_export_retries', 3)
        last_error = None
        temp_path = None
        
        try:
            for attempt in range(max_retries + 1):
                try:
                    # Create a temporary file for export in the same directory as the target
                    temp_fd, temp_path_str = tempfile.mkstemp(
                        suffix='.mp3',
                        dir=str(output_path.parent),
                        prefix=f"temp_{output_path.stem}_"
                    )
                    os.close(temp_fd)  # Close the file descriptor as we'll use the path with AudioSegment
                    temp_path = Path(temp_path_str)
                    
                    # Export to the temporary file
                    logger.debug(f"Exporting audio to temporary file: {temp_path}")
                    audio.export(
                        str(temp_path),
                        **export_params
                    )
                    
                    # Verify the file was created and is not empty
                    if not temp_path.exists():
                        raise AudioProcessingError("Temporary file was not created")
                        
                    if temp_path.stat().st_size == 0:
                        raise AudioProcessingError("Temporary file is empty")
                    
                    # Move the temporary file to the final location
                    if output_path.exists():
                        try:
                            output_path.unlink()
                        except Exception as unlink_error:
                            logger.warning(f"Failed to remove existing output file: {unlink_error}")
                    
                    shutil.move(str(temp_path), str(output_path))
                    temp_path = None  # Prevent cleanup in finally block
                    
                    # Final verification
                    if not output_path.exists():
                        raise AudioProcessingError("Output file was not created after move")
                        
                    output_size = output_path.stat().st_size
                    logger.debug(f"Output file size: {output_size} bytes")
                    
                    if output_size == 0:
                        raise AudioProcessingError("Output file is empty after move")
                        
                    logger.info(f"Successfully exported audio to {output_path} ({output_size} bytes)")
                    return output_path
                    
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed to export audio: {e}"
                    )
                    
                    # Clean up any temporary files
                    if temp_path and temp_path.exists():
                        try:
                            temp_path.unlink()
                            temp_path = None
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to clean up temporary file: {cleanup_error}")
                    
                    # Add a small delay before retry with exponential backoff
                    if attempt < max_retries:
                        await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff
                        continue
                    
                    # If we get here, all attempts failed
                    error_msg = f"Failed to export audio to {output_path} after {max_retries + 1} attempts: {last_error}"
                    raise AudioProcessingError(error_msg) from last_error
            
        except Exception as e:
            # This handles any other unexpected errors
            error_msg = f"Unexpected error exporting audio to {output_path}: {e}"
            logger.error(error_msg)
            raise AudioProcessingError(error_msg) from e
            
        finally:
            # Clean up any remaining temporary files
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary file in finally: {cleanup_error}")
    
    async def _load_audio(
        self, 
        source: Union[str, Path, BinaryIO],
        max_retries: int = 3,
        initial_delay: float = 0.2
    ) -> AudioSegment:
        """Load an audio file into an AudioSegment with retry logic.
        
        Args:
            source: File path or file-like object
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds (will be doubled each retry)
            
        Returns:
            AudioSegment instance
            
        Raises:
            AudioProcessingError: If the file cannot be loaded after all retries
        """
        last_error = None
        delay = initial_delay
        
        for attempt in range(max_retries + 1):
            try:
                if isinstance(source, (str, Path)):
                    if not Path(source).exists() or Path(source).stat().st_size == 0:
                        raise AudioProcessingError(f"Input file is missing or empty: {source}")
                    # Load from file path
                    audio = AudioSegment.from_file(str(source))
                elif hasattr(source, 'read'):
                    # Load from file-like object
                    if hasattr(source, 'seek'):
                        source.seek(0)  # Rewind the file pointer
                    audio = AudioSegment.from_file(source)
                else:
                    raise AudioProcessingError(f"Unsupported source type: {type(source).__name__}")
                
                # Verify the audio was loaded correctly
                if len(audio) == 0:
                    raise AudioProcessingError("Loaded audio is empty")
                    
                return audio
            except Exception as e:
                # Catch pydub specific errors and other exceptions
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed to load audio: {e}")
                
                # Add a delay before retry (exponential backoff)
                if attempt < max_retries:
                    import asyncio
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                
                # Reset file position if it's a file-like object
                if 'audio_file' in locals() and hasattr(audio_file, 'seek'):
                    try:
                        audio_file.seek(0)
                    except Exception as seek_error:
                        logger.warning(f"Failed to reset file position: {seek_error}")
        
        # If we get here, all attempts failed
        error_msg = f"Failed to load audio after {max_retries + 1} attempts: {last_error}"
        logger.error(error_msg)
        raise AudioProcessingError(error_msg) from last_error
    
    async def _add_silence(
        self,
        audio: AudioSegment,
        duration_ms: int,
        position: str = "end"
    ) -> AudioSegment:
        """Add silence to an AudioSegment.
        
        Args:
            audio: Input AudioSegment
            duration_ms: Duration of silence to add in milliseconds
            position: Where to add silence ('start', 'end', or 'both')
            
        Returns:
            New AudioSegment with added silence
        """
        if duration_ms <= 0:
            return audio
            
        silence = AudioSegment.silent(duration=duration_ms)
        
        if position == "start":
            return silence + audio
        elif position == "end":
            return audio + silence
        elif position == "both":
            return silence + audio + silence
        else:
            raise ValueError(f"Invalid position: {position}. Must be 'start', 'end', or 'both'.")
    
    async def _normalize_audio(
        self,
        audio: AudioSegment,
        target_level: float
    ) -> AudioSegment:
        """Normalize audio to a target loudness level.
        
        Args:
            audio: Input AudioSegment
            target_level: Target loudness in LUFS
            
        Returns:
            Normalized AudioSegment
        """
        try:
            # Convert to mono for loudness measurement if needed
            if audio.channels > 1:
                mono_audio = audio.set_channels(1)
            else:
                mono_audio = audio
            
            # Calculate current loudness (RMS)
            samples = np.array(mono_audio.get_array_of_samples())
            if len(samples) == 0:
                return audio
                
            rms = np.sqrt(np.mean(np.square(samples, dtype=np.float64)))
            if rms == 0:
                return audio
                
            # Calculate current loudness in dBFS
            current_level = 20 * np.log10(rms / (2**15))
            
            # Calculate gain needed to reach target level
            gain_db = target_level - current_level
            
            # Apply gain
            return audio.apply_gain(gain_db)
            
        except Exception as e:
            logger.warning(f"Error normalizing audio: {e}")
            return audio  # Return original audio if normalization fails
    
    async def _trim_silence(
        self,
        audio: AudioSegment,
        threshold_db: float = -50.0
    ) -> AudioSegment:
        """Trim silence from the beginning and end of an AudioSegment.
        
        Args:
            audio: Input AudioSegment
            threshold_db: Silence threshold in dB
            
        Returns:
            Trimmed AudioSegment
        """
        try:
            # Convert to mono for silence detection if needed
            if audio.channels > 1:
                mono_audio = audio.set_channels(1)
            else:
                mono_audio = audio
            
            # Find non-silent chunks
            non_silent_ranges = detect_nonsilent(
                mono_audio,
                min_silence_len=100,  # 100ms minimum silence length
                silence_thresh=threshold_db
            )
            
            if not non_silent_ranges:
                # Return a very short silent segment (10ms) instead of empty
                # to ensure we create a valid MP3 file
                return AudioSegment.silent(duration=10, frame_rate=audio.frame_rate)
            
            # Get start and end times of non-silent audio
            start_time = non_silent_ranges[0][0]
            end_time = non_silent_ranges[-1][1]
            
            # Trim the audio
            return audio[start_time:end_time]
            
        except Exception as e:
            logger.warning(f"Error trimming silence: {e}")
            return audio  # Return original audio if trimming fails
