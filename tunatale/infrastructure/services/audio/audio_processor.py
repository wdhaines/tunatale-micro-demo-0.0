"""Audio processing service for TunaTale."""
import logging
import os
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
                normalized_file = Path(tmp.name)
            
            try:
                target_level = kwargs.pop('target_level', -16.0)
                
                # Normalize the audio
                result_path = await self.normalize_audio(
                    input_file=input_file,
                    output_file=normalized_file,
                    target_level=target_level,
                    **kwargs
                )
                
                # Add silence if requested
                if add_silence_ms > 0:
                    silence_position = kwargs.get('silence_position', 'end')
                    result_path = await self.add_silence(
                        input_file=normalized_file,
                        output_file=output_file,
                        duration=add_silence_ms / 1000.0,  # Convert to seconds
                        position=silence_position,
                        **kwargs
                    )
                else:
                    # If no silence to add, just move the normalized file to the output path
                    normalized_file.replace(output_file)
                    result_path = Path(output_file)
                
                return result_path
                
            finally:
                # Clean up the temporary normalized file if it still exists
                if normalized_file.exists():
                    try:
                        normalized_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary file {normalized_file}: {e}")
                
        except Exception as e:
            raise AudioProcessingError(f"Error processing audio: {e}") from e
            
    async def concatenate_audio(
        self,
        input_files: List[Union[str, Path, BinaryIO]],
        output_file: Union[str, Path],
        format: str = None,
        **kwargs
    ) -> Path:
        """Concatenate multiple audio files into a single file.
        
        Args:
            input_files: List of input file paths or file-like objects
            output_file: Output file path
            format: Output audio format (default: 'mp3')
            **kwargs: Additional format-specific parameters
            
        Returns:
            Path to the output file
            
        Raises:
            AudioProcessingError: If there's an error processing the audio
        """
        if not input_files:
            raise AudioProcessingError("No input files provided")
        
        output_path = Path(output_file)
        output_format = format or output_path.suffix[1:] or self.default_format
        
        try:
            # Load the first audio file
            combined = await self._load_audio(input_files[0])
            
            # Append the rest of the files
            for audio_file in input_files[1:]:
                try:
                    audio = await self._load_audio(audio_file)
                    combined += audio
                except Exception as e:
                    logger.warning(f"Error loading audio file {audio_file}: {e}")
                    continue
            
            # Export the combined audio
            return await self._export_audio(combined, output_path, output_format, **kwargs)
            
        except Exception as e:
            raise AudioProcessingError(f"Error concatenating audio files: {e}") from e
    
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
        if duration <= 0:
            raise AudioProcessingError("Duration must be greater than 0")
            
        output_path = Path(output_file)
        output_format = output_path.suffix[1:] or self.default_format
        
        try:
            # Load the audio file
            audio = await self._load_audio(input_file)
            
            # Create silence segment
            silence = AudioSegment.silent(
                duration=int(duration * 1000),  # Convert to milliseconds
                frame_rate=audio.frame_rate
            )
            
            # Add silence at the specified position
            if position == "start":
                result = silence + audio
            elif position == "end":
                result = audio + silence
            elif position == "both":
                result = silence + audio + silence
            else:
                raise ValueError(f"Invalid position: {position}. Must be 'start', 'end', or 'both'")
            
            # Export the result
            return await self._export_audio(result, output_path, output_format, **kwargs)
            
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
        output_path = Path(output_file)
        output_format = output_path.suffix[1:] or self.default_format
        target_level = target_level or self.target_lufs
        
        try:
            # Load the audio file
            audio = await self._load_audio(input_file)
            
            # Normalize loudness
            normalized = effects.normalize(
                audio,
                headroom=abs(target_level)
            )
            
            # Apply peak normalization if needed
            if self.max_peak is not None:
                change_in_db = self.max_peak - normalized.max_dBFS
                normalized = normalized.apply_gain(change_in_db)
            
            # Remove output_format from kwargs to avoid duplicate parameter
            export_kwargs = {k: v for k, v in kwargs.items() if k != 'output_format'}
            
            # Export the result
            return await self._export_audio(normalized, output_path, output_format, **export_kwargs)
            
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
        output_path = Path(output_file)
        output_format = output_path.suffix[1:] or self.default_format
        threshold = threshold or self.silence_threshold
        
        try:
            # Load the audio file
            audio = await self._load_audio(input_file)
            
            # Detect non-silent chunks
            non_silent_ranges = detect_nonsilent(
                audio,
                min_silence_len=100,  # 100ms minimum silence length
                silence_thresh=threshold,
                seek_step=1
            )
            
            if not non_silent_ranges:
                # If no non-silent ranges found, return a very short silent segment
                # (1ms duration) instead of zero-duration to avoid issues with some players
                trimmed = AudioSegment.silent(duration=1, frame_rate=audio.frame_rate)
            else:
                # Get the start and end of non-silent audio
                start = non_silent_ranges[0][0]
                end = non_silent_ranges[-1][1]
                
                # Trim the audio
                trimmed = audio[start:end]
            
            # Export the result
            return await self._export_audio(trimmed, output_path, output_format, **kwargs)
            
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
        output_path = Path(output_file)
        output_format = format or output_path.suffix[1:] or self.default_format
        
        try:
            # Load the input audio
            audio = await self._load_audio(input_file)
            
            # Export to the new format
            return await self._export_audio(audio, output_path, output_format, **kwargs)
            
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
            return len(audio) / 1000.0  # Convert to seconds
        except Exception as e:
            raise AudioProcessingError(f"Error getting audio duration: {e}") from e
    
    async def _validate_mp3_file(self, file_path: Path) -> bool:
        """
        Validate that an MP3 file is not corrupted and has valid audio data.
        Returns True if the file is valid, False otherwise.
        """
        try:
            # Check file exists and has content
            if not file_path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return False
                
            file_size = file_path.stat().st_size
            if file_size < 100:  # MP3 files should be at least 100 bytes
                logger.warning(f"File too small ({file_size} bytes): {file_path}")
                return False
                
            # Read first 1024 bytes to check for valid MP3 header
            with open(file_path, 'rb') as f:
                header = f.read(1024)
                
            # Check for MP3 sync word (starts with 0xFFE or 0xFFF)
            if len(header) < 2 or (header[0] != 0xFF or (header[1] & 0xE0) != 0xE0):
                logger.warning(f"Invalid MP3 header in file: {file_path}")
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Error validating MP3 file {file_path}: {str(e)}")
            return False

    async def _load_audio(
        self, 
        source: Union[str, Path, BinaryIO],
        max_retries: int = 3,  # Increased default retries
        initial_delay: float = 0.2  # Increased initial delay
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
        file_path = str(source) if not hasattr(source, 'read') else None
        
        def log_file_info(path: Path):
            """Log detailed file information for debugging."""
            try:
                if not path.exists():
                    logger.error(f"File does not exist: {path}")
                    return
                
                file_size = path.stat().st_size
                logger.debug(f"File info - Path: {path}, Size: {file_size} bytes")
                
                # Log file permissions
                perms = {
                    'read': os.access(path, os.R_OK),
                    'write': os.access(path, os.W_OK),
                    'execute': os.access(path, os.X_OK)
                }
                logger.debug(f"File permissions - Read: {perms['read']}, Write: {perms['write']}, Execute: {perms['execute']}")
                
                # Log file header for debugging
                try:
                    with open(path, 'rb') as f:
                        header = f.read(16)
                        if header:
                            header_hex = ' '.join([f'\\x{b:02x}' for b in header])
                            logger.debug(f"File header (hex): {header_hex}")
                            
                            # Try to determine file type from magic numbers
                            if len(header) >= 4:
                                if header.startswith(b'\x49\x44\x33'):  # ID3
                                    logger.debug("File type: MP3 with ID3 tag")
                                elif header.startswith(b'\xff\xfb') or header.startswith(b'\xff\xf3') or header.startswith(b'\xff\xfa'):
                                    logger.debug("File type: MP3 frame sync")
                                elif header.startswith(b'RIFF'):
                                    logger.debug("File type: WAV/AVI/RIFF")
                                elif header.startswith(b'OggS'):
                                    logger.debug("File type: OGG")
                                elif header.startswith(b'fLaC'):
                                    logger.debug("File type: FLAC")
                                elif header.startswith(b'\x00\x00\x00 ftyp'):
                                    logger.debug("File type: MP4/M4A")
                except Exception as e:
                    logger.debug(f"Could not read file header: {e}")
                    
            except Exception as e:
                logger.debug(f"Error logging file info: {e}")
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                if file_path:
                    # It's a file path
                    path = Path(file_path)
                    
                    # Log detailed file info on first attempt
                    if attempt == 0:
                        log_file_info(path)
                    
                    if not path.exists():
                        raise FileNotFoundError(f"Audio file not found: {path}")
                    
                    # Get file info for better error reporting
                    file_size = path.stat().st_size
                    if file_size == 0:
                        if attempt < max_retries:
                            logger.warning(f"Audio file is empty, will retry: {path} (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            delay *= 2  # Exponential backoff
                            continue
                        raise AudioProcessingError(f"Audio file is empty: {path} (0 bytes)")
                    
                    # Check file permissions
                    if not os.access(path, os.R_OK):
                        raise AudioProcessingError(f"Insufficient permissions to read audio file: {path}")
                    
                    # Try loading with explicit format first
                    try:
                        logger.debug(f"Attempt {attempt + 1}: Loading audio with format detection: {path}")
                        format = path.suffix[1:].lower() if path.suffix else None
                        audio = AudioSegment.from_file(str(path), format=format)
                        
                        if len(audio) == 0:
                            if attempt < max_retries:
                                logger.warning(f"Loaded empty audio, will retry: {path} (attempt {attempt + 1}/{max_retries})")
                                time.sleep(delay)
                                delay *= 2
                                continue
                            raise AudioProcessingError(f"Loaded audio is empty (0ms duration): {path}")
                        
                        # Verify audio has valid data
                        if audio.frame_rate == 0:
                            raise AudioProcessingError(f"Invalid frame rate (0) in audio file: {path}")
                        
                        logger.debug(f"Successfully loaded audio: {path}, duration: {len(audio)}ms, channels: {audio.channels}, frame_rate: {audio.frame_rate}Hz")
                        return audio
                        
                    except Exception as e:
                        # Fall back to auto-detection if format-specific loading fails
                        if attempt == 0:  # Only log this warning on first attempt
                            logger.warning(f"Failed to load with format detection, trying auto-detect: {e}")
                        
                        try:
                            logger.debug(f"Attempt {attempt + 1}: Loading with auto-detection: {path}")
                            audio = AudioSegment.from_file(str(path))
                            
                            if len(audio) == 0:
                                if attempt < max_retries:
                                    logger.warning(f"Auto-detected empty audio, will retry: {path} (attempt {attempt + 1}/{max_retries})")
                                    time.sleep(delay)
                                    delay *= 2
                                    continue
                                raise AudioProcessingError(f"Auto-detected audio is empty (0ms duration): {path}")
                            
                            logger.debug(f"Successfully loaded with auto-detection: {path}, duration: {len(audio)}ms")
                            return audio
                            
                        except Exception as inner_e:
                            last_error = inner_e
                            if attempt < max_retries:
                                # Log the error and try again
                                logger.warning(
                                    f"Attempt {attempt + 1} failed to load {path}. "
                                    f"Retrying in {delay:.2f}s... Error: {inner_e}"
                                )
                                time.sleep(delay)
                                delay *= 2  # Exponential backoff
                                continue
                            
                            # If we're out of retries, log the file header for debugging
                            log_file_info(path)
                            raise AudioProcessingError(f"Failed to load audio file after {max_retries + 1} attempts: {inner_e}") from inner_e
                else:
                    # Handle file-like objects
                    if hasattr(source, 'seek'):
                        source.seek(0)
                    try:
                        return AudioSegment.from_file(source)
                    except Exception as e:
                        if hasattr(source, 'read'):
                            source.seek(0)
                            data = source.read()
                            if not data:
                                raise AudioProcessingError("No data available from file-like object")
                            from io import BytesIO
                            return AudioSegment.from_file(BytesIO(data))
                        raise
                        
            except Exception as e:
                last_error = e
                if attempt >= max_retries:
                    # If we're out of retries, log the error and re-raise
                    error_msg = str(e)
                    source_info = str(source)
                    
                    # Add file size info if available
                    if file_path and os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        error_msg += f" (File size: {file_size} bytes)"
                        
                    raise AudioProcessingError(
                        f"Failed to load audio from {source_info} after {max_retries + 1} attempts: {error_msg}"
                    ) from e
                
                # Log the warning and retry
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s... Error: {e}"
                )
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                
        # This should never be reached due to the exception above, but just in case
        raise AudioProcessingError(
            f"Failed to load audio after {max_retries + 1} attempts: {last_error}"
        ) from last_error
    
    async def split_audio(
        self,
        input_file: Union[str, Path, BinaryIO],
        output_files: List[Union[str, Path]],
        split_points: List[float],
        **kwargs
    ) -> List[Path]:
        """Split an audio file at specific timestamps.
        
        Args:
            input_file: Input audio file path or file-like object
            output_files: List of output file paths (should have one more element than split_points)
            split_points: List of timestamps in seconds where to split the audio
            **kwargs: Additional format-specific parameters
            
        Returns:
            List of paths to the output files
            
        Raises:
            AudioProcessingError: If there's an error processing the audio
            ValueError: If the number of output files is not one more than the number of split points
        """
        if len(output_files) != len(split_points) + 1:
            raise ValueError(
                f"Number of output files ({len(output_files)}) must be one more than "
                f"the number of split points ({len(split_points)})"
            )
            
        try:
            # Load the audio file
            audio = await self._load_audio(input_file)
            
            # Convert split points from seconds to milliseconds
            split_points_ms = [int(point * 1000) for point in split_points]
            
            # Add start and end points
            split_points_ms = [0] + split_points_ms + [len(audio)]
            
            results = []
            for i in range(len(output_files)):
                # Get the segment for this split
                segment = audio[split_points_ms[i]:split_points_ms[i+1]]
                
                # Export the segment
                output_path = Path(output_files[i])
                result_path = await self._export_audio(
                    segment,
                    output_path,
                    **kwargs
                )
                results.append(result_path)
                
            return results
            
        except Exception as e:
            raise AudioProcessingError(f"Error splitting audio: {e}") from e
    
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
            format: Output format (default: 'mp3')
            **kwargs: Additional format-specific parameters
            
        Returns:
            Path to the exported audio file
            
        Raises:
            AudioProcessingError: If export fails after all retries
        """
        if not isinstance(audio, AudioSegment):
            raise AudioProcessingError(f"Invalid audio type: {type(audio).__name__}, expected AudioSegment")
            
        if not output_path:
            raise AudioProcessingError("Output path cannot be empty")
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        format = format.lower()
        if format not in AUDIO_FORMATS:
            raise AudioProcessingError(f"Unsupported audio format: {format}")
        
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries + 1):
            temp_path = output_path.with_suffix(f".tmp{os.urandom(4).hex()}")
            
            try:
                # Ensure the audio has content
                if len(audio) == 0:
                    raise AudioProcessingError("Cannot export empty audio segment")
                
                # Get format-specific parameters
                format_params = AUDIO_FORMATS.get(format, {}).copy()
                format_params.update(kwargs)
                
                # Remove any None values as they can cause issues with pydub
                format_params = {k: v for k, v in format_params.items() if v is not None}
                
                # Remove format from format_params to avoid duplicate parameter
                format_params.pop('format', None)
                
                # Filter out non-standard parameters that aren't supported by AudioSegment.export()
                # These are known parameters that pydub's export method accepts
                valid_export_params = {
                    'format', 'codec', 'bitrate', 'parameters', 'tags', 'id3v2_version',
                    'cover', 'cover_path', 'artwork', 'artwork_path', 'preprocessor',
                    'ffmpeg_params', 'ffmpeg_pre_params', 'ffmpeg_post_params',
                    'ffmpeg_metadata', 'ffmpeg_log_level', 'ffmpeg_log_path'
                }
                
                # Only keep parameters that are in the valid set
                export_params = {
                    k: v for k, v in format_params.items()
                    if k in valid_export_params
                }
                
                # Log any filtered parameters for debugging
                filtered_params = set(format_params.keys()) - set(export_params.keys())
                if filtered_params:
                    logger.debug(f"Filtered out unsupported export parameters: {filtered_params}")
                
                # Export to temporary file first
                logger.debug(f"Exporting audio to temporary file: {temp_path}")
                
                # Export with error handling
                try:
                    audio.export(
                        str(temp_path),
                        format=format,
                        **export_params
                    )
                except Exception as e:
                    raise AudioProcessingError(f"Failed to export audio: {e}") from e
                
                # Verify the file was created and has content
                if not temp_path.exists():
                    raise AudioProcessingError(f"Temporary file was not created: {temp_path}")
                
                file_size = temp_path.stat().st_size
                if file_size == 0:
                    raise AudioProcessingError(f"Exported file is empty (0 bytes): {temp_path}")
                
                logger.debug(f"Exported {file_size} bytes to {temp_path}")
                
                # Validate the exported file
                try:
                    test_audio = AudioSegment.from_file(str(temp_path), format=format)
                    if len(test_audio) == 0:
                        raise AudioProcessingError(f"Exported audio is empty (0ms duration): {temp_path}")
                    logger.debug(f"Successfully validated audio file: duration={len(test_audio)}ms")
                except Exception as e:
                    # Log file header for debugging
                    try:
                        with open(temp_path, 'rb') as f:
                            header = f.read(16)
                            header_hex = ' '.join([f'\\x{b:02x}' for b in header])
                            logger.error(f"Invalid audio file header (first 16 bytes): {header_hex}")
                    except Exception as read_err:
                        logger.error(f"Failed to read temporary file for debugging: {read_err}")
                    raise AudioProcessingError(f"Invalid audio file created: {e}")
                
                # Move the temporary file to the final location
                try:
                    if output_path.exists():
                        output_path.unlink()
                    
                    # Try atomic rename first
                    try:
                        temp_path.rename(output_path)
                    except OSError as e:
                        if e.errno == 18:  # Cross-device link error
                            # Fall back to copy if rename fails across devices
                            import shutil
                            shutil.copy2(temp_path, output_path)
                            temp_path.unlink()
                            logger.debug(f"Copied audio to {output_path} (cross-device)")
                        else:
                            raise
                    
                    logger.info(f"Successfully exported audio to {output_path}")
                    return output_path
                    
                except OSError as e:
                    raise AudioProcessingError(f"Failed to move temporary file to {output_path}: {e}")
                
            except Exception as e:
                last_error = e
                # Clean up temporary file if it exists
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception as clean_err:
                        logger.warning(f"Failed to clean up temporary file {temp_path}: {clean_err}")
                
                if attempt < max_retries:
                    wait_time = 0.5 * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed to export audio to {output_path}. "
                        f"Retrying in {wait_time:.2f}s... Error: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to export audio to {output_path} after {max_retries + 1} attempts. "
                        f"Last error: {e}"
                    )
        
        # If we get here, all attempts failed
        error_msg = f"Failed to export audio to {output_path} after {max_retries + 1} attempts: {last_error}"
        logger.error(error_msg)
        raise AudioProcessingError(error_msg) from last_error
