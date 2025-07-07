"""Audio processing service for TunaTale."""
import logging
import tempfile
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
    """Service for audio processing operations."""
    
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
            combined = self._load_audio(input_files[0])
            
            # Append the rest of the files
            for audio_file in input_files[1:]:
                try:
                    audio = self._load_audio(audio_file)
                    combined += audio
                except Exception as e:
                    logger.warning(f"Error loading audio file {audio_file}: {e}")
                    continue
            
            # Export the combined audio
            return self._export_audio(combined, output_path, output_format, **kwargs)
            
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
            audio = self._load_audio(input_file)
            
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
            return self._export_audio(result, output_path, output_format, **kwargs)
            
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
            audio = self._load_audio(input_file)
            
            # Normalize loudness
            normalized = effects.normalize(
                audio,
                headroom=abs(target_level)
            )
            
            # Apply peak normalization if needed
            if self.max_peak is not None:
                change_in_db = self.max_peak - normalized.max_dBFS
                normalized = normalized.apply_gain(change_in_db)
            
            # Export the result
            return self._export_audio(normalized, output_path, output_format, **kwargs)
            
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
            audio = self._load_audio(input_file)
            
            # Detect non-silent chunks
            non_silent_ranges = detect_nonsilent(
                audio,
                min_silence_len=100,  # 100ms minimum silence length
                silence_thresh=threshold,
                seek_step=1
            )
            
            if not non_silent_ranges:
                # If the entire audio is silent, return an empty segment
                trimmed = AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)
            else:
                # Get the start and end of non-silent audio
                start = non_silent_ranges[0][0]
                end = non_silent_ranges[-1][1]
                
                # Trim the audio
                trimmed = audio[start:end]
            
            # Export the result
            return self._export_audio(trimmed, output_path, output_format, **kwargs)
            
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
            audio = self._load_audio(input_file)
            
            # Export to the new format
            return self._export_audio(audio, output_path, output_format, **kwargs)
            
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
            audio = self._load_audio(audio_file)
            return len(audio) / 1000.0  # Convert to seconds
        except Exception as e:
            raise AudioProcessingError(f"Error getting audio duration: {e}") from e
    
    def _load_audio(self, source: Union[str, Path, BinaryIO]) -> AudioSegment:
        """Load an audio file into an AudioSegment.
        
        Args:
            source: File path or file-like object
            
        Returns:
            AudioSegment instance
            
        Raises:
            AudioProcessingError: If the file cannot be loaded
        """
        try:
            if hasattr(source, 'read'):
                # It's a file-like object
                if hasattr(source, 'seek'):
                    source.seek(0)
                return AudioSegment.from_file(source)
            else:
                # It's a file path
                path = Path(source)
                if not path.exists():
                    raise FileNotFoundError(f"Audio file not found: {path}")
                return AudioSegment.from_file(str(path))
        except Exception as e:
            raise AudioProcessingError(f"Error loading audio from {source}: {e}") from e
    
    def _export_audio(
        self,
        audio: AudioSegment,
        output_path: Union[str, Path],
        format: str = None,
        **kwargs
    ) -> Path:
        """Export an AudioSegment to a file.
        
        Args:
            audio: AudioSegment to export
            output_path: Output file path
            format: Output format (default: determined from output_path)
            **kwargs: Additional format-specific parameters
            
        Returns:
            Path to the exported file
            
        Raises:
            AudioProcessingError: If the file cannot be exported
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get format from path if not specified
            format = format or output_path.suffix[1:] or self.default_format
            
            # Get export parameters for the format
            export_params = AUDIO_FORMATS.get(format.lower(), AUDIO_FORMATS[DEFAULT_FORMAT]).copy()
            export_params.update(kwargs)
            
            # Remove format from export_params to avoid duplicate parameter
            export_params.pop('format', None)
            
            # Export the audio
            audio.export(
                str(output_path),
                format=format,
                **export_params
            )
            
            return output_path
            
        except Exception as e:
            raise AudioProcessingError(f"Error exporting audio to {output_path}: {e}") from e
