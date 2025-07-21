"""Interface for audio processing services."""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union, BinaryIO
from pathlib import Path


class AudioProcessor(ABC):
    """Abstract base class for audio processing services.
    
    This class defines the interface for audio processing operations such as
    concatenation, normalization, and format conversion.
    """
    
    @abstractmethod
    async def concatenate_audio(
        self,
        input_files: List[Union[str, Path, BinaryIO]],
        output_file: Union[str, Path],
        format: str = "mp3",
        **kwargs: Any
    ) -> Path:
        """Concatenate multiple audio files into a single file.
        
        Args:
            input_files: List of input file paths or file-like objects
            output_file: Output file path
            format: Output audio format (e.g., 'mp3', 'wav')
            **kwargs: Additional format-specific parameters
            
        Returns:
            Path to the output file
            
        Raises:
            AudioProcessingError: If there's an error processing the audio
        """
        ...
    
    @abstractmethod
    async def normalize_audio(
        self,
        input_file: Union[str, Path, BinaryIO],
        output_file: Union[str, Path],
        target_level: float = -20.0,
        **kwargs: Any
    ) -> Path:
        """Normalize audio to a target level.
        
        Args:
            input_file: Input file path or file-like object
            output_file: Output file path
            target_level: Target loudness level in LUFS
            **kwargs: Additional processing parameters
            
        Returns:
            Path to the output file
        """
        ...
    
    @abstractmethod
    async def convert_format(
        self,
        input_file: Union[str, Path, BinaryIO],
        output_file: Union[str, Path],
        format: str = "mp3",
        **kwargs: Any
    ) -> Path:
        """Convert audio file to a different format.
        
        Args:
            input_file: Input file path or file-like object
            output_file: Output file path
            format: Target audio format
            **kwargs: Additional format-specific parameters
            
        Returns:
            Path to the output file
        """
        ...
    
    @abstractmethod
    async def add_silence(
        self,
        input_file: Union[str, Path, BinaryIO],
        output_file: Union[str, Path],
        duration: float,
        position: str = "end",
        **kwargs: Any
    ) -> Path:
        """Add silence to an audio file.
        
        Args:
            input_file: Input file path or file-like object
            output_file: Output file path
            duration: Duration of silence in seconds
            position: Where to add silence ('start', 'end', or 'both')
            **kwargs: Additional processing parameters
            
        Returns:
            Path to the output file
        """
        ...
    
    @abstractmethod
    async def get_audio_duration(
        self,
        audio_file: Union[str, Path, BinaryIO]
    ) -> float:
        """Get the duration of an audio file in seconds.
        
        Args:
            audio_file: Path to audio file or file-like object
            
        Returns:
            Duration in seconds as a float
        """
        ...
    
    @abstractmethod
    async def trim_silence(
        self,
        input_file: Union[str, Path, BinaryIO],
        output_file: Union[str, Path],
        threshold: float = -50.0,
        **kwargs: Any
    ) -> Path:
        """Trim silence from the beginning and end of an audio file.
        
        Args:
            input_file: Input file path or file-like object
            output_file: Output file path
            threshold: Silence threshold in dB
            **kwargs: Additional processing parameters
            
        Returns:
            Path to the output file
        """
        ...
