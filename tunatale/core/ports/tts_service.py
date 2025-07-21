"""Interface for Text-to-Speech services."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Protocol, runtime_checkable
from pathlib import Path

from ..models.voice import Voice


class TTSException(Exception):
    """Base exception for TTS service errors."""
    pass


class TTSValidationError(TTSException):
    """Raised when input validation fails."""
    pass


class TTSRateLimitError(TTSException):
    """Raised when rate limits are exceeded."""
    def __init__(self, message, **kwargs):
        super().__init__(message)
        self.retry_after = kwargs.get('retry_after')
        self.status_code = kwargs.get('status_code')
        self.headers = kwargs.get('headers', {})


class TTSAuthenticationError(TTSException):
    """Raised when authentication fails."""
    pass


class TTSTransientError(TTSException):
    """Raised for transient errors that might succeed on retry."""
    pass


@runtime_checkable
class TTSService(Protocol):
    """Protocol defining the interface for TTS services.
    
    This protocol defines the required methods that any TTS service implementation
    must provide to be compatible with the TunaTale application.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the TTS service.
        
        Returns:
            str: The name of the service (e.g., 'edge_tts', 'google_tts')
        """
        ...
    
    @abstractmethod
    async def get_voices(self, language: Optional[str] = None) -> List[Voice]:
        """Get a list of available voices.
        
        Args:
            language: Optional language code to filter voices by
            
        Returns:
            List[Voice]: List of available voices
            
        Raises:
            TTSConnectionError: If there's an error connecting to the service
            TTSAuthenticationError: If authentication with the service fails
        """
        ...
    
    @abstractmethod
    async def synthesize_speech(
        self,
        text: str,
        voice_id: str,
        output_path: Path,
        rate: float = 1.0,
        pitch: float = 0.0,
        volume: float = 1.0,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Synthesize speech from text.
        
        Args:
            text: The text to synthesize
            voice_id: The ID of the voice to use
            output_path: Path to save the audio file
            rate: Speech rate (0.5-2.0)
            pitch: Pitch adjustment (-20 to 20)
            volume: Volume level (0.0-1.0)
            **kwargs: Additional service-specific parameters
            
        Returns:
            Dict containing metadata about the synthesis
            
        Raises:
            TTSValidationError: If input validation fails
            TTSRateLimitExceeded: If rate limits are exceeded
            TTSServiceError: For other TTS service errors
        """
        ...
    
    @abstractmethod
    def get_voice(self, voice_id: str) -> Optional[Voice]:
        """Get a specific voice by ID.
        
        Args:
            voice_id: The ID of the voice to retrieve
            
        Returns:
            Optional[Voice]: The voice if found, None otherwise
        """
        ...
    
    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Validate that the service credentials are valid.
        
        Returns:
            bool: True if credentials are valid, False otherwise
            
        Raises:
            TTSAuthenticationError: If credentials are invalid or missing
        """
        ...
