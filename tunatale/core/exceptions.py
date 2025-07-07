"""Base exceptions for the TunaTale application."""
from typing import Optional, Dict, Any, Type, TypeVar, Generic, List

T = TypeVar('T')


class TunaTaleError(Exception):
    """Base exception for all TunaTale-specific exceptions."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ) -> None:
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
            cause: The underlying exception that caused this one
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        super().__init__(message)
    
    def __str__(self) -> str:
        """Get a string representation of the error."""
        parts = [f"{self.__class__.__name__}"]
        if self.error_code:
            parts.append(f"[{self.error_code}]")
        parts.append(self.message)
        return " ".join(parts)


class ValidationError(TunaTaleError):
    """Raised when input validation fails."""
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        **kwargs: Any
    ) -> None:
        """Initialize the validation error.
        
        Args:
            message: Error message
            field: The field that failed validation
            value: The invalid value
            **kwargs: Additional error details
        """
        details = kwargs.pop('details', {})
        if field is not None:
            details['field'] = field
        if value is not None:
            details['value'] = value
        
        super().__init__(
            message=message,
            error_code="validation_error",
            details=details,
            **kwargs
        )


class ConfigurationError(TunaTaleError):
    """Raised when there's a configuration error."""
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message=message,
            error_code="configuration_error",
            **kwargs
        )


class TTSServiceError(TunaTaleError):
    """Raised when there's an error with the TTS service."""
    error_code = "tts_service_error"
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message=message, **kwargs)


class TTSError(TunaTaleError):
    """Base class for TTS-related errors."""
    error_code = "tts_error"
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message=message, **kwargs)


class TTSConnectionError(TTSError):
    """Raised when there's a connection error with the TTS service."""
    error_code = "tts_connection_error"
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message=message, **kwargs)


class TTSAuthenticationError(TTSError):
    """Raised when authentication with the TTS service fails."""
    error_code = "tts_authentication_error"
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message=message, **kwargs)


class TTSRateLimitExceeded(TTSError):
    """Raised when rate limits are exceeded for the TTS service."""
    error_code = "tts_rate_limit_exceeded"
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message=message, **kwargs)


class TTSValidationError(TTSError):
    """Raised when TTS input validation fails."""
    error_code = "tts_validation_error"
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message=message, **kwargs)


class VoiceNotAvailableError(TTSError):
    """Raised when a requested voice is not available."""
    def __init__(self, voice_id: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Voice not available: {voice_id}",
            error_code="voice_not_available",
            details={"voice_id": voice_id},
            **kwargs
        )


class AudioProcessingError(TunaTaleError):
    """Raised when there's an error processing audio."""
    error_code = "audio_processing_error"
    
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message=message, **kwargs)


class FileOperationError(TunaTaleError):
    """Raised when there's an error performing file operations."""
    error_code = "file_operation_error"
    
    def __init__(self, message: str, path: Optional[str] = None, **kwargs: Any) -> None:
        details = kwargs.pop('details', {})
        if path is not None:
            details['path'] = path
        super().__init__(
            message=message,
            details=details,
            **kwargs
        )


class ResourceNotFoundError(TunaTaleError):
    """Raised when a requested resource is not found."""
    error_code = "resource_not_found"
    
    def __init__(self, resource_type: str, resource_id: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"{resource_type} not found: {resource_id}",
            details={"resource_type": resource_type, "resource_id": resource_id},
            **kwargs
        )


class ResourceExistsError(TunaTaleError):
    """Raised when trying to create a resource that already exists."""
    error_code = "resource_exists"
    
    def __init__(self, resource_type: str, resource_id: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"{resource_type} already exists: {resource_id}",
            details={"resource_type": resource_type, "resource_id": resource_id},
            **kwargs
        )
