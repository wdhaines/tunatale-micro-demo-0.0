"""Audio configuration model for TTS and audio processing settings."""
from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    """Configuration for audio processing.
    
    Attributes:
        normalize: Whether to normalize audio levels
        trim_silence: Whether to trim silence from the beginning and end
        silence_threshold: Threshold in dB below reference to consider as silence
        silence_duration: Duration of silence to detect in seconds
        fade_in: Fade-in duration in seconds
        fade_out: Fade-out duration in seconds
    """
    normalize: bool = True
    trim_silence: bool = True
    silence_threshold: float = Field(default=-40.0, ge=-100.0, le=0.0)
    silence_duration: float = Field(default=0.1, gt=0.0)
    fade_in: float = Field(default=0.1, ge=0.0)
    fade_out: float = Field(default=0.1, ge=0.0)
