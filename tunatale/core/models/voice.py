"""Voice model for TTS voices."""
from typing import Optional, Dict, Any
from pydantic import Field, validator

from .base import BaseEntity
from .enums import VoiceGender, VoiceAge, Language


class Voice(BaseEntity):
    """Represents a TTS voice with its properties.
    
    Attributes:
        name: The display name of the voice
        provider: The TTS provider (e.g., 'edge_tts', 'google_tts')
        provider_id: The voice ID used by the provider
        language: The language of the voice
        gender: The gender of the voice
        age: The age group of the voice
        sample_rate: The sample rate in Hz
        is_active: Whether the voice is available for use
        metadata: Additional provider-specific metadata
    """
    name: str = Field(..., min_length=1, max_length=100)
    provider: str = Field(..., min_length=1, max_length=50)
    provider_id: str = Field(..., min_length=1, max_length=100)
    language: Language
    gender: VoiceGender
    age: VoiceAge = VoiceAge.ADULT
    sample_rate: int = Field(24000, ge=8000, le=48000)
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """Validate and normalize the voice name."""
        return v.strip()
    
    @validator('provider')
    def validate_provider(cls, v: str) -> str:
        """Validate and normalize the provider name."""
        return v.lower().strip()
    
    @validator('language', pre=True)
    def validate_language(cls, v):
        """Ensure language is a valid Language enum value or code."""
        from .enums import Language
        
        if isinstance(v, Language):
            return v
            
        if isinstance(v, str):
            # Try to get the language from the string (handles 'en', 'fil', etc.)
            lang = Language.from_string(v)
            if lang is not None:
                return lang
                
        raise ValueError(f"Invalid language: {v}. Must be a valid Language enum or code.")
    
    @property
    def full_name(self) -> str:
        """Get a full descriptive name for the voice."""
        return f"{self.name} ({self.language.value.capitalize()}, {self.gender.value.capitalize()})"
    
    def is_compatible_with(self, text_language: str) -> bool:
        """Check if this voice is compatible with the given language."""
        try:
            text_lang = Language.from_string(text_language)
            return text_lang == self.language
        except (ValueError, AttributeError):
            return False
