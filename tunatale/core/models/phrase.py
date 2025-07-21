"""Phrase model for language learning content."""
from typing import Optional, List, Dict, Any
from pydantic import Field, validator, HttpUrl

from .base import BaseEntity
from .voice import Voice
from .enums import Language


class Phrase(BaseEntity):
    """Represents a phrase in a language learning lesson.
    
    Attributes:
        text: The text content of the phrase
        language: The language of the phrase
        voice_id: ID of the voice to use for TTS
        voice_settings: Additional settings for TTS (rate, pitch, etc.)
        position: The position of the phrase in its section
        section_id: ID of the section this phrase belongs to
        metadata: Additional metadata for the phrase
    """
    text: str = Field(..., min_length=1)
    language: Language
    voice_id: Optional[str] = None
    voice_settings: Dict[str, Any] = Field(default_factory=dict)
    position: int = Field(1, ge=1)
    section_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('section_id', pre=True)
    def validate_section_id(cls, v: Any) -> Optional[str]:
        """Ensure section_id is always stored as a string."""
        if v is None:
            return None
        return str(v)
    
    @validator('text')
    def validate_text(cls, v: str) -> str:
        """Validate and normalize the phrase text."""
        return v.strip()
    
    @validator('voice_settings')
    def validate_voice_settings(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize voice settings."""
        settings = v.copy()
        
        # Normalize rate (0.5-2.0)
        if 'rate' in settings:
            rate = float(settings['rate'])
            settings['rate'] = max(0.5, min(2.0, rate))
            
        # Normalize pitch (-20 to 20)
        if 'pitch' in settings:
            pitch = float(settings['pitch'])
            settings['pitch'] = max(-20, min(20, pitch))
            
        # Normalize volume (0-100)
        if 'volume' in settings:
            volume = float(settings['volume'])
            settings['volume'] = max(0, min(100, volume))
            
        return settings
    
    def get_voice_settings_with_defaults(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Get voice settings with defaults for any missing values."""
        settings = defaults.copy()
        settings.update(self.voice_settings)
        return settings
    
    def get_audio_filename(self, extension: str = 'mp3') -> str:
        """Generate a filename for the phrase's audio file."""
        return f"phrase_{self.id}.{extension.lstrip('.')}"
