"""Voice model for TTS voices."""

import hashlib
from typing import Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field, PrivateAttr, field_validator, validator
from datetime import datetime
from uuid import UUID, uuid4
from .enums import VoiceGender, VoiceAge, Language


class Voice(BaseModel):
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
    # ID can be any string from TTS services
    id: str = Field(..., min_length=1, max_length=100, description="Unique identifier for the voice")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    name: str = Field(..., min_length=1, max_length=100)
    provider: str = Field(..., min_length=1, max_length=50)
    provider_id: str = Field(..., min_length=1, max_length=100)
    language: Language = Field(default=Language.ENGLISH, description="The language of the voice")
    gender: VoiceGender = Field(default=VoiceGender.NEUTRAL, description="The gender of the voice")
    age: VoiceAge = Field(default=VoiceAge.ADULT, description="The age group of the voice")
    sample_rate: int = Field(24000, ge=8000, le=48000)
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Store the generated UUID for backward compatibility
    _uuid: Optional[UUID] = PrivateAttr(default=None)
    
    def __init__(self, **data):
        """Initialize the Voice with proper ID handling."""
        # If id is not provided, use provider_id as the ID
        if 'id' not in data and 'provider_id' in data:
            data['id'] = data['provider_id']
            
        super().__init__(**data)
        
        # Generate a deterministic UUID from the string ID for backward compatibility
        if self.id and not self._uuid:
            self._generate_uuid()
    
    def _generate_uuid(self) -> None:
        """Generate a deterministic UUID from the string ID."""
        if not self.id:
            return
            
        # Create a deterministic UUID from the string ID
        hash_obj = hashlib.md5(self.id.encode('utf-8'))
        self._uuid = UUID(bytes=hash_obj.digest())
    
    @property
    def uuid(self) -> UUID:
        """Get the UUID for this voice (for backward compatibility)."""
        if not self._uuid:
            self._generate_uuid()
        return self._uuid
    
    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> 'Voice':
        """Override model_validate to handle string IDs."""
        if isinstance(obj, dict) and 'id' not in obj and 'provider_id' in obj:
            obj['id'] = obj['provider_id']
        return super().model_validate(obj, **kwargs)
    

    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and normalize the voice name."""
        if not v or not isinstance(v, str):
            raise ValueError("Name must be a non-empty string")
        return v.strip()
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate and normalize the provider name."""
        if not v or not isinstance(v, str):
            raise ValueError("Provider must be a non-empty string")
        return v.lower().strip()
    
    @field_validator('language', mode='before')
    @classmethod
    def validate_language(cls, v):
        """Ensure language is a valid Language enum or code."""
        from .enums import Language
        
        if v is None:
            return Language.ENGLISH  # Default to English if not specified
            
        if isinstance(v, str):
            try:
                return Language(v)
            except ValueError:
                # If it's not a valid enum value, try to find by code
                for lang in Language:
                    if lang.value.lower() == v.lower():
                        return lang
                return Language.ENGLISH  # Default to English if not found
        return v
            
        if isinstance(v, Language):
            return v
            
        if isinstance(v, str):
            # Try to get the language from the string (handles 'en', 'fil', etc.)
            lang = Language.from_string(v)
            if lang is not None:
                return lang
                
        # If we get here, the value is invalid
        raise ValueError(f"Invalid language: {v}. Must be a valid Language enum or code.")
        
    @field_validator('gender', mode='before')
    @classmethod
    def validate_gender(cls, v):
        """Ensure gender is a valid VoiceGender enum or string."""
        from .enums import VoiceGender
        
        if v is None:
            return VoiceGender.NEUTRAL  # Default to NEUTRAL if not specified
            
        if isinstance(v, VoiceGender):
            return v
            
        if isinstance(v, str):
            try:
                return VoiceGender(v.upper())
            except ValueError:
                # If it's not a valid enum value, try to find a match
                for gender in VoiceGender:
                    if gender.value.lower() == v.lower():
                        return gender
                return VoiceGender.NEUTRAL  # Default to NEUTRAL if not found
                
        return v
    
    @field_validator('age', mode='before')
    @classmethod
    def validate_age(cls, v):
        """Ensure age is a valid VoiceAge enum or string."""
        from .enums import VoiceAge
        
        if v is None:
            return VoiceAge.ADULT  # Default to ADULT if not specified
            
        if isinstance(v, VoiceAge):
            return v
            
        if isinstance(v, str):
            try:
                return VoiceAge(v.upper())
            except ValueError:
                # If it's not a valid enum value, try to find a match
                for age in VoiceAge:
                    if age.value.lower() == v.lower():
                        return age
                return VoiceAge.ADULT  # Default to ADULT if not found
                
        return v
    
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
