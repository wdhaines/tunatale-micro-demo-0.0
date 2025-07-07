"""Enums for the TunaTale domain model."""
from enum import Enum, auto
from typing import Dict, Optional, Type, TypeVar

T = TypeVar('T', bound='AutoName')

class AutoName(Enum):
    """Enum that automatically generates values as lowercase names."""
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
        return name.lower()
    
    @classmethod
    def from_string(cls: Type[T], value: str) -> Optional[T]:
        """Get enum member from string value (case-insensitive)."""
        try:
            return cls[value.upper()]
        except KeyError:
            # Try to find by value if not found by name
            value_lower = value.lower()
            for member in cls:
                if member.value.lower() == value_lower:
                    return member
            return None

class Environment(AutoName):
    """Application environment types."""
    DEVELOPMENT = auto()
    TESTING = auto()
    STAGING = auto()
    PRODUCTION = auto()


class SectionType(AutoName):
    """Types of sections in a language lesson."""
    KEY_PHRASES = auto()
    NATURAL_SPEED = auto()
    SLOW_SPEED = auto()
    TRANSLATED = auto()

class VoiceGender(AutoName):
    """Gender of a voice."""
    MALE = auto()
    FEMALE = auto()
    NEUTRAL = auto()

class VoiceAge(AutoName):
    """Age group of a voice."""
    CHILD = auto()
    YOUNG_ADULT = auto()
    ADULT = auto()
    SENIOR = auto()

class Language(AutoName):
    """Supported languages."""
    TAGALOG = auto()
    ENGLISH = auto()
    SPANISH = auto()
    
    @property
    def code(self) -> str:
        """Get the language code (using 'fil' for Tagalog instead of 'tl')."""
        return {
            'tagalog': 'fil',
            'english': 'en',
            'spanish': 'es',
        }.get(self.value, 'en')
        
    @classmethod
    def from_string(cls: Type[T], value: str) -> Optional[T]:
        """Get enum member from string value (case-insensitive).
        
        Also accepts language codes 'en', 'fil', and 'es'.
        """
        # First try the standard lookup
        result = super().from_string(value)
        if result is not None:
            return result
            
        # Then try matching by language code
        code_map = {
            'en': 'english',
            'fil': 'tagalog',
            'es': 'spanish'
        }
        
        normalized_value = value.lower()
        if normalized_value in code_map:
            return super().from_string(code_map[normalized_value])
            
        return None
