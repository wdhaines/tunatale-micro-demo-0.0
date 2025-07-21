"""Core models for the TunaTale application."""
from typing import TYPE_CHECKING

# Import base classes and enums first to avoid circular imports
from .base import BaseEntity
from .enums import Language, SectionType

# Import models that don't have circular dependencies
from .voice import Voice
from .phrase import Phrase
from .audio_config import AudioConfig
from .processed_models import ProcessedPhrase, ProcessedSection, ProcessedLesson

# Import models with circular dependencies
from .section import Section
from .lesson import Lesson

# Rebuild models to handle forward references and circular dependencies
if not TYPE_CHECKING:
    # This ensures that model_rebuild() is only called at runtime, not during type checking
    from pydantic import model_validator
    
    # Rebuild models with forward references
    Section.model_rebuild()
    Lesson.model_rebuild()

# Make these available when importing from tunatale.core.models
__all__ = [
    'BaseEntity',
    'Language',
    'SectionType',
    'Voice',
    'Phrase',
    'Section',
    'Lesson',
    'ProcessedPhrase',
    'ProcessedSection',
    'ProcessedLesson',
    'AudioConfig',
]