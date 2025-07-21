"""Processed models for lesson processor results."""
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import Field
from uuid import UUID

from .base import BaseEntity
from .enums import Language


class ProcessedPhrase(BaseEntity):
    """Represents a processed phrase with audio file path.
    
    Attributes:
        id: Unique identifier for the phrase
        text: The original text of the phrase
        translation: Optional translation of the phrase
        language: The language of the phrase
        audio_file: Path to the generated audio file
        metadata: Additional metadata for the phrase
    """
    text: str
    translation: Optional[str] = None
    language: Language
    audio_file: Optional[Path] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessedSection(BaseEntity):
    """Represents a processed section with audio file path.
    
    Attributes:
        id: Unique identifier for the section
        title: Title of the section
        phrases: List of processed phrases in this section
        audio_file: Path to the generated audio file for this section
    """
    title: str
    phrases: List[ProcessedPhrase] = Field(default_factory=list)
    audio_file: Optional[Path] = None


class ProcessedLesson(BaseEntity):
    """Represents a processed lesson with audio files.
    
    Attributes:
        id: Unique identifier for the lesson
        title: Title of the lesson
        language: The target language of the lesson
        level: Difficulty level of the lesson
        sections: List of processed sections in this lesson
        audio_file: Path to the generated audio file for the entire lesson
        metadata_file: Path to the metadata file
        output_dir: Directory where all output files are stored
    """
    title: str
    language: Language
    level: str
    sections: List[ProcessedSection] = Field(default_factory=list)
    audio_file: Optional[Path] = None
    metadata_file: Optional[Path] = None
    output_dir: Optional[Path] = None
