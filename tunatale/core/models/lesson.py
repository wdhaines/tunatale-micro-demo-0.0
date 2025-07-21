"""Lesson model for language learning content."""
from __future__ import annotations

from typing import List, Dict, Optional, Any, TYPE_CHECKING
from datetime import datetime
from pydantic import Field, validator, HttpUrl

from .base import BaseEntity
from .enums import Language, SectionType

if TYPE_CHECKING:
    from .section import Section


class Lesson(BaseEntity):
    """Represents a complete language learning lesson.
    
    Attributes:
        title: The title of the lesson
        description: A brief description of the lesson
        target_language: The language being taught
        native_language: The learner's native language (for translations)
        difficulty: Difficulty level (1-5)
        estimated_duration: Estimated duration in minutes
        sections: List of sections in this lesson
        tags: List of tags for categorization
        is_published: Whether the lesson is published and available
        metadata: Additional metadata for the lesson
    """
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=1000)
    target_language: Language
    native_language: Language = Language.ENGLISH
    difficulty: int = Field(1, ge=1, le=5)
    estimated_duration: int = Field(30, ge=1, description="Duration in minutes")
    sections: List['Section'] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    is_published: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('title')
    def validate_title(cls, v: str) -> str:
        """Validate and normalize the lesson title."""
        return v.strip()
    
    @validator('sections', each_item=True)
    def validate_sections(cls, v: 'Section', values: Dict[str, Any]) -> 'Section':
        """Ensure sections have the correct lesson_id."""
        if 'id' in values:
            v.lesson_id = values['id']
        return v
    
    @property
    def total_phrases(self) -> int:
        """Get the total number of phrases in the lesson."""
        return sum(len(section.phrases) for section in self.sections)
    
    def get_section(self, section_id: str) -> Optional[Section]:
        """Get a section by its ID."""
        for section in self.sections:
            if str(section.id) == str(section_id):
                return section
        return None
    
    def get_section_by_type(self, section_type: SectionType) -> Optional[Section]:
        """Get the first section of a specific type."""
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None
    
    def add_section(self, section: Section) -> None:
        """Add a section to the lesson."""
        section.lesson_id = str(self.id)
        section.position = len(self.sections) + 1
        self.sections.append(section)
    
    def reorder_sections(self, new_order: List[str]) -> None:
        """Reorder sections based on a list of section IDs."""
        # Create a mapping of section IDs to sections
        section_map = {str(s.id): s for s in self.sections}
        
        # Rebuild sections list in new order
        new_sections = []
        for idx, section_id in enumerate(new_order, 1):
            if section_id in section_map:
                section = section_map[section_id]
                section.position = idx
                new_sections.append(section)
        
        # Add any sections that weren't in the new order
        for section in self.sections:
            if str(section.id) not in new_order:
                section.position = len(new_sections) + 1
                new_sections.append(section)
        
        self.sections = new_sections
    
    def get_all_phrases(self) -> List[tuple[Section, Any]]:
        """Get all phrases in the lesson with their sections."""
        return [
            (section, phrase)
            for section in sorted(self.sections, key=lambda s: s.position)
            for phrase in sorted(section.phrases, key=lambda p: p.position)
        ]
