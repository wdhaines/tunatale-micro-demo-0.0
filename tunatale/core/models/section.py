"""Section model for organizing phrases in a lesson."""
from __future__ import annotations

from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pydantic import Field, validator

from .base import BaseEntity
from .enums import SectionType

if TYPE_CHECKING:
    from .phrase import Phrase


class Section(BaseEntity):
    """Represents a section within a language learning lesson.
    
    Attributes:
        title: The title of the section
        section_type: The type of section (e.g., KEY_PHRASES, NATURAL_SPEED)
        lesson_id: ID of the lesson this section belongs to
        position: The position of the section in the lesson
        phrases: List of phrases in this section
        settings: Section-specific settings
    """
    title: str = Field(..., min_length=1, max_length=200)
    section_type: SectionType
    lesson_id: str
    position: int = Field(1, ge=1)
    phrases: List['Phrase'] = Field(default_factory=list)
    settings: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('lesson_id', pre=True)
    def validate_lesson_id(cls, v: Any) -> str:
        """Ensure lesson_id is always stored as a string."""
        if v is None:
            raise ValueError("lesson_id is required")
        return str(v)
    
    @validator('title')
    def validate_title(cls, v: str) -> str:
        """Validate and normalize the section title."""
        return v.strip()
    
    @validator('phrases', each_item=True)
    def validate_phrases(cls, v: 'Phrase', values: Dict[str, Any]) -> 'Phrase':
        """Ensure phrases have the correct section_id."""
        if 'id' in values:
            v.section_id = values['id']
        return v
    
    def add_phrase(self, phrase: Phrase) -> None:
        """Add a phrase to this section."""
        phrase.section_id = str(self.id)
        phrase.position = len(self.phrases) + 1
        self.phrases.append(phrase)
    
    def get_phrase(self, phrase_id: str) -> Optional[Phrase]:
        """Get a phrase by its ID."""
        for phrase in self.phrases:
            if str(phrase.id) == str(phrase_id):
                return phrase
        return None
    
    def reorder_phrases(self, new_order: List[str]) -> None:
        """Reorder phrases based on a list of phrase IDs."""
        # Create a mapping of phrase IDs to phrases
        phrase_map = {str(p.id): p for p in self.phrases}
        
        # Rebuild phrases list in new order
        new_phrases = []
        for idx, phrase_id in enumerate(new_order, 1):
            if phrase_id in phrase_map:
                phrase = phrase_map[phrase_id]
                phrase.position = idx
                new_phrases.append(phrase)
        
        # Add any phrases that weren't in the new order
        for phrase in self.phrases:
            if str(phrase.id) not in new_order:
                phrase.position = len(new_phrases) + 1
                new_phrases.append(phrase)
        
        self.phrases = new_phrases
