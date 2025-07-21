"""Interface for voice selection components."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from tunatale.core.models.voice import Voice
from tunatale.core.models.enums import Language


class VoiceSelector(ABC):
    """Interface for selecting appropriate voices for text-to-speech.
    
    Voice selectors are responsible for choosing the most appropriate voice
    for a given piece of text based on language, gender, and other criteria.
    """
    
    @abstractmethod
    async def get_voice(
        self,
        language: Language,
        gender: Optional[str] = None,
        **kwargs: Any
    ) -> Optional[Voice]:
        """Get the most appropriate voice for the given parameters.
        
        Args:
            language: The language of the text to be spoken
            gender: Preferred gender of the voice (e.g., 'male', 'female')
            **kwargs: Additional parameters for voice selection
            
        Returns:
            A Voice object if a matching voice is found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_voices(
        self,
        language: Optional[Language] = None,
        gender: Optional[str] = None,
        **kwargs: Any
    ) -> List[Voice]:
        """Get all available voices matching the given criteria.
        
        Args:
            language: Filter voices by language
            gender: Filter voices by gender
            **kwargs: Additional filters
            
        Returns:
            A list of Voice objects matching the criteria
        """
        pass
    
    @abstractmethod
    async def get_default_voice(self, language: Language) -> Optional[Voice]:
        """Get the default voice for a given language.
        
        Args:
            language: The language to get the default voice for
            
        Returns:
            The default Voice for the language, or None if none found
        """
        pass
