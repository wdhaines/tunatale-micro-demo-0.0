"""Interface for word selection components."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from tunatale.core.models.phrase import Phrase


class WordSelector(ABC):
    """Interface for selecting words or phrases from a lesson.
    
    Word selectors are responsible for choosing which words or phrases
    from a lesson should be included in the final output, based on
    various criteria like difficulty, frequency, or relevance.
    """
    
    @abstractmethod
    async def select_words(
        self,
        phrases: List[Phrase],
        language: str,
        **kwargs: Any
    ) -> List[Phrase]:
        """Select words or phrases from the given list.
        
        Args:
            phrases: List of phrases to select from
            language: The language of the phrases
            **kwargs: Additional parameters for word selection
            
        Returns:
            A filtered list of phrases
        """
        pass
    
    @abstractmethod
    async def rank_words(
        self,
        phrases: List[Phrase],
        language: str,
        **kwargs: Any
    ) -> List[tuple[Phrase, float]]:
        """Rank words or phrases by relevance or importance.
        
        Args:
            phrases: List of phrases to rank
            language: The language of the phrases
            **kwargs: Additional parameters for ranking
            
        Returns:
            A list of (phrase, score) tuples, sorted by score in descending order
        """
        pass
    
    @abstractmethod
    async def filter_by_difficulty(
        self,
        phrases: List[Phrase],
        difficulty: str,
        **kwargs: Any
    ) -> List[Phrase]:
        """Filter phrases by difficulty level.
        
        Args:
            phrases: List of phrases to filter
            difficulty: Target difficulty level (e.g., 'beginner', 'intermediate', 'advanced')
            **kwargs: Additional parameters for difficulty filtering
            
        Returns:
            A filtered list of phrases matching the difficulty level
        """
        pass
