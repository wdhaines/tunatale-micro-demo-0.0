"""Default implementation of WordSelector for basic word/phrase selection."""
from typing import Any, Dict, List, Optional, Tuple

from tunatale.core.models.phrase import Phrase
from tunatale.core.ports.word_selector import WordSelector


class DefaultWordSelector(WordSelector):
    """Default implementation of WordSelector with basic word selection logic."""

    async def select_words(
        self,
        phrases: List[Phrase],
        language: str,
        **kwargs: Any
    ) -> List[Phrase]:
        """Select words or phrases from the given list.
        
        This basic implementation returns all phrases. In a real application,
        this could implement more sophisticated selection logic.
        
        Args:
            phrases: List of phrases to select from
            language: The language of the phrases
            **kwargs: Additional parameters for word selection
            
        Returns:
            A filtered list of phrases
        """
        # For now, just return all phrases
        return phrases

    async def rank_words(
        self,
        phrases: List[Phrase],
        language: str,
        **kwargs: Any
    ) -> List[Tuple[Phrase, float]]:
        """Rank words or phrases by relevance or importance.
        
        This basic implementation assigns a default score of 1.0 to all phrases.
        
        Args:
            phrases: List of phrases to rank
            language: The language of the phrases
            **kwargs: Additional parameters for ranking
            
        Returns:
            A list of (phrase, score) tuples, sorted by score in descending order
        """
        # Assign a default score of 1.0 to all phrases
        return [(phrase, 1.0) for phrase in phrases]

    async def filter_by_difficulty(
        self,
        phrases: List[Phrase],
        difficulty: str,
        **kwargs: Any
    ) -> List[Phrase]:
        """Filter phrases by difficulty level.
        
        This basic implementation returns all phrases regardless of difficulty.
        
        Args:
            phrases: List of phrases to filter
            difficulty: Target difficulty level (e.g., 'beginner', 'intermediate', 'advanced')
            **kwargs: Additional parameters for difficulty filtering
            
        Returns:
            A filtered list of phrases matching the difficulty level
        """
        # For now, return all phrases regardless of difficulty
        return phrases
