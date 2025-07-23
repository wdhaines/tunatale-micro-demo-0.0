"""Text preprocessing utilities for TTS systems.

This module provides functions to preprocess text before sending it to TTS services
to improve pronunciation and handle special cases like abbreviations and language-specific
pronunciation issues.
"""
import re
import logging
from typing import Dict, Pattern, Union

# Set up logging
logger = logging.getLogger(__name__)

def fix_abbreviation_pronunciation(text: str) -> str:
    """Add dots to abbreviations for proper TTS pronunciation.
    
    Args:
        text: Input text potentially containing abbreviations
        
    Returns:
        Text with abbreviations formatted for better TTS pronunciation
    """
    if not text:
        return text
        
    original_text = text
    # Dictionary of abbreviations and their TTS-friendly versions
    # Keys are regex patterns, values are replacements
    abbreviation_fixes: Dict[Union[str, Pattern], str] = {
        # Try phonetic respelling instead of periods
        r'\bCR\b': 'see are',        # Convenience Room
        r'\bID\b': 'eye dee',        # Identification
        
        # Alternative: Try with hyphens
        # r'\bCR\b': 'C-R',          # Convenience Room
        # r'\bID\b': 'I-D',          # Identification
    }
    
    for pattern, replacement in abbreviation_fixes.items():
        new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if new_text != text:
            logger.debug(f"Abbreviation fix: '{text}' -> '{new_text}'")
            text = new_text
    
    if original_text != text:
        logger.debug(f"Abbreviation preprocessing complete: '{original_text}' -> '{text}'")
    return text

def preprocess_tagalog_for_tts(text: str) -> str:
    """Fix common Tagalog TTS interpretation issues.
    
    Args:
        text: Input Tagalog text
        
    Returns:
        Text with common Tagalog pronunciation issues fixed
    """
    # Common syllables/words that are misread by TTS
    tagalog_fixes: Dict[Union[str, Pattern], str] = {
        # Word boundaries ensure we don't replace parts of other words
        r'\bito\b': 'ee-toh',           # "ito" (this) is often misread as "I" + "2"
        r'\bIto\b': 'Ee-toh',           # Capitalized version
        r'\bITO\b': 'EE-TOH',           # All caps version
    }
    
    for pattern, replacement in tagalog_fixes.items():
        # Use re.IGNORECASE flag for case-insensitive matching
        text = re.sub(pattern, replacement, text)
    
    return text

def preprocess_text_for_tts(text: str, language_code: str) -> str:
    """Preprocess text for TTS based on language.
    
    Applies language-specific preprocessing to improve TTS output quality.
    
    Args:
        text: The text to preprocess
        language_code: BCP-47 language code (e.g., 'en-US', 'fil-PH')
        
    Returns:
        Preprocessed text ready for TTS
    """
    if not text or not isinstance(text, str):
        return text
    
    logger.debug(f"Preprocessing text for language '{language_code}': '{text}'")
    
    # Always apply abbreviation fixes (language-agnostic)
    text = fix_abbreviation_pronunciation(text)
    
    # Apply language-specific preprocessing
    if language_code.lower().startswith('fil'):  # Filipino/Tagalog
        text = preprocess_tagalog_for_tts(text)
    
    logger.debug(f"Preprocessed text: '{text}'")
    return text
