"""Text preprocessing utilities for TTS systems.

This module provides functions to preprocess text before sending it to TTS services
to improve pronunciation and handle special cases like abbreviations and language-specific
pronunciation issues.
"""
import re
import logging
from typing import Dict, Pattern, Union, Optional

# Set up logging
logger = logging.getLogger(__name__)

# =============================================================================
# TAGALOG SYLLABLE PHONETIC CORRECTIONS FOR KEY PHRASES
# =============================================================================
# These patterns fix TTS pronunciation of isolated Tagalog syllables that get
# mispronounced using English phonetics. Only applied in Key Phrases sections.

# Vowel-ending syllables (most problematic - TTS uses English vowel sounds)
VOWEL_SYLLABLE_PATTERNS = {
    # Single vowels - UPDATED BASED ON TESTING
    r'\bi\b': 'eey',          # i → eey (tested and chosen)
    # r'\bo\b': 'o',          # o → unchanged (natural sounds best)
    r'\ba\b': 'ah',           # a → ah (tested and chosen) 
    r'\be\b': 'eh',           # e → eh (tested and chosen)
    r'\bu\b': 'ooh',          # u → ooh (tested and chosen)
    
    # Consonant + i (very common in Tagalog) - using "eey" ending
    r'\bbi\b': 'beey',        # bi → beey (from gabi, etc.)
    r'\bti\b': 'teey',        # ti → teey
    r'\bmi\b': 'meey',        # mi → meey
    r'\bki\b': 'keey',        # ki → keey
    
    # Consonant + o (also very common)
    r'\bto\b': 'toh',         # to → toh (from kwarto, ito, etc.)
    r'\bdo\b': 'doh',         # do → doh
        
    # Consonant + e
    r'\ble\b': 'leh',         # le → leh
    r'\bme\b': 'meh',         # me → meh
    r'\bte\b': 'teh',         # te → teh
    r'\bse\b': 'seh',         # se → seh
    r'\bne\b': 'neh',         # ne → neh
    r'\bre\b': 'reh',         # re → reh
    r'\bke\b': 'keh',         # ke → keh
    r'\bde\b': 'deh',         # de → deh
    r'\bge\b': 'geh',         # ge → geh
    r'\bpe\b': 'peh',         # pe → peh
    
    # Consonant + u - using "ooh" ending (tested and chosen)
    r'\bmu\b': 'mooh',        # mu → mooh
    r'\bdu\b': 'dooh',        # du → dooh
    r'\bgu\b': 'gooh',        # gu → gooh
}

# Consonant-ending syllables (less problematic but still need fixes)
CONSONANT_SYLLABLE_PATTERNS = {
    # Common consonant endings that get mispronounced
}

# All syllable patterns combined for easy reference
ALL_SYLLABLE_PATTERNS = {
    **VOWEL_SYLLABLE_PATTERNS,
    **CONSONANT_SYLLABLE_PATTERNS
}

def fix_tagalog_syllables_for_key_phrases(text: str, section_type: Optional[str] = None, language_code: Optional[str] = None) -> str:
    """Fix TTS pronunciation of Tagalog syllables in Key Phrases sections.
    
    This function applies phonetic respelling to isolated Tagalog syllables that
    get mispronounced when TTS applies English phonetics. Only applies in 
    Key Phrases sections for Tagalog content to avoid false positives.
    
    Args:
        text: Input text potentially containing problematic syllables
        section_type: Type of section ('key_phrases', 'natural_speed', etc.)
        language_code: Language code (e.g., 'fil-PH', 'en-US')
        
    Returns:
        Text with syllables phonetically corrected for TTS
    """
    # Only apply to Key Phrases sections with Tagalog content
    if not text or section_type != 'key_phrases':
        return text
    
    # Only apply to Tagalog/Filipino content, not English narrator lines
    if not language_code or not language_code.lower().startswith('fil'):
        return text
        
    original_text = text
    
    # Apply vowel syllable fixes (most important)
    for pattern, replacement in VOWEL_SYLLABLE_PATTERNS.items():
        new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if new_text != text:
            logger.debug(f"Vowel syllable fix: '{text}' -> '{new_text}'")
            text = new_text
    
    # Apply consonant syllable fixes
    for pattern, replacement in CONSONANT_SYLLABLE_PATTERNS.items():
        new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if new_text != text:
            logger.debug(f"Consonant syllable fix: '{text}' -> '{new_text}'")
            text = new_text
    
    if original_text != text:
        logger.debug(f"Syllable preprocessing complete: '{original_text}' -> '{text}'")
    
    return text

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
    # This function is now primarily used for general Tagalog fixes
    # that aren't handled by the syllable patterns in Key Phrases sections.
    # Currently no additional fixes needed beyond syllable preprocessing.
    return text

def preprocess_text_for_tts(text: str, language_code: str, section_type: Optional[str] = None) -> str:
    """Preprocess text for TTS based on language and section context.
    
    Applies language-specific preprocessing to improve TTS output quality.
    
    Args:
        text: The text to preprocess
        language_code: BCP-47 language code (e.g., 'en-US', 'fil-PH')
        section_type: Type of section ('key_phrases', 'natural_speed', etc.)
        
    Returns:
        Preprocessed text ready for TTS
    """
    if not text or not isinstance(text, str):
        return text
    
    logger.debug(f"Preprocessing text for language '{language_code}', section '{section_type}': '{text}'")
    
    # Apply syllable fixes for Key Phrases sections (before other fixes)
    text = fix_tagalog_syllables_for_key_phrases(text, section_type, language_code)
    
    # Always apply abbreviation fixes (language-agnostic)
    text = fix_abbreviation_pronunciation(text)
    
    # Apply language-specific preprocessing
    if language_code.lower().startswith('fil'):  # Filipino/Tagalog
        text = preprocess_tagalog_for_tts(text)
    
    logger.debug(f"Preprocessed text: '{text}'")
    return text
