"""Text preprocessing utilities for TTS systems.

This module provides functions to preprocess text before sending it to TTS services
to improve pronunciation and handle special cases like abbreviations and language-specific
pronunciation issues. It includes a hybrid SSML system that supports both legacy ellipses
and direct SSML markup.
"""
import re
import logging
from typing import Dict, Pattern, Union, Optional, Tuple, List
from dataclasses import dataclass

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
    r'\ba\b': 'ah',           # a → ah (tested and chosen)
    r'\be\b': 'eh',           # e → eh (tested and chosen)
    r'\bi\b': 'eey',          # i → eey (tested and chosen)
    # r'\bo\b': 'o',          # o → unchanged (natural sounds best)
    r'\bu\b': 'ooh',          # u → ooh (tested and chosen)
    
    # Consonant + i (very common in Tagalog) - using "eey" ending
    r'\bbi\b': 'beey',        # bi → beey (from gabi, etc.)
    r'\bki\b': 'keey',        # ki → keey
    r'\bmi\b': 'meey',        # mi → meey
    r'\bti\b': 'teey',        # ti → teey
    
    # Consonant + o (also very common)
    r'\bdo\b': 'doh',         # do → doh
    r'\bto\b': 'toh',         # to → toh (from kwarto, ito, etc.)
        
    # Consonant + e
    r'\bde\b': 'deh',         # de → deh
    r'\bge\b': 'geh',         # ge → geh
    r'\bke\b': 'keh',         # ke → keh
    r'\ble\b': 'leh',         # le → leh
    r'\bme\b': 'meh',         # me → meh
    r'\bne\b': 'neh',         # ne → neh
    r'\bpe\b': 'peh',         # pe → peh
    r'\bre\b': 'reh',         # re → reh
    r'\bse\b': 'seh',         # se → seh
    r'\bte\b': 'teh',         # te → teh
    
    # Consonant + u - using "ooh" ending (tested and chosen)
    r'\bdu\b': 'dooh',        # du → dooh
    r'\bgu\b': 'gooh',        # gu → gooh
    r'\bmu\b': 'mooh',        # mu → mooh

    # Other words pronounced as English
    r'\bate\b': 'ahteh',      # ate → ahteh
    r'\bpit\b': 'peeyt',      # pit → peeyt
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

# =============================================================================
# UNIVERSAL ABBREVIATION HANDLER
# =============================================================================

# Phonetic pronunciation mapping for letters A-Z
LETTER_PHONETICS = {
    'A': 'eh', 'B': 'bee', 'C': 'see', 'D': 'dee', 'E': 'eey', 'F': 'eff',
    'G': 'gee', 'H': 'aych', 'I': 'eye', 'J': 'jay', 'K': 'kay', 'L': 'ell',
    'M': 'em', 'N': 'en', 'O': 'oh', 'P': 'pee', 'Q': 'cue', 'R': 'are',
    'S': 'ess', 'T': 'tee', 'U': 'you', 'V': 'vee', 'W': 'double-you', 
    'X': 'ex', 'Y': 'wyee', 'Z': 'zee'
}

# Protected words that should NOT be converted even if they're all caps
# These are common words that might appear in caps for emphasis
PROTECTED_WORDS = {
    'TO', 'IN', 'ON', 'AT', 'BY', 'FOR', 'WITH', 'FROM', 'UP', 'OUT', 'OFF',
    'PO', 'ANG', 'NG', 'SA', 'NA', 'KO', 'MO', 'SYA', 'KAMI', 'KAYO', 'SILA',  # Common Tagalog words
    'THE', 'AND', 'OR', 'BUT', 'SO', 'IF', 'AS', 'IS', 'IT', 'BE', 'DO', 'GO',
    'WE', 'HE', 'SHE', 'YOU', 'ME', 'HIM', 'HER', 'US', 'MY', 'HIS', 'OUR',
    'ARE', 'WAS', 'WERE', 'BEEN', 'HAVE', 'HAS', 'HAD', 'WILL', 'WOULD',
    'CAN', 'COULD', 'MAY', 'MIGHT', 'MUST', 'SHALL', 'SHOULD', 'OUGHT'
}

# Common English words that might look like abbreviations but aren't
COMMON_WORDS = {
    'ALL', 'ANY', 'BAD', 'BIG', 'BOY', 'CAR', 'DAY', 'END', 'FAR', 'GET',
    'GOD', 'HOW', 'JOB', 'LAW', 'LOT', 'MAN', 'NEW', 'NOW', 'OLD', 'OWN',
    'PAY', 'PUT', 'RUN', 'SAY', 'SEE', 'SIT', 'TRY', 'TWO', 'USE', 'WAY',
    'WHO', 'WIN', 'YES', 'YET', 'YOU', 'BAG', 'BED', 'BOX', 'BUS', 'CUT',
    'EAR', 'EYE', 'FUN', 'GUN', 'HAT', 'HOT', 'ICE', 'JOB', 'KEY', 'LEG',
    'MAP', 'NET', 'OIL', 'PEN', 'RED', 'SUN', 'TOP', 'VAN', 'WAR', 'WET',
    'CODE', 'DOOR', 'FOOD', 'GOOD', 'HELP', 'HOME', 'HOPE', 'KEEP', 'KNOW',
    'LAST', 'LEFT', 'LIFE', 'LIKE', 'LIVE', 'LONG', 'LOOK', 'MAKE', 'MOVE',
    'NAME', 'NEED', 'NEXT', 'ONCE', 'ONLY', 'OPEN', 'OVER', 'PART', 'PLAY',
    'READ', 'REAL', 'RIGHT', 'ROOM', 'SAME', 'SEEM', 'SHOW', 'SIDE', 'SOME',
    'TAKE', 'TELL', 'THAT', 'THEM', 'THEN', 'THEY', 'THINK', 'THIS', 'TIME',
    'TURN', 'VERY', 'WANT', 'WEEK', 'WELL', 'WERE', 'WHAT', 'WHEN', 'WHERE',
    'WHICH', 'WILL', 'WITH', 'WORD', 'WORK', 'YEAR', 'YOUR'
}


def _is_likely_abbreviation(word: str) -> bool:
    """Determine if a word is likely an abbreviation vs a real word.
    
    Args:
        word: The word to check (should be all caps)
        
    Returns:
        True if likely an abbreviation, False if likely a real word
    """
    # Check protected words first
    if word.upper() in PROTECTED_WORDS:
        return False
    
    # Check common English words
    if word.upper() in COMMON_WORDS:
        return False
    
    # Length-based heuristics
    length = len(word)
    
    # Single letters are usually abbreviations (except protected ones like 'I', 'A')
    if length == 1:
        return word.upper() not in {'I', 'A'}
    
    # 2-letter words: Most are likely abbreviations unless they're common words
    if length == 2:
        return True  # Common 2-letter words already filtered out above
    
    # 3-letter words: Check for vowel patterns
    # Real words usually have at least one vowel, abbreviations often don't
    if length == 3:
        vowels = set('AEIOU')
        has_vowel = any(c in vowels for c in word.upper())
        # If no vowels, likely an abbreviation (like 'BBC', 'CNN')
        # If has vowels but is not in common words, could be either - lean toward abbreviation
        return not has_vowel or word.upper() not in COMMON_WORDS
    
    # 4+ letter words: Check for common abbreviation patterns
    if length >= 4:
        vowels = set('AEIOU')
        vowel_count = sum(1 for c in word.upper() if c in vowels)
        vowel_ratio = vowel_count / length
        
        # Known abbreviation patterns that should be converted despite having vowels
        known_abbreviations = {'NASA', 'JPEG', 'HTML', 'HTTP', 'HTTPS', 'JSON', 'AJAX', 'SOAP'}
        if word.upper() in known_abbreviations:
            return True
        
        # If very few vowels relative to length, likely abbreviation
        if vowel_ratio < 0.3:  # Less than 30% vowels suggests abbreviation
            return True
        
        # For 4-letter words specifically, be more aggressive
        if length == 4 and vowel_ratio <= 0.5:  # 50% or less vowels for 4-letter words
            return True
        
        return False
    
    return False


def fix_abbreviation_pronunciation(text: str) -> str:
    """Universal abbreviation handler that converts abbreviations to phonetic pronunciation.
    
    Automatically detects abbreviations (1-6 capital letters as word boundaries) and 
    converts each letter to its phonetic pronunciation (A→"ay", B→"bee", etc.).
    
    Args:
        text: Input text potentially containing abbreviations
        
    Returns:
        Text with abbreviations converted to phonetic pronunciation
    """
    if not text:
        return text
        
    original_text = text
    
    # Pattern to match potential abbreviations: 1-6 capital letters as word boundaries
    # Use word boundaries to avoid matching parts of longer words
    abbreviation_pattern = r'\b[A-Z]{1,6}\b'
    
    def replace_abbreviation(match):
        """Replace a matched abbreviation with phonetic pronunciation."""
        abbrev = match.group(0)
        
        # Check if this is likely an abbreviation vs a real word
        if not _is_likely_abbreviation(abbrev):
            logger.debug(f"Skipping protected/common word: '{abbrev}'")
            return abbrev  # Keep original
        
        # Convert each letter to phonetic pronunciation
        phonetic_parts = []
        for letter in abbrev.upper():
            if letter in LETTER_PHONETICS:
                phonetic_parts.append(LETTER_PHONETICS[letter])
            else:
                # Fallback for unexpected characters
                phonetic_parts.append(letter.lower())
        
        phonetic_result = ' '.join(phonetic_parts)
        logger.debug(f"Abbreviation conversion: '{abbrev}' -> '{phonetic_result}'")
        return phonetic_result
    
    # Apply the universal abbreviation conversion
    processed_text = re.sub(abbreviation_pattern, replace_abbreviation, text)
    
    if original_text != processed_text:
        logger.debug(f"Universal abbreviation processing complete: '{original_text}' -> '{processed_text}'")
    
    return processed_text

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

# =============================================================================
# FILIPINO NUMBER CLARIFICATION SYSTEM
# =============================================================================

# Spanish time system (authentic Filipino usage)
SPANISH_NUMBERS = {
    1: 'una', 2: 'dos', 3: 'tres', 4: 'kuwatro', 5: 'singko', 
    6: 'seys', 7: 'syete', 8: 'otso', 9: 'nuwebe', 10: 'diyes',
    11: 'onse', 12: 'dose', 20: 'bente', 30: 'treinta', 40: 'kuwarenta', 50: 'singkuwenta'
}

# Tagalog digits for breakdown
TAGALOG_DIGITS = {
    '0': 'sero', '1': 'isa', '2': 'dalawa', '3': 'tatlo', '4': 'apat',
    '5': 'lima', '6': 'anim', '7': 'pito', '8': 'walo', '9': 'siyam'
}

def convert_time_to_spanish(hour: int, minute: int) -> str:
    """Convert time to Filipino Spanish format with pauses for better comprehension.
    
    Args:
        hour: Hour (1-12 or 0-23)
        minute: Minute (0-59)
        
    Returns:
        Spanish time expression with pauses between components
    """
    # Convert 24-hour to 12-hour format
    display_hour = hour if hour <= 12 else hour - 12
    if display_hour == 0:
        display_hour = 12
    
    # Get Spanish hour
    if display_hour in SPANISH_NUMBERS:
        spanish_hour = SPANISH_NUMBERS[display_hour]
    else:
        # Fallback for invalid hours
        spanish_hour = str(display_hour)
    
    # Handle minutes with pauses between components
    if minute == 0:
        return f"... alas... {spanish_hour}."
    elif minute == 15:
        return f"... alas... {spanish_hour}... kinse."
    elif minute == 30:
        return f"... alas... {spanish_hour}... y medya."
    elif minute == 45:
        next_hour = display_hour + 1 if display_hour < 12 else 1
        next_spanish = SPANISH_NUMBERS.get(next_hour, str(next_hour))
        return f"... kinse... para... alas... {next_spanish}."
    else:
        # For other minutes, break apart based on minute value
        if minute < 10:
            # Single digit minutes: 8:05 -> ... alas... otso... singko
            minute_word = SPANISH_NUMBERS.get(minute, str(minute))
            return f"... alas... {spanish_hour}... {minute_word}."
        elif minute % 10 == 0:
            # Round tens: 8:20 -> ... alas... otso... bente
            minute_word = SPANISH_NUMBERS.get(minute, str(minute))
            return f"... alas... {spanish_hour}... {minute_word}."
        else:
            # Compound minutes: 8:35 -> ... alas... otso... treinta't... singko
            tens = (minute // 10) * 10
            ones = minute % 10
            tens_word = SPANISH_NUMBERS.get(tens, str(tens))
            ones_word = SPANISH_NUMBERS.get(ones, str(ones))
            return f"... alas... {spanish_hour}... {tens_word}'t... {ones_word}."

def convert_digits_to_tagalog(number_str: str) -> str:
    """Convert digits to Tagalog pronunciation with pauses between digits.
    
    Args:
        number_str: String of digits (e.g., "150", "203", "5")
        
    Returns:
        Tagalog digit breakdown with pauses
    """
    # Remove any non-digit characters for processing
    clean_digits = ''.join(c for c in number_str if c.isdigit())
    
    if not clean_digits:
        return number_str
    
    # Convert each digit individually
    tagalog_parts = []
    for digit in clean_digits:
        if digit in TAGALOG_DIGITS:
            tagalog_parts.append(TAGALOG_DIGITS[digit])
        else:
            tagalog_parts.append(digit)  # Fallback
    
    # Join with pauses between digits
    return '... '.join(tagalog_parts) + '.'

def clarify_number(number_match: str, context: str = '', force_clarify: bool = False) -> str:
    """Generate clarification for a number based on context.
    
    Args:
        number_match: The number string to clarify
        context: Surrounding context to determine clarification type
        force_clarify: If True, clarify even small numbers (for clarify tags)
        
    Returns:
        Clarified number string
    """
    # Time pattern: HH:MM or H:MM
    time_pattern = r'^(\d{1,2}):(\d{2})$'
    time_match = re.match(time_pattern, number_match)
    
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        spanish_time = convert_time_to_spanish(hour, minute)
        return f"{number_match}. {spanish_time}"
    
    # Phone number pattern: XXX-XXXX or similar
    if '-' in number_match and len(number_match.replace('-', '')) >= 4:
        parts = number_match.split('-')
        tagalog_parts = []
        for part in parts:
            if part.isdigit():
                tagalog_parts.append(convert_digits_to_tagalog(part).rstrip('.'))
        return f"{number_match}... {', '.join(tagalog_parts)}."
    
    # Regular number - check clarification rules
    clean_number = ''.join(c for c in number_match if c.isdigit())
    
    # Always clarify if force_clarify is True (from clarify tags)
    if force_clarify:
        tagalog_digits = convert_digits_to_tagalog(number_match)
        return f"{number_match}... {tagalog_digits}"
    
    # Check for room/kwarto context - these should always be clarified
    if any(keyword in context.lower() for keyword in ['room', 'kwarto', 'numero']):
        tagalog_digits = convert_digits_to_tagalog(number_match)
        return f"{number_match}... {tagalog_digits}"
    
    # Large numbers (3+ digits) get digit breakdown
    if clean_number.isdigit() and len(clean_number) >= 3:
        tagalog_digits = convert_digits_to_tagalog(number_match)
        return f"{number_match}... {tagalog_digits}"
    
    # Small numbers (1-99) don't get clarified by default
    if clean_number.isdigit() and 1 <= int(clean_number) <= 99:
        return number_match
    
    # Default: no clarification
    return number_match

def process_number_clarification(text: str, section_type: Optional[str] = None) -> str:
    """Process number clarification based on section type and clarify tags.
    
    Args:
        text: Input text that may contain numbers and clarify tags
        section_type: Type of section ('slow_speed', 'natural_speed', 'key_phrases', etc.)
        
    Returns:
        Text with appropriate number clarifications applied
    """
    if not text:
        return text
    
    original_text = text
    processed_text = text
    
    # First, handle clarify tags (highest priority)
    clarify_pattern = r'<clarify>(.*?)</clarify>'
    
    def process_clarify_tag(match):
        content = match.group(1)
        logger.debug(f"Processing clarify tag content: '{content}'")
        
        # Simple approach: try patterns in order of specificity and stop after first match
        clarify_patterns = [
            (r'\b\d{1,2}:\d{2}\b', 'time'),           # Time: 8:30, 12:45 (most specific)
            (r'\b\d{3}-\d{3}-\d{4}\b', 'phone'),      # Full phone: 123-456-7890
            (r'\b\d{3,4}-\d{4}\b', 'phone'),          # Phone: 555-1234
            (r'\bRoom\s+\d+\b', 'room'),              # Room numbers: Room 203
            (r'\bKwarto\s+\d+\b', 'room'),            # Filipino room: Kwarto 203
            (r'\b\d+\b', 'number'),                   # Any number (including small ones) - least specific
        ]
        
        # Find the BEST match (most specific) and process only that one
        best_match = None
        for pattern, pattern_type in clarify_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            if matches:
                # Take the first match of this type since it's the most specific pattern that matched
                match_obj = matches[0]
                best_match = (match_obj.start(), match_obj.end(), match_obj.group(0), pattern_type)
                break  # Stop at first matching pattern type (most specific)
        
        # Process the best match if found
        if best_match:
            start, end, number_str, pattern_type = best_match
            context = content  # Use the original clarify tag content as context
            clarified = clarify_number(number_str, context, force_clarify=True)
            
            # Replace in the content
            processed_content = content[:start] + clarified + content[end:]
            logger.debug(f"Clarify tag clarification ({pattern_type}): '{number_str}' → '{clarified}'")
            return processed_content
        else:
            # No number patterns found, return as-is
            return content
    
    # Process clarify tags and remove them
    processed_text = re.sub(clarify_pattern, process_clarify_tag, processed_text, flags=re.IGNORECASE)
    
    # Then handle slow_speed sections for numbers NOT in clarify tags
    if section_type == 'slow_speed':
        logger.debug("Processing slow_speed section: clarifying remaining numbers")
        
        # Pattern to match numbers in various formats (for slow speed)
        slow_speed_patterns = [
            r'\b\d{1,2}:\d{2}\b',          # Time: 8:30, 12:45
            r'\b\d{3,4}-\d{4}\b',          # Phone: 555-1234
            r'\b\d{3}-\d{3}-\d{4}\b',      # Full phone: 123-456-7890
            r'\bRoom\s+\d+\b',             # Room numbers: Room 203
            r'\bKwarto\s+\d+\b',           # Filipino room: Kwarto 203
            r'\b\d{2,}\b',                 # Numbers over 10: 11, 25, 150, 1205
        ]
        
        # Find all matches first to avoid overlap issues
        all_slow_matches = []
        for pattern in slow_speed_patterns:
            for match in re.finditer(pattern, processed_text, re.IGNORECASE):
                all_slow_matches.append((match.start(), match.end(), match.group(0)))
        
        # Sort by start position and remove overlaps (prefer longer/more specific matches)
        all_slow_matches.sort()
        non_overlapping_slow = []
        for start, end, text in all_slow_matches:
            # Check if this overlaps with any existing match
            overlaps = False
            for existing_start, existing_end, _ in non_overlapping_slow:
                if not (end <= existing_start or start >= existing_end):
                    # There's an overlap - prefer the longer match
                    existing_length = existing_end - existing_start
                    current_length = end - start
                    if current_length > existing_length:
                        # Remove the shorter existing match
                        non_overlapping_slow = [(s, e, t) for s, e, t in non_overlapping_slow 
                                              if not (s == existing_start and e == existing_end)]
                    else:
                        # Skip the current shorter match
                        overlaps = True
                        break
            if not overlaps:
                non_overlapping_slow.append((start, end, text))
        
        # Process matches from end to start to preserve indices
        for start, end, number_str in reversed(non_overlapping_slow):
            # Skip if this exact text has already been clarified (more precise check)
            # Look for clarification markers immediately after this number
            text_after = processed_text[end:end+50] if end < len(processed_text) else ""
            if text_after.startswith('. ') and ('alas' in text_after.lower() or 
                any(tagalog in text_after.lower() for tagalog in ['isa', 'dalawa', 'tatlo', 'lima'])):
                continue
            
            context = processed_text[max(0, start-20):end+20]  # Get surrounding context  
            # For slow speed, force clarification of numbers over 10
            force_slow_speed = len(number_str.replace('-', '').replace(':', '')) >= 2 and number_str.replace('-', '').replace(':', '').isdigit()
            clarified = clarify_number(number_str, context, force_clarify=force_slow_speed)
            
            # Only replace if clarification was applied
            if clarified != number_str:
                processed_text = processed_text[:start] + clarified + processed_text[end:]
                logger.debug(f"Slow speed clarification: '{number_str}' → '{clarified}'")
    
    # Final cleanup: remove any remaining clarify tags
    processed_text = re.sub(r'</?clarify>', '', processed_text, flags=re.IGNORECASE)
    
    if original_text != processed_text:
        logger.debug(f"Number clarification complete: '{original_text}' → '{processed_text}'")
    
    return processed_text

def preprocess_text_for_tts(text: str, language_code: str, section_type: Optional[str] = None) -> str:
    """Preprocess text for TTS based on language and section context.
    
    Applies language-specific preprocessing to improve TTS output quality.
    
    Args:
        text: The text to preprocess
        language_code: BCP-47 language code (e.g., 'en-US', 'fil-PH')
        section_type: Type of section ('key_phrases', 'natural_speed', etc.)
        
    Returns:
        Preprocessed text ready for TTS (empty string if text is None)
    """
    if text is None:
        return ""
        
    if not text or not isinstance(text, str):
        return text
    
    logger.debug(f"Preprocessing text for language '{language_code}', section '{section_type}': '{text}'")
    
    # Apply syllable fixes for Key Phrases sections (before other fixes)
    text = fix_tagalog_syllables_for_key_phrases(text, section_type, language_code)
    
    # Apply number clarification (before abbreviation fixes)
    text = process_number_clarification(text, section_type)
    
    # Apply language-specific preprocessing
    if language_code.lower().startswith('fil'):  # Filipino/Tagalog
        # Apply abbreviation fixes for Tagalog TTS only
        text = fix_abbreviation_pronunciation(text)
        text = preprocess_tagalog_for_tts(text)
    
    logger.debug(f"Preprocessed text: '{text}'")
    return text


# =============================================================================
# HYBRID SSML SYSTEM
# =============================================================================

@dataclass
class ProsodySettings:
    """Prosody settings extracted from SSML."""
    rate: Optional[float] = None  # 0.5 = 50% speed, 2.0 = 200% speed
    pitch: Optional[float] = None  # -50 = -50Hz, +100 = +100Hz
    volume: Optional[float] = None  # 0.5 = 50% volume, 2.0 = 200% volume
    emphasis: Optional[str] = None  # 'strong', 'moderate', 'reduced'


@dataclass
class SSMLProcessingResult:
    """Result of SSML processing with metadata."""
    processed_text: str
    has_ellipses: bool
    has_ssml_markup: bool
    conversion_applied: bool
    original_format: str  # 'ellipses', 'ssml', 'mixed', 'plain'
    prosody: Optional[ProsodySettings] = None


# Ellipses to SSML timing mapping
ELLIPSES_TO_SSML_MAPPING = {
    # Note: Single '...' excluded - handled separately by direct conversion
    '....': '<break time="0.5s"/>',     # Short pause (0.5 seconds)
    '.....': '<break time="0.75s"/>',   # Medium pause (0.75 seconds)
    '......': '<break time="1s"/>',     # Medium-long pause (1 second)
    '.......': '<break time="1.25s"/>', # Long pause (1.25 seconds)
    '........': '<break time="1.5s"/>', # Extra long pause (1.5 seconds)
    '.........': '<break time="1.75s"/>',# Very long pause (1.75 seconds)
    '..........': '<break time="2s"/>',  # Very long pause (2 seconds)
    '...........': '<break time="2.25s"/>',# Very long pause (2.25 seconds)
    '............': '<break time="2.5s"/>',# Extra long pause (2.5 seconds)
    '.............': '<break time="2.75s"/>',# Maximum pause (2.75 seconds) - cap at 13 dots
}


def detect_content_format(text: str) -> str:
    """Detect the format of the input text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Format type: 'ellipses', 'ssml', 'mixed', or 'plain'
    """
    if not text:
        return 'plain'
    
    has_ellipses = bool(re.search(r'\.{3,}', text))
    has_ssml = bool(re.search(r'<[^>]+>', text))
    
    if has_ellipses and has_ssml:
        return 'mixed'
    elif has_ellipses:
        return 'ellipses'
    elif has_ssml:
        return 'ssml'
    else:
        return 'plain'


def convert_ellipses_to_ssml(text: str) -> Tuple[str, bool]:
    """Convert ellipses patterns to SSML break tags.
    
    Args:
        text: Input text containing ellipses patterns
        
    Returns:
        Tuple of (converted_text, conversion_applied)
    """
    if not text or not re.search(r'\.{3,}', text):
        return text, False
    
    original_text = text
    conversion_applied = False
    
    # Sort by length (descending) to handle longer patterns first
    sorted_patterns = sorted(ELLIPSES_TO_SSML_MAPPING.items(), 
                           key=lambda x: len(x[0]), reverse=True)
    
    for ellipses_pattern, ssml_break in sorted_patterns:
        # Use regex to find exact ellipses patterns
        pattern = re.escape(ellipses_pattern)
        if re.search(pattern, text):
            text = re.sub(pattern, ssml_break, text)
            conversion_applied = True
            logger.debug(f"Converted ellipses: '{ellipses_pattern}' -> '{ssml_break}'")
    
    if conversion_applied:
        logger.debug(f"Ellipses conversion: '{original_text}' -> '{text}'")
    
    return text, conversion_applied


def extract_prosody_settings(text: str) -> Tuple[str, ProsodySettings]:
    """Extract prosody settings from SSML tags and return clean text.
    
    Args:
        text: Input text potentially containing prosody SSML tags
        
    Returns:
        Tuple of (text_without_prosody_tags, prosody_settings)
    """
    if not text:
        return text, ProsodySettings()
    
    prosody = ProsodySettings()
    processed_text = text
    
    # Extract prosody rate
    rate_pattern = r'<prosody\s+rate="([^"]+)"[^>]*>(.*?)</prosody>'
    rate_matches = re.findall(rate_pattern, text, re.IGNORECASE | re.DOTALL)
    for rate_value, content in rate_matches:
        # Parse rate value (e.g., "slow", "fast", "50%", "2x", "0.8")
        if rate_value.lower() == 'slow':
            prosody.rate = 0.7
        elif rate_value.lower() == 'fast':
            prosody.rate = 1.3
        elif rate_value.lower() == 'x-slow':
            prosody.rate = 0.5
        elif rate_value.lower() == 'x-fast':
            prosody.rate = 1.5
        elif rate_value.endswith('%'):
            prosody.rate = float(rate_value[:-1]) / 100.0
        elif rate_value.endswith('x'):
            prosody.rate = float(rate_value[:-1])
        else:
            try:
                prosody.rate = float(rate_value)
            except ValueError:
                logger.warning(f"Invalid prosody rate value: {rate_value}")
        
        # Replace prosody tag with just the content
        prosody_tag = f'<prosody rate="{rate_value}">{content}</prosody>'
        processed_text = processed_text.replace(prosody_tag, content)
        logger.debug(f"Extracted prosody rate: {rate_value} -> {prosody.rate}")
    
    # Extract prosody pitch
    pitch_pattern = r'<prosody\s+pitch="([^"]+)"[^>]*>(.*?)</prosody>'
    pitch_matches = re.findall(pitch_pattern, text, re.IGNORECASE | re.DOTALL)
    for pitch_value, content in pitch_matches:
        # Parse pitch value (e.g., "high", "low", "+50%", "-20Hz", "+2st")
        if pitch_value.lower() == 'high':
            prosody.pitch = 50.0
        elif pitch_value.lower() == 'low':
            prosody.pitch = -50.0
        elif pitch_value.lower() == 'x-high':
            prosody.pitch = 100.0
        elif pitch_value.lower() == 'x-low':
            prosody.pitch = -100.0
        elif pitch_value.startswith('+') and pitch_value.endswith('%'):
            prosody.pitch = float(pitch_value[1:-1])
        elif pitch_value.startswith('-') and pitch_value.endswith('%'):
            prosody.pitch = float(pitch_value[:-1])
        elif pitch_value.endswith('Hz'):
            prosody.pitch = float(pitch_value[:-2])
        else:
            try:
                prosody.pitch = float(pitch_value)
            except ValueError:
                logger.warning(f"Invalid prosody pitch value: {pitch_value}")
        
        # Replace prosody tag with just the content
        prosody_tag = f'<prosody pitch="{pitch_value}">{content}</prosody>'
        processed_text = processed_text.replace(prosody_tag, content)
        logger.debug(f"Extracted prosody pitch: {pitch_value} -> {prosody.pitch}")
    
    # Extract emphasis tags
    emphasis_pattern = r'<emphasis\s+level="([^"]+)"[^>]*>(.*?)</emphasis>'
    emphasis_matches = re.findall(emphasis_pattern, text, re.IGNORECASE | re.DOTALL)
    for emphasis_value, content in emphasis_matches:
        prosody.emphasis = emphasis_value.lower()
        
        # Replace emphasis tag with just the content
        emphasis_tag = f'<emphasis level="{emphasis_value}">{content}</emphasis>'
        processed_text = processed_text.replace(emphasis_tag, content)
        logger.debug(f"Extracted emphasis: {emphasis_value}")
    
    return processed_text, prosody


def preserve_existing_ssml(text: str) -> Tuple[str, bool]:
    """Preserve and validate existing SSML markup.
    
    Args:
        text: Input text potentially containing SSML markup
        
    Returns:
        Tuple of (processed_text, has_ssml_markup)
    """
    if not text:
        return text, False
    
    # Check for SSML tags (basic validation)
    ssml_pattern = r'<[^>]+>'
    has_ssml = bool(re.search(ssml_pattern, text))
    
    if has_ssml:
        logger.debug(f"Preserving existing SSML markup in: '{text}'")
        # For now, we pass through SSML as-is
        # Future: could add validation and normalization here
    
    return text, has_ssml


def process_hybrid_ssml(text: str) -> SSMLProcessingResult:
    """Process text with hybrid SSML support (ellipses + direct SSML + prosody).
    
    This is the main function for the hybrid SSML system. It:
    1. Detects the content format (ellipses, SSML, mixed, plain)
    2. Extracts prosody settings from SSML tags
    3. Converts ellipses to SSML breaks while preserving existing SSML
    4. Returns comprehensive processing metadata
    
    Args:
        text: Input text with ellipses and/or SSML markup
        
    Returns:
        SSMLProcessingResult with processed text and metadata
    """
    if not text or not isinstance(text, str):
        return SSMLProcessingResult(
            processed_text=text or '',
            has_ellipses=False,
            has_ssml_markup=False,
            conversion_applied=False,
            original_format='plain',
            prosody=None
        )
    
    # Detect original format
    original_format = detect_content_format(text)
    
    # Check for ellipses before processing
    has_ellipses = bool(re.search(r'\.{3,}', text))
    
    # Extract prosody settings first (removes prosody tags from text)
    processed_text, prosody_settings = extract_prosody_settings(text)
    
    # Preserve existing SSML markup (breaks, etc.)
    processed_text, has_ssml_markup = preserve_existing_ssml(processed_text)
    
    # Convert ellipses to SSML breaks
    processed_text, conversion_applied = convert_ellipses_to_ssml(processed_text)
    
    # If we extracted prosody settings, mark as having SSML markup
    if prosody_settings.rate or prosody_settings.pitch or prosody_settings.volume or prosody_settings.emphasis:
        has_ssml_markup = True
    
    result = SSMLProcessingResult(
        processed_text=processed_text,
        has_ellipses=has_ellipses,
        has_ssml_markup=has_ssml_markup,
        conversion_applied=conversion_applied,
        original_format=original_format,
        prosody=prosody_settings if any([prosody_settings.rate, prosody_settings.pitch, 
                                       prosody_settings.volume, prosody_settings.emphasis]) else None
    )
    
    logger.debug(f"Hybrid SSML processing: {original_format} -> {result}")
    if result.prosody:
        logger.debug(f"Extracted prosody settings: {result.prosody}")
    return result


def format_for_tts_provider(text: str, provider_name: str, supports_ssml: bool = True) -> str:
    """Format processed SSML for specific TTS provider.
    
    Args:
        text: Text with SSML markup (from process_hybrid_ssml)
        provider_name: Name of TTS provider ('edge_tts', 'gtts', etc.)
        supports_ssml: Whether the provider supports SSML
        
    Returns:
        Provider-specific formatted text
    """
    if not text:
        return text
    
    # Check if text contains SSML markup
    has_ssml_tags = bool(re.search(r'<[^>]+>', text))
    
    if supports_ssml and has_ssml_tags:
        # Wrap in SSML speak tags if not already wrapped
        if not text.strip().startswith('<speak>'):
            formatted_text = f"<speak>{text}</speak>"
        else:
            formatted_text = text
        logger.debug(f"Formatted for SSML-capable provider ({provider_name}): '{formatted_text}'")
        return formatted_text
    
    elif has_ssml_tags and not supports_ssml:
        # Convert SSML breaks back to provider-specific format
        fallback_text = convert_ssml_to_fallback_pauses(text, provider_name)
        logger.debug(f"Converted SSML to fallback for {provider_name}: '{fallback_text}'")
        return fallback_text
    
    else:
        # No SSML markup, return as-is
        return text


def convert_ssml_to_fallback_pauses(text: str, provider_name: str = 'edge_tts') -> str:
    """Convert SSML break tags to punctuation that TTS providers will respect.
    
    Different providers handle pauses differently:
    - EdgeTTS: Ignores ellipses but respects semicolons and commas for pauses
    - gTTS: Handles ellipses naturally, so we keep those for gTTS
    
    Args:
        text: Text with SSML break tags
        provider_name: Name of TTS provider ('edge_tts', 'gtts', etc.)
        
    Returns:
        Text with break tags converted to appropriate punctuation
    """
    if not text:
        return text
    
    # Convert SSML breaks to appropriate punctuation based on provider
    break_pattern = r'<break\s+time="([^"]+)"\s*/>'
    
    if provider_name == 'edge_tts':
        # EdgeTTS handles ellipses well and creates appropriate long pauses
        TIME_TO_ELLIPSES = {
            '0.5s': '....',       # Short pause
            '0.75s': '.....',     # Medium pause  
            '1s': '......',       # Medium-long pause
            '1.25s': '.......',   # Long pause
            '1.5s': '........',   # Extra long pause
            '1.75s': '.........',  # Very long pause
            '2s': '..........',    # Very long pause  
            '2.25s': '...........',  # Very long pause
            '2.5s': '............',  # Extra long pause
            '2.75s': '.............' # Maximum pause
        }
        
        def replace_break(match):
            duration = match.group(1)
            ellipses = TIME_TO_ELLIPSES.get(duration, '......')  # Default to medium ellipses
            return ellipses
            
    else:
        # For gTTS and other providers, use ellipses (original approach)
        TIME_TO_ELLIPSES = {
            '0.5s': '....',       # Short pause
            '0.75s': '.....',     # Medium pause  
            '1s': '......',       # Medium-long pause
            '1.25s': '.......',   # Long pause
            '1.5s': '........',   # Extra long pause
            '1.75s': '.........',  # Very long pause
            '2s': '..........',   # Very long pause  
            '2.25s': '...........',# Very long pause
            '2.5s': '............', # Extra long pause
            '2.75s': '.............' # Maximum pause
        }
        
        def replace_break(match):
            duration = match.group(1)
            ellipses = TIME_TO_ELLIPSES.get(duration, '.....')  # Default to medium pause
            return ellipses
    
    converted = re.sub(break_pattern, replace_break, text, flags=re.IGNORECASE)
    
    if converted != text:
        if provider_name == 'edge_tts':
            logger.debug(f"SSML to ellipses conversion for EdgeTTS: '{text}' -> '{converted}'")
        else:
            logger.debug(f"SSML to ellipses conversion for {provider_name}: '{text}' -> '{converted}'")
    
    return converted


def convert_single_ellipses_for_edgetts(text: str, provider_name: str) -> str:
    """Keep ellipses for EdgeTTS as they create proper long pauses.
    
    EdgeTTS handles ellipses well and creates longer, more appropriate pauses
    than semicolons. This function now preserves ellipses for EdgeTTS.
    
    Args:
        text: Input text with potential single ellipses
        provider_name: Name of TTS provider
        
    Returns:
        Text unchanged for EdgeTTS (ellipses preserved)
    """
    # Keep ellipses for EdgeTTS - they work better than semicolons
    if provider_name == 'edge_tts':
        logger.debug(f"Preserving ellipses for EdgeTTS: '{text}'")
        return text
    
    # For other providers, convert ellipses to semicolons if needed
    if not text:
        return text
    
    # Convert single ellipses to semicolons for non-EdgeTTS providers
    converted = re.sub(r'(?<!\.)\.{3}(?!\.)', ';', text)
    
    if converted != text:
        logger.debug(f"Single ellipses conversion for {provider_name}: '{text}' -> '{converted}'")
    
    return converted


def enhanced_preprocess_text_for_tts(
    text: str, 
    language_code: str, 
    provider_name: str,
    supports_ssml: bool = True,
    section_type: Optional[str] = None
) -> Tuple[str, SSMLProcessingResult]:
    """Enhanced preprocessing with hybrid SSML support.
    
    This function extends the original preprocessing with hybrid SSML capabilities:
    1. Applies original preprocessing (syllables, abbreviations, language-specific)
    2. Converts single ellipses to semicolons for EdgeTTS (before SSML processing)
    3. Processes hybrid SSML (4+ ellipses + direct SSML)
    4. Formats for specific TTS provider
    
    Args:
        text: Input text to preprocess
        language_code: BCP-47 language code (e.g., 'en-US', 'fil-PH')
        provider_name: Name of TTS provider ('edge_tts', 'gtts', etc.)
        supports_ssml: Whether the provider supports SSML
        section_type: Type of section ('key_phrases', 'natural_speed', etc.)
        
    Returns:
        Tuple of (processed_text, ssml_processing_result)
    """
    if not text or not isinstance(text, str):
        empty_result = SSMLProcessingResult('', False, False, False, 'plain')
        return text or '', empty_result
    
    # Step 1: Apply original preprocessing (syllables, abbreviations, language-specific)
    preprocessed_text = preprocess_text_for_tts(text, language_code, section_type)
    
    # Step 2: Convert single ellipses for EdgeTTS (before SSML processing)
    ellipses_converted_text = convert_single_ellipses_for_edgetts(preprocessed_text, provider_name)
    
    # Step 3: Process hybrid SSML (4+ ellipses and direct SSML)
    ssml_result = process_hybrid_ssml(ellipses_converted_text)
    
    # Step 4: Format for TTS provider
    final_text = format_for_tts_provider(
        ssml_result.processed_text, 
        provider_name, 
        supports_ssml
    )
    
    # Update the result with final text
    ssml_result.processed_text = final_text
    
    logger.debug(f"Enhanced preprocessing complete: '{text}' -> '{final_text}' "
                f"(format: {ssml_result.original_format}, provider: {provider_name})")
    
    return final_text, ssml_result


def split_text_with_pauses(text: str) -> List[Tuple[str, Optional[float]]]:
    """Split text containing pause markers into segments with pause durations.
    
    This function parses text containing [PAUSE:Xs] markers and returns a list
    of text segments along with their associated pause durations.
    
    Args:
        text: Text containing pause markers like [PAUSE:1s] or [PAUSE:500ms]
        
    Returns:
        List of tuples (text_segment, pause_duration_ms) where pause_duration_ms
        is None for the last segment or if no pause follows the segment.
        
    Example:
        "Hello[PAUSE:1s] world[PAUSE:2s] goodbye" returns:
        [("Hello", 1000.0), ("world", 2000.0), ("goodbye", None)]
    """
    if not text or '[PAUSE:' not in text:
        return [(text, None)]
    
    # Pattern to match pause markers: [PAUSE:1s], [PAUSE:500ms], etc.
    pause_pattern = r'\[PAUSE:([0-9.]+)(s|ms)\]'
    
    segments = []
    last_end = 0
    
    for match in re.finditer(pause_pattern, text):
        # Get the text segment before this pause marker
        segment_text = text[last_end:match.start()].strip()
        
        # Parse the pause duration
        duration_str = match.group(1)
        unit = match.group(2)
        
        try:
            duration = float(duration_str)
            if unit == 's':
                duration_ms = duration * 1000  # Convert seconds to milliseconds
            else:  # unit == 'ms'
                duration_ms = duration
        except ValueError:
            logger.warning(f"Invalid pause duration: {match.group(0)}")
            duration_ms = 1000.0  # Default to 1 second
        
        # Add the segment with its pause duration
        if segment_text:  # Only add non-empty segments
            segments.append((segment_text, duration_ms))
        
        last_end = match.end()
    
    # Add any remaining text after the last pause marker
    remaining_text = text[last_end:].strip()
    if remaining_text:
        segments.append((remaining_text, None))
    
    # If no segments were found, return the original text
    if not segments:
        segments = [(text, None)]
    
    return segments
