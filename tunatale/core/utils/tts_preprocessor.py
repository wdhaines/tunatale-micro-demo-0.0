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
    '...': '<break time="0.5s"/>',      # Base pause (0.5 seconds)
    '....': '<break time="0.75s"/>',    # Short pause (0.75 seconds)
    '.....': '<break time="1s"/>',      # Medium pause (1 second)
    '......': '<break time="1.25s"/>',  # Medium-long pause (1.25 seconds)
    '.......': '<break time="1.5s"/>',  # Long pause (1.5 seconds)
    '........': '<break time="1.75s"/>',# Extra long pause (1.75 seconds)
    '.........': '<break time="2s"/>',  # Very long pause (2 seconds)
    '..........': '<break time="2.25s"/>',# Very long pause (2.25 seconds)
    '...........': '<break time="2.5s"/>',# Very long pause (2.5 seconds)
    '............': '<break time="2.75s"/>',# Maximum pause (2.75 seconds)
    '.............': '<break time="3s"/>',# Maximum pause (3 seconds) - cap at 13 dots
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
        # For non-SSML providers, we'll need to handle this in audio post-processing
        fallback_text = convert_ssml_to_fallback_pauses(text)
        logger.debug(f"Converted SSML to fallback for {provider_name}: '{fallback_text}'")
        return fallback_text
    
    else:
        # No SSML markup, return as-is
        return text


def convert_ssml_to_fallback_pauses(text: str) -> str:
    """Convert SSML break tags to fallback pause markers for non-SSML providers.
    
    Args:
        text: Text with SSML break tags
        
    Returns:
        Text with break tags converted to pause markers
    """
    if not text:
        return text
    
    # Convert SSML breaks to pause markers that can be handled in audio post-processing
    # Pattern: <break time="1.5s"/> -> [PAUSE:1.5s]
    break_pattern = r'<break\s+time="([^"]+)"\s*/>'
    
    def replace_break(match):
        duration = match.group(1)
        return f"[PAUSE:{duration}]"
    
    converted = re.sub(break_pattern, replace_break, text, flags=re.IGNORECASE)
    
    if converted != text:
        logger.debug(f"SSML to fallback conversion: '{text}' -> '{converted}'")
    
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
    2. Processes hybrid SSML (ellipses + direct SSML)
    3. Formats for specific TTS provider
    
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
    
    # Step 2: Process hybrid SSML
    ssml_result = process_hybrid_ssml(preprocessed_text)
    
    # Step 3: Format for TTS provider
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
