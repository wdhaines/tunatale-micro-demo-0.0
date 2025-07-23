"""Linguistic boundary detection for natural pause calculation."""
import re
from typing import List, Dict, Tuple
from .natural_pause_calculator import NaturalPauseCalculator


def detect_linguistic_boundaries(text: str) -> List[Tuple[str, int]]:
    """Detect different types of boundaries in text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of tuples containing (boundary_type, position)
    """
    boundaries = []
    
    # Sentence boundaries (longest pauses) - periods, exclamation marks, question marks
    # But exclude periods that are part of abbreviations (like I.D., C.R., etc.)
    for match in re.finditer(r'[.!?]+\s*', text):
        # Check if this period is part of an abbreviation
        period_pos = match.start()
        
        # Look for common abbreviation patterns around this period
        is_abbreviation = False
        
        # Pattern 1: single letter + period + single letter (like "I.D." - the middle period)
        if period_pos > 0 and period_pos + 1 < len(text):
            if (text[period_pos-1].isupper() and text[period_pos+1].isupper()):
                is_abbreviation = True
        
        # Pattern 2: Check if this period is the final period in a 2-letter abbreviation
        # Look for pattern: letter + period + letter + period (like the final period in "I.D.")
        if period_pos >= 3:  # Need at least 3 chars before to check for "X.Y."
            if (text[period_pos-3].isupper() and 
                text[period_pos-2] == '.' and 
                text[period_pos-1].isupper()):
                is_abbreviation = True
        
        # Only add as sentence boundary if it's not part of an abbreviation
        if not is_abbreviation:
            boundaries.append(('sentence', match.end()))
    
    # Phrase boundaries (comma, semicolon, conjunctions)
    for match in re.finditer(r'[,;]\s*|\s+(?:and|but|or|yet|so|for|nor)\s+', text, re.IGNORECASE):
        boundaries.append(('phrase', match.end()))
    
    # Special handling for ellipsis - treat as sentence boundary for now
    for match in re.finditer(r'\.{3,}\s*', text):
        boundaries.append(('sentence', match.end()))
    
    # Word boundaries (spaces between words)
    for match in re.finditer(r'\s+', text):
        # Skip if it's already a phrase/sentence boundary
        if not any(abs(match.start() - pos) < 3 for _, pos in boundaries):
            boundaries.append(('word', match.start()))
    
    return sorted(boundaries, key=lambda x: x[1])


def split_with_natural_pauses(text: str, is_slow: bool = False) -> List[Dict]:
    """Split text with appropriate pauses for natural speech.
    
    Args:
        text: Input text to split
        is_slow: Whether this is slow speech (longer pauses)
        
    Returns:
        List of segments with text and pause information
    """
    if not text or not text.strip():
        return []
    
    segments = []
    boundaries = detect_linguistic_boundaries(text)
    last_pos = 0
    
    calculator = NaturalPauseCalculator()
    complexity = 'slow' if is_slow else 'normal'
    
    for boundary_type, pos in boundaries:
        # Add the text segment before this boundary
        if pos > last_pos:
            segment_text = text[last_pos:pos].strip()
            if segment_text:
                segments.append({
                    'type': 'text',
                    'content': segment_text,
                    'voice_settings': {'rate': 0.5 if is_slow else 1.0}
                })
        
        # Add the pause
        pause_duration = calculator.get_pause_for_boundary(boundary_type, complexity)
        
        segments.append({
            'type': 'pause',
            'duration': pause_duration,
            'boundary': boundary_type
        })
        
        last_pos = pos
    
    # Add remaining text after the last boundary
    if last_pos < len(text):
        remaining = text[last_pos:].strip()
        if remaining:
            segments.append({
                'type': 'text', 
                'content': remaining,
                'voice_settings': {'rate': 0.5 if is_slow else 1.0}
            })
    
    # If no boundaries were found, return the whole text as a single segment
    if not segments and text.strip():
        segments.append({
            'type': 'text',
            'content': text.strip(),
            'voice_settings': {'rate': 0.5 if is_slow else 1.0}
        })
    
    return segments