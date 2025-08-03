"""Natural pause calculator for linguistic boundaries."""
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class NaturalPauseCalculator:
    """Calculate pauses based on linguistic boundaries."""
    
    def __init__(self):
        """Initialize the pause calculator with hierarchical pause levels."""
        self.pause_levels = {
            'syllable': 300,      # Between syllables: ma-gan-da
            'word': 600,          # Between words: maganda hapon  
            'phrase': 1200,       # Between phrases: maganda hapon | po
            'sentence': 2000,     # Between sentences: Kumusta? | Mabuti.
            'section': 3000       # Between sections
        }
    
    def get_pause_for_boundary(self, boundary_type: str, text_complexity: str = 'normal', 
                             audio_duration_seconds: float = None, phrase_text: str = None) -> int:
        """Get pause duration based on boundary type and complexity.
        
        Args:
            boundary_type: Type of linguistic boundary ('syllable', 'word', 'phrase', 'sentence', 'section')
            text_complexity: Complexity level ('normal' or 'slow')
            audio_duration_seconds: Duration of the audio segment in seconds (for dynamic pauses)
            phrase_text: The actual phrase text to analyze for word count-based multipliers
            
        Returns:
            Pause duration in milliseconds
        """
        # If audio duration is provided, use dynamic calculation with adaptive multipliers
        if audio_duration_seconds is not None:
            # Determine dynamic multiplier based on phrase complexity
            multiplier = self._get_dynamic_multiplier(phrase_text, audio_duration_seconds)
            
            # Calculate desired total pause duration
            desired_pause_ms = int(audio_duration_seconds * multiplier)
            
            # Account for the base silence_between_phrases (0.5s = 500ms) that gets added later
            # Subtract it so our total pause timing is exactly what we want
            base_silence_ms = 500  # Default silence_between_phrases from config
            dynamic_pause = max(0, desired_pause_ms - base_silence_ms)
            
            
            # Adjust for slow speech
            if text_complexity == 'slow':
                dynamic_pause = int(dynamic_pause * 1.2)  # 20% longer for slow sections
                
            return dynamic_pause
        
        # Fallback to original fixed pause system
        base_pause = self.pause_levels.get(boundary_type, 600)  # Default to word-level pause
        
        # Adjust for slow speech
        if text_complexity == 'slow':
            return int(base_pause * 1.5)  # 50% longer for slow sections
        
        return base_pause
    
    def _get_dynamic_multiplier(self, phrase_text: str, audio_duration_seconds: float) -> int:
        """Calculate dynamic multiplier based on measured audio duration.
        
        Uses a fixed multiplier that scales with actual audio length for consistent timing.
        
        Args:
            phrase_text: The phrase text to analyze (for debug logging)
            audio_duration_seconds: Duration of the audio segment
            
        Returns:
            Multiplier value in milliseconds per second of audio
        """
        # Fixed 1x multiplier - gives longer phrases proportionally longer pauses
        # This means: 1 second audio = 1 second pause, 2 second audio = 2 second pause
        base_multiplier = 1000
        
        # Reduced logging for performance
        logger.debug(f"🎵 PAUSE CALC: '{phrase_text}' ({audio_duration_seconds:.2f}s audio) → {base_multiplier}x → {(audio_duration_seconds * base_multiplier/1000):.2f}s pause")
        
        return base_multiplier