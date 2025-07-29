"""Natural pause calculator for linguistic boundaries."""
from typing import Dict


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
                             audio_duration_seconds: float = None) -> int:
        """Get pause duration based on boundary type and complexity.
        
        Args:
            boundary_type: Type of linguistic boundary ('syllable', 'word', 'phrase', 'sentence', 'section')
            text_complexity: Complexity level ('normal' or 'slow')
            audio_duration_seconds: Duration of the audio segment in seconds (for dynamic pauses)
            
        Returns:
            Pause duration in milliseconds
        """
        # If audio duration is provided, use dynamic calculation (1.5x audio length + base pause)
        if audio_duration_seconds is not None:
            base_pause = self.pause_levels.get(boundary_type, 600)
            dynamic_pause = int((audio_duration_seconds * 1500) + base_pause)  # 1.5x audio + base
            
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