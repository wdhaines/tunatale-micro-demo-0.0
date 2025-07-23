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
    
    def get_pause_for_boundary(self, boundary_type: str, text_complexity: str = 'normal') -> int:
        """Get pause duration based on boundary type and complexity.
        
        Args:
            boundary_type: Type of linguistic boundary ('syllable', 'word', 'phrase', 'sentence', 'section')
            text_complexity: Complexity level ('normal' or 'slow')
            
        Returns:
            Pause duration in milliseconds
        """
        base_pause = self.pause_levels.get(boundary_type, 600)  # Default to word-level pause
        
        # Adjust for slow speech
        if text_complexity == 'slow':
            return int(base_pause * 1.5)  # 50% longer for slow sections
        
        return base_pause