#!/usr/bin/env python3
"""Test the phonetic respelling approach"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from tunatale.core.utils.tts_preprocessor import preprocess_text_for_tts
from tunatale.core.services.linguistic_boundary_detector import split_with_natural_pauses

def test_phonetic_approach():
    """Test the phonetic respelling approach"""
    print("🔊 Testing Phonetic Respelling Approach")
    print("=" * 50)
    
    test_cases = [
        "Please show your ID",
        "ID po, salamat",
        "Go to the CR please",
        "Show ID at the CR",
        "Valid ID required",
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n{i}. Original: '{test_text}'")
        
        # Apply preprocessing
        preprocessed = preprocess_text_for_tts(test_text, 'fil-PH')
        print(f"   Phonetic: '{preprocessed}'")
        
        # Apply natural pause splitting
        segments = split_with_natural_pauses(preprocessed, is_slow=False)
        
        # Extract text segments that go to TTS
        text_segments = [s['content'] for s in segments if s['type'] == 'text']
        print(f"   TTS Gets: {text_segments}")

if __name__ == "__main__":
    test_phonetic_approach()