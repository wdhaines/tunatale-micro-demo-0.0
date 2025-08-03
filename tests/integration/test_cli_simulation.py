#!/usr/bin/env python3
"""
Complete CLI simulation test to verify pause marker processing.
"""

import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, '.')

class MockProvider:
    """Mock individual provider (EdgeTTS/gTTS) that captures calls."""
    
    def __init__(self, name):
        self.name = name
        self.calls = []
        
    async def synthesize_speech(self, text, voice_id, output_path, **kwargs):
        self.calls.append({
            'text': text,
            'voice_id': voice_id,
            'output_path': output_path
        })
        print(f"🎤 {self.name} received: '{text}'")
        
        # Create dummy MP3 file with valid audio
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create a minimal silent audio segment and export as MP3
        from pydub import AudioSegment
        silent_audio = AudioSegment.silent(duration=1000)  # 1 second of silence
        silent_audio.export(output_path, format="mp3")
            
    def _get_provider_for_voice(self, voice_id):
        return self.name.lower()

async def test_complete_cli_simulation():
    """Test complete CLI simulation with MultiProviderTTSService."""
    
    print("=" * 60)
    print("🔍 COMPLETE CLI SIMULATION TEST")
    print("=" * 60)
    
    # Create mock providers
    edge_provider = MockProvider("EdgeTTS")
    gtts_provider = MockProvider("gTTS")
    
    # Create multi-provider service
    from tunatale.infrastructure.services.tts.multi_provider_tts_service import MultiProviderTTSService
    
    providers = {
        'edge': edge_provider,
        'gtts': gtts_provider
    }
    
    multi_service = MultiProviderTTSService(providers)
    
    # Test input
    test_text = "Magkano[PAUSE:1s] po?"
    voice_id = "fil-PH-BlessicaNeural"  # Should route to EdgeTTS
    
    print(f"📝 Input text: '{test_text}'")
    print(f"🎤 Voice ID: {voice_id}")
    
    # Test the multi-provider service
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test.mp3"
        
        print(f"\n🔧 Calling synthesize_speech_with_pauses...")
        
        try:
            result = await multi_service.synthesize_speech_with_pauses(
                text=test_text,
                voice_id=voice_id,
                output_path=str(output_path)
            )
            
            print(f"✅ Multi-provider call succeeded")
            print(f"📊 Result: {result}")
            
        except Exception as e:
            print(f"❌ Multi-provider call failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Check what each provider received
        print(f"\n📊 PROVIDER CALL SUMMARY:")
        
        edge_calls = edge_provider.calls
        gtts_calls = gtts_provider.calls
        
        print(f"EdgeTTS calls: {len(edge_calls)}")
        for i, call in enumerate(edge_calls, 1):
            text = call['text']
            has_pause = '[PAUSE:' in text or 'PAUSE' in text
            status = "❌ HAS PAUSE MARKERS" if has_pause else "✅ Clean"
            print(f"  {i}. '{text}' {status}")
            
        print(f"gTTS calls: {len(gtts_calls)}")
        for i, call in enumerate(gtts_calls, 1):
            text = call['text']
            has_pause = '[PAUSE:' in text or 'PAUSE' in text
            status = "❌ HAS PAUSE MARKERS" if has_pause else "✅ Clean"
            print(f"  {i}. '{text}' {status}")
        
        # Final verdict
        all_calls = edge_calls + gtts_calls
        any_has_pause = any('[PAUSE:' in call['text'] or 'PAUSE' in call['text'] 
                           for call in all_calls)
        
        print(f"\n🏁 FINAL VERDICT:")
        if any_has_pause:
            print("❌ FAILED: Some providers received pause markers as literal text!")
        else:
            print("✅ SUCCESS: All providers received clean text!")
            print("   The pause marker processing is working correctly.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_complete_cli_simulation())