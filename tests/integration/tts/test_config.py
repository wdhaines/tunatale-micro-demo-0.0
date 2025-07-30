#!/usr/bin/env python3

import sys
import asyncio
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from tunatale.cli.config import load_config
from tunatale.infrastructure.factories import create_tts_service

async def test_config():
    print("Testing configuration loading...")
    
    # Test 1: Load config from current directory
    config = load_config(Path("config.yaml"))
    print(f"Loaded config: {config}")
    print(f"TTS config: {config.tts}")
    print(f"TTS provider: {config.tts.provider}")
    
    # Test 2: Create TTS service
    print("\nTesting TTS service creation...")
    tts_service = create_tts_service(config.tts)
    print(f"TTS service type: {type(tts_service)}")
    
if __name__ == "__main__":
    asyncio.run(test_config())