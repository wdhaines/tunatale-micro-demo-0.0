"""Integration tests for Edge TTS service caching functionality."""
import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import pytest
import pytest_asyncio

from tunatale.infrastructure.services.tts.edge_tts_service import EdgeTTSService

# Test text and voice
TEST_TEXT = "This is a test of the Edge TTS service with caching."
TEST_VOICE = "en-US-JennyNeural"

@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="tts_cache_test_")
    yield Path(temp_dir)
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

@pytest_asyncio.fixture
async def tts_service(temp_cache_dir):
    """Create an EdgeTTSService instance with a temporary cache directory."""
    async with EdgeTTSService(
        cache_dir=temp_cache_dir,
        connection_limit=1,
    ) as service:
        await service.validate_credentials()
        yield service

@pytest.mark.asyncio
async def test_caching_basic(tts_service: EdgeTTSService, temp_cache_dir: Path):
    """Test that synthesized speech is properly cached."""
    # Create a temporary output file
    output_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    
    try:
        # First synthesis - should not be cached
        start_time = time.monotonic()
        result1 = await tts_service.synthesize_speech(
            text=TEST_TEXT,
            voice_id=TEST_VOICE,
            output_path=output_file,
        )
        duration1 = time.monotonic() - start_time
        
        # Verify the file was created
        assert os.path.exists(output_file)
        file_size1 = os.path.getsize(output_file)
        assert file_size1 > 0
        
        # Verify the result indicates it wasn't from cache
        assert not result1.get("cached", True)
        
        # Get the cache key and verify the file exists in cache
        cache_key = tts_service._generate_cache_key(TEST_TEXT, TEST_VOICE, 1.0, 0.0, 1.0)
        cache_file = temp_cache_dir / cache_key  # No need to add .mp3 as it's already included in the key
        assert cache_file.exists(), f"Cache file not found at {cache_file}"
        
        # Second synthesis - should be from cache
        output_file2 = f"{output_file}_cached.mp3"
        start_time = time.monotonic()
        result2 = await tts_service.synthesize_speech(
            text=TEST_TEXT,
            voice_id=TEST_VOICE,
            output_path=output_file2,
        )
        duration2 = time.monotonic() - start_time
        
        # Verify the second file was created and has the same content
        assert os.path.exists(output_file2)
        assert os.path.getsize(output_file2) == file_size1
        
        # Verify the result indicates it was from cache
        assert result2.get("cached", False), "Result should indicate it was served from cache"
        
        # Cached response should be significantly faster
        assert duration2 < duration1, "Cached response should be faster than uncached"
        
    finally:
        # Clean up
        for f in [output_file, f"{output_file}_cached.mp3"]:
            if os.path.exists(f):
                os.unlink(f)

@pytest.mark.asyncio
async def test_cache_invalidation(tts_service: EdgeTTSService, temp_cache_dir: Path):
    """Test that changing parameters creates a new cache entry."""
    output_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    
    try:
        # First synthesis - normal rate (1.0 = 0% change)
        await tts_service.synthesize_speech(
            text=TEST_TEXT,
            voice_id=TEST_VOICE,
            output_path=output_file,
            rate=1.0  # This will be converted to "+0%" by the service
        )
        
        # Get the cache key for normal rate
        normal_cache_key = tts_service._generate_cache_key(TEST_TEXT, TEST_VOICE, 1.0, 0.0, 1.0)
        normal_cache_file = temp_cache_dir / normal_cache_key
        assert normal_cache_file.exists(), f"Normal rate cache file not found at {normal_cache_file}"
        
        # Second synthesis - faster rate (1.2 = +20% rate)
        await tts_service.synthesize_speech(
            text=TEST_TEXT,
            voice_id=TEST_VOICE,
            output_path=f"{output_file}_fast.mp3",
            rate=1.2  # This will be converted to "+20%" by the service
        )      
        
        # Get the cache key for fast rate
        fast_cache_key = tts_service._generate_cache_key(TEST_TEXT, TEST_VOICE, 1.2, 0.0, 1.0)
        fast_cache_file = temp_cache_dir / fast_cache_key  # No .mp3 suffix needed
        
        # Both cache files should exist and be different
        assert normal_cache_file.exists(), f"Normal rate cache file should exist at {normal_cache_file}"
        assert fast_cache_file.exists(), f"Fast rate cache file should exist at {fast_cache_file}"
        assert normal_cache_key != fast_cache_key, "Cache keys should be different for different rates"
        
    finally:
        # Clean up
        for f in [output_file, f"{output_file}_fast.mp3"]:
            if os.path.exists(f):
                os.unlink(f)
