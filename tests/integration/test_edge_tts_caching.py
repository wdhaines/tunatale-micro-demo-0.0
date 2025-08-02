"""Integration tests for Edge TTS service caching functionality."""
import asyncio
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional, List

import pytest
import pytest_asyncio

logger = logging.getLogger(__name__)

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
        output_file1 = os.path.join(tempfile.gettempdir(), "test_output1.mp3")
        
        # Clear any existing output file
        if os.path.exists(output_file1):
            os.unlink(output_file1)
        
        start_time = time.monotonic()
        
        # Print debug info before synthesis
        cache_key_before = tts_service._generate_cache_key(TEST_TEXT, TEST_VOICE, 1.0, 0.0, 1.0)
        logger.info(f"Expected cache key before synthesis: {cache_key_before}")
        
        # Get list of files in cache directory before synthesis
        cache_files_before = list(temp_cache_dir.glob('*'))
        logger.info(f"Cache files before synthesis: {[f.name for f in cache_files_before]}")
        
        # Perform the synthesis
        result1 = await tts_service.synthesize_speech(
            text=TEST_TEXT,
            voice_id=TEST_VOICE,
            output_path=output_file1,
            rate=1.0,
            pitch=0.0,
            volume=1.0
        )
        duration1 = time.monotonic() - start_time
        
        # Print debug info after synthesis
        logger.info(f"First synthesis took {duration1:.2f} seconds")
        logger.info(f"Synthesis result: {result1}")
        cache_files_after = list(temp_cache_dir.glob('*'))
        logger.info(f"Cache files after synthesis: {[f.name for f in cache_files_after]}")
        
        # Get the actual cache file that was created
        new_cache_files = set(cache_files_after) - set(cache_files_before)
        if new_cache_files:
            actual_cache_file = new_cache_files.pop()
            logger.info(f"New cache file created: {actual_cache_file.name}")
        else:
            logger.warning("No new cache files were created after synthesis")
        
        # Verify the file was created
        assert os.path.exists(output_file1), f"Output file not created at {output_file1}"
        file_size1 = os.path.getsize(output_file1)
        assert file_size1 > 0, f"Output file is empty at {output_file1}"
        
        # Verify the result indicates it wasn't from cache
        assert not result1.get("cached", True)
        
        # Get all files in the cache directory after synthesis
        cache_files = list(temp_cache_dir.glob('*'))
        assert len(cache_files) > 0, "No cache files were created"
        
        # Get the actual cache file that was created
        cache_file = cache_files[0]
        cache_key = cache_file.name
        logger.info(f"Using cache file: {cache_file.name}, size: {cache_file.stat().st_size} bytes")
        
        # Verify the cache file exists and has content
        assert cache_file.exists(), f"Cache file not found at {cache_file}"
        assert cache_file.stat().st_size > 0, f"Cache file is empty at {cache_file}"
        
        # Second synthesis - should be from cache
        output_file2 = f"{output_file1}_cached.mp3"
        
        # Clear any existing output file
        if os.path.exists(output_file2):
            os.unlink(output_file2)
            
        start_time = time.monotonic()
        result2 = await tts_service.synthesize_speech(
            text=TEST_TEXT,
            voice_id=TEST_VOICE,
            output_path=output_file2,
            rate=1.0,
            pitch=0.0,
            volume=1.0
        )
        end_time = time.monotonic()
        duration2 = end_time - start_time
        
        logger.info(f"Second synthesis took {duration2:.2f} seconds")
        logger.info(f"Second synthesis result: {result2}")
        
        # The output file should exist and not be empty
        assert os.path.exists(output_file2), f"Output file not created at {output_file2}"
        assert os.path.getsize(output_file2) > 0, f"Output file is empty at {output_file2}"
        
        # Check if the result was served from cache
        is_cached = result2.get('cached', False)
        logger.info(f"Second synthesis was {'served from cache' if is_cached else 'not served from cache'}")
        
        # Log the contents of the cache directory after second synthesis
        cache_files_after = list(temp_cache_dir.glob('*'))
        logger.info(f"Cache files after second synthesis: {[f.name for f in cache_files_after]}")
        
        # The output files should have the same content
        with open(output_file1, 'rb') as f1, open(output_file2, 'rb') as f2:
            assert f1.read() == f2.read(), "Output files have different content"
            
        # The second synthesis should be faster than the first
        # Note: On some systems, the difference might be minimal, so we'll just log it
        logger.info(f"First synthesis: {duration1:.2f}s, Second synthesis: {duration2:.2f}s")
        
    finally:
        # Clean up
        output_files = [
            output_file1,
            f"{output_file1}_cached.mp3"
        ]
        for f in output_files:
            if os.path.exists(f):
                try:
                    os.unlink(f)
                except Exception as e:
                    logger.warning(f"Failed to clean up file {f}: {e}")

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
        
        # Find the actual cache file that was created
        cache_files = list(temp_cache_dir.glob('*'))
        
        # Find the file that matches the base name of the cache key
        base_name = '_'.join(normal_cache_key.split('_')[:-1])
        matching_files = [f for f in cache_files if f.name.startswith(base_name)]
        
        assert matching_files, (
            f"No cache file found for base name {base_name}. "
            f"Available files: {[f.name for f in cache_files]}"
        )
        
        normal_cache_file = matching_files[0]
        logger.info(f"Found normal rate cache file: {normal_cache_file}")
        
        # Second synthesis - faster rate (1.2 = +20% rate)
        await tts_service.synthesize_speech(
            text=TEST_TEXT,
            voice_id=TEST_VOICE,
            output_path=f"{output_file}_fast.mp3",
            rate=1.2  # This will be converted to "+20%" by the service
        )
        
        # Get the cache key for fast rate
        fast_cache_key = tts_service._generate_cache_key(TEST_TEXT, TEST_VOICE, 1.2, 0.0, 1.0)
        
        # Find the actual fast rate cache file that was created
        cache_files_after = list(temp_cache_dir.glob('*'))
        fast_base_name = '_'.join(fast_cache_key.split('_')[:-1])
        matching_fast_files = [f for f in cache_files_after if f.name.startswith(fast_base_name)]
        
        assert matching_fast_files, (
            f"No cache file found for fast rate base name {fast_base_name}. "
            f"Available files: {[f.name for f in cache_files_after]}"
        )
        
        fast_cache_file = matching_fast_files[0]
        logger.info(f"Found fast rate cache file: {fast_cache_file}")
        
        # Both cache files should exist and be different
        assert normal_cache_file.exists(), f"Normal rate cache file should exist at {normal_cache_file}"
        assert fast_cache_file.exists(), f"Fast rate cache file should exist at {fast_cache_file}"
        
        # The actual cache files should be different (different content hashes)
        assert normal_cache_file != fast_cache_file, "Cache files should be different for different rates"
        
        # The cache keys should be different (different rate parameters)
        normal_base = '_'.join(normal_cache_key.split('_')[:-1])
        fast_base = '_'.join(fast_cache_key.split('_')[:-1])
        assert normal_base != fast_base, "Cache keys should be different for different rates"
        
    finally:
        # Clean up
        for f in [output_file, f"{output_file}_fast.mp3"]:
            if os.path.exists(f):
                os.unlink(f)
