"""Integration tests for AudioProcessorService."""
import asyncio
import io
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import pytest_asyncio
from pydub import AudioSegment

from tunatale.core.exceptions import AudioProcessingError
from tunatale.infrastructure.services.audio.audio_processor import AudioProcessorService

# Disable logging during tests
import logging
logging.getLogger().setLevel(logging.CRITICAL)


@pytest.fixture
def audio_processor():
    """Create an AudioProcessorService instance for testing."""
    return AudioProcessorService()


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a sample audio file for testing."""
    # Create a 1-second silent audio file
    audio = AudioSegment.silent(duration=1000)  # 1 second
    
    # Add a small tone in the middle
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    t = np.linspace(0, 1, len(samples), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t) * 32767  # 440 Hz tone
    samples += tone.astype(np.int16)
    
    # Create a new audio segment with the tone
    audio_with_tone = audio._spawn(samples.tobytes())
    
    # Save to a temporary file
    file_path = tmp_path / "sample_audio.wav"
    audio_with_tone.export(file_path, format="wav")
    
    return file_path


@pytest.fixture
def silent_audio_file(tmp_path):
    """Create a silent audio file for testing."""
    # Create a 2-second silent audio file
    audio = AudioSegment.silent(duration=2000)  # 2 seconds
    
    file_path = tmp_path / "silent_audio.wav"
    audio.export(file_path, format="wav")
    
    return file_path


class TestAudioProcessorService:
    """Test cases for AudioProcessorService."""
    
    async def test_concatenate_audio(self, audio_processor, tmp_path, sample_audio_file):
        """Test concatenating audio files."""
        output_file = tmp_path / "output.mp3"
        
        # Concatenate the same file twice
        result = await audio_processor.concatenate_audio(
            [sample_audio_file, sample_audio_file],
            output_file
        )
        
        assert result == output_file
        assert output_file.exists()
        
        # Verify the output duration is approximately 2x the input
        duration = await audio_processor.get_audio_duration(output_file)
        assert 1.9 <= duration <= 2.1  # Allow some tolerance
    
    async def test_concatenate_audio_with_file_objects(self, audio_processor, tmp_path, sample_audio_file):
        """Test concatenating audio from file objects."""
        output_file = tmp_path / "output.mp3"
        
        # Open files in binary mode
        with open(sample_audio_file, 'rb') as f1, open(sample_audio_file, 'rb') as f2:
            result = await audio_processor.concatenate_audio(
                [f1, f2],
                output_file
            )
        
        assert result == output_file
        assert output_file.exists()
    
    async def test_add_silence_to_start(self, audio_processor, tmp_path, sample_audio_file):
        """Test adding silence to the start of an audio file."""
        output_file = tmp_path / "output.mp3"
        silence_duration = 1.0  # 1 second
        
        # Get original duration
        original_duration = await audio_processor.get_audio_duration(sample_audio_file)
        
        # Add silence to start
        result = await audio_processor.add_silence(
            sample_audio_file,
            output_file,
            silence_duration,
            position="start"
        )
        
        assert result == output_file
        assert output_file.exists()
        
        # Verify the output duration
        new_duration = await audio_processor.get_audio_duration(output_file)
        assert abs(new_duration - (original_duration + silence_duration)) < 0.1
    
    async def test_add_silence_to_end(self, audio_processor, tmp_path, sample_audio_file):
        """Test adding silence to the end of an audio file."""
        output_file = tmp_path / "output.mp3"
        silence_duration = 0.5  # 0.5 seconds
        
        # Get original duration
        original_duration = await audio_processor.get_audio_duration(sample_audio_file)
        
        # Add silence to end
        result = await audio_processor.add_silence(
            sample_audio_file,
            output_file,
            silence_duration,
            position="end"
        )
        
        assert result == output_file
        assert output_file.exists()
        
        # Verify the output duration
        new_duration = await audio_processor.get_audio_duration(output_file)
        assert abs(new_duration - (original_duration + silence_duration)) < 0.1
    
    async def test_normalize_audio(self, audio_processor, tmp_path, sample_audio_file):
        """Test normalizing audio levels."""
        output_file = tmp_path / "output.wav"
        
        # Normalize the audio
        result = await audio_processor.normalize_audio(
            sample_audio_file,
            output_file,
            target_level=-20.0
        )
        
        assert result == output_file
        assert output_file.exists()
        
        # Verify the output file is not empty
        assert os.path.getsize(output_file) > 0
    
    async def test_trim_silence(self, audio_processor, tmp_path, sample_audio_file):
        """Test trimming silence from audio."""
        output_file = tmp_path / "output.wav"
        
        # First, add some silence to the start and end
        with_silence = tmp_path / "with_silence.wav"
        await audio_processor.add_silence(
            sample_audio_file,
            with_silence,
            duration=1.0,  # 1 second
            position="both"
        )
        
        # Get duration with silence
        duration_with_silence = await audio_processor.get_audio_duration(with_silence)
        
        # Trim the silence
        result = await audio_processor.trim_silence(
            with_silence,
            output_file,
            threshold=-40
        )
        
        assert result == output_file
        assert output_file.exists()
        
        # Get duration after trimming
        duration_trimmed = await audio_processor.get_audio_duration(output_file)
        
        # Verify the trimmed audio is shorter
        assert duration_trimmed < duration_with_silence
    
    async def test_get_audio_duration(self, audio_processor, sample_audio_file):
        """Test getting audio duration."""
        duration = await audio_processor.get_audio_duration(sample_audio_file)
        assert 0.9 <= duration <= 1.1  # Should be approximately 1 second
    
    async def test_invalid_input_file(self, audio_processor, tmp_path):
        """Test with a non-existent input file."""
        non_existent_file = tmp_path / "nonexistent.wav"
        output_file = tmp_path / "output.wav"
        
        with pytest.raises(AudioProcessingError):
            await audio_processor.concatenate_audio(
                [non_existent_file],
                output_file
            )
    
    async def test_empty_file_list(self, audio_processor, tmp_path):
        """Test with an empty list of input files."""
        output_file = tmp_path / "output.wav"
        
        with pytest.raises(AudioProcessingError):
            await audio_processor.concatenate_audio([], output_file)
    
    async def test_unsupported_format(self, audio_processor, tmp_path, sample_audio_file):
        """Test with an unsupported output format."""
        output_file = tmp_path / "output.xyz"  # Unsupported format
        
        with pytest.raises(AudioProcessingError):
            await audio_processor.concatenate_audio(
                [sample_audio_file],
                output_file,
                format="xyz"
            )
    
    async def test_trim_silence_all_silent(self, audio_processor, silent_audio_file, tmp_path):
        """Test trimming silence from a completely silent audio file."""
        output_file = tmp_path / "output.wav"
        
        # Trim silence
        result = await audio_processor.trim_silence(
            silent_audio_file,
            output_file,
            threshold=-40
        )
        
        assert result == output_file
        assert output_file.exists()
        
        # The output should be very short (effectively empty)
        duration = await audio_processor.get_audio_duration(output_file)
        assert duration < 0.1  # Should be close to 0 seconds
