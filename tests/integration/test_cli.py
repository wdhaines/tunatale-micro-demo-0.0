"""Integration tests for the TunaTale CLI."""
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import nest_asyncio
import pytest
from typer.testing import CliRunner

# Apply nest_asyncio to allow nested event loops for testing
nest_asyncio.apply()

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tunatale.cli.main import app
from tunatale.core.models.lesson import Lesson
from tunatale.core.models.section import Section
from tunatale.core.models.phrase import Phrase
from tunatale.core.models.enums import SectionType, Language

# Test data
SAMPLE_LESSON = """
# Test Lesson

## Introduction
This is a test lesson.

## Dialogue
- Hello, how are you? | Kamusta ka?
- I'm fine, thank you! | Mabuti naman, salamat!
"""

@pytest.fixture
def sample_lesson_file(tmp_path: Path) -> Path:
    """Create a sample lesson file for testing."""
    lesson_file = tmp_path / "test_lesson.txt"
    lesson_file.write_text(SAMPLE_LESSON)
    return lesson_file

@pytest.fixture
def mock_services():
    """Mock TTS and audio processing services."""
    with patch('tunatale.infrastructure.factories.create_tts_service') as mock_tts_factory, \
         patch('tunatale.infrastructure.factories.create_audio_processor') as mock_audio_factory:
        
        # Create mock TTS service
        mock_tts = AsyncMock()
        mock_tts.synthesize_speech = AsyncMock(side_effect=lambda text, voice_id, output_path, **kwargs: {
            'path': str(output_path),
            'cached': False,
            'voice_id': voice_id
        })
        mock_tts.get_voices = AsyncMock(return_value=[])
        mock_tts.get_voice = AsyncMock(return_value={
            'id': 'test-voice',
            'name': 'Test Voice',
            'language': 'en-US',
            'gender': 'Female',
            'provider': 'test'
        })
        mock_tts_factory.return_value = mock_tts
        
        # Create mock audio processor
        mock_audio = AsyncMock()
        mock_audio.concatenate_audio = AsyncMock(return_value=str(Path("output/final_audio.wav")))
        mock_audio.normalize_audio = AsyncMock(return_value=str(Path("output/normalized_audio.wav")))
        mock_audio.convert_format = AsyncMock(side_effect=lambda input_path, output_path, **kwargs: str(output_path))
        mock_audio_factory.return_value = mock_audio
        
        yield mock_tts, mock_audio

@pytest.fixture
def cli_runner():
    """Return a CliRunner instance for testing the CLI."""
    return CliRunner()

def test_cli_help(cli_runner):
    """Test that the CLI shows help information."""
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output

@pytest.mark.asyncio
async def test_generate_command(cli_runner, sample_lesson_file, tmp_path, mock_services):
    """Test the generate command with a sample lesson file."""
    output_dir = tmp_path / "output"
    
    # Create the output directory structure
    (output_dir / 'sections').mkdir(parents=True, exist_ok=True)
    (output_dir / 'phrases').mkdir(parents=True, exist_ok=True)
    
    # Create dummy files
    (output_dir / 'final_audio.wav').touch()
    (output_dir / 'sections' / 'section1.wav').touch()
    (output_dir / 'phrases' / 'phrase1.wav').touch()
    
    from tunatale.core.models.enums import SectionType, Language
    
    # Create a simple metadata file with required language fields
    metadata = {
        'lesson': {
            'title': 'Test Lesson',
            'target_language': Language.TAGALOG.value,  # 'tagalog'
            'native_language': Language.ENGLISH.value,  # 'english'
            'difficulty': 2,
            'estimated_duration': 15,
            'sections': []
        },
        'sections': [{
            'id': 'section1', 
            'title': 'Test Section', 
            'section_type': SectionType.KEY_PHRASES.value,
            'lesson_id': 'test-lesson-1',
            'position': 1,
            'phrases': ['phrase1']
        }],
        'phrases': [{
            'id': 'phrase1',
            'text': 'Test phrase',
            'translation': 'Test translation',
            'target_language': Language.TAGALOG.value,  # 'tagalog'
            'native_language': Language.ENGLISH.value,  # 'english'
            'section_id': 'section1',
            'position': 1
        }]
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    # Set up test environment
    test_env = {'PYTEST_CURRENT_TEST': 'test_generate_command'}
    
    # Mock the process_lesson function to avoid actual processing
    with patch.dict('os.environ', test_env, clear=True), \
         patch('builtins.input', return_value='y'), \
         patch('tunatale.cli.main.process_lesson', new_callable=AsyncMock) as mock_process_lesson, \
         patch('tunatale.cli.main.ProgressReporter') as mock_progress_reporter:
        
        # Setup the progress reporter mock
        mock_progress = MagicMock()
        mock_progress_reporter.return_value = mock_progress
        
        # Create the expected output files first
        (output_dir / 'sections').mkdir(parents=True, exist_ok=True)
        (output_dir / 'phrases').mkdir(parents=True, exist_ok=True)
        final_audio = output_dir / 'final_audio.wav'
        final_audio.touch()
        section_audio = output_dir / 'sections' / 'section1.wav'
        section_audio.parent.mkdir(parents=True, exist_ok=True)
        section_audio.touch()
        phrase_audio = output_dir / 'phrases' / 'phrase1.wav'
        phrase_audio.parent.mkdir(parents=True, exist_ok=True)
        phrase_audio.touch()
        
        # Setup the mock to return a successful result with absolute paths
        mock_process_lesson.return_value = {
            'final_audio_file': str(final_audio),
            'metadata_file': str(output_dir / 'metadata.json'),
            'sections': [
                {'id': 'section1', 'audio_file': str(section_audio), 'title': 'Test Section'}
            ],
            'phrases': [
                {'id': 'phrase1', 'audio_file': str(phrase_audio), 'text': 'Test phrase'}
            ],
            'output_dir': str(output_dir),
            'performance': {'enabled': False},
            'lesson_id': 'test-lesson-1',
            'title': 'Test Lesson'
        }
        
        # Run the CLI command
        result = cli_runner.invoke(
            app,
            [
                "generate",
                str(sample_lesson_file),
                "--output", str(output_dir),
                "--force"
            ]
        )
        
        # Check the command was successful
        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        
        # Check output files were created
        assert (output_dir / "metadata.json").exists(), "Metadata file not found"
        assert (output_dir / "final_audio.wav").exists(), "Final audio file not found"
        
        # Check metadata file contains expected content
        with open(output_dir / "metadata.json") as f:
            loaded_metadata = json.load(f)
        assert "lesson" in loaded_metadata, "Lesson info not in metadata"
        assert "sections" in loaded_metadata, "Sections not in metadata"
        assert len(loaded_metadata["sections"]) > 0, "No sections in metadata"

@pytest.mark.asyncio
async def test_generate_with_invalid_file(cli_runner, tmp_path):
    """Test the generate command with a non-existent input file."""
    invalid_file = tmp_path / "nonexistent.txt"
    output_dir = tmp_path / "output"
    
    result = cli_runner.invoke(
        app,
        [
            "generate",
            str(invalid_file),
            "--output", str(output_dir)
        ]
    )
    
    assert result.exit_code != 0
    assert "does not exist" in result.output

@pytest.mark.asyncio
async def test_list_voices_command(capsys):
    """Test the list-voices command."""
    from tunatale.cli.main import _list_voices_async
    from tunatale.core.models.voice import Voice, VoiceGender
    from tunatale.core.models.enums import Language
    
    # Create a test voice object
    test_voice = Voice(
        name="Test Voice",
        provider_id="test-voice",
        language=Language.ENGLISH,  # Use the Language enum
        gender=VoiceGender.FEMALE,
        provider="edge"
    )
    
    # Mock the TTS service
    with patch('tunatale.cli.main.create_tts_service') as mock_tts_factory:
        # Configure mock TTS service
        mock_tts = AsyncMock()
        mock_tts.get_voices = AsyncMock(return_value=[test_voice])
        mock_tts_factory.return_value = mock_tts
        
        # Call the async function directly
        await _list_voices_async(None)
        
        # Verify the output
        captured = capsys.readouterr()
        assert "Available Voices" in captured.out
        assert "Test Voice" in captured.out
        assert "test-voice" in captured.out

@pytest.mark.asyncio
async def test_config_command_show(cli_runner, tmp_path):
    """Test the config show command."""
    config_file = tmp_path / "config.json"
    config_file.write_text('{"tts": {"provider": "edge"}}')
    
    result = cli_runner.invoke(
        app,
        ["config", "--config", str(config_file), "--show"]
    )
    
    assert result.exit_code == 0
    assert "Current Configuration" in result.output
    assert "edge" in result.output
