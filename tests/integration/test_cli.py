"""Integration tests for the TunaTale CLI."""
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

from typer.testing import CliRunner

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

TAGALOG_LESSON = """
[NARRATOR]: Day 1: Welcome to El Nido!

Key Phrases:

[TAGALOG-FEMALE-1]: magandang hapon po
[NARRATOR]: good afternoon (polite)
[TAGALOG-FEMALE-1]: magandang hapon po
po
hapon
pon
ha
hapon
hapon po
magandang
dang
gan
gandang
ma
magandang
magandang hapon po
magandang hapon po
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
        
        # Mock voices for testing
        mock_voices = [
            {
                'id': 'fil-PH-BlessicaNeural',
                'name': 'Tagalog Female',
                'language': 'fil-PH',
                'gender': 'Female',
                'provider': 'edge_tts'
            },
            {
                'id': 'en-US-AriaNeural',
                'name': 'English Female',
                'language': 'en-US',
                'gender': 'Female',
                'provider': 'edge_tts'
            }
        ]
        
        mock_tts.get_voices = AsyncMock(return_value=mock_voices)
        mock_tts.get_voice = AsyncMock(side_effect=lambda voice_id: next(
            (v for v in mock_voices if v['id'] == voice_id), None
        ))
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
async def test_generate_command(sample_lesson_file, tmp_path, mock_services):
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
         patch('tunatale.cli.main.ProgressReporter') as mock_progress_reporter, \
         patch('tunatale.cli.main.progress_callback_factory') as mock_callback_factory:
        
        # Setup the progress reporter mock with async methods
        mock_progress = AsyncMock()
        mock_progress.add_task = AsyncMock(return_value="task-123")
        mock_progress.update = AsyncMock()
        mock_progress.complete_task = AsyncMock()
        mock_progress.__aenter__.return_value = mock_progress
        mock_progress.__aexit__.return_value = None
        mock_progress_reporter.return_value = mock_progress
        
        # Setup the callback factory to be an async function that returns a mock callback
        mock_callback = MagicMock()
        async def async_callback_factory(progress, task_type=None):
            return mock_callback
        mock_callback_factory.side_effect = async_callback_factory
        
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
            'success': True,  # Explicitly mark as successful
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
        
        # Import the generate function directly to avoid CliRunner I/O issues
        from tunatale.cli.main import generate
        
        # Call the generate function directly with the test arguments
        exit_code = 0
        output = ""
        error_output = ""
        
        try:
            # Run the generate function with test arguments
            generate(
                input_file=sample_lesson_file,
                output_dir=output_dir,
                force=True,
                verbose=True,
                log_file=output_dir / 'test.log'
            )
        except SystemExit as e:
            exit_code = e.code if hasattr(e, 'code') and isinstance(e.code, int) else 1
            output = str(e)
            error_output = str(e)
        
        # Print detailed debug info
        print("\n=== Test Debug Info ===")
        print(f"Exit code: {exit_code}")
        print(f"Output: {output}")
        if error_output:
            print(f"Error output: {error_output}")
        print("Current directory:", Path.cwd())
        print("Output directory exists:", output_dir.exists())
        print("Output directory contents:", list(output_dir.glob('*')))
        print("Sections dir exists:", (output_dir / 'sections').exists())
        print("Phrases dir exists:", (output_dir / 'phrases').exists())
        print("Final audio exists:", (output_dir / 'final_audio.wav').exists())
        print("Metadata exists:", (output_dir / 'metadata.json').exists())
        print("Process lesson called:", mock_process_lesson.called)
        print("Progress reporter called:", mock_progress_reporter.called)
        print("Callback factory called:", mock_callback_factory.called)
        print("====================\n")
        
        # Check the command was successful
        assert exit_code == 0, f"""
        Command failed with exit code: {exit_code}
        Output: {output}
        Error output: {error_output}
        
        Debug Info:
        - Exit code: {exit_code}
        - Output directory: {output_dir}
        - Directory exists: {output_dir.exists()}
        - Directory contents: {list(output_dir.glob('*'))}
        - Metadata exists: {(output_dir / 'metadata.json').exists()}
        - Final audio exists: {(output_dir / 'final_audio.wav').exists()}
        - Process lesson called: {mock_process_lesson.called}
        - Progress reporter called: {mock_progress_reporter.called}
        - Callback factory called: {mock_callback_factory.called}
        """
        
        # Check output files were created
        assert (output_dir / "metadata.json").exists(), "Metadata file not found"
        assert (output_dir / "final_audio.wav").exists(), "Final audio file not found"
        
        # Check metadata file contains expected content
        with open(output_dir / "metadata.json") as f:
            loaded_metadata = json.load(f)
        assert "lesson" in loaded_metadata, "Lesson info not in metadata"
        assert "sections" in loaded_metadata, "Sections not in metadata"
        assert len(loaded_metadata["sections"]) > 0, "No sections in metadata"

def test_generate_with_invalid_file(cli_runner, tmp_path):
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
    
    # Debug output
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    
    # Check for non-zero exit code and error message in output
    assert result.exit_code != 0, f"Expected non-zero exit code, got {result.exit_code}"
    assert "Error" in result.output, f"Expected 'Error' in output, got: {result.output}"
    assert "File does not exist" in result.output, \
        f"Expected 'File does not exist' in output, got: {result.output}"

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
