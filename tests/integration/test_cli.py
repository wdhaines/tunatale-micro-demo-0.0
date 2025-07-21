"""Integration tests for the TunaTale CLI."""
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List
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
    # Test main help
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    
    # Test generate command help
    result = cli_runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--force" in result.output
    
    # Test list-voices command help
    result = cli_runner.invoke(app, ["list-voices", "--help"])
    assert result.exit_code == 0
    assert "List available TTS voices" in result.output

def assert_timestamped_dir_structure(base_dir: Path) -> Path:
    """Assert that the directory structure follows the timestamped pattern."""
    # Check for run_YYYYMMDD_HHMMSS pattern
    dirs = list(base_dir.glob('run_*'))
    assert len(dirs) == 1, f"Expected exactly one run directory, found {len(dirs)}"
    
    run_dir = dirs[0]
    # Verify the directory name matches the expected pattern
    assert re.match(r'run_\d{8}_\d{6}$', run_dir.name), \
        f"Directory name {run_dir.name} does not match expected pattern"
    
    # Verify the directory structure
    assert (run_dir / 'sections').is_dir(), "Sections directory not found"
    assert (run_dir / 'phrases').is_dir(), "Phrases directory not found"
    assert (run_dir / 'metadata').is_dir(), "Metadata directory not found"
    assert (run_dir / 'debug.log').exists(), "Debug log not found"
    
    return run_dir

@pytest.mark.asyncio
async def test_generate_command(sample_lesson_file, tmp_path, mock_services):
    """Test the generate command with a sample lesson file."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a mock for process_lesson that returns a coroutine
    async def mock_process_lesson(lesson_file, output_dir, **kwargs):
        # Ensure output_dir is a Path object
        output_dir = Path(output_dir)
        
        # Create the expected directory structure
        sections_dir = output_dir / 'sections'
        phrases_dir = output_dir / 'phrases'
        metadata_dir = output_dir / 'metadata'
        
        # Create all directories
        for d in [output_dir, sections_dir, phrases_dir, metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Create test files
        final_audio = output_dir / 'final_audio.wav'
        section_audio = sections_dir / 'section1.wav'
        phrase_audio = phrases_dir / 'phrase1.wav'
        
        final_audio.touch()
        section_audio.touch()
        phrase_audio.touch()
        
        metadata_file = metadata_dir / 'lesson_metadata.json'
        metadata_file.write_text(json.dumps({
            'lesson': {'title': 'Test Lesson'},
            'sections': [{'id': 'section1', 'title': 'Test Section'}],
            'phrases': [{'id': 'phrase1', 'text': 'Test phrase'}]
        }))
        
        # Return the expected result structure with proper path handling
        result = {
            'success': True,
            'final_audio_file': str(final_audio.absolute()),
            'metadata_file': str(metadata_file.absolute()),
            'sections': [{
                'id': 'section1', 
                'audio_file': str(section_audio.absolute()),
                'title': 'Test Section'
            }],
            'phrases': [{
                'id': 'phrase1', 
                'audio_file': str(phrase_audio.absolute()),
                'text': 'Test phrase'
            }],
            'output_dir': str(output_dir.absolute()),
            'output_files': [
                str(final_audio.absolute()),
                str(section_audio.absolute()),
                str(phrase_audio.absolute()),
                str(metadata_file.absolute())
            ],
            'performance': {'enabled': False},
            'lesson_id': 'test-lesson-1',
            'title': 'Test Lesson'
        }
        
        return result
    
    # Mock the CLI runner
    runner = CliRunner()
    
    # Patch the process_lesson function with our mock
    with patch('tunatale.cli.main.process_lesson', new_callable=AsyncMock) as mock_pl:
        # Configure the mock to use our function
        mock_pl.side_effect = mock_process_lesson
        
        # Run the CLI command
        result = runner.invoke(
            app,
            [
                "generate",
                str(sample_lesson_file),
                "--output", str(output_dir),
                "--force"
            ],
            catch_exceptions=False
        )
    
    # Verify the command succeeded
    assert result.exit_code == 0, f"CLI command failed with output: {result.output}"
    
    # Verify the output directory structure
    run_dir = assert_timestamped_dir_structure(output_dir)
    
    # Verify the output files were created
    assert (run_dir / 'final_audio.wav').exists(), "Final audio file not found"
    assert any((run_dir / 'sections').glob('*.wav')), "No section audio files found"
    assert any((run_dir / 'phrases').glob('*.wav')), "No phrase audio files found"
    assert (run_dir / 'metadata' / 'lesson_metadata.json').exists(), "Metadata file not found"

@pytest.mark.asyncio
async def test_timestamped_output_directories(sample_lesson_file, tmp_path, mock_services):
    """Test that each run creates a new timestamped output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Helper function to create test directory structure
    def create_test_structure(run_dir):
        """Create the expected directory structure for a test run."""
        # Create directories
        sections_dir = run_dir / 'sections'
        phrases_dir = run_dir / 'phrases'
        metadata_dir = run_dir / 'metadata'
        
        sections_dir.mkdir(parents=True, exist_ok=True)
        phrases_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Create some test files
        (run_dir / 'final_audio.wav').touch()
        (sections_dir / 'section1.wav').touch()
        (phrases_dir / 'phrase1.wav').touch()
        (metadata_dir / 'lesson_metadata.json').write_text(json.dumps({
            'lesson': {'title': 'Test Lesson'},
            'sections': [{'id': 'section1', 'title': 'Test Section'}],
            'phrases': [{'id': 'phrase1', 'text': 'Test phrase'}]
        }))
        
        return {
            'success': True,
            'final_audio_file': str(run_dir / 'final_audio.wav'),
            'metadata_file': str(metadata_dir / 'lesson_metadata.json'),
            'sections': [{'id': 'section1', 'audio_file': str(sections_dir / 'section1.wav')}],
            'phrases': [{'id': 'phrase1', 'audio_file': str(phrases_dir / 'phrase1.wav')}],
            'output_dir': str(run_dir),
            'performance': {'enabled': False},
            'lesson_id': 'test-lesson-1',
            'title': 'Test Lesson'
        }
    
    # Create a mock for process_lesson that returns a coroutine
    async def mock_process_lesson(lesson_file, output_dir, **kwargs):
        # Ensure output_dir is a Path object
        output_dir = Path(output_dir)
        
        # Create the expected directory structure
        sections_dir = output_dir / 'sections'
        phrases_dir = output_dir / 'phrases'
        metadata_dir = output_dir / 'metadata'
        
        # Create all directories
        for d in [output_dir, sections_dir, phrases_dir, metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Create test files
        final_audio = output_dir / 'final_audio.wav'
        section_audio = sections_dir / 'section1.wav'
        phrase_audio = phrases_dir / 'phrase1.wav'
        
        final_audio.touch()
        section_audio.touch()
        phrase_audio.touch()
        
        metadata_file = metadata_dir / 'lesson_metadata.json'
        metadata_file.write_text(json.dumps({
            'lesson': {'title': 'Test Lesson'},
            'sections': [{'id': 'section1', 'title': 'Test Section'}],
            'phrases': [{'id': 'phrase1', 'text': 'Test phrase'}]
        }))
        
        # Return the expected result structure with proper path handling
        result = {
            'success': True,
            'final_audio_file': str(final_audio.absolute()),
            'metadata_file': str(metadata_file.absolute()),
            'sections': [{
                'id': 'section1', 
                'audio_file': str(section_audio.absolute()),
                'title': 'Test Section'
            }],
            'phrases': [{
                'id': 'phrase1', 
                'audio_file': str(phrase_audio.absolute()),
                'text': 'Test phrase'
            }],
            'output_dir': str(output_dir.absolute()),
            'output_files': [
                str(final_audio.absolute()),
                str(section_audio.absolute()),
                str(phrase_audio.absolute()),
                str(metadata_file.absolute())
            ],
            'performance': {'enabled': False},
            'lesson_id': 'test-lesson-1',
            'title': 'Test Lesson'
        }
        
        return result
    
    # Apply the mock using AsyncMock
    with patch('tunatale.cli.main.process_lesson', new_callable=AsyncMock) as mock_pl, \
         patch('tunatale.cli.main.Path') as mock_path, \
         patch('tunatale.cli.main.load_config') as mock_load_config:
        
        # Configure the mocks
        mock_pl.side_effect = mock_process_lesson
        
        # Mock the load_config to return a simple config
        mock_load_config.return_value = {
            'tts': {'provider': 'edge_tts'},
            'audio': {'sample_rate': 24000, 'channels': 1},
            'output_dir': str(output_dir)
        }
        
        # Mock Path behavior
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.is_file.return_value = True
        
        # First run
        runner = CliRunner()
        
        # Enable debug output
        import sys
        print("\n=== Starting test with debug output ===", file=sys.stderr)
        print(f"Output directory: {output_dir}", file=sys.stderr)
        print(f"Sample lesson file: {sample_lesson_file}", file=sys.stderr)
        
        # Create a custom exception handler to get more detailed error information
        def handle_exception(exc_type, exc_value, exc_traceback):
            print("\n=== Unhandled exception ===")
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
            
            # Print local variables at the time of the exception
            import inspect
            tb = exc_traceback
            while tb is not None:
                frame = tb.tb_frame
                print(f"\nFrame: {frame.f_code.co_name} in {frame.f_code.co_filename}:{tb.tb_lineno}")
                print("Locals:")
                for k, v in frame.f_locals.items():
                    print(f"  {k}: {v!r}")
                tb = tb.tb_next
            
            # Call the default exception handler
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        
        # Set the custom exception handler
        sys.excepthook = handle_exception
        
        try:
            # Temporarily capture stderr to see more detailed error
            import io
            from contextlib import redirect_stderr
            
            f = io.StringIO()
            with redirect_stderr(f):
                # Run the CLI command with debug flags
                # First, try with verbose flag only
                result1 = runner.invoke(
                    app,
                    [
                        "generate", 
                        str(sample_lesson_file), 
                        "--output", str(output_dir), 
                        "--force", 
                        "--verbose"
                    ],
                    catch_exceptions=False  # Let exceptions propagate for better debugging
                )
            
            # Print captured stderr for debugging
            stderr_output = f.getvalue()
            print("\n=== Captured stderr ===", file=sys.stderr)
            print(stderr_output, file=sys.stderr)
            
            # Print the result object for debugging
            print("\n=== Result object ===", file=sys.stderr)
            print(f"Exit code: {result1.exit_code}", file=sys.stderr)
            print(f"Exception: {result1.exception}", file=sys.stderr)
            print(f"Output: {result1.output}", file=sys.stderr)
            
            # Print the mock call arguments
            print("\n=== Mock calls ===", file=sys.stderr)
            for i, call in enumerate(mock_pl.call_args_list, 1):
                print(f"\nCall {i}:", file=sys.stderr)
                print(f"  Args: {call[0]}", file=sys.stderr)
                print(f"  Kwargs: {call[1]}", file=sys.stderr)
            
            # Print mock_path calls
            print("\n=== Path mock calls ===", file=sys.stderr)
            for i, call in enumerate(mock_path.mock_calls, 1):
                print(f"Call {i}: {call}", file=sys.stderr)
            
            # Check the result
            assert result1.exit_code == 0, f"First run failed with output: {result1.output}\nStderr: {stderr_output}"
            
        except Exception as e:
            print(f"\n=== Test failed with exception ===", file=sys.stderr)
            print(f"Type: {type(e).__name__}", file=sys.stderr)
            print(f"Message: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            
            # Print the current working directory
            import os
            print(f"\nCurrent working directory: {os.getcwd()}", file=sys.stderr)
            
            # Print directory contents if relevant
            try:
                print("\nOutput directory contents:", file=sys.stderr)
                for p in output_dir.glob('**/*'):
                    print(f"  {p.relative_to(output_dir)}", file=sys.stderr)
            except Exception as dir_err:
                print(f"Could not list output directory: {dir_err}", file=sys.stderr)
            
            # Re-raise the exception to fail the test
            raise
        
        # Get the first run's directory
        first_run_dirs = list(output_dir.glob('run_*'))
        assert len(first_run_dirs) == 1, f"Expected one run directory, found {len(first_run_dirs)}: {first_run_dirs}"
        first_run_dir = first_run_dirs[0]
        first_run_time = first_run_dir.stat().st_mtime
        
        # Second run after a short delay (to ensure different timestamps)
        import time
        time.sleep(1)  # Ensure different timestamps
        
        result2 = runner.invoke(
            app,
            ["generate", str(sample_lesson_file), "--output", str(output_dir), "--force"]
        )
        assert result2.exit_code == 0, f"Second run failed with output: {result2.output}"
        
        # Verify both runs created different directories
        run_dirs = sorted(output_dir.glob('run_*'))
        assert len(run_dirs) == 2, f"Expected two run directories, found {len(run_dirs)}: {run_dirs}"
        assert run_dirs[0] != run_dirs[1], "Run directories should be different"
        
        # Verify the directories are in chronological order
        assert run_dirs[0].stat().st_mtime < run_dirs[1].stat().st_mtime, \
            f"Expected {run_dirs[0]} to be older than {run_dirs[1]}"
    
    from tunatale.core.models.enums import SectionType, Language
    
    # Test that the output directory structure is correct for each run
    for run_dir in run_dirs:
        # Verify the directory structure
        assert (run_dir / 'sections').is_dir(), f"Sections directory not found in {run_dir}"
        assert (run_dir / 'phrases').is_dir(), f"Phrases directory not found in {run_dir}"
        assert (run_dir / 'metadata').is_dir(), f"Metadata directory not found in {run_dir}"
        
        # Verify expected files were created
        assert (run_dir / 'final_audio.wav').exists(), f"Final audio file not found in {run_dir}"
        metadata_file = run_dir / 'metadata' / 'lesson_metadata.json'
        assert metadata_file.exists(), f"Metadata file not found in {run_dir}"
        
        # Verify metadata file has content
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            assert 'sections' in metadata, "No sections in metadata"
            assert len(metadata['sections']) > 0, "No sections in metadata"

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
