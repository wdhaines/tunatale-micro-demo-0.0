"""Integration tests for the simplified TunaTale CLI."""
import asyncio
import json
import re
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from typer.testing import CliRunner

from tunatale.cli.main import app


@pytest.fixture
def cli_runner():
    """Return a CliRunner instance for testing the CLI."""
    return CliRunner()

@pytest.fixture
def sample_lesson_file(tmp_path: Path) -> Path:
    """Create a sample lesson file for testing."""
    lesson_file = tmp_path / "test_lesson.txt"
    lesson_file.write_text("Hello world")
    return lesson_file

def test_cli_help(cli_runner):
    """Test that the simplified CLI shows help information."""
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "generate" in result.output
    # Ensure removed commands are not in help text
    assert "list-voices" not in result.output
    assert "config" not in result.output

@patch('tunatale.cli.main.run_lesson_processing', new_callable=AsyncMock)
def test_generate_command(mock_run_processing, cli_runner, sample_lesson_file, tmp_path):
    """Test the simplified generate command."""
    output_dir = tmp_path / "output"

    result = cli_runner.invoke(
        app,
        [str(sample_lesson_file), "--output", str(output_dir)],
        catch_exceptions=False
    )

    assert result.exit_code == 0
    # Assert that our core processing function was called
    mock_run_processing.assert_awaited_once()
    
    # Check that the correct arguments were passed
    args, kwargs = mock_run_processing.call_args
    assert args[0] == sample_lesson_file
    # The output path will be a timestamped subdirectory
    assert args[1].parent == output_dir


def test_generate_with_nonexistent_file(cli_runner):
    """Test the generate command with a non-existent input file."""
    result = cli_runner.invoke(app, ["generate", "nonexistent.txt"])
    assert result.exit_code != 0
    assert "Error" in result.output
    assert "does not exist" in result.output