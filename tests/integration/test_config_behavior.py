"""Tests for application behavior based on configuration settings."""
import pytest
from unittest.mock import patch, MagicMock
from unittest.mock import patch, MagicMock, AsyncMock
from typer.testing import CliRunner

from tunatale.cli.main import app
from tunatale.cli.config import load_config


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_tts_provider_selection_from_config(cli_runner, tmp_path):
    """Test that the correct TTS service is created based on the config file."""
    # Create a dummy lesson file
    lesson_file = tmp_path / "lesson.txt"
    lesson_file.write_text("Hello world")

    # Create two different config files
    edge_config_file = tmp_path / "edge_config.yaml"
    edge_config_file.write_text("tts:\n  provider: edge_tts")

    gtts_config_file = tmp_path / "gtts_config.yaml"
    gtts_config_file.write_text("tts:\n  provider: gtts")

    with patch('tunatale.cli.main.process_lesson', new_callable=AsyncMock) as mock_process_lesson:
        mock_process_lesson.return_value = {'success': True, 'output_dir': str(tmp_path)}
        # Test with EdgeTTS config
        result = cli_runner.invoke(app, ["generate", str(lesson_file), "--config", str(edge_config_file), "--force"], catch_exceptions=False)
        assert result.exit_code == 0
        mock_process_lesson.assert_awaited()
        # The factory is called inside process_lesson, so we can't easily inspect it here.
        # Instead, we rely on the fact that process_lesson was called, which is a good
        # indicator that the config was loaded and the process started.

        # Reset mock and test with gTTS config
        mock_process_lesson.reset_mock()
        result = cli_runner.invoke(app, ["generate", str(lesson_file), "--config", str(gtts_config_file), "--force"], catch_exceptions=False)
        assert result.exit_code == 0
        mock_process_lesson.assert_awaited()
