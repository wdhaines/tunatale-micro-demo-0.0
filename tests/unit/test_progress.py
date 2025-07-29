"""Unit tests for progress reporting functionality."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from tunatale.cli.main import ProgressReporter

class TestProgressReporter:
    """Tests for the ProgressReporter class."""

    @pytest.mark.asyncio
    async def test_progress_with_zero_total(self):
        """Test that progress bar handles zero total correctly."""
        # Setup
        progress = ProgressReporter()
        task_id = "test_task"
        
        # Mock console
        mock_console = MagicMock()
        progress.console = mock_console
        
        # Test with zero total
        await progress.update_async(
            task_id=task_id,
            completed=0,
            total=0,  # This would cause division by zero in the original code
            description="Test Task"
        )
        
        # Verify no division by zero occurred
        # The actual assertion is that we get here without raising an exception
        assert True
        
        # Verify the progress bar was updated with safe values
        mock_console.print.assert_called()
        
    @pytest.mark.asyncio
    async def test_progress_with_empty_file(self):
        """Test that progress bar handles empty file processing correctly."""
        # Setup
        progress = ProgressReporter()
        task_id = "empty_file_task"
        
        # Mock console
        mock_console = MagicMock()
        progress.console = mock_console
        
        # Simulate progress with empty file (0 total items)
        await progress.update_async(
            task_id=task_id,
            completed=0,
            total=0,
            description="Empty File Processing"
        )
        
        # Complete the task
        await progress.complete_task(task_id, total=0)
        
        # Verify no division by zero occurred
        # The actual assertion is that we get here without raising an exception
        assert True
