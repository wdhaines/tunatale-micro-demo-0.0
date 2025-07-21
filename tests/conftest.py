"""Pytest configuration and fixtures for TunaTale tests."""
import asyncio
import pytest
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

# Add the project root to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Configure asyncio to be less verbose
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# Pytest configuration
def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (dependencies required)"
    )
    config.addinivalue_line(
        "markers",
        "e2e: mark test as end-to-end test (external services required)"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow-running"
    )

# Fixtures
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Create an instance of the default event loop for the test session.
    
    This ensures that all async tests use the same event loop.
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent

@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Return the path to the test data directory."""
    return project_root / "tests" / "fixtures"

# Sample data fixtures
@pytest.fixture
def sample_voice_data() -> Dict[str, Any]:
    """Return sample voice data for testing."""
    return {
        "name": "Test Voice",
        "provider": "test_provider",
        "provider_id": "test-voice-1",
        "language": "en",
        "gender": "female",
        "age": "adult",
        "sample_rate": 24000,
        "is_active": True,
        "metadata": {}
    }

@pytest.fixture
def sample_phrase_data() -> Dict[str, Any]:
    """Return sample phrase data for testing."""
    return {
        "text": "Hello, world!",
        "language": "en",
        "voice_id": "test-voice-1",
        "position": 1,
        "voice_settings": {
            "rate": 1.0,
            "pitch": 0.0,
            "volume": 1.0
        },
        "metadata": {}
    }
