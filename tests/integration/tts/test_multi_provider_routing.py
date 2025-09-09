"""Tests for the MultiProviderTTSService to ensure correct voice routing."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from tunatale.infrastructure.services.tts.multi_provider_tts_service import MultiProviderTTSService
from tunatale.infrastructure.services.tts.edge_tts_service import EdgeTTSService
from tunatale.infrastructure.services.tts.gtts_service import GTTSService


@pytest.fixture
def mock_edge_tts_service():
    """Mock EdgeTTSService."""
    service = MagicMock(spec=EdgeTTSService)
    service.synthesize_speech = AsyncMock()
    service.get_voice = MagicMock()
    return service

@pytest.fixture
def mock_gtts_service():
    """Mock GTTSService."""
    service = MagicMock(spec=GTTSService)
    service.synthesize_speech = AsyncMock()
    service.get_voice = MagicMock()
    return service

@pytest.fixture
def multi_provider_service(mock_edge_tts_service, mock_gtts_service):
    """Instantiate MultiProviderTTSService with mocked providers."""
    providers = {
        'edge': mock_edge_tts_service,
        'gtts': mock_gtts_service
    }
    return MultiProviderTTSService(providers)

@pytest.mark.asyncio
async def test_voice_routing(multi_provider_service: MultiProviderTTSService, mock_edge_tts_service: MagicMock, mock_gtts_service: MagicMock):
    """Test that voices are correctly routed to the appropriate TTS provider."""
    # --- Test Case 1: Route to EdgeTTS --- 
    edge_voice_id = "fil-PH-BlessicaNeural"  # This voice should be handled by EdgeTTS
    mock_edge_tts_service.get_voice.return_value = True # Simulate voice exists

    await multi_provider_service.synthesize_speech(
        text="Hello from EdgeTTS",
        voice_id=edge_voice_id,
        output_path="/tmp/edge.mp3"
    )

    # Verify EdgeTTS was called and gTTS was not
    mock_edge_tts_service.synthesize_speech.assert_called_once()
    mock_gtts_service.synthesize_speech.assert_not_called()

    # --- Test Case 2: Route to gTTS --- 
    gtts_voice_id = "fil-com.ph"  # This voice ID is specific to gTTS in the parser
    mock_gtts_service.get_voice.return_value = True # Simulate voice exists

    # Reset mocks
    mock_edge_tts_service.synthesize_speech.reset_mock()

    await multi_provider_service.synthesize_speech(
        text="Hello from gTTS",
        voice_id=gtts_voice_id,
        output_path="/tmp/gtts.mp3"
    )

    # Verify gTTS was called and EdgeTTS was not
    mock_gtts_service.synthesize_speech.assert_called_once()
    mock_edge_tts_service.synthesize_speech.assert_not_called()
