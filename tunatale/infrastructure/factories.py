"""Factory functions for creating service instances."""
import logging
from typing import Any, Dict, Optional

from tunatale.core.ports.audio_processor import AudioProcessor
from tunatale.core.ports.tts_service import TTSService
from tunatale.infrastructure.services.audio.audio_processor import AudioProcessorService
from tunatale.infrastructure.services.tts.edge_tts_service import EdgeTTSService

# Configure logging
logger = logging.getLogger(__name__)


def create_tts_service(config: Dict[str, Any]) -> TTSService:
    """Create a TTS service based on configuration.
    
    Args:
        config: TTS configuration dictionary
        
    Returns:
        An instance of a TTS service
        
    Raises:
        ValueError: If the TTS provider is not supported
    """
    provider = config.get('provider', 'edge').lower()
    
    if provider == 'edge':
        logger.info("Using Edge TTS service")
        return EdgeTTSService(config.get('edge_tts', {}))
    
    elif provider == 'google':
        logger.info("Using Google TTS service")
        # Import here to avoid dependency if not used
        from tunatale.infrastructure.services.tts.google_tts_service import GoogleTTSService
        return GoogleTTSService(config.get('google_tts', {}))
    
    else:
        raise ValueError(f"Unsupported TTS provider: {provider}")


def create_audio_processor(config: Dict[str, Any]) -> AudioProcessor:
    """Create an audio processor based on configuration.
    
    Args:
        config: Audio processing configuration dictionary
        
    Returns:
        An instance of an audio processor
    """
    logger.info("Creating audio processor")
    return AudioProcessorService(config)


def create_lesson_processor(
    tts_config: Optional[Dict[str, Any]] = None,
    audio_config: Optional[Dict[str, Any]] = None,
) -> 'LessonProcessor':
    """Create a lesson processor with the specified services.
    
    Args:
        tts_config: TTS service configuration
        audio_config: Audio processor configuration
        
    Returns:
        An instance of a lesson processor
    """
    from tunatale.core.services.lesson_processor import LessonProcessor
    
    tts_config = tts_config or {}
    audio_config = audio_config or {}
    
    logger.info("Creating lesson processor")
    tts_service = create_tts_service(tts_config)
    audio_processor = create_audio_processor(audio_config)
    
    return LessonProcessor(
        tts_service=tts_service,
        audio_processor=audio_processor,
        config={
            'output_format': audio_config.get('output_format', 'mp3'),
            'silence_between_phrases': audio_config.get('silence_between_phrases', 0.5),
            'silence_between_sections': audio_config.get('silence_between_sections', 1.0),
            'cleanup_temp_files': audio_config.get('cleanup_temp_files', True),
        }
    )
