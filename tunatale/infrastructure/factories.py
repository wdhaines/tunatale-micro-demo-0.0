"""Factory functions for creating service instances."""
import logging
from typing import Any, Dict, Optional

from tunatale.core.ports.audio_processor import AudioProcessor
from tunatale.core.ports.tts_service import TTSService
from tunatale.infrastructure.services.audio.audio_processor import AudioProcessorService
from tunatale.infrastructure.services.tts.edge_tts_service import EdgeTTSService

# Configure logging
logger = logging.getLogger(__name__)


def create_tts_service(config: Any) -> TTSService:
    """Create a TTS service based on configuration.
    
    Args:
        config: TTS configuration (dict or Pydantic model)
        
    Returns:
        An instance of a TTS service
        
    Raises:
        ValueError: If the TTS provider is not supported
    """
    # Convert Pydantic model to dict if needed
    if hasattr(config, 'model_dump'):  # Pydantic v2
        config_dict = config.model_dump()
    elif hasattr(config, 'dict'):  # Pydantic v1
        config_dict = config.dict()
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")
    
    provider = config_dict.get('provider', 'edge').lower()
    
    if provider == 'edge':
        logger.info("Using Edge TTS service with caching disabled")
        # Create edge_tts config with cache disabled by default
        edge_config = config_dict.get('edge_tts', {})
        if 'cache_dir' not in edge_config:
            edge_config['cache_dir'] = None
        return EdgeTTSService(edge_config)
    
    elif provider == 'google':
        logger.info("Using Google TTS service")
        # Import here to avoid dependency if not used
        from tunatale.infrastructure.services.tts.google_tts_service import GoogleTTSService
        return GoogleTTSService(config_dict.get('google_tts', {}))
    
    else:
        raise ValueError(f"Unsupported TTS provider: {provider}")


def create_audio_processor(config: Any, **kwargs) -> AudioProcessor:
    """Create an audio processor based on configuration.
    
    Args:
        config: Audio processing configuration (dict or Pydantic model)
        **kwargs: Additional keyword arguments to pass to AudioProcessorService
        
    Returns:
        An instance of an audio processor
    """
    # Convert Pydantic model to dict if needed
    if config is None:
        config_dict = {}
    elif hasattr(config, 'model_dump'):  # Pydantic v2
        config_dict = config.model_dump()
    elif hasattr(config, 'dict'):  # Pydantic v1
        config_dict = config.dict()
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")
    
    # Merge with any additional kwargs
    config_dict.update(kwargs)
    
    # Only keep the expected arguments for AudioProcessorService
    expected_args = {
        'default_format',
        'silence_duration_ms',
        'silence_threshold',
        'target_lufs',
        'max_peak'
    }
    
    # Filter the config to only include expected arguments
    filtered_config = {
        k: v for k, v in config_dict.items() 
        if k in expected_args or k == 'default_format'  # Always include default_format
    }
    
    logger.info(f"Creating audio processor with config: {filtered_config}")
    return AudioProcessorService(filtered_config)


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
    from tunatale.infrastructure.services.voice.default_voice_selector import DefaultVoiceSelector
    from tunatale.infrastructure.services.word.default_word_selector import DefaultWordSelector
    
    tts_config = tts_config or {}
    audio_config = audio_config or {}
    
    logger.info("Creating lesson processor")
    tts_service = create_tts_service(tts_config)
    audio_processor = create_audio_processor(audio_config)
    
    # Create voice selector with default voices
    voice_selector = DefaultVoiceSelector()
    word_selector = DefaultWordSelector()
    
    return LessonProcessor(
        tts_service=tts_service,
        audio_processor=audio_processor,
        voice_selector=voice_selector,
        word_selector=word_selector,
        max_workers=4,  # Default value from LessonProcessor.__init__
        output_dir="output"  # Default value from LessonProcessor.__init__
    )
