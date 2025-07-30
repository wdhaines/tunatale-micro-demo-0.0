"""Multi-provider TTS service that routes voice IDs to appropriate TTS services."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from tunatale.core.exceptions import TTSServiceError, VoiceNotAvailableError
from tunatale.core.models.voice import Voice
from tunatale.core.ports.tts_service import TTSService

logger = logging.getLogger(__name__)


class MultiProviderTTSService:
    """
    A composite TTS service that routes voice IDs to appropriate providers.
    
    This allows mixing different TTS providers within the same lesson,
    enabling users to use EdgeTTS for some voices and gTTS for others.
    """
    
    def __init__(self, providers: Dict[str, TTSService]):
        """
        Initialize the multi-provider TTS service.
        
        Args:
            providers: Dictionary mapping provider names to TTS service instances
                      e.g., {'edge': EdgeTTSService(), 'gtts': GTTSService()}
        """
        self.providers = providers
        self._voice_to_provider_map: Dict[str, str] = {}
        self._all_voices: List[Voice] = []
        self._voices_loaded = False
        
        # Voice routing rules - maps voice ID patterns to providers
        self.voice_routing = {
            # EdgeTTS voices
            'fil-PH-': 'edge',
            'en-US-': 'edge',
            'en-GB-': 'edge',
            'es-MX-': 'edge',
            
            # gTTS voices  
            'fil-com.ph': 'gtts',
            'en-com': 'gtts',
            'es-com': 'gtts',
        }
    
    @property
    def name(self) -> str:
        """Get the name of the TTS service."""
        provider_names = ', '.join(self.providers.keys())
        return f"multi_provider({provider_names})"
    
    def _get_provider_for_voice(self, voice_id: str) -> str:
        """
        Determine which provider should handle a given voice ID.
        
        Args:
            voice_id: The voice ID to route
            
        Returns:
            The provider name that should handle this voice
            
        Raises:
            VoiceNotAvailableError: If no provider can handle the voice
        """
        # Check exact matches first
        for pattern, provider in self.voice_routing.items():
            if voice_id == pattern:
                return provider
        
        # Check prefix matches
        for pattern, provider in self.voice_routing.items():
            if pattern.endswith('-') and voice_id.startswith(pattern):
                return provider
        
        # If no routing rule matches, try each provider in order
        for provider_name, provider in self.providers.items():
            try:
                # Try to get the voice from this provider
                voice = provider.get_voice(voice_id)
                if voice is not None:
                    return provider_name
            except Exception:
                continue
        
        raise VoiceNotAvailableError(f"No provider found for voice: {voice_id}")
    
    async def get_voices(self, language: Optional[str] = None) -> List[Voice]:
        """Get a list of available voices from all providers."""
        if not self._voices_loaded:
            await self._load_all_voices()
        
        voices = self._all_voices
        
        if language:
            voices = [v for v in voices if v.language.value.lower() == language.lower()]
        
        return voices
    
    async def _load_all_voices(self):
        """Load voices from all providers."""
        self._all_voices = []
        self._voice_to_provider_map = {}
        
        for provider_name, provider in self.providers.items():
            try:
                if hasattr(provider, 'get_voices'):
                    provider_voices = await provider.get_voices()
                else:
                    # For providers that don't have async get_voices, skip
                    continue
                
                for voice in provider_voices:
                    self._all_voices.append(voice)
                    self._voice_to_provider_map[voice.id] = provider_name
                    if hasattr(voice, 'provider_id') and voice.provider_id != voice.id:
                        self._voice_to_provider_map[voice.provider_id] = provider_name
                
                logger.info(f"Loaded {len(provider_voices)} voices from {provider_name}")
                
            except Exception as e:
                logger.warning(f"Failed to load voices from provider {provider_name}: {e}")
        
        self._voices_loaded = True
        logger.info(f"Total voices loaded: {len(self._all_voices)}")
    
    def get_voice(self, voice_id: str) -> Optional[Voice]:
        """Get a specific voice by ID from the appropriate provider."""
        try:
            provider_name = self._get_provider_for_voice(voice_id)
            provider = self.providers[provider_name]
            return provider.get_voice(voice_id)
        except Exception as e:
            logger.debug(f"Could not get voice {voice_id}: {e}")
            return None
    
    async def validate_voice(self, voice_id: str) -> bool:
        """Validate that a voice is available from one of the providers."""
        try:
            provider_name = self._get_provider_for_voice(voice_id)
            provider = self.providers[provider_name]
            
            if hasattr(provider, 'validate_voice'):
                await provider.validate_voice(voice_id)
            else:
                # For providers without validate_voice, just check if voice exists
                voice = provider.get_voice(voice_id)
                if voice is None:
                    raise VoiceNotAvailableError(f"Voice not found: {voice_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Voice validation failed for {voice_id}: {e}")
            raise
    
    async def synthesize_speech(
        self,
        text: str,
        voice_id: str,
        output_path: Union[str, Path],
        rate: float = 1.0,
        pitch: float = 0.0,
        volume: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Synthesize speech using the appropriate provider for the voice.
        
        Args:
            text: The text to synthesize
            voice_id: The ID of the voice to use
            output_path: Path to save the audio file
            rate: Speech rate
            pitch: Pitch adjustment  
            volume: Volume level
            **kwargs: Additional arguments
            
        Returns:
            Dict containing metadata about the synthesis
            
        Raises:
            TTSServiceError: If synthesis fails
            VoiceNotAvailableError: If the voice is not available
        """
        try:
            # Route to the appropriate provider
            provider_name = self._get_provider_for_voice(voice_id)
            provider = self.providers[provider_name]
            
            logger.debug(f"Routing voice {voice_id} to provider {provider_name}")
            
            # Call the provider's synthesize_speech method
            result = await provider.synthesize_speech(
                text=text,
                voice_id=voice_id,
                output_path=output_path,
                rate=rate,
                pitch=pitch,
                volume=volume,
                **kwargs
            )
            
            # Add provider info to the result
            if isinstance(result, dict):
                result['provider'] = provider_name
            else:
                result = {
                    'provider': provider_name,
                    'output_path': str(output_path),
                    'voice_id': voice_id
                }
            
            return result
            
        except VoiceNotAvailableError:
            raise
        except Exception as e:
            logger.error(f"Speech synthesis failed for voice {voice_id}: {e}")
            raise TTSServiceError(f"Failed to synthesize speech: {e}") from e
    
    async def validate_credentials(self) -> bool:
        """Validate credentials for all providers."""
        all_valid = True
        
        for provider_name, provider in self.providers.items():
            try:
                if hasattr(provider, 'validate_credentials'):
                    is_valid = await provider.validate_credentials()
                    if not is_valid:
                        logger.warning(f"Invalid credentials for provider {provider_name}")
                        all_valid = False
                else:
                    logger.debug(f"Provider {provider_name} does not support credential validation")
            except Exception as e:
                logger.warning(f"Credential validation failed for provider {provider_name}: {e}")
                all_valid = False
        
        return all_valid