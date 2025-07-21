"""TTS service configuration models."""
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from pydantic import Field, validator, HttpUrl, AnyUrl
from pydantic import ConfigDict

from ..models.enums import Language
from .base import BaseConfig


class TTSProviderConfig(BaseConfig):
    """Base configuration for TTS providers."""
    
    PROVIDER_NAME: str
    ENABLED: bool = True
    PRIORITY: int = 10  # Lower number = higher priority
    TIMEOUT: int = 30  # Request timeout in seconds
    MAX_TEXT_LENGTH: int = 5000  # Maximum characters per request
    
    model_config = ConfigDict(
        extra="ignore",
        env_prefix="TTS_"
    )


class EdgeTTSConfig(TTSProviderConfig):
    """Configuration for Edge TTS service."""
    
    PROVIDER_NAME: str = "edge_tts"
    BASE_URL: str = "https://speech.platform.bing.com/"
    DEFAULT_VOICE: str = "en-US-GuyNeural"  # Changed to male voice
    RATE: str = "+0%"
    PITCH: str = "+0Hz"
    VOLUME: str = "+0%"
    
    model_config = ConfigDict(
        extra="ignore",
        env_prefix="TTS_EDGE_"
    )


class GoogleTTSConfig(TTSProviderConfig):
    """Configuration for Google TTS service."""
    
    PROVIDER_NAME: str = "google_tts"
    CREDENTIALS_FILE: Optional[Path] = None
    PROJECT_ID: Optional[str] = None
    DEFAULT_VOICE: str = "en-US-Standard-C"
    AUDIO_ENCODING: str = "MP3"  # MP3, LINEAR16, OGG_OPUS, etc.
    SPEAKING_RATE: float = 1.0  # 0.25 to 4.0
    PITCH: float = 0.0  # -20.0 to 20.0
    VOLUME_GAIN_DB: float = 0.0  # -96.0 to 16.0
    
    model_config = ConfigDict(
        extra="ignore",
        env_prefix="TTS_GOOGLE_"
    )
    
    @validator("CREDENTIALS_FILE", pre=True)
    def validate_credentials_file(cls, v: Optional[Union[str, Path]]) -> Optional[Path]:
        """Validate and convert credentials file path."""
        if not v:
            return None
        path = Path(v).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Credentials file not found: {v}")
        return path


class TTSConfig(BaseConfig):
    """Main TTS configuration model."""
    
    # Provider configurations
    PROVIDERS: Dict[str, TTSProviderConfig] = Field(
        default_factory=dict,
        description="Available TTS providers and their configurations"
    )
    
    # Default settings
    DEFAULT_PROVIDER: str = "edge_tts"
    DEFAULT_LANGUAGE: Language = Language.ENGLISH
    DEFAULT_VOICE: Optional[str] = None
    
    # Caching settings
    CACHE_ENABLED: bool = True
    CACHE_DIR: Path = Path("data/tts_cache")
    CACHE_TTL: int = 60 * 60 * 24 * 7  # 1 week in seconds
    
    # Performance settings
    MAX_CONCURRENT_REQUESTS: int = 5
    REQUEST_TIMEOUT: int = 30  # seconds
    
    # Voice mapping (language -> provider -> voice_id)
    VOICE_MAPPING: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Mapping of language codes to provider-specific voice IDs"
    )
    
    @validator("PROVIDERS", pre=True)
    def parse_providers(
        cls, 
        v: Union[Dict[str, Dict], List[Dict], None],
        values: Dict[str, Any],
    ) -> Dict[str, TTSProviderConfig]:
        """Parse and validate provider configurations."""
        if v is None:
            return {}
            
        providers = {}
        provider_map = {
            "edge_tts": EdgeTTSConfig,
            "google_tts": GoogleTTSConfig,
        }
        
        if isinstance(v, list):
            # Convert list of provider configs to dict
            v = {p["PROVIDER_NAME"]: p for p in v if "PROVIDER_NAME" in p}
        
        for provider_name, config in v.items():
            provider_name = provider_name.lower()
            if provider_name not in provider_map:
                continue
                
            provider_cls = provider_map[provider_name]
            if isinstance(config, dict):
                providers[provider_name] = provider_cls(**config)
            elif isinstance(config, TTSProviderConfig):
                providers[provider_name] = config
        
        # Add default providers if none configured
        if not providers:
            providers["edge_tts"] = EdgeTTSConfig()
            
        return providers
    
    @validator("CACHE_DIR", pre=True)
    def validate_cache_dir(cls, v: Union[str, Path]) -> Path:
        """Ensure cache directory exists."""
        path = Path(v).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @validator("DEFAULT_PROVIDER")
    def validate_default_provider(cls, v: str, values: Dict[str, Any]) -> str:
        """Validate that the default provider is enabled."""
        providers = values.get("PROVIDERS", {})
        if v not in providers or not providers[v].ENABLED:
            # Find first enabled provider
            for name, config in providers.items():
                if config.ENABLED:
                    return name
            raise ValueError("No enabled TTS providers configured")
        return v
    
    model_config = ConfigDict(
        extra="ignore",
        env_prefix="TTS_"
    )
    
    def get_provider_config(self, provider_name: Optional[str] = None) -> TTSProviderConfig:
        """Get configuration for a specific provider."""
        provider_name = provider_name or self.DEFAULT_PROVIDER
        if provider_name not in self.PROVIDERS:
            raise ValueError(f"Unknown TTS provider: {provider_name}")
        return self.PROVIDERS[provider_name]
    
    def get_voice_for_language(
        self, 
        language: Union[str, Language],
        provider: Optional[str] = None,
    ) -> Optional[str]:
        """Get the default voice for a language and optional provider."""
        if isinstance(language, Language):
            language = language.code
            
        if provider:
            # Get voice for specific provider
            return self.VOICE_MAPPING.get(language, {}).get(provider)
        
        # Try to find a voice for the language from any provider
        for provider_voices in self.VOICE_MAPPING.get(language, {}).values():
            if provider_voices:
                return provider_voices[0] if isinstance(provider_voices, list) else provider_voices
                
        return None
