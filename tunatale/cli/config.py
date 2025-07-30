"""Configuration management for TunaTale CLI."""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field

# Default configuration paths
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "tunatale"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.yaml"


class TTSSettings(BaseModel):
    """TTS service configuration."""
    
    provider: str = Field(
        default="edge",
        description="TTS provider to use (e.g., 'edge', 'google', 'gtts', 'multi')",
    )
    
    # Edge TTS specific settings
    edge_tts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Edge TTS specific configuration",
    )
    
    # Google TTS specific settings
    google_tts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Google TTS specific configuration",
    )
    
    # Google Translate TTS specific settings
    gtts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Google Translate TTS specific configuration",
    )
    
    # Generic TTS settings
    cache_dir: Optional[str] = Field(
        default=None,
        description="Cache directory for TTS files",
    )
    rate: float = Field(
        default=1.0,
        description="Speech rate",
        ge=0.1,
        le=3.0,
    )
    pitch: float = Field(
        default=0.0,
        description="Pitch adjustment",
        ge=-20.0,
        le=20.0,
    )
    volume: float = Field(
        default=1.0,
        description="Volume level",
        ge=0.0,
        le=2.0,
    )


class AudioSettings(BaseModel):
    """Audio processing configuration."""
    
    output_format: str = Field(
        default="mp3",
        description="Output audio format (e.g., 'mp3', 'wav')",
    )
    
    silence_between_phrases: float = Field(
        default=0.5,
        description="Silence between phrases in seconds",
        ge=0.0,
    )
    
    silence_between_sections: float = Field(
        default=1.0,
        description="Silence between sections in seconds",
        ge=0.0,
    )
    
    normalize_audio: bool = Field(
        default=True,
        description="Whether to normalize audio levels",
    )
    
    trim_silence: bool = Field(
        default=True,
        description="Whether to trim silence from the beginning and end of audio",
    )
    
    cleanup_temp_files: bool = Field(
        default=True,
        description="Whether to clean up temporary files after processing",
    )


class AppConfig(BaseModel):
    """Main application configuration."""
    
    tts: TTSSettings = Field(
        default_factory=TTSSettings,
        description="TTS service configuration",
    )
    
    audio: AudioSettings = Field(
        default_factory=AudioSettings,
        description="Audio processing configuration",
    )
    
    # Add other configuration sections here as needed
    
    class Config:
        """Pydantic config."""
        
        json_encoders = {
            Path: str,
        }
        
        @classmethod
        def schema_extra(cls, schema: Dict[str, Any], model: Any) -> None:
            """Add example to schema."""
            if "properties" in schema:
                # Add example to the schema
                schema["example"] = {
                    "tts": {
                        "provider": "edge",
                        "edge_tts": {
                            "voice": "en-US-AriaNeural",
                            "rate": "+0%",
                            "pitch": "+0Hz",
                            "volume": "+0%",
                        },
                        "google_tts": {
                            "voice": "en-US-Wavenet-A",
                            "speaking_rate": 1.0,
                            "pitch": 0.0,
                            "volume_gain_db": 0.0,
                        },
                    },
                    "audio": {
                        "output_format": "mp3",
                        "silence_between_phrases": 0.5,
                        "silence_between_sections": 1.0,
                        "normalize_audio": True,
                        "trim_silence": True,
                        "cleanup_temp_files": True,
                    },
                }


def ensure_config_dir() -> Path:
    """Ensure the configuration directory exists.
    
    Returns:
        Path to the configuration directory
    """
    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CONFIG_DIR


def get_default_config_path() -> Path:
    """Get the default configuration file path.
    
    Returns:
        Path to the default configuration file
    """
    return DEFAULT_CONFIG_FILE


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load configuration from a file or create a default one.
    
    Args:
        config_path: Path to the configuration file. If None, uses default location.
        
    Returns:
        Loaded configuration
    """
    if config_path is None:
        config_path = get_default_config_path()
    
    # If the config file doesn't exist, create a default one
    if not config_path.exists():
        ensure_config_dir()
        config = AppConfig()
        save_config(config, config_path)
        return config
    
    # Load the config file
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ('.yaml', '.yml'):
                config_data = yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Create config object
        return AppConfig(**config_data)
    
    except Exception as e:
        raise ValueError(f"Failed to load configuration from {config_path}: {e}") from e


def save_config(config: AppConfig, config_path: Optional[Path] = None) -> None:
    """Save configuration to a file.
    
    Args:
        config: Configuration to save
        config_path: Path to save the configuration to. If None, uses default location.
    """
    if config_path is None:
        config_path = get_default_config_path()
    
    # Ensure the directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict and remove None values
    config_dict = config.dict(exclude_unset=True, exclude_none=True)
    
    # Save to file
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ('.yaml', '.yml'):
                yaml.dump(config_dict, f, sort_keys=False, default_flow_style=False)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    except Exception as e:
        raise ValueError(f"Failed to save configuration to {config_path}: {e}") from e


def update_config(config: AppConfig, updates: Dict[str, Any]) -> AppConfig:
    """Update a configuration with new values.
    
    Args:
        config: Configuration to update
        updates: Dictionary of updates to apply
        
    Returns:
        Updated configuration
    """
    config_dict = config.dict()
    
    # Apply updates
    for key, value in updates.items():
        if value is not None:  # Only update if value is not None
            keys = key.split('.')
            current = config_dict
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the value
            current[keys[-1]] = value
    
    # Create a new config object with the updated values
    return AppConfig(**config_dict)
