"""Application configuration module.

This module provides the main application configuration and utilities for
loading configuration from various sources (environment variables, config files, etc.).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

import yaml
from pydantic import BaseModel, Field, validator, ConfigDict

from .base import BaseConfig
from .tts import TTSConfig, TTSProviderConfig

# Type variable for generic configuration models
T = TypeVar('T', bound=BaseConfig)

# Default configuration file paths
DEFAULT_CONFIG_PATHS = [
    Path("config/config.yaml"),
    Path("config/config.yml"),
    Path("config/config.json"),
]


def load_config_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML or JSON file.
    
    Args:
        file_path: Path to the configuration file.
        
    Returns:
        Dict containing the configuration.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    suffix = path.suffix.lower()
    with open(path, 'r', encoding='utf-8') as f:
        if suffix in ('.yaml', '.yml'):
            return yaml.safe_load(f) or {}
        elif suffix == '.json':
            return json.load(f) or {}
        else:
            raise ValueError(f"Unsupported config file format: {suffix}")


def find_config_file(search_paths: Optional[list[Union[str, Path]]] = None) -> Optional[Path]:
    """Find a configuration file in common locations.
    
    Args:
        search_paths: Optional list of paths to search. If not provided,
            DEFAULT_CONFIG_PATHS will be used.
            
    Returns:
        Path to the first found config file, or None if none found.
    """
    search_paths = search_paths or DEFAULT_CONFIG_PATHS
    for path in search_paths:
        path = Path(path).resolve()
        if path.exists():
            return path
    return None


class AppConfig(BaseConfig):
    """Main application configuration model.
    
    This class combines all the different configuration components into a single
    configuration object that can be used throughout the application.
    """
    
    # Core application settings
    APP_NAME: str = "TunaTale"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent
    CONFIG_DIR: Path = BASE_DIR / "config"
    DATA_DIR: Path = BASE_DIR / "data"
    CACHE_DIR: Path = DATA_DIR / "cache"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    model_config = ConfigDict(
        env_nested_delimiter="__",
        env_prefix="TUNATALE_",
    )
    
    # TTS configuration
    TTS: TTSConfig = Field(default_factory=TTSConfig)
    
    @classmethod
    def load(
        cls,
        config_file: Optional[Union[str, Path]] = None,
        env_file: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> 'AppConfig':
        """Load configuration from file and environment variables.
        
        Args:
            config_file: Path to a YAML or JSON config file. If not provided,
                will search in common locations.
            env_file: Path to a .env file to load.
            **kwargs: Additional configuration overrides.
            
        Returns:
            AppConfig instance with loaded configuration.
        """
        # If no config file provided, try to find one in default locations
        if config_file is None:
            config_file = find_config_file()
        
        # Load config from file if found
        config_data: Dict[str, Any] = {}
        if config_file and os.path.isfile(config_file):
            config_file = Path(config_file)
            if config_file.suffix.lower() in ('.yaml', '.yml'):
                config_data = yaml.safe_load(config_file.read_text(encoding='utf-8')) or {}
            elif config_file.suffix.lower() == '.json':
                config_data = json.loads(config_file.read_text(encoding='utf-8'))
            
        # Override with any kwargs
        config_data.update(kwargs)
        
        # Create and return the config instance
        if env_file:
            return cls(_env_file=env_file, **config_data)
        return cls(**config_data)
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v: str) -> str:
        """Validate and normalize environment name."""
        return v.lower()


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance.
    
    If no configuration has been loaded, a default configuration will be created.
    
    Returns:
        The global AppConfig instance.
    """
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def init_config(
    config_file: Optional[Union[str, Path]] = None,
    env_file: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> AppConfig:
    """Initialize the global configuration.
    
    This should be called once at application startup.
    
    Args:
        config_file: Path to a YAML or JSON config file.
        env_file: Path to a .env file.
        **kwargs: Additional configuration overrides.
        
    Returns:
        The initialized AppConfig instance.
    """
    global _config
    _config = AppConfig.load(config_file=config_file, env_file=env_file, **kwargs)
    return _config
