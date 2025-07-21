"""Base configuration models for TunaTale."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Tuple
from pathlib import Path
from pydantic import Field, validator, HttpUrl, DirectoryPath, FilePath, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource

from ...core.models.enums import Environment

T = TypeVar('T', bound='BaseConfig')

class BaseConfig(BaseSettings):
    """Base configuration model with common settings.
    
    This class provides common configuration settings and utilities that are used
    throughout the application. It supports loading settings from environment
    variables, .env files, and configuration files.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=True,
        validate_default=True,
        env_prefix="TUNATALE_",
        use_enum_values=True,
    )
    
    # Application settings
    APP_NAME: str = "TunaTale"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"
    
    # API settings
    API_PREFIX: str = "/api/v1"
    API_TITLE: str = "TunaTale API"
    API_DESCRIPTION: str = "API for TunaTale language learning application"
    API_VERSION: str = "0.1.0"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    CORS_METHODS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["*"]
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATEFORMAT: str = "%Y-%m-%d %H:%M:%S"
    
    # File paths
    BASE_DIR: DirectoryPath = Path(__file__).parent.parent.parent.parent
    CONFIG_DIR: DirectoryPath = BASE_DIR / "config"
    DATA_DIR: DirectoryPath = BASE_DIR / "data"
    CACHE_DIR: DirectoryPath = DATA_DIR / "cache"
    LOG_DIR: DirectoryPath = BASE_DIR / "logs"
    
    # Feature flags
    FEATURE_CACHING: bool = True
    FEATURE_METRICS: bool = True
    FEATURE_TRACING: bool = False
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """Customize the sources for configuration values."""
        # Order of precedence:
        # 1. Environment variables
        # 2. .env file
        # 3. init_settings (passed to constructor)
        # 4. file_secret_settings
        return env_settings, dotenv_settings, init_settings, file_secret_settings
    
    @validator("CORS_ORIGINS", "CORS_METHODS", "CORS_HEADERS", pre=True)
    def parse_comma_separated_list(
        cls, 
        v: Union[str, List[str]]
    ) -> List[str]:
        """Parse comma-separated strings into lists."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        return v
    
    @validator("ENVIRONMENT", pre=True)
    def validate_environment(cls, v: Any) -> Environment:
        """Validate and convert environment string to enum."""
        if isinstance(v, str):
            try:
                return Environment[v.upper()]
            except KeyError:
                pass
        return v
    
    @validator("BASE_DIR", "CONFIG_DIR", "DATA_DIR", "CACHE_DIR", "LOG_DIR", pre=True)
    def ensure_directory_exists(cls, v: Any) -> Path:
        """Ensure directories exist and return Path objects."""
        path = Path(v).resolve()
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Convert the config to a dictionary, excluding sensitive fields."""
        exclude = kwargs.pop("exclude", {"SECRET_KEY"})
        return super().model_dump(*args, exclude=exclude, **kwargs)
    
    @classmethod
    def from_yaml(cls: Type[T], file_path: Union[str, Path], **kwargs: Any) -> T:
        """Create a config instance from a YAML file."""
        import yaml
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
            
        with file_path.open("r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
            
        return cls(**{**config_data, **kwargs})
    
    @classmethod
    def from_json(cls: Type[T], file_path: Union[str, Path], **kwargs: Any) -> T:
        """Create a config instance from a JSON file."""
        import json
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
            
        with file_path.open("r", encoding="utf-8") as f:
            config_data = json.load(f)
            
        return cls(**{**config_data, **kwargs})
