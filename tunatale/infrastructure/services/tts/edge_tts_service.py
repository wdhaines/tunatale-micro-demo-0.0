"""Edge TTS service implementation with connection pooling and optimized request handling."""
import aiofiles
import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import aiohttp
import edge_tts
from edge_tts import VoicesManager
from edge_tts.communicate import Communicate as EdgeTTSCommunicate
from pydantic import BaseModel, Field, ValidationError, validator

from tunatale.core.config.tts import TTSConfig
from tunatale.core.exceptions import (
    TTSAuthenticationError,
    TTSConnectionError,
    TTSRateLimitExceeded,
    TTSServiceError,
    TTSValidationError,
    TunaTaleError,
)
from tunatale.core.models.voice import Voice
from tunatale.core.models.enums import VoiceGender, VoiceAge, Language
from tunatale.core.ports.tts_service import TTSService
from tunatale.core.config import get_config

# Default configuration constants
DEFAULT_CONNECTION_LIMIT = 10
DEFAULT_TIMEOUT = 60  # seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0

@dataclass
class EdgeTTSConfig:
    """Configuration for Edge TTS service."""
    default_voice: Optional[str] = None
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: str = "+0%"
    cache_dir: str = "./data/tts_cache"
    connection_limit: int = DEFAULT_CONNECTION_LIMIT
    timeout: int = DEFAULT_TIMEOUT


logger = logging.getLogger(__name__)


class EdgeTTSService(TTSService):
    """Text-to-speech service using Microsoft Edge TTS with connection pooling."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        connection_limit: int = DEFAULT_CONNECTION_LIMIT,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        communicate_class: Optional[type] = None,
    ) -> None:
        """Initialize the Edge TTS service.
        
        Args:
            config: Configuration dictionary containing TTS settings.
            cache_dir: Directory to store cached audio files.
            connection_limit: Maximum number of concurrent connections.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts for failed requests.
            retry_delay: Initial delay between retries in seconds.
            communicate_class: Optional custom Communicate class for testing.
        """
        self.config = config or {}
        
        # Determine cache directory - prioritize parameter, then config, then default
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        elif 'cache_dir' in self.config and self.config['cache_dir']:
            self.cache_dir = Path(self.config['cache_dir'])
        else:
            # Use a default cache directory in the system's cache directory
            self.cache_dir = Path.home() / '.cache' / 'tunatale' / 'tts'
        
        # Ensure the cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.connection_limit = connection_limit
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session = None
        self._connector = None
        self._voice_cache = {}
        self._voices = {}  # Initialize _voices as an empty dictionary
        self._voice_cache_time = 0
        self._voice_cache_ttl = 3600  # 1 hour
        
        # Debug logging for initialization
        logger.debug(f"[DEBUG] Initializing EdgeTTSService with config: {config}")
        logger.debug(f"[DEBUG] Using cache_dir: {self.cache_dir.absolute()}")
        logger.debug(f"[DEBUG] communicate_class: {communicate_class}")
        
        # Use the provided Communicate class or the default one
        self._communicate_class = communicate_class or EdgeTTSCommunicate
        logger.debug(f"[DEBUG] Using communicate_class: {self._communicate_class}")
        
        # Ensure the communicate_class has the expected interface
        if not hasattr(self._communicate_class, '__aenter__') or not hasattr(self._communicate_class, '__aexit__'):
            logger.warning("communicate_class does not implement the async context manager protocol")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        await self.close()
    
    async def close(self):
        """Close the HTTP session and clean up resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session with connection pooling.
        
        Returns:
            An aiohttp ClientSession instance with connection pooling enabled.
            
        Raises:
            TTSConnectionError: If session creation fails.
        """
        if self._session is None or self._session.closed:
            async with asyncio.Lock():
                if self._session is None or self._session.closed:
                    try:
                        timeout = aiohttp.ClientTimeout(total=self.timeout)
                        connector = aiohttp.TCPConnector(
                            limit=self.connection_limit,
                            force_close=True,
                            enable_cleanup_closed=True
                        )
                        self._session = aiohttp.ClientSession(
                            timeout=timeout,
                            connector=connector,
                            trust_env=True
                        )
                        logger.debug("Created new aiohttp session with connection pooling")
                    except Exception as e:
                        error_msg = f"Failed to create aiohttp session: {e}"
                        logger.error(error_msg, exc_info=True)
                        raise TTSConnectionError(error_msg) from e
        return self._session
    
    @property
    def name(self) -> str:
        """Get the name of the TTS service."""
        return "edge_tts"
        
    @property
    def _voice_cache_file(self) -> Path:
        """Get the path to the voice cache file."""
        return self.cache_dir / "edge_voices.json"
    
    async def initialize(self) -> None:
        """Initialize the service and load available voices."""
        if not self._voice_cache:
            await self._load_voices()
    
    async def _load_voices(self) -> None:
        """Load available voices from the service or cache."""
        logger.debug("Entering _load_voices")
        async with asyncio.Lock():
            if self._voice_cache:
                logger.debug("Voices already loaded, returning early")
                return
                
            try:
                # Try to load from cache first
                logger.debug("Attempting to load voices from cache")
                if await self._load_voices_from_cache():
                    self._voice_cache_time = asyncio.get_event_loop().time()
                    logger.debug("Successfully loaded voices from cache")
                    return
                else:
                    logger.debug("Failed to load voices from cache, will try service")
                
                # Load from the service
                logger.info("Fetching available voices from Edge TTS service...")
                try:
                    voices = await VoicesManager.create()
                    # Convert to our Voice model
                    self._process_voices(voices.voices)
                    
                    # Cache the voices
                    if self._voice_cache:  # Only save if we have voices
                        await self._save_voices_to_cache()
                    
                    self._voice_cache_time = asyncio.get_event_loop().time()
                    logger.debug("Successfully loaded voices from service")
                except Exception as e:
                    logger.error("Failed to load voices from service: %s", str(e), exc_info=True)
                    if not self._voice_cache and self._voice_cache_file.exists():
                        logger.warning("Falling back to cached voices")
                        if not await self._load_voices_from_cache():
                            logger.error("Failed to load voices from cache")
                            raise TTSConnectionError(f"Failed to load voices: {str(e)}") from e
                
            except Exception as e:
                logger.error("Unexpected error loading voices: %s", str(e), exc_info=True)
                if not self._voice_cache:
                    if self._voice_cache_file.exists():
                        logger.warning("Falling back to cached voices due to error")
                        if not await self._load_voices_from_cache():
                            raise TTSConnectionError("Failed to load voices and no cache available") from e
                    else:
                        raise TTSConnectionError("Failed to load voices and no cache available") from e
    
    def _process_voices(self, voices_data) -> None:
        """Process voices data and populate self._voices.
        
        Args:
            voices_data: List of voice data dictionaries from the TTS service.
        """
        from tunatale.core.models.enums import Language, VoiceGender, VoiceAge
        
        self._voice_cache.clear()
        logger.debug(f"Starting _process_voices with {len(voices_data)} voices")
        
        # Log the first few voice objects for inspection at debug level
        for i, voice in enumerate(voices_data[:3]):
            logger.debug(f"Voice sample {i+1} type: {type(voice)}")
            if hasattr(voice, '__dict__'):
                logger.debug(f"Voice attrs: {vars(voice)}")
            elif isinstance(voice, dict):
                logger.debug(f"Voice dict: {voice}")
            else:
                logger.debug(f"Voice str: {str(voice)}")
        
        for voice_data in voices_data:
            try:
                logger.debug(f"Processing voice data: {str(voice_data)[:200]}...")
                
                # Extract voice attributes with robust fallbacks
                if hasattr(voice_data, 'get'):  # Dictionary-like access
                    short_name = voice_data.get("ShortName")
                    locale = voice_data.get("Locale")
                    gender = voice_data.get("Gender")
                    friendly_name = voice_data.get("FriendlyName")
                    status = voice_data.get("Status")
                    voice_type = voice_data.get("VoiceType")
                else:  # Object attribute access
                    short_name = getattr(voice_data, "ShortName", None)
                    locale = getattr(voice_data, "Locale", None)
                    gender = getattr(voice_data, "Gender", None)
                    friendly_name = getattr(voice_data, "FriendlyName", 
                                         getattr(voice_data, "LocalName", None))
                    status = getattr(voice_data, "Status", None)
                    voice_type = getattr(voice_data, "VoiceType", None)
                
                # Skip if required fields are missing
                if not short_name or not locale:
                    logger.warning(f"Skipping voice with missing required fields: {voice_data}")
                    continue
                
                # Normalize locale and extract language code
                language_code = None
                if locale:
                    # Handle both 'en-US' and 'en_US' formats
                    locale = str(locale).replace('_', '-').lower()
                    language_code = locale.split('-')[0].lower()
                
                # Default to English if no valid language code
                if not language_code or language_code not in ['en', 'fil', 'es']:
                    logger.warning(f"Voice {short_name} has unsupported locale '{locale}', defaulting to 'en'")
                    language_code = 'en'
                
                # Map gender to our VoiceGender enum with case-insensitive matching
                gender_map = {
                    'male': 'male',
                    'female': 'female',
                    'neutral': 'neutral',
                    'm': 'male',
                    'f': 'female',
                    'n': 'neutral',
                    '': 'neutral',
                    None: 'neutral'
                }
                
                # Determine voice gender with fallback to neutral
                voice_gender = gender_map.get(str(gender or '').lower(), 'neutral')
                
                # Set default values for required fields
                voice_dict = {
                    'name': friendly_name or short_name or f'Voice-{short_name}',
                    'provider': 'edge_tts',
                    'provider_id': short_name,
                    'language': language_code,
                    'gender': voice_gender,
                    'age': VoiceAge.ADULT,  # Default to adult if not specified
                    'is_active': status is None or str(status).lower() != 'disabled',
                    'metadata': {
                        'locale': locale,
                        'status': status,
                        'voice_type': voice_type,
                        'original_gender': gender,
                        'original_data': str(voice_data)[:200]  # Truncate to avoid huge logs
                    }
                }
                
                # Validate and clean the voice data
                try:
                    # Ensure language is set and valid
                    if not language_code:
                        logger.warning(f"No language code for voice {short_name}, defaulting to 'en'")
                        language_code = 'en'
                    
                    # Convert language code to Language enum with proper error handling
                    try:
                        language = Language.from_string(language_code)
                        if language is None:
                            logger.warning(f"Unsupported language code: {language_code} for voice {short_name}, defaulting to 'en'")
                            language = Language.ENGLISH
                        voice_dict['language'] = language
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Invalid language code '{language_code}' for voice {short_name}, defaulting to 'en': {e}")
                        voice_dict['language'] = Language.ENGLISH
                    
                    # Ensure gender is a valid VoiceGender
                    if 'gender' not in voice_dict or not voice_dict['gender']:
                        voice_dict['gender'] = 'neutral'
                    
                    try:
                        gender = VoiceGender.from_string(voice_dict['gender'])
                        if gender is None:
                            logger.warning(f"Could not determine gender from '{voice_dict['gender']}' for voice {short_name}, defaulting to 'neutral'")
                            gender = VoiceGender.NEUTRAL
                        voice_dict['gender'] = gender
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Invalid gender '{voice_dict['gender']}' for voice {short_name}, defaulting to 'neutral': {e}")
                        voice_dict['gender'] = VoiceGender.NEUTRAL
                    
                    # Ensure age is a valid VoiceAge
                    if 'age' not in voice_dict or not voice_dict['age']:
                        voice_dict['age'] = VoiceAge.ADULT
                    elif not isinstance(voice_dict['age'], VoiceAge):
                        try:
                            age = VoiceAge.from_string(str(voice_dict['age']))
                            voice_dict['age'] = age if age is not None else VoiceAge.ADULT
                        except (ValueError, AttributeError) as e:
                            logger.warning(f"Invalid age '{voice_dict['age']}' for voice {short_name}, defaulting to 'adult': {e}")
                            voice_dict['age'] = VoiceAge.ADULT
                    
                    # Create the Voice object with validation
                    try:
                        # Ensure required fields are present
                        if 'name' not in voice_dict or not voice_dict['name']:
                            voice_dict['name'] = f"Voice-{short_name}"
                        if 'provider' not in voice_dict or not voice_dict['provider']:
                            voice_dict['provider'] = 'edge_tts'
                        if 'provider_id' not in voice_dict or not voice_dict['provider_id']:
                            voice_dict['provider_id'] = short_name
                        
                        logger.debug(f"Creating Voice with data: {json.dumps(voice_dict, default=str, indent=2)}")
                        voice = Voice(
                            name=voice_dict['name'],
                            provider=voice_dict['provider'],
                            provider_id=voice_dict['provider_id'],
                            language=voice_dict['language'],
                            gender=voice_dict['gender'],
                            age=voice_dict['age'],
                            is_active=voice_dict.get('is_active', True),
                            metadata=voice_dict.get('metadata', {})
                        )
                        
                        self._voice_cache[voice.provider_id] = voice
                        logger.debug(f"Successfully created voice: {voice.provider_id} - {voice.name}")
                    except Exception as e:
                        logger.error(f"Failed to create Voice object: {e}\nData: {json.dumps(voice_dict, default=str, indent=2)}", exc_info=True)
                        raise
                    
                except Exception as e:
                    logger.warning(f"Failed to create voice {short_name}: {str(e)}", exc_info=True)
                    continue
                
            except Exception as e:
                logger.error(f"[DEBUG] Failed to process voice data: {e}\nVoice data: {str(voice_data)[:500]}", exc_info=True)
                raise
        
        logger.info(f"Successfully loaded {len(self._voice_cache)} voices from Edge TTS")
        
        # Log first few voices for debugging
        for i, (voice_id, voice) in enumerate(list(self._voice_cache.items())[:3]):
            logger.debug(f"Sample voice {i+1}: {voice_id} - {voice.name} "
                        f"(lang: {voice.language}, gender: {voice.gender}, age: {voice.age})")
    
    async def _load_voices_from_cache(self) -> bool:
        """Load voices from cache."""
        logger.debug(f"Attempting to load voices from cache: {self._voice_cache_file}")
        if not self._voice_cache_file.exists():
            logger.debug("Cache file does not exist")
            return False
            
        try:
            with open(self._voice_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            logger.debug(f"Cache file content: {json.dumps(data, indent=2, default=str)[:200]}...")  # Log first 200 chars of cache
            
            self._voice_cache.clear()  # Clear any existing voices
            voices_loaded = 0
            for voice_data in data.get('voices', []):
                try:
                    # Convert string enums back to proper enum types
                    if 'language' in voice_data and isinstance(voice_data['language'], str):
                        from tunatale.core.models.enums import Language
                        voice_data['language'] = Language.from_string(voice_data['language'])
                    
                    if 'gender' in voice_data and isinstance(voice_data['gender'], str):
                        from tunatale.core.models.enums import VoiceGender
                        voice_data['gender'] = VoiceGender.from_string(voice_data['gender'])
                    
                    # Create the Voice object
                    voice = Voice(**voice_data)
                    # Populate both caches
                    self._voice_cache[voice.provider_id] = voice
                    self._voices[voice.provider_id] = voice
                    voices_loaded += 1
                    logger.debug(f"Loaded voice from cache: {voice.provider_id}")
                    
                    # Debug logging
                    logger.debug(f"Voice loaded into _voices: {voice.provider_id} - {voice.name}")
                except Exception as e:
                    logger.warning("Failed to parse cached voice: %s", e, exc_info=True)
            
            logger.info("Loaded %d voices from cache", voices_loaded)
            logger.debug(f"Total voices in cache: {len(self._voice_cache)}")
            return bool(self._voice_cache)  # Return True only if we loaded voices
            
        except Exception as e:
            logger.warning("Failed to load voices from cache: %s", e, exc_info=True)
            return False
    
    async def _save_voices_to_cache(self) -> None:
        """Save voices to cache."""
        logger.debug(f"_save_voices_to_cache: Starting with {len(self._voice_cache)} voices")
        logger.debug(f"_save_voices_to_cache: Cache dir: {self.cache_dir.absolute()}")
        logger.debug(f"_save_voices_to_cache: Cache file: {self._voice_cache_file.absolute()}")
        
        if not self._voice_cache:
            logger.warning("No voices to cache")
            return
            
        try:
            # Ensure cache directory exists
            logger.debug("_save_voices_to_cache: Ensuring cache directory exists")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"_save_voices_to_cache: Cache directory exists: {self.cache_dir.exists()}")
            
            # Prepare data to cache - convert Voice objects to dictionaries
            logger.debug("_save_voices_to_cache: Preparing voice data for caching")
            data = {
                'voices': [
                    voice.dict(by_alias=True, exclude={'id', 'created_at', 'updated_at'})
                    for voice in self._voice_cache.values()
                ]
            }
            
            # Write to a temporary file first, then rename atomically
            temp_file = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode='w', 
                    encoding='utf-8',
                    dir=str(self.cache_dir),
                    delete=False,
                    suffix='.tmp',
                    prefix='edge_voices_'
                ) as f:
                    temp_file = f.name
                    logger.debug(f"_save_voices_to_cache: Writing to temp file: {temp_file}")
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                    logger.debug("_save_voices_to_cache: Successfully wrote to temp file")
                
                temp_path = Path(temp_file)
                logger.debug(f"_save_voices_to_cache: Temp file exists: {temp_path.exists()}")
                
                # On Windows, we need to remove the destination file first
                if self._voice_cache_file.exists():
                    logger.debug("_save_voices_to_cache: Removing existing cache file")
                    self._voice_cache_file.unlink()
                
                # Rename the temp file to the final name
                logger.debug(f"_save_voices_to_cache: Renaming {temp_path} to {self._voice_cache_file}")
                temp_path.rename(self._voice_cache_file)
                logger.debug("_save_voices_to_cache: Successfully renamed file")
                logger.debug(f"_save_voices_to_cache: Cache file exists: {self._voice_cache_file.exists()}")
                logger.debug("Saved voices to cache")
                
            except Exception as e:
                # Clean up temp file if rename fails
                if temp_path.exists():
                    temp_path.unlink()
                raise
            
        except Exception as e:
            logger.error("Failed to save voices to cache: %s", e, exc_info=True)
            raise
    
    async def get_voices(self, language: Optional[Union[str, Language]] = None) -> List[Voice]:
        """Get a list of available voices.

        Args:
            language: Optional language code or Language enum to filter voices by.

        Returns:
            List of available Voice objects.

        Raises:
            TTSConnectionError: If there's an error connecting to the service.
            TTSServiceError: For other TTS service errors.
        """
        try:
            await self.initialize()

            if not self._voice_cache:
                raise TTSConnectionError("Failed to load voices from service or cache")
            
            if language is not None:
                # Convert language to Language enum if it's a string
                from tunatale.core.models.enums import Language
                if isinstance(language, str):
                    try:
                        language_enum = Language.from_string(language)
                        if language_enum is None:
                            return []
                        language = language_enum
                    except (ValueError, AttributeError):
                        return []
                
                # Filter by language
                return [
                    v for v in self._voice_cache.values()
                    if v.language == language or (isinstance(v.language, Language) and v.language.code == language.code)
                ]
                
            return list(self._voice_cache.values())
            
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as e:
            logger.error(f"Connection error getting voices: {e}")
            raise TTSConnectionError(f"Failed to connect to TTS service: {e}") from e
        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            if not isinstance(e, (TTSConnectionError, TTSServiceError)):
                raise TTSServiceError(f"Failed to get voices: {e}") from e
            raise
    
    async def get_voice(self, voice_id: str) -> Optional[Voice]:
        """Get a specific voice by ID.
        
        Args:
            voice_id: The ID of the voice to retrieve.
            
        Returns:
            The Voice object if found, None otherwise.
        """
        if not self._voice_cache:
            await self._load_voices()
            
        # Convert voice_id to string to ensure consistent comparison
        voice_id_str = str(voice_id).strip()
        
        # Debug: Log the voice lookup
        logger.debug(f"Looking up voice with ID: '{voice_id_str}' (original type: {type(voice_id)})")
        
        # Check for exact match first
        voice = self._voice_cache.get(voice_id_str)
        
        # If not found, try case-insensitive match on provider_id
        if not voice:
            for vid, v in self._voice_cache.items():
                if vid.lower() == voice_id_str.lower():
                    voice = v
                    logger.debug(f"Found case-insensitive match by provider_id: '{vid}' for '{voice_id_str}'")
                    break
        
        # If still not found, try matching by name or provider_id
        if not voice:
            for v in self._voice_cache.values():
                if (v.name and v.name.lower() == voice_id_str.lower()) or \
                   (v.provider_id and v.provider_id.lower() == voice_id_str.lower()):
                    voice = v
                    logger.debug(f"Found match by name/provider_id: {v.name} / {v.provider_id}")
                    break
        
        if not voice:
            available_voices = [f"{v.name} ({v.provider_id})" for v in list(self._voice_cache.values())[:5]]
            logger.warning(
                f"Voice not found: '{voice_id_str}'. "
                f"Available voices ({len(self._voice_cache)}): {', '.join(available_voices)}..."
            )
        else:
            logger.debug(f"Found voice: {voice.name} (provider_id: {voice.provider_id})")
            
        return voice
    
    async def synthesize_speech(
        self,
        text: str,
        voice_id: str,
        output_path: Union[str, Path],
        rate: float = 1.0,
        pitch: float = 0.0,
        volume: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Synthesize speech from text using Edge TTS.

        Args:
            text: The text to synthesize.
            voice_id: The ID of the voice to use.
            output_path: The path to save the synthesized audio to.
            rate: The speech rate (0.5-3.0).
            pitch: The pitch adjustment (-20.0 to 20.0).
            volume: The volume (0.0-1.0).
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Dictionary containing the result with the following keys:
                - audio_file: Path to the generated audio file
                - voice: The voice ID used
                - text_length: Length of the input text
                - cached: Whether the result was served from cache

        Raises:
            TTSValidationError: If the input is invalid.
            TTSRateLimitExceeded: If the rate limit is exceeded.
            TTSConnectionError: If there is a connection error.
            TTSServiceError: For other TTS service errors.
        """
        logger.debug(f"[synthesize_speech] Entry point. voice_id={voice_id}, output_path={output_path}")
        logger.debug(f"[synthesize_speech] Starting synthesis for text: {text[:50]}...")
        logger.debug(f"[synthesize_speech] voice_id: {voice_id}, output_path: {output_path}")
        logger.debug(f"[synthesize_speech] rate: {rate}, pitch: {pitch}, volume: {volume}")
        logger.debug(f"[synthesize_speech] self._communicate_class: {self._communicate_class}")
        logger.debug(f"[synthesize_speech] dir(self._communicate_class): {dir(self._communicate_class)}")
        
        # Validate inputs
        if not voice_id or not isinstance(voice_id, str):
            raise TTSValidationError("Voice ID must be a non-empty string")
            
        if not text or not isinstance(text, str):
            raise TTSValidationError("Text must be a non-empty string")
            
        # Validate rate, pitch, and volume ranges
        if not (0.5 <= rate <= 3.0):
            raise TTSValidationError("Rate must be between 0.5 and 3.0")
            
        if not (-20.0 <= pitch <= 20.0):
            raise TTSValidationError("Pitch must be between -20.0 and 20.0")
            
        if not (0.0 <= volume <= 1.0):
            raise TTSValidationError("Volume must be between 0.0 and 1.0")
        
        output_path = Path(output_path).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get the voice
        logger.debug(f"[synthesize_speech] Getting voice: {voice_id}")
        voice = await self.get_voice(voice_id)
        if not voice:
            raise TTSValidationError(f"Voice not found: {voice_id}")
        
        # Generate a cache key
        cache_key = self._generate_cache_key(text, voice_id, rate, pitch, volume)
        logger.debug(f"[synthesize_speech] Cache key: {cache_key}")
        
        # Check cache first
        if self.cache_dir:
            cached_file = self.cache_dir / f"{cache_key}.mp3"
            if cached_file.exists():
                logger.debug(f"[synthesize_speech] Using cached TTS result: {cached_file}")
                # Copy the cached file to the output path
                shutil.copy2(cached_file, output_path)
                return {
                    "audio_file": output_path,
                    "voice": voice_id,
                    "text_length": len(text),
                    "cached": True
                }
        
        # Create a temporary file for the output with a specific prefix
        temp_path = None
        try:
            # Create a temporary file with a specific prefix
            temp_fd, temp_path_str = tempfile.mkstemp(
                prefix='tts_temp_',
                suffix='.mp3',
                dir=output_path.parent
            )
            os.close(temp_fd)  # Close the file descriptor as we'll reopen it later
            temp_path = Path(temp_path_str)
            
            logger.debug(f"[synthesize_speech] Created temp file: {temp_path}")
            
            # Format parameters for Edge TTS
            rate_str = f"+{int((rate - 1.0) * 100)}%"  # Convert 0.5-3.0 to -50% to +200%
            
            # Convert pitch (-20.0 to 20.0) to -2000Hz to +2000Hz with sign prefix
            pitch_value = int(pitch * 100)
            pitch_str = f"{pitch_value:+}Hz"  # The '+' formatter adds the sign for both positive and negative
            
            # Convert volume (0.0-1.0) to -100% to +0% (EdgeTTS expects relative volume change)
            # This ensures we always have a sign prefix
            if volume <= 1.0:
                volume_str = f"-{int((1.0 - volume) * 100)}%"
            else:
                volume_str = f"+{int((volume - 1.0) * 100)}%"
            
            logger.debug(f"[synthesize_speech] Parameters - rate: {rate_str}, pitch: {pitch_str}, volume: {volume_str}")
            
            # Create the communicate instance with the formatted parameters
            communicate = self._communicate_class(
                text=text,
                voice=voice_id,
                rate=rate_str,
                pitch=pitch_str,
                volume=volume_str
            )
            
            # Save the audio to the temporary file
            await communicate.save(str(temp_path))
            
            # Verify the file was created and has content
            if not temp_path.exists():
                raise TTSServiceError("Temporary file was not created")
                
            if temp_path.stat().st_size == 0:
                raise TTSServiceError("TTS service generated an empty file")
            
            # Move the temporary file to the final location
            if output_path.exists():
                output_path.unlink()  # Remove existing file if it exists
            shutil.move(str(temp_path), str(output_path))
            temp_path = None  # Prevent cleanup since we moved the file
            
            # Cache the result for future use
            if self.cache_dir:
                try:
                    await self._cache_result(cache_key, output_path.read_bytes())
                    logger.debug("Successfully cached TTS result")
                except Exception as e:
                    logger.warning(f"Failed to cache TTS result: {e}", exc_info=True)
            
            return {
                "audio_file": output_path,
                "voice": voice_id,
                "text_length": len(text),
                "cached": False
            }
            
        except Exception as e:
            # Log detailed error information
            logger.error(f"[ERROR] Exception in synthesize_speech: {e.__class__.__name__}: {str(e)}")
            
            # Helper function to get status code from various error types
            def get_status_code(err):
                if hasattr(err, 'status'):
                    return err.status
                if hasattr(err, 'code'):
                    return err.code
                if hasattr(err, 'status_code'):
                    return err.status_code
                if hasattr(err, 'response') and hasattr(err.response, 'status'):
                    return err.response.status
                if hasattr(err, 'response') and hasattr(err.response, 'status_code'):
                    return err.response.status_code
                return None
            
            # Get status code using the helper
            status_code = get_status_code(e)
            logger.error(f"[DEBUG] Status code from error: {status_code}")
            
            # Check if this is a rate limit error (status code 429 or contains 'rate limit' in message)
            is_rate_limit = (
                status_code == 429 or 
                'rate limit' in str(e).lower() or 
                (hasattr(e, 'message') and 'rate limit' in str(e.message).lower())
            )
            
            if is_rate_limit:
                # Get headers from the error object
                headers = {}
                if hasattr(e, 'headers') and e.headers is not None:
                    headers = dict(e.headers)
                elif hasattr(e, 'response') and hasattr(e.response, 'headers'):
                    headers = dict(e.response.headers)
                
                # Get retry_after from headers or use default
                retry_after = str(headers.get('Retry-After', '60')).strip()
                error_msg = f"Rate limit exceeded, retry after {retry_after} seconds"
                
                # Log the rate limit error
                logger.error(f"[RATE_LIMIT] {error_msg}")
                
                # Create and raise the rate limit error
                rate_limit_error = TTSRateLimitExceeded(error_msg)
                rate_limit_error.retry_after = retry_after
                raise rate_limit_error from e
            
            # Handle connection errors
            if isinstance(e, (aiohttp.ClientError, ConnectionError, asyncio.TimeoutError)):
                raise TTSConnectionError(f"Network error during TTS request: {str(e)}") from e
                
            # Handle any other errors
            raise TTSServiceError(f"Failed to synthesize speech: {str(e)}") from e
            
        finally:
            # Clean up temporary file if it still exists
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_path}: {e}")
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Starting TTS synthesis (attempt {attempt + 1}/{max_retries})...")
                logger.debug(f"Text to synthesize (length: {len(text)}): {text[:100]}{'...' if len(text) > 100 else ''}")
                
                # Create a temporary file in the same directory as the output to ensure same filesystem
                temp_dir = output_path.parent
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                with tempfile.NamedTemporaryFile(
                    dir=str(temp_dir),
                    prefix="tts_temp_",
                    suffix='.mp3',
                    delete=False
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    logger.debug(f"Using temporary file: {temp_path}")
                    
                    # Convert numeric values to string format expected by Edge TTS
                    rate_str = f"{int(rate * 100)}%" if rate != 1.0 else "+0%"
                    pitch_str = f"{int(pitch * 100)}Hz" if pitch != 0.0 else "+0Hz"
                    vol_str = f"{int(volume * 100)}%" if volume != 1.0 else "+0%"
                    
                    # Debug the Communicate class itself
                    logger.debug(f"[DEBUG] Communicate class before instantiation: {self._communicate_class}")
                    logger.debug(f"[DEBUG] Communicate class module: {self._communicate_class.__module__}")
                    logger.debug(f"[DEBUG] EdgeTTS module: {edge_tts.__file__}")
                    
                    # Create and use the communicate object with async context manager
                    logger.debug(f"[DEBUG] Creating Communicate instance with text: {text[:100]}...")
                    logger.debug(f"[DEBUG] Voice ID: {voice_id}")
                    logger.debug(f"[DEBUG] About to create Communicate instance with:")
                    logger.debug(f"[DEBUG]   text: {text}")
                    logger.debug(f"[DEBUG]   voice: {voice_id}")
                    logger.debug(f"[DEBUG]   rate: {rate_str}")
                    logger.debug(f"[DEBUG]   pitch: {pitch_str}")
                    logger.debug(f"[DEBUG]   volume: {vol_str}")
                    logger.debug(f"[DEBUG]   Using communicate_class: {self._communicate_class}")
                    logger.debug(f"[DEBUG]   communicate_class module: {getattr(self._communicate_class, '__module__', 'unknown')}")
                    
                    # Debug the class attributes
                    logger.debug(f"[DEBUG] communicate_class attributes: {dir(self._communicate_class)}")
                    
                    try:
                        # Create the Communicate instance
                        communicate = self._communicate_class(
                            text=text,
                            voice=voice_id,
                            rate=rate_str,
                            pitch=pitch_str,
                            volume=vol_str,
                            proxy=None
                        )
                        
                        logger.debug(f"[DEBUG] Successfully created Communicate instance: {communicate}")
                        logger.debug(f"[DEBUG] Instance type: {type(communicate)}")
                        logger.debug(f"[DEBUG] Instance attributes: {dir(communicate)}")
                        logger.debug(f"[DEBUG] Instance module: {getattr(communicate, '__module__', 'unknown')}")
                    except Exception as e:
                        logger.error(f"Failed to create Communicate instance: {e}", exc_info=True)
                        raise
                
                # Set the session on the communicate object if it has the attribute
                if hasattr(communicate, '_session'):
                    communicate._session = await self.get_session()
                
                # Save the audio to the temporary file
                try:
                    logger.debug(f"Saving audio to temporary file: {temp_path}")
                    await communicate.save(str(temp_path))
                    
                    # Verify the file was created and has content
                    if not temp_path.exists():
                        raise TTSServiceError(f"Temporary file was not created: {temp_path}")
                        
                    file_size = temp_path.stat().st_size
                    logger.debug(f"Temporary file size: {file_size} bytes")
                    
                    if file_size == 0:
                        raise TTSServiceError(f"TTS service generated an empty file: {temp_path}")
                    
                    # If we get here, the save was successful
                    # Call assert_called_once if it exists (for testing)
                    # This is used in tests to verify the save method was called
                    if hasattr(communicate, 'assert_called_once') and callable(communicate.assert_called_once):
                        # Call the method on the instance to verify save was called once
                        communicate.assert_called_once()
                except Exception as e:
                    # Handle any errors that occur during save or verification
                    # This will raise an appropriate exception with the cause preserved
                    await self._handle_tts_error(e, temp_path=temp_path, output_path=output_path)
                
                # Cache the result for future use
                try:
                    shutil.copy2(temp_path, cache_path)
                    logger.debug(f"Cached TTS result to {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to cache TTS result: {e}", exc_info=True)
                
                # Move temp file to final location
                try:
                    # Ensure output directory exists
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # On Windows, we need to remove the destination file first if it exists
                    if output_path.exists():
                        output_path.unlink()
                    
                    # Use rename for atomic operation (same filesystem)
                    temp_path.rename(output_path)
                    logger.info(f"Saved TTS output to {output_path} ({output_path.stat().st_size} bytes)")
                    
                    return {
                        "audio_file": output_path,
                        "voice": voice_id,
                        "text_length": len(text),
                        "cached": False
                    }
                    
                except Exception as e:
                    error_msg = f"Failed to move temporary file to {output_path}: {e}"
                    logger.error(error_msg, exc_info=True)
                    raise FileOperationError(error_msg) from e
                        
            except Exception as e:
                # Check if this is a rate limit error
                if self._is_rate_limit_error(e):
                    logger.error(f"Rate limit exceeded: {e}")
                    # This will raise an appropriate exception with the cause preserved
                    await self._handle_tts_error(e, temp_path=temp_path, output_path=output_path)
                    return  # This line should never be reached as _handle_tts_error raises an exception
                    
                if attempt == max_retries - 1:
                    # On final attempt, clean up and re-raise with proper error handling
                    # This will raise an appropriate exception with the cause preserved
                    await self._handle_tts_error(e, temp_path=temp_path, output_path=output_path)
                    return  # This line should never be reached as _handle_tts_error raises an exception
                
                # Log the error and wait before retrying
                if isinstance(e, (aiohttp.ClientError, asyncio.TimeoutError)):
                    error_type = "network error"
                else:
                    error_type = "error"
                
                # Calculate exponential backoff with jitter
                backoff = min(base_delay * (2 ** attempt) + random.uniform(0, 0.1 * base_delay), max_delay)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed with {error_type}: {e}. "
                    f"Retrying in {backoff:.2f}s..."
                )
                await asyncio.sleep(backoff)
        
        # This should never be reached due to the raise in the except block above
        raise TTSServiceError("Unexpected error: Reached end of retry loop without success or proper error handling")
        
    async def batch_synthesize(
        self,
        texts: List[Tuple[str, str, Path]],  # List of (text, voice_id, output_path) tuples
        rate: float = 1.0,
        pitch: float = 0.0,
        volume: float = 1.0,
        max_concurrent: Optional[int] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Synthesize multiple texts in a batch with optimized connection handling.
        
        This method processes multiple TTS requests concurrently using a connection pool
        and implements rate limiting to avoid overwhelming the TTS service.
        
        Args:
            texts: List of tuples containing (text, voice_id, output_path)
            rate: Speech rate (0.5-3.0).
            pitch: Pitch adjustment (-20 to 20).
            volume: Volume level (0.0-1.0).
            max_concurrent: Maximum number of concurrent TTS requests. If None, uses the
                          connection limit from the config.
            **kwargs: Additional parameters for the TTS service.
            
        Returns:
            List of dictionaries containing metadata about each synthesis. Each dictionary
            will contain either the synthesis results or error information.
        """
        if not texts:
            return []
            
        # Use configured connection limit if max_concurrent is not specified
        if max_concurrent is None:
            max_concurrent = self.config.connection_limit
            
        logger.info(f"Starting batch synthesis of {len(texts)} texts with {max_concurrent} concurrent requests")
        
        # Process texts in parallel using semaphore to limit concurrency
        sem = asyncio.Semaphore(max_concurrent)
        
        async def process_one(text_item: Tuple[str, str, Path]) -> Dict[str, Any]:
            text, voice_id, output_path = text_item
            async with sem:
                try:
                    start_time = asyncio.get_event_loop().time()
                    logger.debug(f"Starting synthesis for text (length: {len(text)}) with voice: {voice_id}")
                    
                    result = await self.synthesize_speech(
                        text=text,
                        voice_id=voice_id,
                        output_path=output_path,
                        rate=rate,
                        pitch=pitch,
                        volume=volume,
                        **kwargs
                    )
                    
                    duration = asyncio.get_event_loop().time() - start_time
                    logger.debug(
                        f"Completed synthesis in {duration:.2f}s - "
                        f"Text length: {len(text)}, Output: {output_path}"
                    )
                    
                    return {
                        **result,
                        "success": True,
                        "duration": duration,
                        "text_length": len(text)
                    }
                    
                except Exception as e:
                    duration = asyncio.get_event_loop().time() - start_time
                    error_msg = f"Failed to synthesize text (length: {len(text)}): {e}"
                    logger.error(error_msg, exc_info=True)
                    
                    return {
                        "success": False,
                        "error": str(e),
                        "error_type": e.__class__.__name__,
                        "text_length": len(text),
                        "voice_id": voice_id,
                        "output_path": str(output_path),
                        "duration": duration
                    }
        
        # Process all texts concurrently with progress tracking
        start_time = asyncio.get_event_loop().time()
        results = []
        
        try:
            # Process in chunks to avoid creating too many tasks at once
            chunk_size = max_concurrent * 2
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i + chunk_size]
                chunk_results = await asyncio.gather(
                    *[process_one(item) for item in chunk],
                    return_exceptions=False
                )
                results.extend(chunk_results)
                
                # Log progress
                processed = min(i + len(chunk), len(texts))
                logger.info(
                    f"Processed {processed}/{len(texts)} texts "
                    f"({processed/len(texts):.1%})"
                )
                
        except Exception as e:
            logger.error(f"Fatal error during batch synthesis: {e}", exc_info=True)
            raise
            
        finally:
            # Log summary
            total_duration = asyncio.get_event_loop().time() - start_time
            success_count = sum(1 for r in results if r.get('success', False))
            avg_duration = total_duration / len(results) if results else 0
            
            logger.info(
                f"Batch synthesis completed in {total_duration:.2f}s - "
                f"{success_count}/{len(results)} successful, "
                f"average {avg_duration:.2f}s per request"
            )
        
        return results

    def _is_rate_limit_error(self, error: Optional[Exception]) -> bool:
        """Check if the given error is a rate limit error.
        
        Args:
            error: The exception to check.
            
        Returns:
            bool: True if the error indicates a rate limit was exceeded.
        """
        if error is None:
            return False
            
        # Log the error type for debugging
        error_type = type(error).__name__
        logger.debug(f"_is_rate_limit_error: Checking error of type {error_type}")
        
        # Check if this is an aiohttp.ClientResponseError with status 429 or our custom RateLimitError
        if (error_type == 'ClientResponseError' and hasattr(error, 'status') and error.status == 429) or \
           (error_type == 'RateLimitError' and getattr(error, 'status', None) == 429):
            logger.debug(f"_is_rate_limit_error: Detected {error_type} with status 429")
            return True
            
        # Check status code directly on the error object
        status_code = getattr(error, 'status', getattr(error, 'code', getattr(error, 'status_code', None)))
        if status_code == 429:
            logger.debug("_is_rate_limit_error: Detected status code 429 in error attributes")
            return True
            
        # Check response object if it exists
        if hasattr(error, 'response'):
            response = error.response
            status = getattr(response, 'status', getattr(response, 'status_code', None))
            if status == 429:
                logger.debug("_is_rate_limit_error: Detected response with status 429")
                return True
                
        # Check error message for rate limit indicators
        error_msg = str(error).lower()
        if 'rate limit' in error_msg or 'too many requests' in error_msg or '429' in error_msg:
            logger.debug(f"_is_rate_limit_error: Detected rate limit in error message: {error_msg[:200]}")
            return True
            
        # Check nested exceptions
        if hasattr(error, '__cause__') and error.__cause__ is not None:
            return self._is_rate_limit_error(error.__cause__)
            
        return False

    async def _handle_tts_error(self, error: Exception, temp_path: Optional[Path] = None, output_path: Optional[Path] = None) -> None:
        """Handle TTS errors and raise appropriate exceptions.
        
        Args:
            error: The exception that was raised.
            temp_path: Path to temporary file that may need cleanup.
            output_path: Path to output file that may be partially written.
            
        Raises:
            TTSRateLimitExceeded: If the request was rate limited.
            TTSConnectionError: If there was a connection error.
            TTSValidationError: If there was a validation error.
            TTSServiceError: For other TTS service errors.
        """
        logger.debug(f"_handle_tts_error: Handling error: {error}")
        logger.debug(f"_handle_tts_error: Error type: {type(error).__name__}")
        logger.debug(f"_handle_tts_error: Error dir: {dir(error)}")
        
        # Log all attributes of the error for debugging
        for attr in dir(error):
            try:
                if not attr.startswith('__'):
                    logger.debug(f"_handle_tts_error: error.{attr} = {getattr(error, attr, 'N/A')}")
            except Exception as e:
                logger.debug(f"_handle_tts_error: Could not get attribute {attr}: {e}")
        
        # Clean up any temporary files
        try:
            if temp_path and temp_path.exists():
                logger.debug(f"_handle_tts_error: Cleaning up temporary file: {temp_path}")
                temp_path.unlink()
                logger.debug(f"_handle_tts_error: Cleaned up temporary file: {temp_path}")
            else:
                logger.debug("_handle_tts_error: No temporary file to clean up")
        except Exception as cleanup_error:
            logger.warning(f"_handle_tts_error: Failed to clean up temporary file: {cleanup_error}")
            
        try:
            if output_path and output_path.exists():
                logger.debug(f"_handle_tts_error: Cleaning up output file: {output_path}")
                output_path.unlink()
                logger.debug(f"_handle_tts_error: Cleaned up partial output file: {output_path}")
            else:
                logger.debug("_handle_tts_error: No output file to clean up")
        except Exception as cleanup_error:
            logger.warning(f"_handle_tts_error: Failed to clean up partial output file: {cleanup_error}")
        
        # Check if this is a rate limit error
        logger.debug("_handle_tts_error: Checking if error is a rate limit error")
        try:
            is_rate_limit = self._is_rate_limit_error(error)
            logger.debug(f"_handle_tts_error: is_rate_limit: {is_rate_limit}")
            
            if is_rate_limit:
                logger.debug("_handle_tts_error: Raising TTSRateLimitExceeded")
                # Get status code if available
                status_code = getattr(error, 'status', getattr(error, 'code', None))
                status_msg = f" (Status: {status_code})" if status_code is not None else ""
                error_message = f"Rate limit exceeded{status_msg}: {str(error)}"
                logger.debug(f"_handle_tts_error: Creating TTSRateLimitExceeded with message: {error_message}")
                
                # Create the exception with the original error as the cause
                exc = TTSRateLimitExceeded(error_message)
                # Set the cause to the original error to preserve the full traceback
                exc.__cause__ = error
                raise exc
                
        except Exception as e:
            logger.error(f"_handle_tts_error: Error in rate limit detection: {e}", exc_info=True)
            # If there's an error in rate limit detection, continue to handle other error types
            
        # Handle specific error types
        if isinstance(error, (asyncio.TimeoutError, aiohttp.ClientError, ConnectionError, TTSConnectionError)):
            if isinstance(error, TTSConnectionError):
                # If it's already a TTSConnectionError, re-raise it directly
                raise error
            # Otherwise, create a new TTSConnectionError
            raise TTSConnectionError(
                f"Connection error during TTS synthesis: {str(error)}"
            ) from error
            
        if isinstance(error, (ValueError, TypeError, KeyError, AttributeError, TTSValidationError)):
            if isinstance(error, TTSValidationError):
                # If it's already a TTSValidationError, re-raise it directly
                raise error
            # Otherwise, create a new TTSValidationError
            raise TTSValidationError(
                f"Invalid TTS request: {str(error)}"
            ) from error
            
        # If it's already a TTSServiceError, re-raise it directly
        if isinstance(error, TTSServiceError):
            raise error
            
        # Default to generic service error
        raise TTSServiceError(
            f"Failed to synthesize speech: {str(error)}"
        ) from error
    
    async def _get_voice(self, voice_id: str) -> Voice:
        """Get a voice by ID.
        
        Args:
            voice_id: The ID of the voice to retrieve.
            
        Returns:
            The Voice object for the specified ID.
            
        Raises:
            TTSValidationError: If the voice ID is not found.
        """
        # First, try to get the voice from the cache
        if hasattr(self, '_voices') and voice_id in self._voices:
            return self._voices[voice_id]
            
        # If not in cache, load all voices
        voices = await self.get_voices()
        
        # Try to find the voice by ID
        for voice in voices:
            if voice.provider_id == voice_id or voice.id == voice_id:
                # Cache the voice for future lookups
                if not hasattr(self, '_voices'):
                    self._voices = {}
                self._voices[voice_id] = voice
                return voice
                
        # If we get here, the voice was not found
        raise TTSValidationError(f"Voice not found: {voice_id}")
        
    async def _cache_result(self, cache_key: str, audio_data: bytes) -> Path:
        """Cache the result of a TTS synthesis operation.
        
        Args:
            cache_key: The cache key for the audio file.
            audio_data: The audio data to cache.
            
        Returns:
            Path to the cached file.
            
        Raises:
            TTSServiceError: If caching fails.
        """
        if not self.cache_dir:
            raise TTSServiceError("Cache directory not configured")
            
        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Write audio data to cache file
            cache_file = self.cache_dir / cache_key
            
            async with aiofiles.open(cache_file, 'wb') as f:
                await f.write(audio_data)
                
            logger.debug(f"Cached TTS result to {cache_file}")
            return cache_file
            
        except Exception as e:
            logger.error(f"Failed to cache TTS result: {e}")
            raise TTSServiceError(f"Failed to cache TTS result: {e}") from e
            
    def _get_cached_file(self, cache_key: str) -> Optional[Path]:
        """Get the path to a cached audio file if it exists.
        
        Args:
            cache_key: The cache key for the audio file.
            
        Returns:
            Path to the cached file if it exists, None otherwise.
        """
        if not self.cache_dir:
            return None
            
        cache_file = self.cache_dir / cache_key
        return cache_file if cache_file.exists() else None

    def _generate_cache_key(self, text: str, voice_id: str, rate: str | float = 1.0, pitch: str | float = 0.0, volume: str | float = 1.0) -> str:
        """Generate a cache key for the given text and voice settings.

        Args:
            text: The text to synthesize.
            voice_id: The voice ID to use.
            rate: Speech rate as a string (e.g., "+0%") or float (0.5-3.0).
            pitch: Pitch adjustment as a string (e.g., "+0Hz") or float (-20 to 20).
            volume: Volume level as a string (e.g., "+0%") or float (0.0-1.0).

        Returns:
            A unique cache key string.
        """
        def _parse_value(value, is_percent=False, is_hz=False):
            """Helper to parse a value that might be a string with units."""
            if isinstance(value, str):
                # Remove any non-numeric characters except +, -, and .
                cleaned = ''.join(c for c in value if c.isdigit() or c in '+-.')
                try:
                    return float(cleaned)
                except (ValueError, TypeError):
                    return 0.0 if is_percent or is_hz else 1.0
            return float(value)
        
        # Parse values with appropriate unit handling
        rate_val = _parse_value(rate, is_percent=True)
        pitch_val = _parse_value(pitch, is_hz=True)
        volume_val = _parse_value(volume, is_percent=True)
        
        # Create a unique key based on the text and voice settings
        key_parts = [
            f"voice={voice_id}",
            f"rate={rate_val:.1f}",
            f"pitch={pitch_val:+.1f}",
            f"volume={volume_val:.1f}",
            f"text={text}"
        ]
        
        # Join the parts with a delimiter that's unlikely to appear in the values
        return "|".join(key_parts)

    async def validate_credentials(self) -> bool:
        """Validate that the service credentials are valid.
        
        Returns:
            bool: True if credentials are valid, False otherwise.
            
        Raises:
            TTSAuthenticationError: If credentials are invalid or missing.
        """
        try:
            # Edge TTS doesn't require credentials, so we just check if we can get voices
            voices = await self.get_voices()
            return len(voices) > 0
        except Exception as e:
            raise TTSAuthenticationError(f"Failed to validate credentials: {e}") from e
