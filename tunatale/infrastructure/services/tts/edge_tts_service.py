"""Edge TTS service implementation for TunaTale."""

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from tunatale.core.utils.tts_preprocessor import (
    preprocess_text_for_tts,
    enhanced_preprocess_text_for_tts,
    SSMLProcessingResult
)

import aiofiles
import aiohttp
import edge_tts
from edge_tts import VoicesManager
from edge_tts.exceptions import NoAudioReceived, WebSocketError

from tunatale.core.ports.tts_service import (
    TTSService,
    TTSValidationError,
    TTSRateLimitError,
    TTSTransientError,
    TTSAuthenticationError,
)
from tunatale.core.models.voice import Voice
from tunatale.core.models.enums import Language, VoiceGender, VoiceAge
from tunatale.utils.file_utils import ensure_directory, sanitize_filename

logger = logging.getLogger(__name__)

# Default voice to use if none is specified
DEFAULT_VOICE = "en-US-JennyNeural"

# Maximum text length for a single TTS request (in characters)
MAX_TEXT_LENGTH = 5000

# Maximum number of retries for failed requests
MAX_RETRIES = 3

# Delay between retries in seconds (exponential backoff)
RETRY_DELAY = 1.0

# Timeout for TTS requests in seconds
REQUEST_TIMEOUT = 30.0

# Rate limiting: minimum delay between TTS requests (seconds)
MIN_REQUEST_DELAY = 0.2

# Rate limiting: additional delay for connection errors (seconds)
CONNECTION_ERROR_DELAY = 2.0

# Maximum concurrent TTS operations
MAX_CONCURRENT_REQUESTS = 3

# Minimum audio file size to be considered valid (bytes)
MIN_AUDIO_SIZE = 100


class EdgeTTSService(TTSService):
    """Text-to-speech service using Microsoft Edge TTS."""

    @property
    def name(self) -> str:
        """Get the name of the TTS service.
        
        Returns:
            str: The name of the service (e.g., 'edge_tts')
        """
        return "edge_tts"

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path, Dict[str, Any]]] = None,
        connection_limit: int = MAX_CONCURRENT_REQUESTS,
        rate: float = 1.0,
        pitch: float = 0.0,
        volume: float = 1.0,
        communicate_class: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Edge TTS service.

        Args:
            cache_dir: Directory to store cached audio files. If None, a default cache directory
                      will be used in the system's temp directory. Can be a string, Path, or a 
                      dictionary containing a 'cache_dir' key.
            connection_limit: Maximum number of concurrent connections to the TTS service.
            rate: Speech rate (0.5-3.0).
            pitch: Pitch adjustment (-20 to 20).
            volume: Volume level (0.0-1.0).
            communicate_class: Class to use for communication with the Edge TTS service.
                              If None, uses the default edge_tts.Communicate class.
            **kwargs: Additional arguments passed to the base class.
        """
        # Handle case where cache_dir is a dictionary (from config)
        if isinstance(cache_dir, dict):
            cache_dir = cache_dir.get('cache_dir')
        
        # Only set up caching if cache_dir is provided
        if cache_dir is not None:
            # Convert cache_dir to Path if it's a string
            if not isinstance(cache_dir, Path):
                cache_dir = Path(cache_dir)
            
            # Create cache directory if it doesn't exist
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized TTS cache at: {cache_dir}")
        else:
            logger.info("TTS caching is disabled")
            
        # Initialize the base class with the processed cache_dir
        super().__init__(cache_dir=cache_dir, **kwargs)
        
        self.connection_limit = connection_limit
        self.rate = rate
        self.pitch = pitch
        self.volume = volume
        self._communicate_class = communicate_class or edge_tts.Communicate
        self._voices: Dict[str, Dict[str, Any]] = {}
        self._voice_objects: Dict[str, Voice] = {}
        self._voices_fetched = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        
        # Rate limiting state
        self._last_request_time: float = 0.0
        self._request_count: int = 0
        self._rate_limit_semaphore: Optional[asyncio.Semaphore] = None
        
        # Store cache directory
        self.cache_dir = cache_dir

    async def __aenter__(self) -> "EdgeTTSService":
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        self._semaphore = asyncio.Semaphore(self.connection_limit)
        self._rate_limit_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None

    def _get_cached_file(self, cache_key: str) -> Optional[Path]:
        """Get the path to a cached file if it exists.
        
        Args:
            cache_key: The cache key to look up (expected format: "{voice_id}_{text_hash}" or "{voice_id}_{text_hash}.mp3").
            
        Returns:
            Path to the cached file if it exists, None otherwise.
        """
        if not self.cache_dir:
            return None
        
        # If the cache key already ends with .mp3, use it as is
        if cache_key.endswith('.mp3'):
            cache_path = self.cache_dir / cache_key
        else:
            # Otherwise, append .mp3 to the cache key
            cache_path = self.cache_dir / f"{cache_key}.mp3"
        
        # Verify the file exists and has content
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return cache_path
        return None
    
    def _convert_to_voice(self, voice_data: Dict[str, Any]) -> Optional[Voice]:
        """Convert Edge TTS voice data to a Voice object.
        
        Args:
            voice_data: Raw voice data from Edge TTS
            
        Returns:
            Voice: Converted Voice object, or None if conversion fails
            
        Raises:
            ValueError: If voice data is invalid
        """
        if not voice_data or not isinstance(voice_data, dict):
            logger.warning("Invalid voice data: expected a dictionary")
            return None
            
        try:
            # Get required fields with fallbacks
            voice_id = (voice_data.get('ShortName') or voice_data.get('Name', '')).strip()
            if not voice_id:
                logger.warning("Voice data missing required 'ShortName' or 'Name' field")
                return None
                
            # Extract locale and language
            locale = (voice_data.get('Locale') or '').strip()
            
            # Map locale to Language enum with debug logging
            language = self._map_locale_to_language(locale, voice_id)
            
            # If language couldn't be determined from locale, try to extract from voice ID
            if not language:
                voice_id_lower = voice_id.lower()
                if any(tagalog_code in voice_id_lower for tagalog_code in ['fil', 'tagalog', 'tl']):
                    language = Language.TAGALOG
                    logger.debug(f"Mapped voice ID {voice_id} to Tagalog based on ID pattern")
                elif any(en_code in voice_id_lower for en_code in ['en', 'eng', 'english']):
                    language = Language.ENGLISH
                    logger.debug(f"Mapped voice ID {voice_id} to English based on ID pattern")
            
            # If still no language, default to English
            if not language:
                logger.warning(f"Could not determine language for voice {voice_id}, defaulting to English")
                language = Language.ENGLISH
            
            # Extract gender with better handling of different formats
            gender = self._extract_gender(voice_data)
            
            # Get voice name with fallback to ID
            voice_name = voice_data.get('FriendlyName') or voice_data.get('Name', voice_id)
            
            # Create Voice object with consistent ID format (lowercase)
            voice = Voice(
                id=voice_id.lower(),  # Use lowercase for consistent lookups
                name=str(voice_name).strip(),
                provider="edge_tts",
                provider_id=voice_id,
                language=language,
                gender=gender,
                age=VoiceAge.ADULT,  # Edge TTS doesn't provide age information
                sample_rate=24000,  # Default sample rate for Edge TTS
                is_active=True,
                metadata={
                    "locale": locale,
                    "voice_type": str(voice_data.get("VoiceType", "")),
                    "status": str(voice_data.get("Status", "")),
                    "original_data": {k: v for k, v in voice_data.items() 
                                    if k not in {'Name', 'ShortName', 'FriendlyName'}}
                }
            )
            
            logger.debug(f"Created voice: id={voice.id}, name={voice.name}, "
                       f"language={voice.language}, locale={locale}, gender={gender}")
            return voice
            
        except Exception as e:
            logger.error(f"Failed to convert voice data: {e}\n"
                       f"Voice data: {json.dumps(voice_data, default=str, indent=2) if voice_data else 'None'}", 
                       exc_info=True)
            return None
    
    def _map_locale_to_language(self, locale: str, voice_id: str) -> Optional[Language]:
        """Map a locale string to a Language enum value.
        
        Args:
            locale: The locale string (e.g., 'en-US', 'fil-PH')
            voice_id: The voice ID for additional context in logging
            
        Returns:
            Optional[Language]: The mapped Language enum, or None if no mapping found
        """
        if not locale:
            logger.debug(f"Empty locale for voice {voice_id}, will try to determine from voice ID")
            return None
            
        # Normalize locale (lowercase, strip whitespace)
        locale = locale.strip().lower()
        
        # Debug log the input
        logger.debug(f"Mapping locale '{locale}' for voice {voice_id}")
        
        # Handle special cases first
        if locale in {'fil', 'tl', 'fil-ph', 'tl-ph', 'fil-latn', 'tl-latn'} or 'tagalog' in locale:
            logger.debug(f"Mapped locale '{locale}' to Tagalog")
            return Language.TAGALOG
            
        # Handle English variants
        if locale.startswith('en'):
            logger.debug(f"Mapped locale '{locale}' to English")
            return Language.ENGLISH
            
        # Try exact match with Language enum values
        try:
            lang = Language(locale)
            logger.debug(f"Mapped locale '{locale}' to {lang} via direct match")
            return lang
        except ValueError:
            pass
            
        # Try matching just the language part (e.g., 'en' from 'en-US')
        if '-' in locale:
            lang_part = locale.split('-')[0].strip()
            try:
                lang = Language(lang_part)
                logger.debug(f"Mapped locale '{locale}' to {lang} via language part '{lang_part}'")
                return lang
            except ValueError:
                pass
                
        # Try matching any part of the voice ID that might indicate language
        voice_id_lower = str(voice_id).lower()
        if any(tagalog_code in voice_id_lower for tagalog_code in ['fil', 'tagalog', 'tl']):
            logger.debug(f"Mapped voice ID '{voice_id}' to Tagalog based on ID pattern")
            return Language.TAGALOG
            
        if any(en_code in voice_id_lower for en_code in ['en', 'eng', 'english']):
            logger.debug(f"Mapped voice ID '{voice_id}' to English based on ID pattern")
            return Language.ENGLISH
            
        logger.debug(f"Could not map locale '{locale}' for voice {voice_id} to a known language")
        return None
    
    def _extract_gender(self, voice_data: Dict[str, Any]) -> VoiceGender:
        """Extract and normalize gender information from voice data.
        
        Args:
            voice_data: The raw voice data dictionary
            
        Returns:
            VoiceGender: The extracted gender, or UNSPECIFIED if not determinable
        """
        gender_str = str(voice_data.get('Gender', '')).lower()
        
        # Check for gender in common fields
        if any(female_term in gender_str for female_term in ['female', 'woman', 'girl', 'f']):
            return VoiceGender.FEMALE
            
        if any(male_term in gender_str for male_term in ['male', 'man', 'boy', 'm']):
            return VoiceGender.MALE
            
        # Check voice ID for gender hints
        voice_id = str(voice_data.get('ShortName', '') or voice_data.get('Name', '')).lower()
        if any(female_term in voice_id for female_term in ['female', 'woman', 'girl', 'f-']):
            return VoiceGender.FEMALE
            
        if any(male_term in voice_id for male_term in ['male', 'man', 'boy', 'm-']):
            return VoiceGender.MALE
            
        # Check voice name for gender hints
        voice_name = str(voice_data.get('FriendlyName', '') or voice_data.get('Name', '')).lower()
        if any(female_term in voice_name for female_term in ['female', 'woman', 'girl']):
            return VoiceGender.FEMALE
            
        if any(male_term in voice_name for male_term in ['male', 'man', 'boy']):
            return VoiceGender.MALE
            
        logger.debug(f"Could not determine gender from voice data, defaulting to UNSPECIFIED. "
                   f"Gender string: '{gender_str}', Voice ID: {voice_id}")
        return VoiceGender.UNSPECIFIED
        
    async def _process_voices(self, voices_data: List[Dict[str, Any]]) -> None:
        """Process raw voice data into internal dictionaries.
        
        Args:
            voices_data: List of voice data dictionaries.
        """
        if not voices_data:
            logger.warning("No voice data provided to _process_voices")
            return
            
        # First filter to only include English (en) and Tagalog (fil) voices
        # This is done before any processing to avoid unnecessary work
        filtered_voices = []
        skipped_voices = []
        for voice in voices_data:
            locale = (voice.get('Locale') or voice.get('locale') or '').lower()
            voice_name = voice.get('Name') or voice.get('ShortName') or 'unknown'
            
            # Check if this is a voice we want to include
            if locale.startswith(('en-', 'fil-', 'tl-')):
                filtered_voices.append(voice)
                logger.debug(f"Including voice: {voice_name} (locale: {locale})")
            else:
                skipped_voices.append(f"{voice_name} (locale: {locale})")
        
        # Log some debug info about filtered voices
        if skipped_voices:
            logger.debug(f"Skipped {len(skipped_voices)} voices not matching language filter. "
                       f"Sample: {', '.join(skipped_voices[:5])}...")
        
        total_voices = len(voices_data)
        filtered_count = len(filtered_voices)
        
        if filtered_count == 0:
            logger.warning("No English or Tagalog voices found after filtering")
            return
            
        logger.info(f"Filtered {total_voices} voices down to {filtered_count} English and Tagalog voices")
        voices_data = filtered_voices
        
        # Process each filtered voice
        processed_count = 0
        for voice_data in voices_data:
            if not isinstance(voice_data, dict):
                logger.warning(f"Skipping invalid voice data (expected dict, got {type(voice_data)}): {voice_data}")
                continue
                
            # At this point we know the voice is either English or Tagalog
            voice = self._convert_to_voice(voice_data)
            if not voice:
                logger.debug(f"Skipping invalid voice data: {voice_data}")
                continue
                
            processed_count += 1
                
            # Ensure ID is lowercase for consistent lookups
            voice_id = voice.id
            
            # Store in both dictionaries
            self._voices[voice_id] = voice_data
            self._voice_objects[voice_id] = voice
                    
            # Store the raw voice data for caching
            logger.debug(f"Storing voice in cache: {voice_id}")
            self._voice_cache[voice_id] = voice_data
                    
        self._voices_fetched = True
        logger.info(f"Successfully processed {processed_count} out of {filtered_count} filtered voices")
        
        if self._voice_objects:
            available_voices = list(self._voice_objects.keys())
            logger.debug(f"Available voice IDs: {available_voices}")
            
            # Log sample of available voices for debugging
            sample_size = min(5, len(self._voice_objects))
            if sample_size > 0:
                sample_voices = list(self._voice_objects.values())[:sample_size]
                logger.debug(f"Sample of {sample_size} available voices:")
                for i, voice in enumerate(sample_voices, 1):
                    logger.debug(f"  {i}. {voice.id}: {voice.name} ({voice.language}, {voice.gender})")
        
        if not self._voice_objects:
            logger.warning("No voices were successfully loaded after filtering and processing")

    async def _save_voices_to_cache(self) -> None:
        """Save the current voices to the cache file.
        
        Raises:
            TTSValidationError: If there's an error saving to cache.
        """
        if not self.cache_dir or not hasattr(self, '_voice_cache_file') or not self._voices:
            return
            
        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Save voices in the expected format with a 'voices' key
            cache_data = {
                'version': '1.0',
                'cached_at': time.time(),
                'voices': list(self._voices.values())
            }
            
            # Write to a temporary file first, then rename for atomicity
            temp_file = self._voice_cache_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            # Rename temp file to final name (atomic on POSIX)
            temp_file.replace(self._voice_cache_file)
            logger.debug(f"Successfully saved {len(self._voices)} voices to cache at {self._voice_cache_file}")
            
        except Exception as e:
            logger.error(f"Failed to save voices to cache: {e}")
            raise TTSValidationError(f"Failed to save voices to cache: {e}") from e

    async def _load_voices_from_cache(self) -> bool:
        """Load voices from cache file if available.
        
        Returns:
            bool: True if voices were loaded from cache, False otherwise.
        """
        if not hasattr(self, '_voice_cache_file') or not self._voice_cache_file.exists():
            return False
            
        try:
            with open(self._voice_cache_file, 'r') as f:
                cache_data = json.load(f)
                
            # Handle different cache formats
            if isinstance(cache_data, dict) and 'voices' in cache_data:
                # New format with metadata
                voices_data = cache_data['voices']
            elif isinstance(cache_data, list):
                # Old format - just a list of voices
                voices_data = cache_data
            else:
                # Unknown format, try to process as dict of voices
                voices_data = list(cache_data.values())
                
            await self._process_voices(voices_data)
            logger.debug(f"Loaded {len(self._voices)} voices from cache")
            return True
            
        except Exception as cache_err:
            logger.warning(f"Failed to load voices from cache: {cache_err}")
            return False

    async def _load_voices(self) -> None:
        """Load voices from the service or cache.
        
        Raises:
            TTSValidationError: If there's an error loading voices.
        """
        if self._voices_fetched:
            logger.debug("Voices already loaded, skipping load")
            return
            
        # Reset state
        self._voices = {}
        self._voice_objects = {}
        self._voice_cache = {}
            
        # Try to load from cache first
        logger.debug("Attempting to load voices from cache...")
        if await self._load_voices_from_cache():
            logger.info(f"Successfully loaded {len(self._voice_objects)} voices from cache")
            self._voices_fetched = True
            return
            
        logger.debug("Cache not available or empty, fetching voices from service...")
        
        # If we get here, we need to fetch from the service
        try:
            logger.info("Fetching voices from Edge TTS service...")
            start_time = time.time()
            
            # Use VoicesManager to get all available voices
            logger.debug("Creating VoicesManager...")
            voices_manager = await VoicesManager.create()
            voices = voices_manager.voices
            logger.debug(f"Retrieved {len(voices)} voices from VoicesManager")
            
            if not voices:
                logger.warning("No voices returned from VoicesManager")
                raise TTSValidationError("No voices available from the TTS service")
                
            await self._process_voices(voices)
            
            # Save to cache if cache_dir is set
            if self.cache_dir and hasattr(self, '_voice_cache_file'):
                logger.debug("Saving voices to cache...")
                await self._save_voices_to_cache()
                logger.debug(f"Saved {len(self._voice_objects)} voices to cache")
            
            if not self._voice_objects:
                raise TTSValidationError("No valid voices were processed")
                
            logger.info(f"Successfully loaded {len(self._voice_objects)} voices from service in {time.time() - start_time:.2f} seconds")
            self._voices_fetched = True
                    
        except Exception as e:
            logger.error(f"Failed to fetch voices: {e}", exc_info=True)
            if not self._voice_objects:  # Only raise if we have no voices to work with
                raise TTSValidationError(f"Failed to fetch voices: {e}") from e

    async def get_voices(self, language: Optional[Union[str, Language]] = None) -> List[Voice]:
        """Get available voices.

        Args:
            language: Optional language to filter voices by.

        Returns:
            List of Voice objects.
            
        Raises:
            TTSValidationError: If there's an error fetching voices.
        """
        logger.debug(f"Getting voices with language filter: {language}")
        
        if not self._voices_fetched:
            logger.debug("Voices not yet loaded, loading now...")
            await self._load_voices()
        
        # If no voices were loaded, log a warning and return empty list
        if not self._voice_objects:
            logger.warning("No voices available in _voice_objects")
            if self._voices:
                logger.warning(f"Found {len(self._voices)} raw voices but no Voice objects. This indicates a problem with voice conversion.")
            return []
            
        logger.debug(f"Found {len(self._voice_objects)} voice objects")
        
        # Get all voices from _voice_objects
        voices = list(self._voice_objects.values())
        
        # Log all available voices for debugging
        for i, voice in enumerate(voices, 1):
            logger.debug(f"Voice {i}: id={voice.id}, language={voice.language}, "
                       f"locale={voice.metadata.get('locale', 'N/A')}")
        
        # Filter by language if specified
        if language is not None:
            logger.debug(f"Filtering voices by language: {language}")
            
            # Convert language to string if it's a Language enum
            if isinstance(language, Language):
                language_str = language.value.lower()
            else:
                language_str = str(language).lower()
            
            # Special case for Tagalog to handle both 'fil' and 'tl' codes
            if language_str in {'fil', 'tagalog', 'tl'}:
                language_codes = {'fil', 'tl'}
            else:
                language_codes = {language_str}
            
            logger.debug(f"Language codes to match: {language_codes}")
            
            filtered_voices = []
            for voice in voices:
                if not voice.language:
                    logger.debug(f"Skipping voice with no language: {voice.id}")
                    continue
                
                # Get the voice's language code (e.g., 'en-US')
                voice_lang = voice.language.value.lower()
                
                # Extract the base language code (e.g., 'en' from 'en-US')
                voice_base_lang = voice_lang.split('-')[0] if '-' in voice_lang else voice_lang
                
                # Get the voice's provider ID in lowercase for matching
                voice_id_lower = voice.provider_id.lower()
                
                # Log voice details for debugging
                logger.debug(f"Checking voice: id={voice.id}, language={voice_lang}, "
                           f"base_lang={voice_base_lang}, provider_id={voice_id_lower}")
                
                # Check for matches against all possible language codes
                matched = False
                for code in language_codes:
                    # Check for exact match with voice language (e.g., 'en' == 'en' or 'en-us' == 'en-us')
                    if code == voice_lang:
                        logger.debug(f"  -> Exact match: {code} == {voice_lang}")
                        matched = True
                        break
                        
                    # Check for base language match (e.g., 'en' matches 'en-us')
                    if code == voice_base_lang:
                        logger.debug(f"  -> Base language match: {code} == {voice_base_lang}")
                        matched = True
                        break
                        
                    # Special case for Tagalog - check if code is in voice ID (e.g., 'fil' in 'fil-PH-BlessicaNeural')
                    if code in {'fil', 'tl'} and ('fil' in voice_id_lower or 'tl' in voice_id_lower):
                        logger.debug(f"  -> Tagalog match: found {code} in {voice_id_lower}")
                        matched = True
                        break
                
                if matched:
                    filtered_voices.append(voice)
                    logger.debug(f"  -> ADDED to filtered list")
                else:
                    logger.debug(f"  -> SKIPPED (no match)")
            
            logger.info(f"Filtered to {len(filtered_voices)}/{len(voices)} voices for language: {language} ({language_str})")
            voices = filtered_voices
            
        return voices

    async def validate_voice(self, voice_id: str) -> None:
        """Validate that the specified voice ID is available.

        Args:
            voice_id: The voice ID to validate (case-insensitive).

        Raises:
            TTSValidationError: If the voice ID is invalid or not available.
        """
        if not voice_id:
            raise TTSValidationError("Voice ID cannot be empty")
            
        # Normalize the voice_id for comparison
        voice_id = voice_id.strip().lower()
        
        # First try to get the voice directly
        voice = await self.get_voice(voice_id)
        if voice is not None:
            return
            
        # If still not found, try loading voices and check again
        if not self._voices_fetched:
            await self._load_voices()
            voice = await self.get_voice(voice_id)
            if voice is not None:
                return
        
        # If we still can't find it, get available voices for error message
        voices = await self.get_voices()
        
        # Check if the voice exists in any case variation
        voice_found = any(v.id.lower() == voice_id for v in voices)
        if not voice_found:
            # Get available voice IDs for the error message
            available_voices = [v.id for v in voices]
            available_samples = ", ".join(sorted(available_voices)[:5])
            if len(voices) > 5:
                available_samples += ", ..."
                
            # Check if this might be a case-sensitivity issue
            case_variations = [v for v in available_voices if v.lower() == voice_id]
            if case_variations:
                raise TTSValidationError(
                    f"Voice '{voice_id}' not found. Did you mean '{case_variations[0]}'? "
                    f"Available voices: {available_samples}"
                )
            else:
                raise TTSValidationError(
                    f"Voice '{voice_id}' is not available. "
                    f"Available voices: {available_samples}"
                )

    async def _validate_audio_file(self, file_path: Union[str, Path]) -> bool:
        """Validate that the generated audio file is not empty and has valid content.
        
        Args:
            file_path: Path to the audio file to validate.
            
        Returns:
            bool: True if the file is valid, False otherwise.
            
        Raises:
            TTSValidationError: If the file is empty or invalid.
        """
        file_path = Path(file_path)
        
        # Check if file exists and has content
        if not file_path.exists():
            logger.error(f"Audio file does not exist: {file_path}")
            return False
            
        file_size = file_path.stat().st_size
        if file_size == 0:
            logger.error(f"Audio file is empty: {file_path}")
            return False
            
        # Check minimum file size (1KB)
        if file_size < 1024:
            logger.warning(f"Audio file is very small ({file_size} bytes), may be corrupted")
            
        try:
            # Check file extension to determine format
            if file_path.suffix.lower() == '.wav':
                # Validate WAV file
                import wave
                import audioop
                
                with wave.open(str(file_path), 'rb') as wav_file:
                    # Check basic WAV format
                    if wav_file.getnchannels() not in [1, 2]:
                        logger.error(f"Invalid number of channels: {wav_file.getnchannels()}")
                        return False
                        
                    sample_rate = wav_file.getframerate()
                    if sample_rate not in [8000, 16000, 24000, 44100, 48000]:
                        logger.warning(f"Unexpected sample rate: {sample_rate} Hz")
                        
                    # Verify we can read at least some frames
                    frames = wav_file.readframes(10)
                    if not frames:
                        logger.error("Could not read audio frames")
                        return False
                        
                    # Verify sample width is valid
                    sample_width = wav_file.getsampwidth()
                    if sample_width not in [1, 2, 3, 4]:
                        logger.error(f"Invalid sample width: {sample_width}")
                        return False
            
            elif file_path.suffix.lower() == '.mp3':
                # For MP3, check file size and basic content
                file_size = file_path.stat().st_size
                if file_size < 100:  # Very small file is likely invalid
                    logger.error(f"MP3 file is too small: {file_size} bytes")
                    return False
                    
                # Read first few bytes to check for MP3 header
                with open(file_path, 'rb') as f:
                    header = f.read(10)  # Read more bytes to catch more header types
                    
                # Check for common MP3 headers:
                # 1. ID3 header (starts with 'ID3')
                # 2. MPEG frame sync (starts with 0xFF 0xFB or 0xFF 0xFA)
                # 3. MPEG frame sync with other bitrates (0xFF 0xF2, 0xFF 0xF3, etc.)
                is_valid = (
                    header.startswith(b'ID3') or  # ID3v2 header
                    (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0)  # MPEG frame sync
                )
                
                if not is_valid:
                    logger.error(f"Invalid MP3 header: {header[:4].hex(' ')}")
                    logger.error(f"File size: {file_size} bytes")
                    return False
            
            else:
                # For other formats, just check if we can read the file
                try:
                    with open(file_path, 'rb') as f:
                        if not f.read(1024):
                            logger.error("Could not read file content")
                            return False
                except Exception as e:
                    logger.error(f"Error reading file: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error validating audio file {file_path}: {e}")
            return False
            
        return True

    async def _rate_limit_delay(self, connection_error: bool = False) -> None:
        """Apply rate limiting delay before making TTS requests.
        
        Args:
            connection_error: If True, applies additional delay for connection errors
        """
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        # Calculate required delay
        min_delay = MIN_REQUEST_DELAY
        if connection_error:
            min_delay += CONNECTION_ERROR_DELAY
            logger.debug(f"Applying connection error delay: {CONNECTION_ERROR_DELAY}s")
        
        # Apply delay if needed
        if time_since_last < min_delay:
            delay = min_delay - time_since_last
            logger.debug(f"Rate limiting: waiting {delay:.2f}s (last request {time_since_last:.2f}s ago)")
            await asyncio.sleep(delay)
        
        # Update last request time
        self._last_request_time = time.time()
        self._request_count += 1
        logger.debug(f"TTS request #{self._request_count}")

    async def _synthesize_single(
        self,
        text: str,
        voice_id: str,
        output_path: Union[str, Path],
        rate: float = 1.0,
        pitch: float = 0.0,
        volume: float = 1.0,
        speaker_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Synthesize speech for a single text string.

        Args:
            text: The text to synthesize.
            voice_id: The voice ID to use.
            output_path: Path to save the output audio file.
            rate: Speech rate (0.5-3.0).
            pitch: Pitch adjustment (-20 to 20).
            volume: Volume level (0.0-1.0).
            **kwargs: Additional arguments for the TTS service.

        Returns:
            Dictionary containing metadata about the synthesis.

        Raises:
            TTSValidationError: If the input is invalid.
            TTSRateLimitError: If the rate limit is exceeded.
            TTSTransientError: If a transient error occurs.
            TTSAuthenticationError: If authentication fails.
            Exception: For other errors.
        """
        if not text.strip():
            raise TTSValidationError("Text cannot be empty")

        output_path = Path(output_path)
        # Ensure the output path has .mp3 extension
        if output_path.suffix.lower() != '.mp3':
            output_path = output_path.with_suffix('.mp3')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Log the voice and speaker being used for synthesis
        logger.info(f"Using voice_id: {voice_id}, speaker_id: {speaker_id} for synthesis")
        
        # Validate the voice ID before proceeding
        try:
            await self.validate_voice(voice_id)
        except TTSValidationError as e:
            logger.error(f"Invalid voice ID '{voice_id}': {str(e)}")
            # Try to get a list of available voices
            try:
                available_voices = await self.get_voices()
                available_voice_ids = [v.provider_id for v in available_voices]
                logger.error(f"Available voices: {available_voice_ids}")
            except Exception as ve:
                logger.error(f"Could not retrieve available voices: {str(ve)}")
            raise
            
        # Generate cache key and check cache
        cache_key = self._generate_cache_key(
            text=text,
            voice_id=voice_id,
            rate=rate,
            pitch=pitch,
            volume=volume,
            speaker_id=speaker_id  # Include speaker_id in cache key
        )
        logger.debug(f"Generated cache key: {cache_key}")
        
        # Only check cache if caching is enabled
        cached_file = None
        if self.cache_dir:
            cached_file = self._get_cached_file(cache_key)
            logger.debug(f"Cached file path: {cached_file}")
            
            if cached_file and cached_file.exists():
                logger.debug(f"Cache hit for key: {cache_key}")
                # Copy cached file to output path
                shutil.copy2(cached_file, output_path)
                logger.debug(f"Copied cached file to {output_path}")
                return {
                    "audio_file": str(output_path),
                    "voice": voice_id,
                    "text_length": len(text),
                    "cached": True,
                }

        # Synthesize speech
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
                temp_file = temp.name

            # Apply custom voice settings for specific voices
            voice_id_lower = voice_id.lower()
            
            # Get speaker ID from kwargs if available (passed from lesson processor)
            # Use the speaker_id parameter if provided, otherwise check kwargs
            original_speaker_id = speaker_id
            if not speaker_id and 'speaker_id' in kwargs:
                speaker_id = kwargs.get('speaker_id')
            
            # Convert to string and normalize
            if speaker_id is not None:
                speaker_id = str(speaker_id).lower()
            
            # Enhanced debug logging for voice settings
            logger.debug(f"[DEBUG] Raw input - rate: {rate}, pitch: {pitch}, speaker_id: {speaker_id} (original: {original_speaker_id}), voice_id: {voice_id}")
            logger.debug(f"[DEBUG] All kwargs: {kwargs}")
            
            # Initialize with default values
            custom_rate = rate
            custom_pitch = pitch
            
            # Apply voice settings based on speaker_id
            logger.debug(f"[DEBUG] Processing speaker_id: {speaker_id}, voice_id: {voice_id_lower}")
            
            # Log all available speaker IDs for debugging
            available_speaker_ids = ['tagalog-female-1', 'tagalog-female-2']
            logger.debug(f"[DEBUG] Available speaker IDs: {available_speaker_ids}")
            
            # Convert voice_id to lowercase for consistent comparison
            voice_id_lower = voice_id.lower()
            
            # Determine language code based on voice ID
            if 'tagalog' in voice_id_lower or 'fil' in voice_id_lower:
                language_code = 'fil-PH'  # Tagalog/Filipino language code
            elif 'en-' in voice_id_lower:
                # Extract the full language code if it follows 'en-' (e.g., 'en-US')
                language_code = 'en-US'  # Default English variant
            else:
                language_code = 'en-US'  # Default to US English
                
            logger.debug(f"Using language code '{language_code}' for voice ID: {voice_id}")
            
            # Log original text before preprocessing
            logger.debug(f"Original text for TTS: '{text}'")
            logger.debug(f"Using language code: '{language_code}'")
            
            # Apply enhanced text preprocessing with hybrid SSML support
            text, ssml_result = enhanced_preprocess_text_for_tts(
                text=text,
                language_code=language_code,
                provider_name='edge_tts',
                supports_ssml=False,  # EdgeTTS does NOT support SSML
                section_type=kwargs.get('section_type')
            )
            logger.debug(f"Enhanced preprocessed text for TTS: '{text}'")
            logger.debug(f"SSML processing metadata: {ssml_result}")
            
            # Check if voice_id contains tagalog but speaker_id is not set
            if ('tagalog' in voice_id_lower or 'fil' in voice_id_lower) and not speaker_id:
                speaker_id = 'tagalog-female-1'  # Default to female-1 for Tagalog voices
                logger.debug(f"[DEBUG] Auto-detected Tagalog voice, setting default speaker_id: {speaker_id}")
                # Update the original speaker_id to reflect the default value
                original_speaker_id = speaker_id
            
            # Use the pitch/rate values provided by the lesson parser (from phrase metadata)
            # These values should come from the lesson parser's voice assignment logic
            custom_rate = rate
            custom_pitch = pitch
            logger.debug(f"[DEBUG] Using lesson parser voice settings - rate: {custom_rate}, pitch: {custom_pitch}, speaker_id: {speaker_id}")
            
            # Format rate as integer percentage without decimal point
            rate_percent = int(round((custom_rate - 1.0) * 100))
            rate_str = f"{rate_percent:+d}%"
            
            # Format pitch - ensure we're using the custom_pitch value
            pitch_value = int(custom_pitch)
            pitch_str = f"{pitch_value:+d}Hz"
            
            # Log the final values being used
            logger.warning(f"[TTS PARAMS] Final values - rate: {rate_str} (from {custom_rate}), "
                         f"pitch: {pitch_str} (from {custom_pitch}), voice: {voice_id}, "
                         f"speaker_id: {speaker_id} (original: {original_speaker_id})")
            logger.debug(f"[DEBUG] Rate string: {rate_str}, Pitch string: {pitch_str}")
            
            # Log the full voice configuration including speaker ID
            voice_config = {
                'voice': voice_id,
                'rate': rate_str,
                'pitch': pitch_str,
                'volume': f"+{int((volume - 1) * 100)}%" if volume != 1.0 else "+0%",
                'text_length': len(text)
            }
            logger.debug(f"[DEBUG] Voice configuration: {voice_config}")
            
            # Implement retry logic with rate limiting and exponential backoff
            retry_count = 0
            last_error = None
            
            # Apply initial rate limiting
            await self._rate_limit_delay()
            
            while retry_count <= MAX_RETRIES:
                try:
                    # Apply rate limiting semaphore to control concurrent requests
                    if self._rate_limit_semaphore is None:
                        self._rate_limit_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
                    
                    async with self._rate_limit_semaphore:
                        logger.debug(f"TTS attempt {retry_count + 1}/{MAX_RETRIES + 1} for text: '{text[:50]}...'")
                        
                        communicate = self._communicate_class(
                            text=text,
                            voice=voice_id,
                            rate=rate_str,
                            pitch=pitch_str,
                            volume=f"+{int((volume - 1) * 100)}%" if volume != 1.0 else "+0%",
                        )

                        # Save the audio file with timeout
                        try:
                            await asyncio.wait_for(
                                communicate.save(temp_file), 
                                timeout=REQUEST_TIMEOUT
                            )
                            # Success! Break out of retry loop
                            break
                            
                        except asyncio.TimeoutError:
                            raise TTSTransientError(f"TTS request timed out after {REQUEST_TIMEOUT}s")
                            
                except (aiohttp.ClientConnectionError, aiohttp.ClientError, ConnectionError, 
                        TTSTransientError, asyncio.TimeoutError) as e:
                    last_error = e
                    retry_count += 1
                    
                    # Log the connection error
                    error_type = type(e).__name__
                    logger.warning(f"TTS connection error (attempt {retry_count}/{MAX_RETRIES + 1}): {error_type}: {e}")
                    
                    if retry_count <= MAX_RETRIES:
                        # Calculate exponential backoff delay
                        backoff_delay = RETRY_DELAY * (2 ** (retry_count - 1))
                        logger.info(f"Retrying TTS in {backoff_delay:.1f}s...")
                        
                        # Apply connection error delay + exponential backoff
                        await self._rate_limit_delay(connection_error=True)
                        await asyncio.sleep(backoff_delay)
                    else:
                        # All retries exhausted
                        logger.error(f"TTS failed after {MAX_RETRIES + 1} attempts. Last error: {last_error}")
                        raise TTSTransientError(f"Network error during TTS: {last_error}")
                        
                except Exception as e:
                    # Non-retriable error, fail immediately
                    logger.error(f"Non-retriable TTS error: {type(e).__name__}: {e}")
                    raise
            
            # Verify the generated file exists and has content
            if not os.path.exists(temp_file):
                raise TTSValidationError(f"Audio file was not created: {temp_file}")
                
            file_size = os.path.getsize(temp_file)
            if file_size == 0:
                raise TTSValidationError(f"Generated audio file is empty: {temp_file}")
            
            # Log file info for debugging
            logger.debug(f"Generated audio file: {temp_file}, size: {file_size} bytes")
            
            # Verify the file format
            if not await self._validate_audio_file(temp_file):
                # Read first few bytes for debugging
                with open(temp_file, 'rb') as f:
                    header = f.read(16)
                logger.error(f"Audio file header: {header.hex(' ')}")
                logger.error(f"File size: {file_size} bytes")
                raise TTSValidationError(
                    f"Generated audio file failed validation. "
                    f"File size: {file_size} bytes, "
                    f"Header: {header.hex(' ')}"
                )

            # Move the temporary file to the final output path
            shutil.move(temp_file, output_path)
            temp_file = None  # Prevent cleanup in the except block
            
            # Cache the result if caching is enabled
            if self.cache_dir:
                # Ensure cache directory exists
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate cache key with all parameters including speaker_id
                cache_key = self._generate_cache_key(
                    text=text,
                    voice_id=voice_id,
                    rate=rate,
                    pitch=pitch,
                    volume=volume,
                    speaker_id=speaker_id  # Include speaker_id in cache key
                )
                cache_path = self.cache_dir / cache_key
                
                # Copy the output file to the cache
                shutil.copy2(output_path, cache_path)
                logger.debug(f"Cached audio to {cache_path} with speaker_id: {speaker_id}")
                
            # Return the result with output_path as a Path object
            return {
                "audio_file": output_path,  # Keep as Path object
                "voice": voice_id,
                "text_length": len(text),
                "cached": False,
            }

        except Exception as e:
            logger.debug(f"[ERROR] Caught exception in _synthesize_single: {e!r}")
            logger.debug(f"[ERROR] Exception type: {type(e).__name__}")
            logger.debug(f"[ERROR] Exception str: {str(e)!r}")
            logger.debug(f"[ERROR] Exception dir: {dir(e)}")
            
            # Log all attributes of the exception
            for attr in dir(e):
                if not attr.startswith('_'):
                    try:
                        value = getattr(e, attr)
                        logger.debug(f"[ERROR] Exception attr {attr}: {value!r}")
                    except Exception as attr_err:
                        logger.debug(f"[ERROR] Could not get attribute {attr}: {attr_err}")
            
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as cleanup_err:
                    logger.debug(f"[ERROR] Error cleaning up temp file: {cleanup_err}")

            # Log the exception hierarchy for debugging
            logger.debug("[ERROR] Exception hierarchy:")
            for i, exc_type in enumerate(type(e).__mro__):
                logger.debug(f"  {i}. {exc_type.__module__}.{exc_type.__name__}")
                
            # Check for specific error types
            if isinstance(e, (aiohttp.ClientError, asyncio.TimeoutError)):
                logger.debug("[ERROR] Handling as ClientError/TimeoutError")
                logger.debug(f"[ERROR] Exception has status: {getattr(e, 'status', 'N/A')}")
                logger.debug(f"[ERROR] Exception has status_code: {getattr(e, 'status_code', 'N/A')}")
                logger.debug(f"[ERROR] Exception headers: {getattr(e, 'headers', 'N/A')}")
                
                if hasattr(e, 'status') and e.status == 429:
                    logger.debug("[ERROR] Detected status=429 in ClientError")
                    raise TTSRateLimitError(f"Rate limit exceeded: {e}", retry_after=60) from e
                elif hasattr(e, 'status_code') and e.status_code == 429:
                    logger.debug("[ERROR] Detected status_code=429 in ClientError")
                    raise TTSRateLimitError(f"Rate limit exceeded: {e}", retry_after=60) from e
                elif "429" in str(e):
                    logger.debug("[ERROR] Found '429' in exception string")
                    raise TTSRateLimitError(f"Rate limit exceeded: {e}", retry_after=60) from e
                else:
                    logger.debug("[ERROR] Raising as generic ClientError")
                    raise TTSTransientError(f"Network error during TTS: {e}") from e
                    
            elif isinstance(e, NoAudioReceived):
                logger.debug("[ERROR] Handling as NoAudioReceived")
                raise TTSTransientError("No audio received from TTS service") from e
                
            elif isinstance(e, WebSocketError):
                logger.debug("[ERROR] Handling as WebSocketError")
                raise TTSTransientError(f"WebSocket error during TTS: {e}") from e
                
            elif hasattr(e, 'status') and e.status == 429:
                logger.debug(f"[ERROR] Detected status code 429 in exception: {e}")
                raise TTSRateLimitError(f"Rate limit exceeded: {e}", retry_after=60) from e
                
            elif hasattr(e, 'status_code') and e.status_code == 429:
                logger.debug(f"[ERROR] Detected status_code 429 in exception: {e}")
                raise TTSRateLimitError(f"Rate limit exceeded: {e}", retry_after=60) from e
                
            elif "429" in str(e):
                logger.debug("[ERROR] Found '429' in exception string")
                raise TTSRateLimitError(f"Rate limit exceeded: {e}", retry_after=60) from e
                
            elif "401" in str(e) or "403" in str(e):
                logger.debug("[ERROR] Handling as authentication error")
                raise TTSAuthenticationError(f"Authentication failed: {e}") from e
                
            else:
                logger.debug("[ERROR] Raising original exception")
                raise

    async def synthesize_speech(
        self,
        text: str,
        voice_id: str,
        output_path: Union[str, Path],
        rate: float = 1.0,
        pitch: float = 0.0,
        volume: float = 1.0,
        speaker_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Synthesize speech from text.

        Args:
            text: The text to synthesize.
            voice_id: The voice ID to use.
            output_path: Path to save the output audio file.
            rate: Speech rate (0.5-3.0).
            pitch: Pitch adjustment (-20 to 20).
            volume: Volume level (0.0-1.0).
            speaker_id: Optional speaker ID for voice customization.
            **kwargs: Additional arguments for the TTS service.

        Returns:
            Dictionary containing metadata about the synthesis.

        Raises:
            TTSValidationError: If the input is invalid.
            TTSRateLimitError: If the rate limit is exceeded.
            TTSTransientError: If a transient error occurs.
            TTSAuthenticationError: If authentication fails.
            Exception: For other errors.
        """
        return await self._synthesize_single(
            text=text,
            voice_id=voice_id,
            output_path=output_path,
            rate=rate,
            pitch=pitch,
            volume=volume,
            speaker_id=speaker_id,
            **kwargs,
        )

    async def synthesize_speech_batch(
        self,
        texts: List[Tuple[str, str, Union[str, Path]]],
        rate: float = 1.0,
        pitch: float = 0.0,
        volume: float = 1.0,
        max_concurrent: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Synthesize speech for multiple texts in a batch.

        Args:
            texts: List of tuples containing (text, voice_id, output_path).
            rate: Speech rate (0.5-3.0).
            pitch: Pitch adjustment (-20 to 20).
            volume: Volume level (0.0-1.0).
            max_concurrent: Maximum number of concurrent requests.
            **kwargs: Additional arguments for the TTS service.

        Returns:
            List of dictionaries containing metadata about each synthesis.
        """
        if not texts:
            return []

        if max_concurrent is None:
            max_concurrent = self.connection_limit

        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def process_one(
            text_item: Union[Tuple[str, str, Union[str, Path]], Tuple[str, str, Union[str, Path], Optional[str]]]
        ) -> Dict[str, Any]:
            # Handle both (text, voice_id, output_path) and (text, voice_id, output_path, speaker_id) formats
            if len(text_item) == 3:
                text, voice_id, output_path = text_item
                item_speaker_id = None
            else:
                text, voice_id, output_path, item_speaker_id = text_item
                
            async with semaphore:
                try:
                    # Use the item-specific speaker_id if provided, otherwise fall back to the one from kwargs
                    speaker_id_to_use = item_speaker_id if item_speaker_id is not None else kwargs.get('speaker_id')
                    
                    return await self._synthesize_single(
                        text=text,
                        voice_id=voice_id,
                        output_path=output_path,
                        rate=rate,
                        pitch=pitch,
                        volume=volume,
                        speaker_id=speaker_id_to_use,
                        **{k: v for k, v in kwargs.items() if k != 'speaker_id'},  # Remove speaker_id from kwargs to avoid duplication
                    )
                except Exception as e:
                    return {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "text": text,
                        "voice_id": voice_id,
                        "output_path": str(output_path),
                    }

        tasks = [process_one(item) for item in texts]
        return await asyncio.gather(*tasks)

    def _generate_cache_key(
        self, text: str, voice_id: str, rate: float, pitch: float, volume: float, speaker_id: Optional[str] = None
    ) -> str:
        """Generate a cache key for the given text and voice settings.

        Args:
            text: The text to generate a cache key for.
            voice_id: The voice ID.
            rate: Speech rate.
            pitch: Pitch adjustment.
            volume: Volume level.
            speaker_id: Optional speaker ID for voice variations.

        Returns:
            A string that can be used as a cache key.
        """
        # Normalize text by removing extra whitespace
        text = re.sub(r"\s+", " ", text.strip())
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

        # Create a unique key based on the hash and voice settings
        key_parts = [
            voice_id[:8],  # First 8 chars of voice ID
            f"r{rate:.1f}".replace(".", ""),
            f"p{pitch:+.1f}".replace("+", "p").replace("-", "m").replace(".", ""),
            f"v{volume:.1f}".replace(".", ""),
        ]
        
        # Add speaker ID to the key if provided
        if speaker_id:
            # Create a short hash of the speaker ID to keep the filename reasonable
            speaker_hash = hashlib.md5(speaker_id.encode("utf-8")).hexdigest()[:4]
            key_parts.append(f"s{speaker_hash}")
        
        # Add text hash last
        key_parts.append(text_hash[:8])

        # Join with underscores to create a safe filename
        return "_".join(key_parts) + ".mp3"

    async def validate_credentials(self) -> bool:
        """Validate that the service credentials are valid.
        
        Returns:
            bool: True if credentials are valid
            
        Raises:
            TTSAuthenticationError: If credentials are invalid.
            TTSValidationError: If there's an error validating credentials.
        """
        try:
            # Try to get voices to validate credentials
            await self._load_voices()
            return True
        except aiohttp.ClientResponseError as e:
            if e.status in (401, 403):
                raise TTSAuthenticationError(
                    f"401 Unauthorized"
                ) from e
            raise TTSValidationError(
                f"Failed to validate credentials: {e}"
            ) from e
        except Exception as e:
            if "401" in str(e) or "403" in str(e):
                raise TTSAuthenticationError(
                    f"401 Unauthorized"
                ) from e
            raise TTSValidationError(
                f"Failed to validate credentials: {e}"
            ) from e
            
    async def get_voice(self, voice_id: str) -> Optional[Voice]:
        """Get a specific voice by ID.
        
        Args:
            voice_id: The ID of the voice to retrieve (case-insensitive).
            
        Returns:
            Optional[Voice]: The Voice object if found, None otherwise.
            
        Raises:
            TTSValidationError: If there's an error loading voices.
        """
        if not voice_id:
            return None
            
        # Normalize the voice_id for comparison
        voice_id = voice_id.strip()
        logger.debug(f"Looking up voice: {voice_id}")
        
        # Check if we've already loaded this voice (case-insensitive)
        voice_lower = voice_id.lower()
        for v_id, voice in self._voice_objects.items():
            if v_id.lower() == voice_lower:
                logger.debug(f"Found voice by case-insensitive match: {v_id} (requested: {voice_id})")
                return voice
            
        # If voices haven't been loaded yet, try loading them
        if not self._voices_fetched:
            logger.debug("Voices not loaded yet, loading now...")
            await self._load_voices()
            # Try again after loading
            for v_id, voice in self._voice_objects.items():
                if v_id.lower() == voice_lower:
                    logger.debug(f"Found voice after loading: {v_id} (requested: {voice_id})")
                    return voice
        
        # If still not found, try loading voices again in case they weren't loaded properly
        if not self._voice_objects:
            logger.warning("No voices loaded, attempting to reload...")
            await self._load_voices()
            # Try one more time after reloading
            for v_id, voice in self._voice_objects.items():
                if v_id.lower() == voice_lower:
                    logger.debug(f"Found voice after reloading: {v_id} (requested: {voice_id})")
                    return voice
        
        logger.debug(f"Voice not found: {voice_id}. Loaded voices: {list(self._voice_objects.keys())}")
        return None
