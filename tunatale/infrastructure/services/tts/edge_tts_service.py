"""Edge TTS service implementation."""
import asyncio
import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import edge_tts
from edge_tts import VoicesManager

from tunatale.core.exceptions import (
    TTSConnectionError,
    TTSAuthenticationError,
    TTSRateLimitExceeded,
    TTSValidationError,
    TTSServiceError,
    FileOperationError
)
from tunatale.core.models.voice import Voice
from tunatale.core.models.enums import VoiceGender, VoiceAge, Language
from tunatale.core.ports.tts_service import TTSService
from tunatale.core.config import get_config


logger = logging.getLogger(__name__)


class EdgeTTSService(TTSService):
    """Text-to-speech service using Microsoft Edge TTS."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Edge TTS service.
        
        Args:
            config: Configuration dictionary with the following keys:
                - base_url: Base URL for the Edge TTS service
                - default_voice: Default voice ID to use
                - rate: Default speech rate (e.g., "+0%")
                - pitch: Default pitch (e.g., "+0Hz")
                - volume: Default volume (e.g., "+0%")
        """
        self.config = config or {}
        self._voices: Dict[str, Voice] = {}
        self._voices_loaded = False
        self._lock = asyncio.Lock()
        
        # Initialize cache directory
        self.cache_dir = Path(self.config.get("cache_dir", "./data/tts_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
# Initialize voice cache
    
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
        if not self._voices_loaded:
            await self._load_voices()
    
    async def _load_voices(self) -> None:
        """Load available voices from the service or cache."""
        logger.debug("Entering _load_voices")
        async with self._lock:
            if self._voices_loaded:
                logger.debug("Voices already loaded, returning early")
                return
                
            try:
                # Try to load from cache first
                logger.debug("Attempting to load voices from cache")
                if await self._load_voices_from_cache():
                    self._voices_loaded = bool(self._voices)  # Only set to True if we have voices
                    logger.debug("Successfully loaded %d voices from cache", len(self._voices))
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
                    if self._voices:  # Only save if we have voices
                        await self._save_voices_to_cache()
                    
                    self._voices_loaded = True
                    logger.debug("Successfully loaded %d voices from service", len(self._voices))
                except Exception as e:
                    logger.error("Failed to load voices from service: %s", str(e), exc_info=True)
                    if not self._voices and self._voice_cache_file.exists():
                        logger.warning("Falling back to cached voices")
                        if not await self._load_voices_from_cache():
                            logger.error("Failed to load voices from cache")
                            raise TTSConnectionError(f"Failed to load voices: {str(e)}") from e
                
            except Exception as e:
                logger.error("Unexpected error loading voices: %s", str(e), exc_info=True)
                if not self._voices:
                    if self._voice_cache_file.exists():
                        logger.warning("Falling back to cached voices due to error")
                        if not await self._load_voices_from_cache():
                            raise TTSConnectionError(f"Failed to load voices: {str(e)}") from e
                    else:
                        raise TTSConnectionError("Failed to load voices and no cache available") from e
    
    def _process_voices(self, voices_data) -> None:
        """Process voices data and populate self._voices.
        
        Args:
            voices_data: List of voice data dictionaries from the TTS service.
        """
        self._voices.clear()
        
        for voice_data in voices_data:
            try:
                # Handle both direct dict access and object attribute access
                if hasattr(voice_data, 'get'):
                    # It's a dictionary
                    short_name = voice_data.get("ShortName")
                    locale = voice_data.get("Locale")
                    gender = voice_data.get("Gender", "").lower()
                    voice_type = voice_data.get("VoiceType", "")
                    sample_rate = voice_data.get("SampleRateHertz", 24000)
                    friendly_name = voice_data.get("FriendlyName") or voice_data.get("LocalName")
                    status = voice_data.get("Status")
                else:
                    # It's an object with attributes
                    short_name = getattr(voice_data, "ShortName", None)
                    locale = getattr(voice_data, "Locale", None)
                    gender = getattr(voice_data, "Gender", "").lower()
                    voice_type = getattr(voice_data, "VoiceType", "")
                    sample_rate = getattr(voice_data, "SampleRateHertz", 24000)
                    friendly_name = getattr(voice_data, "FriendlyName", None) or getattr(voice_data, "LocalName", None)
                    status = getattr(voice_data, "Status", None)
                
                if not all([short_name, locale, gender]):
                    logger.warning("Skipping voice with missing required fields: %s", voice_data)
                    continue
                
                # Convert language code to Language enum
                from tunatale.core.models.enums import Language
                language_code = locale.split('-')[0].split('_')[0]  # Extract language code
                try:
                    language = Language.from_string(language_code)
                    if not language:
                        logger.warning(f"Unsupported language code: {language_code}")
                        continue
                        
                    # Convert gender to VoiceGender
                    voice_gender = VoiceGender.from_string(gender) or VoiceGender.NEUTRAL
                    
                    voice = Voice(
                        name=short_name,
                        provider=self.name,
                        provider_id=short_name,
                        language=language,
                        gender=voice_gender,
                        locale=locale,
                        is_neural="Neural" in str(voice_type),
                        sample_rate=sample_rate,
                        metadata={
                            "friendly_name": friendly_name,
                            "status": status,
                            "voice_type": voice_type,
                        }
                    )
                except ValueError as e:
                    logger.warning(f"Failed to create voice {short_name}: {str(e)}")
                    continue
                self._voices[voice.provider_id] = voice
                
            except Exception as e:
                logger.warning("Failed to parse voice data: %s - %s", str(e), voice_data, exc_info=True)
    
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
            
            self._voices.clear()  # Clear any existing voices
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
                    self._voices[voice.provider_id] = voice
                    voices_loaded += 1
                    logger.debug(f"Loaded voice from cache: {voice.provider_id}")
                except Exception as e:
                    logger.warning("Failed to parse cached voice: %s", e, exc_info=True)
            
            logger.info("Loaded %d voices from cache", voices_loaded)
            logger.debug(f"Total voices in cache: {len(self._voices)}")
            return bool(self._voices)  # Return True only if we loaded voices
            
        except Exception as e:
            logger.warning("Failed to load voices from cache: %s", e, exc_info=True)
            return False
    
    async def _save_voices_to_cache(self) -> None:
        """Save voices to cache."""
        logger.debug(f"_save_voices_to_cache: Starting with {len(self._voices)} voices")
        logger.debug(f"_save_voices_to_cache: Cache dir: {self.cache_dir.absolute()}")
        logger.debug(f"_save_voices_to_cache: Cache file: {self._voice_cache_file.absolute()}")
        
        if not self._voices:
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
                    for voice in self._voices.values()
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
                logger.debug("Saved %d voices to cache", len(self._voices))
                
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
        """
        await self.initialize()

        if not self._voices_loaded and not self._voices:
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
                v for v in self._voices.values()
                if v.language == language or (isinstance(v.language, Language) and v.language.code == language.code)
            ]
            
        return list(self._voices.values())
    
    async def get_voice(self, voice_id: str) -> Optional[Voice]:
        """Get a specific voice by ID.
        
        Args:
            voice_id: The ID of the voice to retrieve.
            
        Returns:
            The Voice object if found, None otherwise.
        """
        await self.initialize()
        return self._voices.get(voice_id)
    
    async def synthesize_speech(
        self,
        text: str,
        voice_id: str,
        output_path: Path,
        rate: float = 1.0,
        pitch: float = 0.0,
        volume: float = 1.0,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Synthesize speech from text.
        
        Args:
            text: The text to synthesize.
            voice_id: The ID of the voice to use.
            output_path: Path to save the audio file.
            rate: Speech rate (0.5-3.0).
            pitch: Pitch adjustment (-20 to 20).
            volume: Volume level (0.0-1.0).
            **kwargs: Additional parameters for the TTS service.
            
        Returns:
            Dictionary containing metadata about the synthesis.
            
        Raises:
            TTSValidationError: If input validation fails.
            TTSRateLimitExceeded: If rate limits are exceeded.
            TTSServiceError: For other TTS service errors.
        """
        await self.initialize()
        
        # Validate inputs
        if not text or not text.strip():
            raise TTSValidationError("Text cannot be empty")
            
        if not voice_id or not voice_id.strip():
            raise TTSValidationError("Voice ID cannot be empty")
            
        # Check if voice exists
        voice = await self.get_voice(voice_id)
        if not voice:
            raise TTSValidationError(f"Voice not found: {voice_id}")
            
        # Validate output path
        try:
            output_path = Path(output_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise FileOperationError(f"Invalid output path: {e}") from e
            
        # Validate parameters
        if not 0.5 <= rate <= 3.0:
            raise TTSValidationError("Rate must be between 0.5 and 3.0")
            
        if not -20.0 <= pitch <= 20.0:
            raise TTSValidationError("Pitch must be between -20 and 20")
            
        if not 0.0 <= volume <= 1.0:
            raise TTSValidationError("Volume must be between 0.0 and 1.0")
            
        # Check cache first
        cache_key = self._generate_cache_key(text, voice_id, rate, pitch, volume)
        cache_path = self.cache_dir / f"{cache_key}.mp3"
        
        if cache_path.exists():
            # Copy cached file to output path
            import shutil
            try:
                shutil.copy2(cache_path, output_path)
                return {
                    "cached": True,
                    "path": str(output_path),
                    "voice": voice_id,  # For backward compatibility with tests
                    "voice_id": voice_id,  # New standard key
                    "text_length": len(text)
                }
            except Exception as e:
                logger.warning(f"Failed to use cached audio: {e}")
                # Continue with synthesis if cache fails
        
        # Generate SSML
        ssml = self._build_ssml(text, voice_id, rate, pitch, volume)
        
        # Generate speech
        try:
            communicate = edge_tts.Communicate(ssml, voice_id)
            await communicate.save(str(output_path))
            
            # Cache the result if output was generated successfully
            if output_path.exists() and output_path.stat().st_size > 0:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(output_path, cache_path)
                except Exception as e:
                    logger.warning(f"Failed to cache audio: {e}")
            
            return {
                "cached": False,
                "path": str(output_path),
                "voice": voice_id,  # For backward compatibility with tests
                "voice_id": voice_id,  # New standard key
                "text_length": len(text)
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "too many requests" in error_msg:
                raise TTSRateLimitExceeded(f"Rate limit exceeded: {e}") from e
            elif "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
                raise TTSConnectionError(f"Network error: {e}") from e
            elif "invalid" in error_msg or "not found" in error_msg:
                raise TTSValidationError(f"Invalid request: {e}") from e
            else:
                raise TTSServiceError(f"Failed to synthesize speech: {e}") from e
    
    def _generate_cache_key(self, text: str, voice_id: str, rate: float = 1.0, pitch: float = 0.0, volume: float = 1.0) -> str:
        """Generate a cache key for the given text and voice settings.
        
        Args:
            text: The text to synthesize.
            voice_id: The voice ID to use.
            rate: Speech rate (0.5-3.0).
            pitch: Pitch adjustment (-20 to 20).
            volume: Volume level (0.0-1.0).
            
        Returns:
            A unique cache key string.
        """
        key = f"{text}:{voice_id}:{rate}:{pitch}:{volume}"
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    
    def _build_ssml(
        self, 
        text: str,
        voice_id: str,
        rate: float = 1.0,
        pitch: float = 0.0,
        volume: float = 1.0
    ) -> str:
        """Build SSML for the given text and voice."""
        # Edge TTS doesn't use SSML directly in the same way as other services,
        # but we can use it to control some aspects of the speech
        return text
    
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
