"""Google Cloud Text-to-Speech service implementation."""
import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from google.api_core.exceptions import GoogleAPICallError, RetryError
from google.cloud import texttospeech
from google.oauth2 import service_account

from tunatale.core.exceptions import (
    TTSServiceError,
    TTSServiceInitializationError,
    VoiceNotAvailableError,
)
from tunatale.core.models.voice import AudioFormat, Gender, Voice
from tunatale.core.ports.tts_service import TTSService, TTSVoice

# Configure logging
logger = logging.getLogger(__name__)

# Default audio configuration
DEFAULT_AUDIO_CONFIG = {
    "audio_encoding": "MP3",
    "speaking_rate": 1.0,
    "pitch": 0.0,
    "volume_gain_db": 0.0,
}

# Voice name mapping for Google TTS
VOICE_NAME_MAPPING = {
    "en-US": {
        "en-US-Wavenet-A": {"gender": Gender.FEMALE, "natural_sample_rate_hertz": 24000},
        "en-US-Wavenet-B": {"gender": Gender.MALE, "natural_sample_rate_hertz": 24000},
        "en-US-Wavenet-C": {"gender": Gender.FEMALE, "natural_sample_rate_hertz": 24000},
        "en-US-Wavenet-D": {"gender": Gender.MALE, "natural_sample_rate_hertz": 24000},
        "en-US-Wavenet-E": {"gender": Gender.FEMALE, "natural_sample_rate_hertz": 24000},
        "en-US-Wavenet-F": {"gender": Gender.FEMALE, "natural_sample_rate_hertz": 24000},
        "en-US-Wavenet-G": {"gender": Gender.MALE, "natural_sample_rate_hertz": 24000},
        "en-US-Wavenet-H": {"gender": Gender.FEMALE, "natural_sample_rate_hertz": 24000},
        "en-US-Wavenet-I": {"gender": Gender.MALE, "natural_sample_rate_hertz": 24000},
        "en-US-Wavenet-J": {"gender": Gender.MALE, "natural_sample_rate_hertz": 24000},
        "en-US-Standard-A": {"gender": Gender.MALE, "natural_sample_rate_hertz": 24000},
        "en-US-Standard-B": {"gender": Gender.FEMALE, "natural_sample_rate_hertz": 24000},
        "en-US-Standard-C": {"gender": Gender.FEMALE, "natural_sample_rate_hertz": 24000},
        "en-US-Standard-D": {"gender": Gender.MALE, "natural_sample_rate_hertz": 24000},
        "en-US-Standard-E": {"gender": Gender.FEMALE, "natural_sample_rate_hertz": 24000},
        "en-US-Standard-F": {"gender": Gender.FEMALE, "natural_sample_rate_hertz": 24000},
        "en-US-Standard-G": {"gender": Gender.MALE, "natural_sample_rate_hertz": 24000},
        "en-US-Standard-H": {"gender": Gender.FEMALE, "natural_sample_rate_hertz": 24000},
        "en-US-Standard-I": {"gender": Gender.MALE, "natural_sample_rate_hertz": 24000},
        "en-US-Standard-J": {"gender": Gender.MALE, "natural_sample_rate_hertz": 24000},
    },
    "fil-PH": {
        "fil-PH-Wavenet-A": {"gender": Gender.FEMALE, "natural_sample_rate_hertz": 24000},
        "fil-PH-Wavenet-B": {"gender": Gender.MALE, "natural_sample_rate_hertz": 24000},
        "fil-PH-Wavenet-C": {"gender": Gender.FEMALE, "natural_sample_rate_hertz": 24000},
        "fil-PH-Wavenet-D": {"gender": Gender.MALE, "natural_sample_rate_hertz": 24000},
        "fil-PH-Standard-A": {"gender": Gender.FEMALE, "natural_sample_rate_hertz": 24000},
        "fil-PH-Standard-B": {"gender": Gender.MALE, "natural_sample_rate_hertz": 24000},
        "fil-PH-Standard-C": {"gender": Gender.FEMALE, "natural_sample_rate_hertz": 24000},
        "fil-PH-Standard-D": {"gender": Gender.MALE, "natural_sample_rate_hertz": 24000},
    },
}


@dataclass
class GoogleTTSVoice(TTSVoice):
    """Google TTS voice model."""
    
    language_code: str = ""
    ssml_gender: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert voice to dictionary."""
        return {
            **super().to_dict(),
            "language_code": self.language_code,
            "ssml_gender": self.ssml_gender,
        }
    
    @classmethod
    def from_google_voice(cls, voice: texttospeech.Voice) -> "GoogleTTSVoice":
        """Create from Google Voice object."""
        # Extract language and region from language code (e.g., "en-US" -> "en", "US")
        language_parts = voice.language_codes[0].split("-")
        language = language_parts[0] if len(language_parts) > 0 else ""
        region = language_parts[1] if len(language_parts) > 1 else ""
        
        # Get additional voice info from our mapping
        voice_info = VOICE_NAME_MAPPING.get(voice.language_codes[0], {}).get(
            voice.name, {}
        )
        
        return cls(
            provider_id=voice.name,
            name=voice.name,
            language=language,
            region=region,
            language_code=voice.language_codes[0],
            gender=voice_info.get("gender", Gender.UNSPECIFIED),
            ssml_gender=voice.ssml_gender.name,
            provider="google",
            natural_sample_rate_hertz=voice_info.get("natural_sample_rate_hertz", 24000),
        )


class GoogleTTSService(TTSService):
    """Google Cloud Text-to-Speech service implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Google TTS service.
        
        Args:
            config: Configuration dictionary with the following keys:
                - credentials_path: Path to Google Cloud service account JSON file
                - credentials_json: JSON string with service account credentials
                - default_voice: Default voice ID to use
                - default_audio_config: Default audio configuration
        """
        self.config = config or {}
        self._client = None
        self._voices = []
        self._voices_loaded = False
        self._credentials = None
        
        # Set default voice
        self.default_voice_id = self.config.get("default_voice", "en-US-Wavenet-D")
        
        # Set audio config
        self.audio_config = {
            **DEFAULT_AUDIO_CONFIG,
            **(self.config.get("audio_config") or {}),
        }
    
    async def initialize(self) -> None:
        """Initialize the TTS service."""
        try:
            await self._initialize_client()
            await self.load_voices()
        except Exception as e:
            raise TTSServiceInitializationError(
                f"Failed to initialize Google TTS service: {e}"
            ) from e
    
    async def _initialize_client(self) -> None:
        """Initialize the Google TTS client."""
        if self._client is not None:
            return
        
        try:
            # Try to get credentials from config
            credentials = None
            
            # Option 1: Credentials from JSON string
            if "credentials_json" in self.config:
                credentials_info = json.loads(self.config["credentials_json"])
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_info
                )
            
            # Option 2: Credentials from file
            elif "credentials_path" in self.config:
                credentials_path = Path(self.config["credentials_path"]).expanduser()
                if not credentials_path.exists():
                    raise ValueError(
                        f"Credentials file not found: {credentials_path}"
                    )
                credentials = service_account.Credentials.from_service_account_file(
                    str(credentials_path)
                )
            
            # Option 3: Use application default credentials
            else:
                # This will use GOOGLE_APPLICATION_CREDENTIALS environment variable
                pass
            
            # Create the client
            self._client = texttospeech.TextToSpeechAsyncClient(credentials=credentials)
            
        except Exception as e:
            raise TTSServiceInitializationError(
                f"Failed to initialize Google TTS client: {e}"
            ) from e
    
    async def load_voices(self, force_refresh: bool = False) -> None:
        """Load available voices from Google TTS.
        
        Args:
            force_refresh: If True, force a refresh of the voices cache
        """
        if self._voices_loaded and not force_refresh:
            return
        
        try:
            await self._initialize_client()
            
            # List all available voices
            response = await self._client.list_voices()
            
            # Convert to our voice model
            self._voices = [
                GoogleTTSVoice.from_google_voice(voice) for voice in response.voices
            ]
            
            self._voices_loaded = True
            logger.info("Loaded %d Google TTS voices", len(self._voices))
            
        except (GoogleAPICallError, RetryError) as e:
            logger.error("Failed to load Google TTS voices: %s", e)
            raise TTSServiceError(f"Failed to load voices: {e}") from e
    
    async def get_voices(
        self, language: Optional[str] = None, gender: Optional[Gender] = None
    ) -> List[TTSVoice]:
        """Get a list of available voices.
        
        Args:
            language: Filter by language code (e.g., 'en', 'fil')
            gender: Filter by gender
            
        Returns:
            List of available voices
        """
        await self.load_voices()
        
        # Filter voices
        voices = self._voices
        
        if language:
            voices = [v for v in voices if v.language.lower() == language.lower()]
        
        if gender:
            voices = [v for v in voices if v.gender == gender]
        
        return voices
    
    async def get_voice(self, voice_id: str) -> TTSVoice:
        """Get a specific voice by ID.
        
        Args:
            voice_id: The ID of the voice to get
            
        Returns:
            The voice with the specified ID
            
        Raises:
            VoiceNotAvailableError: If the voice is not found
        """
        await self.load_voices()
        
        for voice in self._voices:
            if voice.provider_id == voice_id:
                return voice
        
        raise VoiceNotAvailableError(f"Voice not found: {voice_id}")
    
    async def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        output_format: Optional[Union[str, AudioFormat]] = None,
        **kwargs,
    ) -> bytes:
        """Synthesize speech from text.
        
        Args:
            text: The text to synthesize
            voice_id: The ID of the voice to use
            output_format: The desired output format (e.g., 'mp3', 'wav')
            **kwargs: Additional arguments for the TTS service
                - language_code: Language code (e.g., 'en-US')
                - speaking_rate: Speaking rate/speed (0.25 to 4.0)
                - pitch: Pitch adjustment (-20.0 to 20.0)
                - volume_gain_db: Volume gain in decibels (-96.0 to 16.0)
                - effects_profile_id: Audio profile ID
                
        Returns:
            The synthesized audio as bytes
            
        Raises:
            TTSServiceError: If synthesis fails
        """
        try:
            await self._initialize_client()
            
            # Get voice
            voice_config = texttospeech.VoiceSelectionParams()
            
            if voice_id:
                voice = await self.get_voice(voice_id)
                voice_config.language_code = voice.language_code
                voice_config.name = voice.provider_id
            else:
                # Use default voice
                try:
                    voice = await self.get_voice(self.default_voice_id)
                    voice_config.language_code = voice.language_code
                    voice_config.name = voice.provider_id
                except VoiceNotAvailableError:
                    # Fall back to just using the language code
                    voice_config.language_code = kwargs.get("language_code", "en-US")
            
            # Set SSML voice gender
            ssml_gender = getattr(
                texttospeech.SsmlVoiceGender,
                getattr(voice, "ssml_gender", "SSML_VOICE_GENDER_UNSPECIFIED"),
                texttospeech.SsmlVoiceGender.SSML_VOICE_GENDER_UNSPECIFIED,
            )
            voice_config.ssml_gender = ssml_gender
            
            # Set audio config
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=kwargs.get("speaking_rate", self.audio_config["speaking_rate"]),
                pitch=kwargs.get("pitch", self.audio_config["pitch"]),
                volume_gain_db=kwargs.get(
                    "volume_gain_db", self.audio_config["volume_gain_db"]
                ),
                effects_profile_id=kwargs.get("effects_profile_id", None),
            )
            
            # Set the text input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Perform the text-to-speech request
            response = await self._client.synthesize_speech(
                input=synthesis_input,
                voice=voice_config,
                audio_config=audio_config,
            )
            
            return response.audio_content
            
        except Exception as e:
            logger.error("Failed to synthesize speech: %s", e, exc_info=True)
            raise TTSServiceError(f"Failed to synthesize speech: {e}") from e
    
    async def synthesize_to_file(
        self,
        text: str,
        output_file: Union[str, Path],
        voice_id: Optional[str] = None,
        output_format: Optional[Union[str, AudioFormat]] = None,
        **kwargs,
    ) -> Path:
        """Synthesize speech from text and save to a file.
        
        Args:
            text: The text to synthesize
            output_file: Path to save the audio file
            voice_id: The ID of the voice to use
            output_format: The desired output format (e.g., 'mp3', 'wav')
            **kwargs: Additional arguments for the TTS service
                
        Returns:
            Path to the generated audio file
            
        Raises:
            TTSServiceError: If synthesis fails
        """
        try:
            # Synthesize speech
            audio_content = await self.synthesize_speech(
                text=text,
                voice_id=voice_id,
                output_format=output_format,
                **kwargs,
            )
            
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write audio to file
            with open(output_path, "wb") as f:
                f.write(audio_content)
            
            logger.debug("Saved synthesized speech to %s", output_path)
            return output_path
            
        except Exception as e:
            logger.error("Failed to save synthesized speech: %s", e, exc_info=True)
            raise TTSServiceError(f"Failed to save synthesized speech: {e}") from e
