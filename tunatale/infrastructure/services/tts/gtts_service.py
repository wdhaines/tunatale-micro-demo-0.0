"""Google Translate Text-to-Speech service implementation (free)."""
import asyncio
import io
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from gtts import gTTS
from gtts.lang import tts_langs

from tunatale.core.exceptions import (
    TTSServiceError,
    VoiceNotAvailableError,
)
from tunatale.core.models.voice import Voice
from tunatale.core.models.enums import VoiceGender
from tunatale.core.ports.tts_service import TTSService

# Configure logging
logger = logging.getLogger(__name__)

# Language mapping for gTTS (supports fewer languages but is free)
GTTS_SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'gender': VoiceGender.NEUTRAL},
    'fil': {'name': 'Filipino', 'gender': VoiceGender.NEUTRAL},
    'tl': {'name': 'Tagalog', 'gender': VoiceGender.NEUTRAL},  # Alternative code for Tagalog
    'es': {'name': 'Spanish', 'gender': VoiceGender.NEUTRAL},
    'fr': {'name': 'French', 'gender': VoiceGender.NEUTRAL},
    'de': {'name': 'German', 'gender': VoiceGender.NEUTRAL},
    'it': {'name': 'Italian', 'gender': VoiceGender.NEUTRAL},
    'pt': {'name': 'Portuguese', 'gender': VoiceGender.NEUTRAL},
    'ja': {'name': 'Japanese', 'gender': VoiceGender.NEUTRAL},
    'ko': {'name': 'Korean', 'gender': VoiceGender.NEUTRAL},
    'zh': {'name': 'Chinese', 'gender': VoiceGender.NEUTRAL},
    'hi': {'name': 'Hindi', 'gender': VoiceGender.NEUTRAL},
    'ar': {'name': 'Arabic', 'gender': VoiceGender.NEUTRAL},
    'ru': {'name': 'Russian', 'gender': VoiceGender.NEUTRAL},
}

# TLD mapping for different regional variants/voices
TLD_VARIANTS = {
    'en': ['com', 'co.uk', 'com.au', 'ca'],  # US, UK, Australia, Canada
    'fil': ['com.ph'],  # Philippines
    'tl': ['com.ph'],   # Philippines (Tagalog)
    'es': ['com', 'es', 'com.mx'],  # Global, Spain, Mexico
    'fr': ['fr', 'ca'],  # France, Canada
    'de': ['de'],  # Germany
    'it': ['it'],  # Italy
    'pt': ['com.br', 'pt'],  # Brazil, Portugal
    'ja': ['co.jp'],  # Japan
    'ko': ['co.kr'],  # South Korea
    'zh': ['com', 'com.tw'],  # Global, Taiwan
    'hi': ['co.in'],  # India
    'ar': ['com'],  # Global
    'ru': ['ru'],  # Russia
}


class GTTSVoice(Voice):
    """Google Translate TTS voice model extending base Voice."""
    
    def __init__(self, **data):
        # Map gTTS specific fields to Voice fields
        lang_code = None
        tld = None
        
        if 'language_code' in data:
            lang_code = data.pop('language_code')
        if 'tld' in data:
            tld = data.pop('tld')
            
        # Store TLD and language_code in metadata
        if 'metadata' not in data:
            data['metadata'] = {}
        if tld:
            data['metadata']['tld'] = tld
        if lang_code:
            data['metadata']['language_code'] = lang_code
        
        super().__init__(**data)


class GTTSService:
    """Google Translate Text-to-Speech service implementation (free)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the gTTS service.
        
        Args:
            config: Configuration dictionary with the following keys:
                - default_voice: Default voice ID to use
                - slow: Whether to use slow speech by default
                - timeout: Request timeout in seconds
        """
        self.config = config or {}
        self._voices = []
        self._voices_loaded = False
        
        # Set default voice
        self.default_voice_id = self.config.get("default_voice", "en-com")
        
        # Set default options
        self.default_slow = self.config.get("slow", False)
        self.timeout = self.config.get("timeout", 30)
    
    @property
    def name(self) -> str:
        """Get the name of the TTS service."""
        return "gtts"
    
    async def initialize(self) -> None:
        """Initialize the TTS service."""
        try:
            await self.load_voices()
        except Exception as e:
            raise TTSServiceError(
                f"Failed to initialize gTTS service: {e}"
            ) from e
    
    async def load_voices(self, force_refresh: bool = False) -> None:
        """Load available voices from gTTS.
        
        Args:
            force_refresh: If True, force a refresh of the voices cache
        """
        if self._voices_loaded and not force_refresh:
            return
        
        try:
            self._generate_voices()
            logger.info("Loaded %d gTTS voices", len(self._voices))
            
        except Exception as e:
            logger.error("Failed to load gTTS voices: %s", e)
            raise TTSServiceError(f"Failed to load voices: {e}") from e
    
    async def get_voices(
        self, language: Optional[str] = None
    ) -> List[Voice]:
        """Get a list of available voices.
        
        Args:
            language: Filter by language code (e.g., 'en', 'fil')
            gender: Filter by gender (Note: gTTS doesn't provide gender info)
            
        Returns:
            List of available voices
        """
        await self.load_voices()
        
        # Filter voices
        voices = self._voices
        
        if language:
            voices = [v for v in voices if v.language.value.lower() == language.lower()]
        
        return voices
    
    def get_voice(self, voice_id: str) -> Optional[Voice]:
        """Get a specific voice by ID.
        
        Args:
            voice_id: The ID of the voice to get (format: lang-tld, e.g., 'fil-com.ph')
            
        Returns:
            The voice with the specified ID
            
        Raises:
            VoiceNotAvailableError: If the voice is not found
        """
        # Load voices synchronously if needed (convert to simple lookup)
        if not self._voices_loaded:
            # For simplicity, generate voices on demand
            self._generate_voices()
        
        for voice in self._voices:
            if voice.id == voice_id or voice.provider_id == voice_id:
                return voice
        
        return None
    
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
        """Synthesize speech from text and save to file.
        
        Args:
            text: The text to synthesize
            voice_id: The ID of the voice to use (format: lang-tld)
            output_path: Path to save the audio file
            rate: Speech rate (ignored by gTTS)
            pitch: Pitch adjustment (ignored by gTTS)
            volume: Volume level (ignored by gTTS)
            **kwargs: Additional arguments:
                - slow: Whether to use slow speech
                - lang_check: Whether to check if language is supported
                
        Returns:
            Dict containing metadata about the synthesis
            
        Raises:
            TTSServiceError: If synthesis fails
        """
        try:
            # Parse voice_id to get language and tld
            voice = self.get_voice(voice_id)
            if voice and 'language_code' in voice.metadata:
                lang = voice.metadata['language_code']
                tld = voice.metadata.get('tld', 'com')
            else:
                # Fallback: try to parse voice_id directly
                if '-' in voice_id:
                    lang, tld = voice_id.split('-', 1)
                else:
                    lang = voice_id
                    tld = 'com'
            
            # Map language codes for gTTS compatibility
            if lang == 'fil':
                # gTTS doesn't support 'fil', use 'tl' for Tagalog
                lang = 'tl'
            elif lang not in GTTS_SUPPORTED_LANGUAGES:
                # Try Tagalog alternatives
                if lang in ['tagalog', 'filipino']:
                    lang = 'tl'
                else:
                    raise TTSServiceError(f"Language '{lang}' not supported by gTTS")
            
            # Configure gTTS options
            slow = kwargs.get('slow', self.default_slow)
            lang_check = kwargs.get('lang_check', True)
            
            # Create gTTS object
            tts = gTTS(
                text=text,
                lang=lang,
                tld=tld,
                slow=slow,
                lang_check=lang_check
            )
            
            # Ensure output directory exists
            output_path = Path(output_path)
            
            # Ensure the output path has .mp3 extension
            if output_path.suffix.lower() != '.mp3':
                output_path = output_path.with_suffix('.mp3')
                
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save audio to file
            tts.save(str(output_path))
            
            # Return metadata dict like EdgeTTS
            return {
                'voice_id': voice_id,
                'output_path': str(output_path),
                'file_size': output_path.stat().st_size if output_path.exists() else 0,
                'language': lang,
                'tld': tld,
                'provider': 'gtts'
            }
            
        except Exception as e:
            logger.error("Failed to synthesize speech with gTTS: %s", e, exc_info=True)
            raise TTSServiceError(f"Failed to synthesize speech: {e}") from e
    
    async def validate_credentials(self) -> bool:
        """Validate credentials (gTTS doesn't require credentials)."""
        return True
    
    def _generate_voices(self):
        """Generate voices synchronously."""
        if self._voices_loaded:
            return
        
        self._voices = []
        
        for lang_code, lang_info in GTTS_SUPPORTED_LANGUAGES.items():
            tld_variants = TLD_VARIANTS.get(lang_code, ['com'])
            
            for i, tld in enumerate(tld_variants):
                voice_id = f"{lang_code}-{tld}"
                voice_name = f"{lang_info['name']} ({tld.upper()})"
                
                # Map language code to Language enum
                from tunatale.core.models.enums import Language
                # Special handling for Filipino/Tagalog
                if lang_code == 'fil':
                    lang_enum = Language.TAGALOG
                else:
                    lang_enum = Language.from_string(lang_code) or Language.ENGLISH
                
                voice = GTTSVoice(
                    id=voice_id,
                    provider_id=voice_id,
                    name=voice_name,
                    language=lang_enum,
                    gender=lang_info['gender'],
                    provider="gtts",
                    sample_rate=22050,
                    language_code=lang_code,
                    tld=tld
                )
                
                self._voices.append(voice)
        
        self._voices_loaded = True