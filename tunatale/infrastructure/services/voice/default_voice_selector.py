"""Default implementation of VoiceSelector for basic voice selection."""
from typing import Any, List, Optional

from tunatale.core.models.voice import Voice
from tunatale.core.models.enums import Language
from tunatale.core.ports.voice_selector import VoiceSelector


class DefaultVoiceSelector(VoiceSelector):
    """Default implementation of VoiceSelector with basic voice selection."""

    def __init__(self):
        """Initialize the default voice selector with some default voices."""
        self.voices = [
            # English voices
            Voice(
                id="en-US-GuyNeural",
                provider_id="en-US-GuyNeural",
                name="Guy",
                language=Language.ENGLISH,
                gender="male",
                provider="edge",
            ),
            Voice(
                id="en-US-JennyNeural",
                provider_id="en-US-JennyNeural",
                name="Jenny",
                language=Language.ENGLISH,
                gender="female",
                provider="edge",
            ),
            # Tagalog voices
            Voice(
                id="fil-PH-BlessicaNeural",
                provider_id="fil-PH-BlessicaNeural",
                name="Blessica",
                language=Language.TAGALOG,
                gender="female",
                provider="edge",
            ),
            Voice(
                id="fil-PH-AngeloNeural",
                provider_id="fil-PH-AngeloNeural",
                name="Angelo",
                language=Language.TAGALOG,
                gender="male",
                provider="edge",
            ),
            # Spanish voices
            Voice(
                id="es-MX-JorgeNeural",
                provider_id="es-MX-JorgeNeural",
                name="Jorge",
                language=Language.SPANISH,
                gender="male",
                provider="edge",
            ),
            Voice(
                id="es-MX-DaliaNeural",
                provider_id="es-MX-DaliaNeural",
                name="Dalia",
                language=Language.SPANISH,
                gender="female",
                provider="edge",
            ),
        ]

    async def get_voice(
        self,
        language: Language,
        gender: Optional[str] = None,
        **kwargs: Any
    ) -> Optional[Voice]:
        """Get the most appropriate voice for the given parameters.
        
        Args:
            language: The language of the text to be spoken
            gender: Preferred gender of the voice (e.g., 'male', 'female')
            **kwargs: Additional parameters for voice selection
            
        Returns:
            A Voice object if a matching voice is found, None otherwise
        """
        voices = await self.get_voices(language=language, gender=gender, **kwargs)
        return voices[0] if voices else None

    async def get_voices(
        self,
        language: Optional[Language] = None,
        gender: Optional[str] = None,
        **kwargs: Any
    ) -> List[Voice]:
        """Get all available voices matching the given criteria.
        
        Args:
            language: Filter voices by language
            gender: Filter voices by gender
            **kwargs: Additional filters
            
        Returns:
            A list of Voice objects matching the criteria
        """
        filtered_voices = self.voices
        
        if language is not None:
            filtered_voices = [v for v in filtered_voices if v.language == language]
            
        if gender is not None:
            filtered_voices = [v for v in filtered_voices if v.gender.lower() == gender.lower()]
            
        return filtered_voices

    async def get_default_voice(self, language: Language) -> Optional[Voice]:
        """Get the default voice for a given language.
        
        Args:
            language: The language to get the default voice for
            
        Returns:
            The default Voice for the language, or None if none found
        """
        voices = await self.get_voices(language=language)
        return voices[0] if voices else None
        
    async def get_voice_id(
        self,
        language: Language,
        gender: Optional[str] = None,
        speaker_id: Optional[str] = None,
        **kwargs: Any
    ) -> Optional[str]:
        """Get the voice ID for the given parameters.
        
        Args:
            language: The language of the text to be spoken
            gender: Preferred gender of the voice (e.g., 'male', 'female')
            speaker_id: Optional speaker ID to use for voice selection
            **kwargs: Additional parameters for voice selection
            
        Returns:
            A voice ID string if a matching voice is found, None otherwise
        """
        # If speaker_id is provided, try to find a matching voice
        if speaker_id:
            voices = await self.get_voices(language=language, gender=gender)
            for voice in voices:
                if voice.id == speaker_id or voice.provider_id == speaker_id:
                    return voice.id
        
        # Otherwise, get the default voice for the language and gender
        voice = await self.get_voice(language=language, gender=gender, **kwargs)
        return voice.id if voice else None
