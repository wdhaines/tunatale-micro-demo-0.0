"""Interfaces for lesson processing components."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable, Union

from tunatale.core.models.lesson import Lesson
from tunatale.core.models.phrase import Phrase
from tunatale.core.models.section import Section
from tunatale.core.ports.audio_processor import AudioProcessor
from tunatale.core.ports.tts_service import TTSService


class ProgressCallback(Protocol):
    """Protocol for progress callbacks."""

    def __call__(
        self,
        current: int,
        total: int,
        status: str,
        **kwargs: Any
    ) -> None:
        """Callback for progress updates.

        Args:
            current: Current item being processed
            total: Total number of items to process
            status: Current status message
            **kwargs: Additional progress information
        """

    def on_start(self) -> None:
        """Called when processing starts."""

    def on_complete(self) -> None:
        """Called when processing completes."""


@runtime_checkable
class LessonProcessor(Protocol):
    """Interface for lesson processors."""

    async def process_lesson(
        self,
        lesson: Union[Lesson, Path, str],
        output_dir: Path,
        progress_callback: Optional[ProgressCallback] = None,
        **options: Any
    ) -> Dict[str, Any]:
        """Process a lesson and generate audio files.

        Args:
            lesson: Lesson object or path to lesson file
            output_dir: Directory to save output files
            progress_callback: Optional callback for progress updates
            **options: Additional processing options

        Returns:
            Dictionary with processing results
        """

    async def process_phrase(
        self,
        phrase: Phrase,
        output_dir: Path,
        **options: Any
    ) -> Dict[str, Any]:
        """Process a single phrase and generate audio.

        Args:
            phrase: Phrase to process
            output_dir: Directory to save output files
            **options: Additional processing options

        Returns:
            Dictionary with processing results
        """

    async def process_section(
        self,
        section: Section,
        output_dir: Path,
        **options: Any
    ) -> Dict[str, Any]:
        """Process a section and generate audio.

        Args:
            section: Section to process
            output_dir: Directory to save output files
            **options: Additional processing options

        Returns:
            Dictionary with processing results
        """


class LessonProcessorBase(ABC):
    """Base class for lesson processors."""

    def __init__(
        self,
        tts_service: TTSService,
        audio_processor: AudioProcessor,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the lesson processor.

        Args:
            tts_service: TTS service for speech synthesis
            audio_processor: Audio processor for audio manipulation
            config: Configuration dictionary
        """
        self.tts_service = tts_service
        self.audio_processor = audio_processor
        self.config = config or {}

        # Default configuration
        self.default_config = {
            'output_format': 'mp3',
            'silence_between_phrases': 0.5,  # seconds
            'silence_between_sections': 1.5,  # seconds
            'normalize_audio': True,
            'trim_silence': True,
            'max_parallel': 4,  # Maximum parallel TTS requests
        }

        # Update with user config
        self.default_config.update(self.config)

    async def initialize(self) -> None:
        """Initialize the processor and its dependencies."""
        # Initialize TTS service if needed
        if hasattr(self.tts_service, 'initialize'):
            await self.tts_service.initialize()

    @abstractmethod
    async def process_lesson(
        self,
        lesson: Union[Lesson, Path, str],
        output_dir: Path,
        progress_callback: Optional[ProgressCallback] = None,
        **options: Any
    ) -> Dict[str, Any]:
        """Process a lesson and generate audio files."""

    @abstractmethod
    async def process_section(
        self,
        section: Section,
        output_dir: Path,
        **options: Any
    ) -> Dict[str, Any]:
        """Process a section and generate audio."""

    @abstractmethod
    async def process_phrase(
        self,
        phrase: Phrase,
        output_dir: Path,
        **options: Any
    ) -> Dict[str, Any]:
        """Process a single phrase and generate audio."""

    def _merge_config(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Merge default config with provided options."""
        config = self.default_config.copy()
        config.update(options)
        return config
