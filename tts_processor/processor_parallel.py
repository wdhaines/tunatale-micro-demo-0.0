"""Parallel TTS processing implementation."""

import asyncio
import re
import tempfile
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import soundfile as sf
import edge_tts
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Constants
EDGE_TTS_SAMPLE_RATE = 24000  # Edge TTS generates audio at 24kHz

class SectionType(Enum):
    """Types of sections in the script."""
    KEY_PHRASES = "Key Phrases"
    NATURAL_SPEED = "Natural Speed"
    SLOW_SPEED = "Slow Speed"
    TRANSLATED = "Translated"

@dataclass
class VoiceLine:
    """Represents a single line of text with voice attributes."""
    voice_tag: str
    text: str
    section: SectionType
    speed: float = 1.0
    pitch_shift: int = 0

@dataclass
class AudioSegmentInfo:
    """Container for audio segment and its metadata."""
    audio_data: np.ndarray
    sample_rate: int
    voice_tag: str
    section: SectionType
    text: str
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        return (len(self.audio_data) / self.sample_rate) * 1000

class ParallelTTSProcessor:
    """Processes multi-voice TTS generation from scripts with parallel processing."""
    
    def __init__(self, output_dir: str = "output", output_format: str = "mp3", bitrate: str = "192k"):
        """Initialize the TTS processor."""
        self.voices = {
            'NARRATOR': 'en-US-AriaNeural',
            'TAGALOG-FEMALE-1': 'fil-PH-BlessicaNeural',
            'TAGALOG-FEMALE-2': 'fil-PH-BlessicaNeural',
            'TAGALOG-MALE-1': 'fil-PH-AngeloNeural',
            'TAGALOG-MALE-2': 'fil-PH-AngeloNeural'
        }
        
        # Voice-specific adjustments
        self.voice_adjustments = {
            'TAGALOG-FEMALE-2': {'pitch_shift': 2, 'speed': 0.95},
            'TAGALOG-MALE-2': {'pitch_shift': -1, 'speed': 0.95}
        }
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio settings
        self.silence_between_lines = 300  # ms
        self.silence_between_sections = 1000  # ms
        self.practice_syllable_pause = 500  # ms
        self.practice_phrase_pause = 1000  # ms
        
        # Audio format settings
        self.audio_format = output_format.lower()
        self.bitrate = bitrate
        
        # Validate output format
        if self.audio_format not in ['wav', 'mp3']:
            logger.warning(f"Unsupported audio format: {self.audio_format}. Defaulting to 'wav'.")
            self.audio_format = "wav"
    
    async def _generate_tts_single(self, text: str, voice: str, rate: str = '+0%', pitch: str = '+0Hz') -> Tuple[np.ndarray, int]:
        """Generate TTS audio for a single text chunk with the given voice and settings."""
        # Create a temporary WAV file for the TTS output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
        
        try:
            # Generate TTS using edge-tts
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
                pitch=pitch,
                volume='+0%'  # Keep volume at normal level
            )
            
            # Save the TTS output to the temporary file
            await communicate.save(temp_wav_path)
            
            # Verify the file was created and has content
            file_size = os.path.getsize(temp_wav_path)
            if file_size == 0:
                raise ValueError("Generated TTS file is empty")
            
            # Load the generated audio - Edge TTS generates at 24kHz
            audio_data, sample_rate = sf.read(temp_wav_path, dtype='float32')
            
            return audio_data, sample_rate
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_wav_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_wav_path}: {e}")
    
    async def generate_tts(self, text: str, voice: str, rate: str = '+0%', pitch: str = '+0Hz',
                          max_concurrent: int = 3) -> Tuple[np.ndarray, int]:
        """
        Generate TTS audio for the given text and voice with parallel processing.
        
        Args:
            text: The text to convert to speech
            voice: The voice to use for TTS
            rate: The speech rate adjustment (e.g., '+0%', '+10%', '-5%')
            pitch: The pitch adjustment (e.g., '+0Hz', '+2st', '-1st')
            max_concurrent: Maximum number of concurrent TTS requests
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        # Split text into sentences for parallel processing
        sentences = self._split_into_sentences(text)
        
        # If there's only one sentence, process it directly
        if len(sentences) <= 1:
            return await self._generate_tts_single(text, voice, rate, pitch)
        
        # Process sentences in parallel with limited concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_sentence(sentence: str):
            async with semaphore:
                try:
                    return await self._generate_tts_sentence(sentence, voice, rate, pitch)
                except Exception as e:
                    logger.error(f"Error processing sentence: {sentence[:50]}... - {e}")
                    # Return silence on error to allow processing to continue
                    silence_duration = 1000  # 1 second of silence
                    silence_samples = int(silence_duration * EDGE_TTS_SAMPLE_RATE / 1000)
                    return np.zeros(silence_samples, dtype='float32'), EDGE_TTS_SAMPLE_RATE
        
        # Process all sentences in parallel
        tasks = [process_sentence(s) for s in sentences if s.strip()]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Combine results
        audio_segments = []
        sample_rate = EDGE_TTS_SAMPLE_RATE
        
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                audio_data, sr = result
                if audio_data is not None and len(audio_data) > 0:
                    # Ensure all segments have the same sample rate
                    if sr != sample_rate:
                        logger.warning(f"Sample rate mismatch: expected {sample_rate}, got {sr}")
                        # Resample if needed (implementation depends on your requirements)
                        # For now, we'll just proceed with the first sample rate
                    audio_segments.append(audio_data)
        
        if not audio_segments:
            raise ValueError("No valid audio segments were generated")
            
        # Concatenate all audio segments
        combined_audio = np.concatenate(audio_segments)
        return combined_audio, sample_rate
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for parallel processing."""
        if not text.strip():
            return []
            
        # Simple sentence splitting - can be enhanced with NLTK for better accuracy
        sentences = []
        
        # Split on common sentence boundaries
        parts = re.split(r'([.!?]\s*)', text)
        
        # Reconstruct sentences with their terminators
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                sentence = (parts[i] + parts[i+1]).strip()
                if sentence:
                    sentences.append(sentence)
            else:
                remaining = parts[i].strip()
                if remaining:
                    if sentences and not any(sentences[-1].endswith(p) for p in ('.', '!', '?')):
                        sentences[-1] = f"{sentences[-1]} {remaining}"
                    else:
                        sentences.append(remaining)
        
        # If no sentences were found, return the original text as a single sentence
        if not sentences:
            return [text.strip()] if text.strip() else []
            
        return sentences
    
    async def _generate_tts_sentence(self, sentence: str, voice: str, rate: str, pitch: str) -> Tuple[np.ndarray, int]:
        """Generate TTS for a single sentence with error handling and retries."""
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return await self._generate_tts_single(sentence, voice, rate, pitch)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Attempt {attempt + 1} failed for sentence: {sentence[:50]}... "
                        f"Retrying in {wait_time}s... Error: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to generate TTS after {max_retries + 1} attempts "
                        f"for sentence: {sentence[:50]}... Error: {e}"
                    )
                    raise
    
    def _save_audio(self, audio_data: np.ndarray, sample_rate: int, output_file: Path) -> None:
        """Save audio data to a file.
        
        Args:
            audio_data: Audio data as a numpy array
            sample_rate: Sample rate of the audio
            output_file: Path to save the audio file to
        """
        try:
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to int16 if needed
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            # Save as WAV first (temporary)
            temp_wav = output_file.with_suffix('.wav')
            sf.write(
                str(temp_wav),
                audio_data,
                sample_rate,
                format='WAV',
                subtype='PCM_16'
            )
            
            # Convert to desired format if needed
            if output_file.suffix.lower() != '.wav':
                from pydub import AudioSegment
                sound = AudioSegment.from_wav(str(temp_wav))
                sound.export(str(output_file), format=output_file.suffix[1:], bitrate=self.bitrate)
                # Remove temporary WAV file
                temp_wav.unlink()
            
            logger.info(f"Saved audio to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving audio to {output_file}: {e}")
            # Clean up any partially written files
            if 'temp_wav' in locals() and temp_wav.exists():
                temp_wav.unlink()
            raise
