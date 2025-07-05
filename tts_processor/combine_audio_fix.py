from typing import List
import numpy as np
from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)

def combine_audio_segments(segments, output_file, audio_format='mp3', bitrate='192k'):
    """Combine multiple audio segments into a single audio file.
    
    Args:
        segments: List of AudioSegment objects to combine
        output_file: Path where the output file should be saved
        audio_format: Output format ('mp3' or 'wav')
        bitrate: Bitrate for the output file
    """
    if not segments:
        logger.warning("No audio segments to combine")
        return None
        
    logger.info(f"Combining {len(segments)} audio segments...")
    
    # Convert all segments to the same format (16-bit, 24kHz, mono)
    processed_segments = []
    for i, segment in enumerate(segments):
        try:
            # Convert to numpy array if needed
            if not hasattr(segment, 'audio_data'):
                logger.warning(f"Segment {i} has no audio_data, skipping")
                continue
                
            # Get audio data as numpy array
            audio_data = segment.audio_data
            
            # Convert to 16-bit PCM if needed
            if audio_data.dtype != np.int16:
                if np.issubdtype(audio_data.dtype, np.floating):
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)
            
            # Create AudioSegment
            audio_segment = AudioSegment(
                audio_data.tobytes(),
                frame_rate=24000,  # Edge TTS uses 24kHz
                sample_width=2,    # 16-bit
                channels=1         # Mono
            )
            processed_segments.append(audio_segment)
            
        except Exception as e:
            logger.error(f"Error processing segment {i}: {e}")
            continue
    
    if not processed_segments:
        logger.error("No valid audio segments to combine")
        return None
        
    # Combine all segments
    combined = processed_segments[0]
    for segment in processed_segments[1:]:
        combined += segment
        
    # Normalize the final audio to prevent clipping
    combined = combined.normalize()
    
    try:
        # Export to the desired format
        if audio_format == 'wav':
            combined.export(
                str(output_file),
                format="wav",
                parameters=["-ar", "24000"]  # 24kHz sample rate
            )
        else:  # MP3
            combined.export(
                str(output_file),
                format="mp3",
                bitrate=bitrate,
                parameters=["-ar", "24000"],  # 24kHz sample rate
                tags={
                    'title': 'Day 1 - Complete',
                    'artist': 'TunaTale TTS',
                    'album': 'Tagalog Language Learning',
                    'track': '1/1'
                }
            )
        
        logger.info(f"Combined audio saved to {output_file} (duration: {len(combined)/1000:.2f}s)")
        return combined
        
    except Exception as e:
        logger.error(f"Error saving combined audio: {e}")
        raise
