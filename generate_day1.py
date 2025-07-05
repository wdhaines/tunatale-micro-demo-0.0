"""Generate TTS audio for Tagalog Day 1 content."""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from tts_processor.processor import MultiVoiceTTS

# Configure logging with both console and file output
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"tts_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Console handler (INFO level)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handler (DEBUG level)
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info(f"Logging to file: {log_file.absolute()}")
logger = logging.getLogger(__name__)

async def main():
    """Generate TTS audio for Day 1 content using parallel processing."""
    # Input and output paths
    script_path = Path("tagalog/demo-0.0.3-day-1.txt")
    output_dir = Path("output/day1")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting TTS generation for {script_path}")
    logger.info(f"Output will be saved to: {output_dir.absolute()}")
    
    try:
        # Initialize the TTS processor with parallel processing
        tts = MultiVoiceTTS(
            output_dir=str(output_dir),
            output_format="mp3",
            bitrate="192k",
            max_concurrent=3  # Process up to 3 TTS requests in parallel
        )
        
        # Parse the script
        logger.info("Parsing script...")
        tts.parse_script(script_path)
        
        # Generate audio with parallel processing
        logger.info("Generating audio with parallel processing...")
        await tts.generate_audio()
        
        logger.info("TTS generation completed successfully!")
        logger.info(f"Output files are in: {output_dir.absolute()}")
        
    except Exception as e:
        logger.error(f"Error during TTS generation: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
