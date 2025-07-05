#!/usr/bin/env python3
"""Command-line interface for the TTS processor."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from tts_processor.processor import MultiVoiceTTS

logger = logging.getLogger(__name__)

async def process_script(script_path: str, output_dir: str, verbose: bool = False) -> None:
    """Process a single script file.
    
    Args:
        script_path: Path to the script file
        output_dir: Directory to save output files
        verbose: Enable debug logging
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Process the script
    tts = MultiVoiceTTS(output_dir=output_dir)
    
    try:
        logger.info(f"Processing script: {script_path}")
        tts.parse_script(script_path)
        await tts.generate_audio()
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Error processing script: {e}", exc_info=verbose)
        sys.exit(1)

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description='Generate multi-voice TTS from script')
    
    # Required arguments
    parser.add_argument('script', help='Path to the script file to process')
    parser.add_argument('-o', '--output', default='output', help='Output directory')
    parser.add_argument('--format', choices=['wav', 'mp3'], default='mp3', help='Output audio format')
    parser.add_argument('--bitrate', default='192k', help='Audio bitrate (e.g., 128k, 192k, 256k)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging after parsing arguments
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Enable debug logging for edge-tts if in debug mode
    if args.debug:
        logging.getLogger('edge_tts').setLevel(logging.DEBUG)
    
    # Validate script path
    script_path = Path(args.script).resolve()
    if not script_path.exists():
        logger.error(f"Script file not found: {script_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process the script
    asyncio.run(process_script(args.script, str(output_dir), args.debug))

if __name__ == "__main__":
    main()
