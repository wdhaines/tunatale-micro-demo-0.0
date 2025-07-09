"""TunaTale command-line interface."""
"""TunaTale main module."""
import logging
import os
from pathlib import Path
from typing import Optional

from tunatale.cli.main import app

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Set up logging configuration.
    
    Args:
        verbose: If True, set console log level to DEBUG
        log_file: Optional path to log file. If not provided, uses 'logs/tunatale_debug.log'
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Determine log file path
    log_file_path = Path(log_file) if log_file else log_dir / 'tunatale_debug.log'
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler (DEBUG level)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
        '    [%(pathname)s:%(lineno)d]\n    [%(funcName)s]\n',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Set specific log levels for our modules
    logging.getLogger('tunatale').setLevel(logging.DEBUG)
    logging.getLogger('edge_tts').setLevel(logging.INFO)  # Can be noisy at DEBUG
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logger.debug("Logging configured with debug output")
    
    # Add file handler if log_file is provided
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            root_logger.addHandler(file_handler)
            root_logger.debug(f"Logging to file: {log_file}")
        except Exception as e:
            root_logger.error(f"Failed to set up file logging: {e}")
    
    # Set log level for specific loggers
    logging.getLogger('edge_tts').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    # Enable debug logging for our application
    logging.getLogger('tunatale').setLevel(logging.DEBUG if verbose else logging.INFO)

if __name__ == "__main__":
    setup_logging()
    app()
