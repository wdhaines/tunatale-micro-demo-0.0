"""File utility functions for TunaTale."""
import os
import re
import shutil
from pathlib import Path
from typing import Union, Optional


def ensure_directory(directory: Union[str, Path]) -> Path:
    """Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: The directory path to ensure exists.
        
    Returns:
        Path: The path to the directory, guaranteed to exist.
        
    Raises:
        OSError: If the directory cannot be created.
    """
    path = Path(directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize a string to be used as a filename.
    
    Args:
        filename: The input string to sanitize.
        max_length: Maximum length of the resulting filename.
        
    Returns:
        str: A sanitized filename.
    """
    # Replace invalid characters with underscores
    filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
    
    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    
    # Remove leading/trailing whitespace and dots
    filename = filename.strip('. ')
    
    # Truncate to max length
    if len(filename) > max_length:
        # Keep the file extension if present
        if '.' in filename:
            name, ext = os.path.splitext(filename)
            ext_len = len(ext)
            if ext_len < max_length:
                name = name[:max_length - ext_len - 1]
                filename = f"{name}{ext}"
        filename = filename[:max_length]
    
    # If the filename is empty after sanitization, use a default
    if not filename:
        filename = 'untitled'
    
    return filename
