# Tagalog Language Learning Tools

This repository contains tools for processing Tagalog language learning materials, including a multi-voice TTS processor for generating natural-sounding audio from scripts.

## Tools

### 1. Multi-Voice TTS Processor

A Python package for generating natural-sounding multi-voice audio from language learning scripts.

#### Features:
- Multiple voice support with different Filipino and English voices
- Automatic section detection (Key Phrases, Natural Speed, Slow Speed, Translated)
- Pitch and speed adjustments for voice differentiation
- Precise timing control for pauses and pacing
- Audio normalization and quality enhancements
- Section-based audio splitting
- Timestamp generation for synchronization

### 2. Edge TTS Converter

Converts daily Tagalog-English stories from markdown files into MP3 audio files using Microsoft Edge TTS.

### 3. Google Cloud TTS Converter

Alternative converter using Google Cloud Text-to-Speech (requires API credentials).

### Google Cloud TTS Converter
- Higher quality Wavenet voices
- More natural sounding speech
- Advanced SSML support for better prosody and pronunciation
- Handles long texts with automatic chunking

## Installation

1. Install Python 3.7 or higher
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Edge TTS Converter
```bash
python edge_tts_converter.py input.md -o output_directory
```

### Google Cloud TTS Converter
```bash
python google_cloud_tts_converter.py input.md -o output_directory --credentials path/to/credentials.json
```

### Common Arguments
- `input.md`: Path to the markdown file containing the stories
- `-o, --output-dir`: (Optional) Directory to save output MP3 files (default: 'output' for Edge TTS, 'cloud_tts_output' for Google Cloud TTS)

### Google Cloud TTS Specific Arguments
- `--credentials`: Path to Google Cloud service account JSON credentials file
- `--day`: (Optional) Process only a specific day (e.g., 1 for Day 1)

### Examples

#### Edge TTS Example
```bash
python edge_tts_converter.py tagalog/demo-0.0.3.txt -o edge_tts_output
```

#### Google Cloud TTS Example
```bash
python google_cloud_tts_converter.py tagalog/demo-0.0.3.txt -o google_tts_output --credentials tagalog-tts-credentials.json
```

## Markdown Format

The script expects markdown files in the following format:

```markdown
## Day 1: Title
**Target collocations:** [list]
### [Story Title]
[Story text with Tagalog and English]
**Word count: XXX**
```

## Requirements

- Python 3.7+
- Internet connection (for TTS service)
- See `requirements.txt` for Python dependencies

## Notes

- The script uses Microsoft Edge TTS for high-quality voice synthesis
- Common English loanwords in Tagalog are automatically detected and pronounced with Tagalog TTS
- The script creates the output directory if it doesn't exist
- Progress is shown in the console during conversion
