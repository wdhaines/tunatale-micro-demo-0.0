# TunaTale: Tagalog Language Learning Tools

TunaTale is a comprehensive toolkit for creating engaging language learning materials with natural-sounding multi-voice audio. It's designed specifically for Tagalog language instruction but can be adapted for other languages.

## 🚀 Features

- **Multi-Voice Support**: Different Filipino and English voices for natural dialogue
- **Multiple TTS Providers**: EdgeTTS (premium) and Google Translate TTS (free) support
- **Structured Lessons**: Automatic section detection (Key Phrases, Natural Speed, Slow Speed, Translated)
- **High-Quality Audio**: Professional audio processing with normalization and enhancements
- **Dynamic Pauses**: Smart pause duration based on audio length for key phrases repetition
- **Flexible Configuration**: Customize voices, speeds, and processing options
- **Modern Architecture**: Clean architecture with clear separation of concerns
- **Extensible**: Easy to add new TTS providers and audio processing features

## 🏗️ Project Structure

```
tunatale/
├── core/                # Core domain logic and business rules
│   ├── models/         # Domain models (Phrase, Section, Lesson)
│   └── ports/          # Interface definitions (TTSService, etc.)
├── infrastructure/     # External implementations
│   └── services/       # Service implementations (EdgeTTS, GoogleTTS)
├── cli/                # Command line interface
└── tests/              # Test suite
```

## 🛠️ Installation

1. **Prerequisites**
   - Python 3.10 or higher
   - [Poetry](https://python-poetry.org/) (recommended) or pip
   - FFmpeg (for audio processing)

2. **Using Poetry (recommended)**
   ```bash
   # Install Poetry if you don't have it
   curl -sSL https://install.python-poetry.org | python3 -
   
   # Clone the repository
   git clone https://github.com/yourusername/tunatale.git
   cd tunatale
   
   # Install dependencies
   poetry install
   
   # Activate the virtual environment
   poetry shell
   ```

3. **Using pip**
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/tunatale.git
   cd tunatale
   
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

1. **Basic Usage**
   ```bash
   # Process a lesson file
   tunatale generate path/to/lesson.txt -o output/
   
   # Validate a lesson file
   tunatale validate path/to/lesson.txt
   
   # List available voices
   tunatale list-voices
   ```

2. **Example Lesson Format**
   ```
   Day 1: Greetings
   
   [KEY PHRASES]
   [TAGALOG-FEMALE-1] Magandang umaga po!
   [TAGALOG-MALE-1] Magandang umaga rin po!
   
   [NATURAL SPEED]
   [TAGALOG-FEMALE-1] Kumusta po kayo?
   [TAGALOG-MALE-1] Mabuti naman po, salamat!
   ```

## 📚 Documentation

For detailed documentation, please see the [docs](docs/) directory.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft Edge TTS for high-quality voice synthesis
- The language learning community for inspiration and feedback
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
