# Google Translate TTS (gTTS) Setup

## Overview

The TunaTale system now supports Google Translate TTS (gTTS) as a free alternative to EdgeTTS. This provides additional voice options, including the Filipino voice "fil-com.ph".

## Features

- **Free Service**: No API keys or payment required
- **Multiple Languages**: 22+ voices across multiple languages
- **Regional Variants**: Different TLD variants (US, UK, Philippines, etc.)
- **Tagalog Support**: Native Filipino/Tagalog voice support

## Available gTTS Voices

- **Tagalog**: `fil-com.ph` (Filipino, Philippines)
- **English**: `en-com`, `en-co.uk`, `en-com.au`, `en-ca`
- **Spanish**: `es-com`, `es-es`, `es-com.mx`
- **And more...**: French, German, Portuguese, Japanese, Korean, etc.

## Installation

The gTTS dependency is already included in `requirements.txt`:

```bash
pip install gTTS>=2.3.0
```

## Usage in Lesson Files

To use the Google Translate TTS voice in lesson files, use the `TAGALOG-GTTS-1` speaker pattern:

```
[TAGALOG-GTTS-1]: Salamat po!
[NARRATOR]: Thank you!
```

This maps to voice ID `fil-com.ph` which uses the Google Translate TTS service.

## Current Status

✅ **Implemented:**
- Complete gTTS service (`GTTSService`)
- Voice mapping in lesson parser (`TAGALOG-GTTS-1` → `fil-com.ph`)
- Factory integration for 'gtts' provider
- Voice selector with gTTS voices
- Language code mapping (`fil` → `tl` for gTTS compatibility)

⚠️ **Architecture Note:**
The current system is configured with a single TTS provider (EdgeTTS). To fully enable multi-provider support, the lesson processor would need to be enhanced to route different voice IDs to their appropriate TTS services.

## Testing

You can test the gTTS implementation directly:

```python
from tunatale.infrastructure.services.tts.gtts_service import GTTSService

service = GTTSService()
voices = await service.get_voices()
result = await service.synthesize_speech(
    text="Salamat po!",
    voice_id="fil-com.ph", 
    output_path="/tmp/test.mp3"
)
```

## Future Enhancement

To enable full multi-provider support in lesson processing, consider implementing:

1. **Multi-Provider TTS Service**: A composite service that routes voice IDs to appropriate providers
2. **Provider Detection**: Automatic detection of provider based on voice ID patterns
3. **Fallback Logic**: Graceful fallback between providers if one is unavailable

This would allow seamless mixing of EdgeTTS and gTTS voices within the same lesson.