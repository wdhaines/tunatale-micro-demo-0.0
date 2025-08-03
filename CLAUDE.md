# Claude Memory - TunaTale Project

## Testing
- **IMPORTANT**: Do NOT use `make test` to run tests
- **Correct way**: Use `pytest` with the correct virtual environment
- The project uses pytest for testing, not make test

## TTS System Architecture
- Multi-provider TTS system with EdgeTTS and other providers
- SSML (Speech Synthesis Markup Language) support for enhanced pause processing
- Ellipsis-to-pause conversion system for natural speech timing
- Rate limiting implemented to prevent EdgeTTS service overload

## Recent Optimizations Completed
- Single ellipsis (`...`) now handled naturally by TTS (no artificial segmentation)
- Custom pause durations reduced by 0.25s (e.g., `....` = 0.5s instead of 0.75s)
- Comprehensive rate limiting with 200ms delays and connection pooling
- All 15 ellipsis handling tests passing
- **NEW**: Universal abbreviation handler implemented (automatically detects and converts abbreviations to phonetic pronunciation)
- **NEW**: Filipino number clarification system with authentic Spanish time and Tagalog digit breakdown
- **FIXED**: TTS cache collision bug that served wrong audio after preprocessing changes
- **FIXED**: Cache lookup fallback logic that was causing wrong audio to be served
- **NEW**: Dynamic pause multipliers based on phrase length for better collocation timing

## Key Files
- `tunatale/core/utils/tts_preprocessor.py` - ELLIPSES_TO_SSML_MAPPING configuration + Universal abbreviation handler
- `tunatale/infrastructure/services/tts/edge_tts_service.py` - Rate limiting implementation
- `tunatale/core/services/lesson_processor.py` - Long ellipsis detection logic
- `tests/unit/core/services/test_ellipsis_handling.py` - Test coverage for ellipsis handling

## Universal Abbreviation Handler Features
- **Auto-detection**: Regex pattern matches 1-6 capital letters as word boundaries
- **Phonetic conversion**: A→"ay", B→"bee", C→"see", etc. for all letters A-Z
- **Smart filtering**: Protects common words (TO, IN, PO, etc.) and real English words (CODE, HELP, etc.)
- **Language support**: Includes common Tagalog words in protected list
- **Heuristic validation**: Uses vowel ratios and word patterns to distinguish abbreviations from real words
- **Examples**: CR→"see are", FBI→"eff bee eye", NASA→"en ay ess ay", but CODE→"CODE" (unchanged)
- **Test coverage**: 92 comprehensive unit tests covering all functionality and edge cases

## Comprehensive Test Suite Added
- **TestUniversalAbbreviationHandler**: 77 dedicated tests for abbreviation handling
- **Real demo content**: Tests using actual examples from tagalog/demo-*.txt files  
- **Coverage**: Travel (CR, ID), time (AM, PM), currency (USD, PHP), technology (GPS, WIFI, HTML)
- **Protection**: Ensures PO, TO, CODE, HELP, etc. remain unchanged
- **Edge cases**: Punctuation, boundaries, mixed case, empty strings
- **Parametrized tests**: Systematic testing of all common abbreviations
- **Integration**: Works seamlessly with existing ellipsis handling (107/107 total tests pass)

## Filipino Number Clarification System
- **Implementation**: `process_number_clarification()` in `tts_preprocessor.py`
- **Spanish time system**: Authentic Filipino usage (alas otso y medya, kinse para alas una)
- **Tagalog digit breakdown**: isa, dalawa, tatlo for large numbers (150 → "isa lima zero")
- **Section-based behavior**: Auto-clarify in slow_speed, tag-controlled in natural_speed
- **SSML tags**: Uses `<clarify></clarify>` tags for manual control, removes from output
- **Integration**: Runs BEFORE abbreviation fixes but AFTER text preprocessing
- **Test coverage**: 35+ comprehensive unit tests covering all scenarios

## TTS Cache System
- **Cache location**: `cache/` directory with voice-specific subdirectories
- **Cache key format**: `{voice_id}_{rate}_{pitch}_{volume}_{text_hash}.mp3`
- **Hash algorithm**: SHA-256 (upgraded from MD5 for collision resistance)
- **Hash length**: 16 characters (upgraded from 8 to reduce collisions)
- **Preprocessing versioning**: v2 includes Filipino number clarification changes
- **IMPORTANT**: Cache keys include preprocessing version to invalidate when logic changes

## Dynamic Pause Multipliers for Collocations
- **Word count-based scaling**: Longer phrases get progressively longer pause multipliers
  - 1 word: 1.5x multiplier (baseline)
  - 2 words: 1.8x multiplier  
  - 3 words: 2.2x multiplier
  - 4 words: 2.6x multiplier
  - 5 words: 3.0x multiplier
  - 6+ words: 3.5x multiplier
- **Audio duration bonus**: Extra 200ms per second of audio over 3 seconds
- **Implementation**: `natural_pause_calculator.py:_get_dynamic_multiplier()`

## EdgeTTS Service Issues
- Previously had 52 ClientConnectionResetError failures from too many concurrent requests
- Fixed with rate limiting: MIN_REQUEST_DELAY=0.2s, MAX_CONCURRENT_REQUESTS=3
- Enhanced pause processing increased TTS calls from ~172 to ~280+, requiring optimization
- **Cache collision bug**: Fixed MD5 8-char hash collisions that served wrong audio after preprocessing changes
- **Cache lookup bug**: Fixed fallback logic that matched wrong files with same base pattern