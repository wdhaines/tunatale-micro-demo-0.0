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

## EdgeTTS Service Issues
- Previously had 52 ClientConnectionResetError failures from too many concurrent requests
- Fixed with rate limiting: MIN_REQUEST_DELAY=0.2s, MAX_CONCURRENT_REQUESTS=3
- Enhanced pause processing increased TTS calls from ~172 to ~280+, requiring optimization