"""
This module contains a predefined list of known voices for the Edge TTS service.
This is used to avoid fetching the entire voice list from the network, which can be
slow and error-prone.
"""

# A predefined list of known voices that the application uses.
# This avoids the need to fetch the entire voice list from the network.
# The voice data here is a subset of the data returned by the Edge TTS service.
KNOWN_VOICES = [
    {
        "Name": "Microsoft Server Speech Text to Speech Voice (fil-PH, BlessicaNeural)",
        "ShortName": "fil-PH-BlessicaNeural",
        "Gender": "Female",
        "Locale": "fil-PH",
        "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
        "FriendlyName": "Microsoft Blessica Online (Natural) - Filipino (Philippines)",
        "Status": "GA",
        "VoiceTag": {
            "ContentCategories": [
                "General"
            ],
            "VoicePersonalities": [
                "Friendly",
                "Positive"
            ]
        }
    },
    {
        "Name": "Microsoft Server Speech Text to Speech Voice (fil-PH, RosaNeural)",
        "ShortName": "fil-PH-RosaNeural",
        "Gender": "Female",
        "Locale": "fil-PH",
        "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
        "FriendlyName": "Microsoft Rosa Online (Natural) - Filipino (Philippines)",
        "Status": "GA",
        "VoiceTag": {
            "ContentCategories": [
                "General"
            ],
            "VoicePersonalities": [
                "Friendly",
                "Positive"
            ]
        }
    },
    {
        "Name": "Microsoft Server Speech Text to Speech Voice (fil-PH, AngeloNeural)",
        "ShortName": "fil-PH-AngeloNeural",
        "Gender": "Male",
        "Locale": "fil-PH",
        "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
        "FriendlyName": "Microsoft Angelo Online (Natural) - Filipino (Philippines)",
        "Status": "GA",
        "VoiceTag": {
            "ContentCategories": [
                "General"
            ],
            "VoicePersonalities": [
                "Friendly",
                "Positive"
            ]
        }
    },
    {
        "Name": "Microsoft Server Speech Text to Speech Voice (en-US, GuyNeural)",
        "ShortName": "en-US-GuyNeural",
        "Gender": "Male",
        "Locale": "en-US",
        "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
        "FriendlyName": "Microsoft Guy Online (Natural) - English (United States)",
        "Status": "GA",
        "VoiceTag": {
            "ContentCategories": [
                "General"
            ],
            "VoicePersonalities": [
                "Friendly",
                "Positive"
            ]
        }
    },
    {
        "Name": "Microsoft Server Speech Text to Speech Voice (en-US, AriaNeural)",
        "ShortName": "en-US-AriaNeural",
        "Gender": "Female",
        "Locale": "en-US",
        "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
        "FriendlyName": "Microsoft Aria Online (Natural) - English (United States)",
        "Status": "GA",
        "VoiceTag": {
            "ContentCategories": [
                "General"
            ],
            "VoicePersonalities": [
                "Friendly",
                "Positive"
            ]
        }
    },
    {
        "Name": "Microsoft Server Speech Text to Speech Voice (en-US, JennyNeural)",
        "ShortName": "en-US-JennyNeural",
        "Gender": "Female",
        "Locale": "en-US",
        "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
        "FriendlyName": "Microsoft Jenny Online (Natural) - English (United States)",
        "Status": "GA",
        "VoiceTag": {
            "ContentCategories": [
                "General"
            ],
            "VoicePersonalities": [
                "Friendly",
                "Positive"
            ]
        }
    }
]
