"""Utility modules for the testing framework."""

from tests.utils.audio_utils import (
    AudioChunk,
    AudioFile,
    chunk_audio,
    get_audio_duration,
    load_audio,
    save_audio,
)
from tests.utils.api_clients import (
    LMStudioClient,
    OpenAITranslationClient,
    OpenAITTSClient,
    VibeVoiceClient,
    WhisperCppClient,
)
from tests.utils.evaluation import (
    calculate_bleu,
    calculate_comet,
    calculate_wer,
)

__all__ = [
    # Audio utilities
    "AudioChunk",
    "AudioFile",
    "chunk_audio",
    "get_audio_duration",
    "load_audio",
    "save_audio",
    # API clients
    "LMStudioClient",
    "OpenAITranslationClient",
    "OpenAITTSClient",
    "VibeVoiceClient",
    "WhisperCppClient",
    # Evaluation
    "calculate_bleu",
    "calculate_comet",
    "calculate_wer",
]
