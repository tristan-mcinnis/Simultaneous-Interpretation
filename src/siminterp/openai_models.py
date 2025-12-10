"""Canonical OpenAI model identifiers used by the CLI."""

from __future__ import annotations

from typing import Tuple

# Translation/chat models that are known to work well with the interpreter pipeline.
TRANSLATION_MODELS: Tuple[str, ...] = (
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
)

# Default translation model keeps compatibility with previous releases while
# allowing users to upgrade explicitly.
DEFAULT_TRANSLATION_MODEL = "gpt-4o"

# Latest text-to-speech models available from OpenAI.
TTS_MODELS: Tuple[str, ...] = (
    "gpt-4o-mini-tts",
    "tts-1",
    "tts-1-hd",
)

DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
