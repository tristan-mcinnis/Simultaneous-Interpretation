from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(slots=True)
class AppConfig:
    """Container for user configurable runtime options."""

    api_key: str
    input_device_index: Optional[int]
    output_device_index: Optional[int]
    input_language: str
    translation_language: str
    enable_translation: bool
    enable_tts: bool
    dictionary_path: Optional[Path]
    topic: str
    openai_model: str
    tts_voice: str
    tts_model: str
    transcriber: str
    whisper_model: str
    whisper_threads: Optional[int]
    chunk_history: int
    phrase_time_limit: int
    ambient_duration: float
    tts_speed: float
    log_file: Path
    translation_temperature: float


def load_environment() -> None:
    """Load environment variables from .env files if present."""

    load_dotenv(override=False)


def build_config(args) -> AppConfig:
    """Create an :class:`AppConfig` instance from parsed CLI arguments."""

    load_environment()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. Configure it in your environment or .env file."
        )

    dictionary_path: Optional[Path] = None
    if getattr(args, "dictionary", None):
        dictionary_path = Path(args.dictionary).expanduser()
        if not dictionary_path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {dictionary_path}")

    log_file = Path(getattr(args, "log_file", "logfile.txt")).expanduser()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    whisper_threads = getattr(args, "whisper_threads", None)
    if whisper_threads is not None and whisper_threads <= 0:
        raise ValueError("--whisper-threads must be a positive integer if provided")

    chunk_history = max(1, getattr(args, "history", 10))

    return AppConfig(
        api_key=api_key,
        input_device_index=getattr(args, "input_device", None),
        output_device_index=getattr(args, "output_device", None),
        input_language=getattr(args, "input_language", "en"),
        translation_language=getattr(args, "target_language", "fr"),
        enable_translation=bool(getattr(args, "translate", False)),
        enable_tts=bool(getattr(args, "tts", False)),
        dictionary_path=dictionary_path,
        topic=getattr(args, "topic", ""),
        openai_model=getattr(args, "model", "gpt-4o"),
        tts_voice=getattr(args, "voice", "alloy"),
        tts_model=getattr(args, "tts_model", "gpt-4o-mini-tts"),
        transcriber=getattr(args, "transcriber", "whispercpp"),
        whisper_model=getattr(args, "whisper_model", "base.en"),
        whisper_threads=whisper_threads,
        chunk_history=chunk_history,
        phrase_time_limit=max(1, int(getattr(args, "phrase_time_limit", 8))),
        ambient_duration=max(0.0, float(getattr(args, "ambient_duration", 2.0))),
        tts_speed=max(0.25, float(getattr(args, "tts_speed", 1.0))),
        log_file=log_file,
        translation_temperature=float(getattr(args, "temperature", 0.0)),
    )
