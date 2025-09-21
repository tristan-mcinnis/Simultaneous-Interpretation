from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Protocol

from ..config import AppConfig


class Transcriber(Protocol):
    def transcribe_file(self, audio_path: Path, language: str) -> str:
        """Return the recognised text for the audio file."""


class WhisperCppTranscriber:
    def __init__(self, model: str, threads: Optional[int] = None):
        try:
            from whispercpp import Whisper  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "whispercpp package not installed. Install it with 'pip install whispercpp'."
            ) from exc

        if Path(model).exists():
            self.model = Whisper(model)
        else:
            self.model = Whisper.from_pretrained(model)
        self.threads = threads

    def transcribe_file(self, audio_path: Path, language: str) -> str:
        kwargs = {}
        if self.threads:
            kwargs["threads"] = self.threads
        if language:
            kwargs["language"] = language

        result = self.model.transcribe(str(audio_path), **kwargs)
        return _segments_to_text(result)


class FasterWhisperTranscriber:
    def __init__(self, model_size: str, threads: Optional[int] = None):
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        from faster_whisper import WhisperModel  # type: ignore

        cpu_threads = threads or max(1, (os.cpu_count() or 2) // 2)
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
            num_workers=cpu_threads,
        )

    def transcribe_file(self, audio_path: Path, language: str) -> str:
        segments, _ = self.model.transcribe(str(audio_path), language=language)
        return _segments_to_text(segments)


def _segments_to_text(result: Iterable) -> str:
    def extract_text(segment) -> str:
        if isinstance(segment, str):
            return segment
        if isinstance(segment, dict):
            return str(segment.get("text", ""))
        return str(getattr(segment, "text", ""))

    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        segments = result.get("segments", [])
        return "".join(extract_text(segment) for segment in segments)
    if hasattr(result, "__iter__"):
        return "".join(extract_text(segment) for segment in result)
    return str(result)


def create_transcriber(config: AppConfig) -> Transcriber:
    if config.transcriber == "whispercpp":
        return WhisperCppTranscriber(config.whisper_model, config.whisper_threads)
    return FasterWhisperTranscriber(config.whisper_model, config.whisper_threads)
