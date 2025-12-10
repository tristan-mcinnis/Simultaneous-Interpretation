"""
API clients for different engines with timing and metrics collection.

Provides unified interfaces for:
- WhisperCpp (local STT)
- OpenAI Translation (cloud)
- LM Studio (local translation via OpenAI-compatible API)
- OpenAI TTS (cloud)
- VibeVoice (local TTS)
"""

import asyncio
import io
import json
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class TimedResult:
    """Base class for results with timing information."""

    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class STTResult(TimedResult):
    """Result from speech-to-text processing."""

    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    word_timestamps: Optional[list[dict]] = None


@dataclass
class TranslationResult(TimedResult):
    """Result from translation processing."""

    text: str
    source_text: str
    tokens_used: Optional[int] = None
    tokens_per_sec: Optional[float] = None


@dataclass
class TTSResult(TimedResult):
    """Result from text-to-speech processing."""

    audio_data: bytes
    sample_rate: int
    first_audio_latency_ms: float  # Time to first audio chunk
    total_latency_ms: float  # Time to complete audio


class BaseSTTClient(ABC):
    """Abstract base class for STT clients."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path | str,
        language: str = "zh",
    ) -> STTResult:
        """Transcribe audio file to text."""
        pass

    @abstractmethod
    def transcribe_chunk(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        language: str = "zh",
    ) -> STTResult:
        """Transcribe audio data directly."""
        pass


class BaseTranslationClient(ABC):
    """Abstract base class for translation clients."""

    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str = "zh",
        target_lang: str = "en",
        context: Optional[list[str]] = None,
    ) -> TranslationResult:
        """Translate text from source to target language."""
        pass


class BaseTTSClient(ABC):
    """Abstract base class for TTS clients."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Synthesize speech from text."""
        pass


class WhisperCppClient(BaseSTTClient):
    """Client for whisper.cpp local transcription."""

    def __init__(
        self,
        model: str = "ggml-medium.bin",
        model_path: Optional[str] = None,
        threads: Optional[int] = None,
        language: str = "zh",
    ):
        """
        Initialize WhisperCpp client.

        Args:
            model: Model name or filename
            model_path: Path to model file (optional, for custom paths)
            threads: Number of CPU threads (None for auto)
            language: Default language for transcription
        """
        self.model_name = model
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.threads = threads
        self.default_language = language
        self._model = None

    def _load_model(self):
        """Lazy load the whisper model."""
        if self._model is not None:
            return

        try:
            from whispercpp import Whisper
        except ImportError:
            raise ImportError(
                "whispercpp is required. Install with: pip install whispercpp"
            )

        if self.model_path and self.model_path.exists():
            self._model = Whisper(str(self.model_path))
        else:
            # Try loading from pretrained models
            self._model = Whisper.from_pretrained(self.model_name.replace(".bin", ""))

    def transcribe(
        self,
        audio_path: Path | str,
        language: str = "zh",
    ) -> STTResult:
        """Transcribe audio file to text."""
        self._load_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        start_time = time.perf_counter()

        # Perform transcription
        result = self._model.transcribe(
            str(audio_path),
            lang=language or self.default_language,
            n_threads=self.threads,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract text from result
        text = self._extract_text(result)

        return STTResult(
            text=text,
            latency_ms=latency_ms,
            language=language,
        )

    def transcribe_chunk(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        language: str = "zh",
    ) -> STTResult:
        """Transcribe audio data directly by saving to temp file."""
        from scipy.io import wavfile

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)
            wavfile.write(temp_path, sample_rate, audio_data)

        try:
            return self.transcribe(temp_path, language)
        finally:
            temp_path.unlink()

    def _extract_text(self, result: Any) -> str:
        """Extract text from various whisper result formats."""
        if isinstance(result, str):
            return result.strip()

        if isinstance(result, dict):
            return result.get("text", "").strip()

        # Handle segment-based results
        if hasattr(result, "__iter__"):
            segments = []
            for segment in result:
                if isinstance(segment, str):
                    segments.append(segment)
                elif hasattr(segment, "text"):
                    segments.append(segment.text)
                elif isinstance(segment, dict):
                    segments.append(segment.get("text", ""))
            return " ".join(segments).strip()

        return str(result).strip()


class OpenAITranslationClient(BaseTranslationClient):
    """Client for OpenAI GPT translation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize OpenAI translation client.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: Model to use for translation
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            system_prompt: Custom system prompt for translation
        """
        import os

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are a professional simultaneous interpreter. "
            "Translate the following text naturally and accurately. "
            "Maintain the speaker's tone and intent. "
            "Output only the translation, no explanations."
        )

    def translate(
        self,
        text: str,
        source_lang: str = "zh",
        target_lang: str = "en",
        context: Optional[list[str]] = None,
    ) -> TranslationResult:
        """Translate text using OpenAI API."""
        # Build prompt with context
        user_content = self._build_prompt(text, source_lang, target_lang, context)

        start_time = time.perf_counter()

        result = self._translate_chat_completions(user_content)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return TranslationResult(
            text=result["text"],
            source_text=text,
            latency_ms=latency_ms,
            tokens_used=result.get("tokens"),
            tokens_per_sec=result.get("tokens") / (latency_ms / 1000)
            if result.get("tokens")
            else None,
        )

    def _build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[list[str]],
    ) -> str:
        """Build the translation prompt with context."""
        parts = []

        if context:
            parts.append("Previous translations for context:")
            for prev in context[-5:]:  # Last 5 for context
                parts.append(f"- {prev}")
            parts.append("")

        lang_map = {"zh": "Chinese", "en": "English"}
        source_name = lang_map.get(source_lang, source_lang)
        target_name = lang_map.get(target_lang, target_lang)

        parts.append(f"Translate from {source_name} to {target_name}:")
        parts.append(text)

        return "\n".join(parts)

    def _translate_chat_completions(self, user_content: str) -> dict:
        """Use Chat Completions API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return {
            "text": response.choices[0].message.content.strip(),
            "tokens": response.usage.total_tokens if response.usage else None,
        }


class LMStudioClient(BaseTranslationClient):
    """Client for LM Studio local translation (OpenAI-compatible API)."""

    def __init__(
        self,
        endpoint: str = "http://localhost:1234/v1/chat/completions",
        model: str = "qwen2.5-7b-instruct",
        temperature: float = 0.3,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize LM Studio client.

        Args:
            endpoint: LM Studio server endpoint
            model: Model identifier (for logging)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            system_prompt: Custom system prompt
            timeout: Request timeout in seconds
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required. Install with: pip install httpx")

        self.endpoint = endpoint
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system_prompt = system_prompt or (
            "You are a professional simultaneous interpreter. "
            "Translate the following text naturally and accurately. "
            "Maintain the speaker's tone and intent. "
            "Output only the translation, no explanations."
        )
        self._client = httpx.Client(timeout=timeout)

    def translate(
        self,
        text: str,
        source_lang: str = "zh",
        target_lang: str = "en",
        context: Optional[list[str]] = None,
    ) -> TranslationResult:
        """Translate text using LM Studio."""
        user_content = self._build_prompt(text, source_lang, target_lang, context)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        start_time = time.perf_counter()

        response = self._client.post(self.endpoint, json=payload)
        response.raise_for_status()

        latency_ms = (time.perf_counter() - start_time) * 1000

        data = response.json()
        result_text = data["choices"][0]["message"]["content"].strip()
        tokens = data.get("usage", {}).get("total_tokens")

        return TranslationResult(
            text=result_text,
            source_text=text,
            latency_ms=latency_ms,
            tokens_used=tokens,
            tokens_per_sec=tokens / (latency_ms / 1000) if tokens else None,
        )

    def _build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[list[str]],
    ) -> str:
        """Build the translation prompt with context."""
        parts = []

        if context:
            parts.append("Previous translations for context:")
            for prev in context[-5:]:
                parts.append(f"- {prev}")
            parts.append("")

        lang_map = {"zh": "Chinese", "en": "English"}
        source_name = lang_map.get(source_lang, source_lang)
        target_name = lang_map.get(target_lang, target_lang)

        parts.append(f"Translate from {source_name} to {target_name}:")
        parts.append(text)

        return "\n".join(parts)

    def check_health(self) -> bool:
        """Check if LM Studio server is running."""
        try:
            # Try the models endpoint
            models_endpoint = self.endpoint.replace(
                "/chat/completions", "/models"
            ).replace("/v1/v1", "/v1")
            response = self._client.get(models_endpoint)
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        """Close the HTTP client."""
        self._client.close()


class OpenAITTSClient(BaseTTSClient):
    """Client for OpenAI TTS."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini-tts",
        voice: str = "nova",
        speed: float = 1.0,
        response_format: str = "pcm",
    ):
        """
        Initialize OpenAI TTS client.

        Args:
            api_key: OpenAI API key
            model: TTS model to use
            voice: Voice preset
            speed: Speech speed multiplier
            response_format: Audio format (pcm, mp3, opus, etc.)
        """
        import os

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.voice = voice
        self.speed = speed
        self.response_format = response_format
        self.sample_rate = 24000  # OpenAI TTS outputs 24kHz

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Synthesize speech from text."""
        voice = voice or self.voice
        speed = speed or self.speed

        start_time = time.perf_counter()
        first_chunk_time = None
        audio_chunks = []

        # Use streaming for timing measurement
        with self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=voice,
            input=text,
            speed=speed,
            response_format=self.response_format,
        ) as response:
            for chunk in response.iter_bytes(chunk_size=4096):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                audio_chunks.append(chunk)

        total_time = time.perf_counter()

        audio_data = b"".join(audio_chunks)
        first_latency_ms = (first_chunk_time - start_time) * 1000 if first_chunk_time else 0
        total_latency_ms = (total_time - start_time) * 1000

        return TTSResult(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            latency_ms=total_latency_ms,
            first_audio_latency_ms=first_latency_ms,
            total_latency_ms=total_latency_ms,
        )


class VibeVoiceClient(BaseTTSClient):
    """Client for VibeVoice local TTS via WebSocket."""

    def __init__(
        self,
        endpoint: str = "ws://localhost:8765",
        model: str = "vibevoice-realtime-0.5b",
        sample_rate: int = 24000,
        speaker_id: Optional[int] = None,
    ):
        """
        Initialize VibeVoice client.

        Args:
            endpoint: WebSocket server endpoint
            model: Model identifier (for logging)
            sample_rate: Output sample rate
            speaker_id: Speaker ID for multi-speaker models
        """
        self.endpoint = endpoint
        self.model = model
        self.sample_rate = sample_rate
        self.speaker_id = speaker_id

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Synthesize speech from text using VibeVoice WebSocket."""
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets is required. Install with: pip install websockets"
            )

        # Run async synthesis
        return asyncio.get_event_loop().run_until_complete(
            self._synthesize_async(text, speed)
        )

    async def _synthesize_async(self, text: str, speed: float) -> TTSResult:
        """Async implementation of synthesis."""
        import websockets

        start_time = time.perf_counter()
        first_chunk_time = None
        audio_chunks = []

        async with websockets.connect(self.endpoint) as ws:
            # Send synthesis request
            request = {
                "text": text,
                "speed": speed,
                "sample_rate": self.sample_rate,
            }
            if self.speaker_id is not None:
                request["speaker_id"] = self.speaker_id

            await ws.send(json.dumps(request))

            # Receive audio chunks
            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=30.0)

                    if isinstance(message, bytes):
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter()
                        audio_chunks.append(message)
                    else:
                        # JSON message - could be status or end marker
                        data = json.loads(message)
                        if data.get("status") == "complete":
                            break

                except asyncio.TimeoutError:
                    break

        total_time = time.perf_counter()

        audio_data = b"".join(audio_chunks)
        first_latency_ms = (
            (first_chunk_time - start_time) * 1000 if first_chunk_time else 0
        )
        total_latency_ms = (total_time - start_time) * 1000

        return TTSResult(
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            latency_ms=total_latency_ms,
            first_audio_latency_ms=first_latency_ms,
            total_latency_ms=total_latency_ms,
        )

    def check_health(self) -> bool:
        """Check if VibeVoice server is running."""
        try:
            import websockets

            async def _check():
                try:
                    async with websockets.connect(
                        self.endpoint,
                        close_timeout=2,
                    ) as ws:
                        await ws.ping()
                        return True
                except Exception:
                    return False

            return asyncio.get_event_loop().run_until_complete(_check())
        except Exception:
            return False
