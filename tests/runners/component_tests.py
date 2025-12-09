"""
Component-level test runner.

Tests each component (STT, Translation, TTS) in isolation
to establish baseline performance.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tests.configs import load_config
from tests.metrics.latency import LatencyCollector, LatencyMetrics, LatencyStats
from tests.metrics.quality import QualityCollector, QualityMetrics
from tests.metrics.resources import ResourceCollector
from tests.utils.api_clients import (
    BaseSTTClient,
    BaseTTSClient,
    BaseTranslationClient,
    LMStudioClient,
    OpenAITranslationClient,
    OpenAITTSClient,
    STTResult,
    TranslationResult,
    TTSResult,
    VibeVoiceClient,
    WhisperCppClient,
)
from tests.utils.audio_utils import AudioFile, get_audio_duration, load_audio


@dataclass
class ComponentTestResult:
    """Result from a component test."""

    component: str  # "stt", "translation", "tts"
    config_name: str
    latency: LatencyStats
    quality: Optional[dict] = None
    resources: Optional[dict] = None
    raw_results: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "component": self.component,
            "config_name": self.config_name,
            "latency": self.latency.to_dict(),
            "quality": self.quality,
            "resources": self.resources,
            "error_count": len(self.errors),
            "errors": self.errors[:10],  # First 10 errors
            "timestamp": self.timestamp.isoformat(),
        }


class ComponentTestRunner:
    """Runner for component-level tests."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        config: Optional[dict] = None,
    ):
        """
        Initialize component test runner.

        Args:
            config_path: Path to configuration YAML file
            config: Configuration dictionary (alternative to config_path)
        """
        if config_path:
            self.config = load_config(config_path)
        elif config:
            self.config = config
        else:
            raise ValueError("Either config_path or config must be provided")

        self.config_name = self.config.get("name", "unknown")

        # Initialize collectors
        self.latency_collector = LatencyCollector(self.config_name)
        self.quality_collector = QualityCollector(self.config_name)
        self.resource_collector = ResourceCollector(self.config_name)

        # Clients (lazy initialized)
        self._stt_client: Optional[BaseSTTClient] = None
        self._translation_client: Optional[BaseTranslationClient] = None
        self._tts_client: Optional[BaseTTSClient] = None

    def _get_stt_client(self) -> BaseSTTClient:
        """Get or create STT client."""
        if self._stt_client is None:
            stt_config = self.config.get("stt", {})
            engine = stt_config.get("engine", "whispercpp")

            if engine == "whispercpp":
                self._stt_client = WhisperCppClient(
                    model=stt_config.get("model", "ggml-medium.bin"),
                    model_path=stt_config.get("model_path"),
                    threads=stt_config.get("threads"),
                    language=stt_config.get("language", "zh"),
                )
            else:
                raise ValueError(f"Unknown STT engine: {engine}")

        return self._stt_client

    def _get_translation_client(self) -> BaseTranslationClient:
        """Get or create translation client."""
        if self._translation_client is None:
            trans_config = self.config.get("translation", {})
            engine = trans_config.get("engine", "openai")

            if engine == "openai":
                self._translation_client = OpenAITranslationClient(
                    model=trans_config.get("model", "gpt-4o-mini"),
                    temperature=trans_config.get("temperature", 0.3),
                    max_tokens=trans_config.get("max_tokens", 512),
                    system_prompt=trans_config.get("system_prompt"),
                )
            elif engine == "lmstudio":
                self._translation_client = LMStudioClient(
                    endpoint=trans_config.get(
                        "endpoint", "http://localhost:1234/v1/chat/completions"
                    ),
                    model=trans_config.get("model", "qwen2.5-7b-instruct"),
                    temperature=trans_config.get("temperature", 0.3),
                    max_tokens=trans_config.get("max_tokens", 512),
                    system_prompt=trans_config.get("system_prompt"),
                )
            else:
                raise ValueError(f"Unknown translation engine: {engine}")

        return self._translation_client

    def _get_tts_client(self) -> BaseTTSClient:
        """Get or create TTS client."""
        if self._tts_client is None:
            tts_config = self.config.get("tts", {})
            engine = tts_config.get("engine", "openai")

            if engine == "openai":
                self._tts_client = OpenAITTSClient(
                    model=tts_config.get("model", "gpt-4o-mini-tts"),
                    voice=tts_config.get("voice", "nova"),
                    speed=tts_config.get("speed", 1.0),
                )
            elif engine == "vibevoice":
                self._tts_client = VibeVoiceClient(
                    endpoint=tts_config.get("endpoint", "ws://localhost:8765"),
                    model=tts_config.get("model", "vibevoice-realtime-0.5b"),
                    sample_rate=tts_config.get("sample_rate", 24000),
                )
            else:
                raise ValueError(f"Unknown TTS engine: {engine}")

        return self._tts_client

    def test_stt(
        self,
        audio_path: Path | str,
        ground_truth: Optional[str] = None,
    ) -> tuple[STTResult, Optional[QualityMetrics]]:
        """
        Test STT on a single audio file.

        Args:
            audio_path: Path to audio file
            ground_truth: Ground truth transcription for quality evaluation

        Returns:
            Tuple of (STTResult, QualityMetrics or None)
        """
        client = self._get_stt_client()
        stt_config = self.config.get("stt", {})

        # Get audio duration for context
        audio_duration_ms = get_audio_duration(audio_path) * 1000

        # Record start of measurement
        self.latency_collector.start_measurement()
        self.latency_collector.set_context(audio_duration_ms=audio_duration_ms)

        # Perform transcription with timing
        self.latency_collector.start_component("stt")
        result = client.transcribe(
            audio_path,
            language=stt_config.get("language", "zh"),
        )
        self.latency_collector.end_component("stt")
        self.latency_collector.finish_measurement()

        # Quality evaluation if ground truth provided
        quality = None
        if ground_truth:
            quality = self.quality_collector.evaluate_stt(
                hypothesis=result.text,
                reference=ground_truth,
            )

        return result, quality

    def test_stt_batch(
        self,
        audio_dir: Path | str,
        ground_truth_dir: Optional[Path | str] = None,
    ) -> ComponentTestResult:
        """
        Test STT on all audio files in a directory.

        Args:
            audio_dir: Directory containing audio files
            ground_truth_dir: Directory containing ground truth transcriptions
                              (expected format: {audio_name}.txt)

        Returns:
            ComponentTestResult with aggregated metrics
        """
        audio_dir = Path(audio_dir)
        ground_truth_dir = Path(ground_truth_dir) if ground_truth_dir else None

        audio_files = list(audio_dir.glob("*.wav"))
        results = []
        errors = []

        self.latency_collector.clear()
        self.quality_collector.clear()

        for audio_path in audio_files:
            try:
                # Look for ground truth
                ground_truth = None
                if ground_truth_dir:
                    gt_path = ground_truth_dir / f"{audio_path.stem}.txt"
                    if gt_path.exists():
                        ground_truth = gt_path.read_text().strip()

                result, quality = self.test_stt(audio_path, ground_truth)
                results.append({
                    "audio": str(audio_path),
                    "text": result.text,
                    "latency_ms": result.latency_ms,
                    "quality": quality.to_dict() if quality else None,
                })

            except Exception as e:
                errors.append({
                    "audio": str(audio_path),
                    "error": str(e),
                })

        return ComponentTestResult(
            component="stt",
            config_name=self.config_name,
            latency=self.latency_collector.get_stats("stt_ms"),
            quality=self.quality_collector.get_aggregated().to_dict(),
            raw_results=results,
            errors=errors,
        )

    def test_translation(
        self,
        text: str,
        reference: Optional[str] = None,
        context: Optional[list[str]] = None,
    ) -> tuple[TranslationResult, Optional[QualityMetrics]]:
        """
        Test translation on a single text.

        Args:
            text: Source text (Chinese) to translate
            reference: Reference translation for quality evaluation
            context: Previous translations for context

        Returns:
            Tuple of (TranslationResult, QualityMetrics or None)
        """
        client = self._get_translation_client()

        self.latency_collector.start_measurement()
        self.latency_collector.set_context(text_length=len(text))

        self.latency_collector.start_component("translation")
        result = client.translate(
            text=text,
            source_lang="zh",
            target_lang="en",
            context=context,
        )
        self.latency_collector.end_component("translation")
        self.latency_collector.finish_measurement()

        # Quality evaluation if reference provided
        quality = None
        if reference:
            quality = self.quality_collector.evaluate_translation(
                hypothesis=result.text,
                reference=reference,
                source=text,
            )

        return result, quality

    def test_translation_batch(
        self,
        texts: list[str],
        references: Optional[list[str]] = None,
    ) -> ComponentTestResult:
        """
        Test translation on multiple texts.

        Args:
            texts: List of source texts
            references: List of reference translations (parallel to texts)

        Returns:
            ComponentTestResult with aggregated metrics
        """
        results = []
        errors = []

        self.latency_collector.clear()
        self.quality_collector.clear()

        for i, text in enumerate(texts):
            try:
                reference = references[i] if references and i < len(references) else None

                result, quality = self.test_translation(text, reference)
                results.append({
                    "source": text,
                    "translation": result.text,
                    "latency_ms": result.latency_ms,
                    "tokens_per_sec": result.tokens_per_sec,
                    "quality": quality.to_dict() if quality else None,
                })

            except Exception as e:
                errors.append({
                    "source": text,
                    "error": str(e),
                })

        return ComponentTestResult(
            component="translation",
            config_name=self.config_name,
            latency=self.latency_collector.get_stats("translation_ms"),
            quality=self.quality_collector.get_aggregated().to_dict(),
            raw_results=results,
            errors=errors,
        )

    def test_tts(
        self,
        text: str,
        output_path: Optional[Path] = None,
    ) -> TTSResult:
        """
        Test TTS on a single text.

        Args:
            text: Text to synthesize
            output_path: Optional path to save audio output

        Returns:
            TTSResult with audio and timing
        """
        client = self._get_tts_client()

        self.latency_collector.start_measurement()
        self.latency_collector.set_context(text_length=len(text))

        self.latency_collector.start_component("tts_first")
        self.latency_collector.start_component("tts_complete")

        result = client.synthesize(text)

        # Record latencies from result
        self.latency_collector.record_tts_latencies(
            first_ms=result.first_audio_latency_ms,
            complete_ms=result.total_latency_ms,
        )
        self.latency_collector.finish_measurement()

        # Save audio if path provided
        if output_path and result.audio_data:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(result.audio_data)

        return result

    def test_tts_batch(
        self,
        texts: list[str],
        output_dir: Optional[Path] = None,
    ) -> ComponentTestResult:
        """
        Test TTS on multiple texts.

        Args:
            texts: List of texts to synthesize
            output_dir: Optional directory to save audio outputs

        Returns:
            ComponentTestResult with aggregated metrics
        """
        results = []
        errors = []

        self.latency_collector.clear()

        for i, text in enumerate(texts):
            try:
                output_path = None
                if output_dir:
                    output_path = Path(output_dir) / f"tts_{i:04d}.pcm"

                result = self.test_tts(text, output_path)
                results.append({
                    "text": text,
                    "first_latency_ms": result.first_audio_latency_ms,
                    "total_latency_ms": result.total_latency_ms,
                    "audio_size_bytes": len(result.audio_data),
                })

            except Exception as e:
                errors.append({
                    "text": text,
                    "error": str(e),
                })

        return ComponentTestResult(
            component="tts",
            config_name=self.config_name,
            latency=self.latency_collector.get_stats("tts_complete_ms"),
            raw_results=results,
            errors=errors,
        )

    def run_all_component_tests(
        self,
        audio_dir: Path,
        ground_truth_dir: Optional[Path] = None,
        translation_texts: Optional[list[str]] = None,
        translation_refs: Optional[list[str]] = None,
        tts_texts: Optional[list[str]] = None,
    ) -> dict[str, ComponentTestResult]:
        """
        Run all component tests.

        Args:
            audio_dir: Directory with test audio files
            ground_truth_dir: Directory with transcription ground truths
            translation_texts: Texts to test translation
            translation_refs: Reference translations
            tts_texts: Texts to test TTS

        Returns:
            Dictionary mapping component name to results
        """
        results = {}

        # STT tests
        print(f"Running STT tests with {self.config_name}...")
        results["stt"] = self.test_stt_batch(audio_dir, ground_truth_dir)

        # Translation tests
        if translation_texts:
            print(f"Running translation tests with {self.config_name}...")
            results["translation"] = self.test_translation_batch(
                translation_texts, translation_refs
            )

        # TTS tests
        if tts_texts:
            print(f"Running TTS tests with {self.config_name}...")
            results["tts"] = self.test_tts_batch(tts_texts)

        return results

    def save_results(
        self,
        results: dict[str, ComponentTestResult],
        output_dir: Path,
    ) -> Path:
        """
        Save test results to JSON file.

        Args:
            results: Dictionary of component test results
            output_dir: Directory to save results

        Returns:
            Path to saved results file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"component_tests_{self.config_name}_{timestamp}.json"
        output_path = output_dir / filename

        data = {
            "config_name": self.config_name,
            "timestamp": datetime.now().isoformat(),
            "results": {k: v.to_dict() for k, v in results.items()},
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path
