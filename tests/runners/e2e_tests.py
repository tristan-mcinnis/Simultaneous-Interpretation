"""
End-to-end pipeline test runner.

Tests complete interpretation pipelines with realistic audio input,
measuring full pipeline latency and quality.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tests.configs import load_config
from tests.metrics.latency import LatencyCollector, LatencyStats
from tests.metrics.quality import QualityCollector
from tests.metrics.resources import CostEstimator, ResourceCollector
from tests.utils.api_clients import (
    LMStudioClient,
    OpenAITranslationClient,
    OpenAITTSClient,
    VibeVoiceClient,
    WhisperCppClient,
)
from tests.utils.audio_utils import AudioFile, chunk_audio, get_audio_duration, load_audio


@dataclass
class PipelineResult:
    """Result from a single pipeline run."""

    audio_path: str
    audio_duration_ms: float

    # Outputs
    transcription: str
    translation: str
    tts_audio_size: int

    # Latencies (all in ms)
    stt_latency_ms: float
    translation_latency_ms: float
    tts_first_latency_ms: float
    tts_complete_latency_ms: float
    e2e_latency_ms: float

    # Quality (if references provided)
    stt_wer: Optional[float] = None
    translation_bleu: Optional[float] = None

    # Realtime factor (< 1.0 is faster than realtime)
    realtime_factor: Optional[float] = None

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "audio_path": self.audio_path,
            "audio_duration_ms": self.audio_duration_ms,
            "transcription": self.transcription,
            "translation": self.translation,
            "tts_audio_size": self.tts_audio_size,
            "stt_latency_ms": self.stt_latency_ms,
            "translation_latency_ms": self.translation_latency_ms,
            "tts_first_latency_ms": self.tts_first_latency_ms,
            "tts_complete_latency_ms": self.tts_complete_latency_ms,
            "e2e_latency_ms": self.e2e_latency_ms,
            "stt_wer": self.stt_wer,
            "translation_bleu": self.translation_bleu,
            "realtime_factor": self.realtime_factor,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BatchResult:
    """Aggregated results from batch testing."""

    config_name: str
    audio_count: int
    total_audio_duration_ms: float

    # Latency statistics
    stt_latency: LatencyStats
    translation_latency: LatencyStats
    tts_latency: LatencyStats
    e2e_latency: LatencyStats

    # Quality aggregates
    stt_wer_mean: Optional[float] = None
    translation_bleu_mean: Optional[float] = None

    # Resource usage
    cpu_mean: Optional[float] = None
    memory_max_mb: Optional[float] = None

    # Cost estimates
    estimated_cost_usd: Optional[float] = None
    estimated_hourly_cost_usd: Optional[float] = None

    # Individual results
    results: list[PipelineResult] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "config_name": self.config_name,
            "audio_count": self.audio_count,
            "total_audio_duration_ms": self.total_audio_duration_ms,
            "stt_latency": self.stt_latency.to_dict(),
            "translation_latency": self.translation_latency.to_dict(),
            "tts_latency": self.tts_latency.to_dict(),
            "e2e_latency": self.e2e_latency.to_dict(),
            "stt_wer_mean": self.stt_wer_mean,
            "translation_bleu_mean": self.translation_bleu_mean,
            "cpu_mean": self.cpu_mean,
            "memory_max_mb": self.memory_max_mb,
            "estimated_cost_usd": self.estimated_cost_usd,
            "estimated_hourly_cost_usd": self.estimated_hourly_cost_usd,
            "error_count": len(self.errors),
            "timestamp": self.timestamp.isoformat(),
        }


class E2ETestRunner:
    """Runner for end-to-end pipeline tests."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        config: Optional[dict] = None,
    ):
        """
        Initialize E2E test runner.

        Args:
            config_path: Path to configuration YAML file
            config: Configuration dictionary
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
        self.cost_estimator = CostEstimator()

        # Initialize clients
        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize all pipeline clients based on config."""
        # STT
        stt_config = self.config.get("stt", {})
        self.stt_client = WhisperCppClient(
            model=stt_config.get("model", "ggml-medium.bin"),
            model_path=stt_config.get("model_path"),
            threads=stt_config.get("threads"),
            language=stt_config.get("language", "zh"),
        )

        # Translation
        trans_config = self.config.get("translation", {})
        engine = trans_config.get("engine", "openai")

        if engine == "openai":
            self.translation_client = OpenAITranslationClient(
                model=trans_config.get("model", "gpt-4o-mini"),
                temperature=trans_config.get("temperature", 0.3),
                max_tokens=trans_config.get("max_tokens", 512),
            )
        else:
            self.translation_client = LMStudioClient(
                endpoint=trans_config.get("endpoint", "http://localhost:1234/v1/chat/completions"),
                model=trans_config.get("model", "qwen2.5-7b-instruct"),
                temperature=trans_config.get("temperature", 0.3),
                max_tokens=trans_config.get("max_tokens", 512),
            )

        # TTS
        tts_config = self.config.get("tts", {})
        engine = tts_config.get("engine", "openai")

        if engine == "openai":
            self.tts_client = OpenAITTSClient(
                model=tts_config.get("model", "gpt-4o-mini-tts"),
                voice=tts_config.get("voice", "nova"),
                speed=tts_config.get("speed", 1.0),
            )
        else:
            self.tts_client = VibeVoiceClient(
                endpoint=tts_config.get("endpoint", "ws://localhost:8765"),
                model=tts_config.get("model", "vibevoice-realtime-0.5b"),
                sample_rate=tts_config.get("sample_rate", 24000),
            )

    def run_pipeline(
        self,
        audio_path: Path | str,
        ground_truth_transcript: Optional[str] = None,
        reference_translation: Optional[str] = None,
        context: Optional[list[str]] = None,
    ) -> PipelineResult:
        """
        Run full pipeline on a single audio file.

        Args:
            audio_path: Path to input audio
            ground_truth_transcript: Expected transcription for WER
            reference_translation: Expected translation for BLEU
            context: Previous translations for context

        Returns:
            PipelineResult with all metrics
        """
        audio_path = Path(audio_path)
        audio_duration_ms = get_audio_duration(audio_path) * 1000

        self.latency_collector.start_measurement()
        self.latency_collector.set_context(audio_duration_ms=audio_duration_ms)
        self.latency_collector.start_component("e2e")

        # Step 1: STT
        self.latency_collector.start_component("stt")
        stt_result = self.stt_client.transcribe(
            audio_path,
            language=self.config.get("stt", {}).get("language", "zh"),
        )
        stt_latency = self.latency_collector.end_component("stt")

        transcription = stt_result.text

        # Step 2: Translation
        self.latency_collector.start_component("translation")
        trans_result = self.translation_client.translate(
            text=transcription,
            source_lang="zh",
            target_lang="en",
            context=context,
        )
        translation_latency = self.latency_collector.end_component("translation")

        translation = trans_result.text

        # Record cost for cloud translation
        trans_config = self.config.get("translation", {})
        if trans_config.get("engine") == "openai" and trans_result.tokens_used:
            self.cost_estimator.record_translation(
                model=trans_config.get("model", "gpt-4o-mini"),
                input_tokens=trans_result.tokens_used // 2,  # Rough split
                output_tokens=trans_result.tokens_used // 2,
            )

        # Step 3: TTS
        tts_result = self.tts_client.synthesize(translation)
        tts_first_latency = tts_result.first_audio_latency_ms
        tts_complete_latency = tts_result.total_latency_ms

        self.latency_collector.record_tts_latencies(
            first_ms=tts_first_latency,
            complete_ms=tts_complete_latency,
        )

        # Record cost for cloud TTS
        tts_config = self.config.get("tts", {})
        if tts_config.get("engine") == "openai":
            self.cost_estimator.record_tts(
                model=tts_config.get("model", "gpt-4o-mini-tts"),
                text=translation,
            )

        # End E2E timing
        e2e_latency = self.latency_collector.end_component("e2e")
        self.latency_collector.finish_measurement()

        # Quality evaluation
        stt_wer = None
        if ground_truth_transcript:
            quality = self.quality_collector.evaluate_stt(
                hypothesis=transcription,
                reference=ground_truth_transcript,
            )
            stt_wer = quality.wer

        translation_bleu = None
        if reference_translation:
            quality = self.quality_collector.evaluate_translation(
                hypothesis=translation,
                reference=reference_translation,
                source=transcription,
            )
            translation_bleu = quality.bleu

        # Calculate realtime factor
        realtime_factor = e2e_latency / audio_duration_ms if audio_duration_ms > 0 else None

        return PipelineResult(
            audio_path=str(audio_path),
            audio_duration_ms=audio_duration_ms,
            transcription=transcription,
            translation=translation,
            tts_audio_size=len(tts_result.audio_data),
            stt_latency_ms=stt_latency,
            translation_latency_ms=translation_latency,
            tts_first_latency_ms=tts_first_latency,
            tts_complete_latency_ms=tts_complete_latency,
            e2e_latency_ms=e2e_latency,
            stt_wer=stt_wer,
            translation_bleu=translation_bleu,
            realtime_factor=realtime_factor,
        )

    def run_batch(
        self,
        audio_dir: Path | str,
        ground_truth_dir: Optional[Path | str] = None,
        reference_dir: Optional[Path | str] = None,
        monitor_resources: bool = True,
    ) -> BatchResult:
        """
        Run pipeline on all audio files in a directory.

        Args:
            audio_dir: Directory containing audio files
            ground_truth_dir: Directory with transcription ground truths
            reference_dir: Directory with reference translations
            monitor_resources: Whether to monitor system resources

        Returns:
            BatchResult with aggregated metrics
        """
        audio_dir = Path(audio_dir)
        ground_truth_dir = Path(ground_truth_dir) if ground_truth_dir else None
        reference_dir = Path(reference_dir) if reference_dir else None

        audio_files = sorted(audio_dir.glob("*.wav"))
        results = []
        errors = []

        # Reset collectors
        self.latency_collector.clear()
        self.quality_collector.clear()

        # Start resource monitoring
        if monitor_resources:
            self.resource_collector.start_monitoring()

        total_audio_duration_ms = 0
        translation_context = []

        for audio_path in audio_files:
            try:
                # Load ground truth if available
                ground_truth = None
                if ground_truth_dir:
                    gt_path = ground_truth_dir / f"{audio_path.stem}.txt"
                    if gt_path.exists():
                        ground_truth = gt_path.read_text().strip()

                # Load reference translation if available
                reference = None
                if reference_dir:
                    ref_path = reference_dir / f"{audio_path.stem}.txt"
                    if ref_path.exists():
                        reference = ref_path.read_text().strip()

                # Run pipeline
                result = self.run_pipeline(
                    audio_path=audio_path,
                    ground_truth_transcript=ground_truth,
                    reference_translation=reference,
                    context=translation_context[-10:],  # Last 10 for context
                )

                results.append(result)
                total_audio_duration_ms += result.audio_duration_ms

                # Update context
                translation_context.append(result.translation)

            except Exception as e:
                errors.append({
                    "audio_path": str(audio_path),
                    "error": str(e),
                })

        # Stop resource monitoring
        duration_sec = 0
        if monitor_resources:
            duration_sec = self.resource_collector.stop_monitoring()

        # Aggregate quality metrics
        quality_agg = self.quality_collector.get_aggregated()

        # Aggregate resource metrics
        resource_agg = self.resource_collector.get_aggregated()

        return BatchResult(
            config_name=self.config_name,
            audio_count=len(results),
            total_audio_duration_ms=total_audio_duration_ms,
            stt_latency=self.latency_collector.get_stats("stt_ms"),
            translation_latency=self.latency_collector.get_stats("translation_ms"),
            tts_latency=self.latency_collector.get_stats("tts_complete_ms"),
            e2e_latency=self.latency_collector.get_stats("e2e_ms"),
            stt_wer_mean=quality_agg.wer_mean,
            translation_bleu_mean=quality_agg.bleu_mean,
            cpu_mean=resource_agg.cpu_mean,
            memory_max_mb=resource_agg.memory_max_mb,
            estimated_cost_usd=self.cost_estimator.get_total_cost(),
            estimated_hourly_cost_usd=self.cost_estimator.estimate_hourly_cost(duration_sec)
            if duration_sec > 0
            else None,
            results=results,
            errors=errors,
        )

    def run_chunked_pipeline(
        self,
        audio_path: Path | str,
        chunk_duration_sec: float = 3.0,
    ) -> list[PipelineResult]:
        """
        Run pipeline on audio in chunks (simulating real-time).

        Args:
            audio_path: Path to audio file
            chunk_duration_sec: Duration of each chunk

        Returns:
            List of PipelineResult for each chunk
        """
        audio = load_audio(audio_path)
        chunks = list(chunk_audio(audio, chunk_duration_sec))

        results = []
        context = []

        for chunk in chunks:
            # Save chunk to temp file for processing
            import tempfile
            from scipy.io import wavfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = Path(f.name)
                wavfile.write(temp_path, chunk.sample_rate, chunk.data)

            try:
                result = self.run_pipeline(
                    audio_path=temp_path,
                    context=context[-10:],
                )
                results.append(result)
                context.append(result.translation)

            finally:
                temp_path.unlink()

        return results

    def save_results(
        self,
        results: BatchResult,
        output_dir: Path,
    ) -> Path:
        """
        Save batch results to JSON file.

        Args:
            results: BatchResult to save
            output_dir: Directory to save results

        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"e2e_tests_{self.config_name}_{timestamp}.json"
        output_path = output_dir / filename

        data = {
            "summary": results.to_dict(),
            "individual_results": [r.to_dict() for r in results.results],
            "errors": results.errors,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path


def compare_architectures(
    configs: list[Path],
    audio_dir: Path,
    ground_truth_dir: Optional[Path] = None,
    reference_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, BatchResult]:
    """
    Compare multiple architecture configurations.

    Args:
        configs: List of config file paths
        audio_dir: Directory with test audio
        ground_truth_dir: Directory with transcription ground truths
        reference_dir: Directory with reference translations
        output_dir: Directory to save results

    Returns:
        Dictionary mapping config name to BatchResult
    """
    results = {}

    for config_path in configs:
        print(f"\nTesting configuration: {config_path.stem}")
        runner = E2ETestRunner(config_path=config_path)

        batch_result = runner.run_batch(
            audio_dir=audio_dir,
            ground_truth_dir=ground_truth_dir,
            reference_dir=reference_dir,
        )

        results[runner.config_name] = batch_result

        if output_dir:
            runner.save_results(batch_result, output_dir)

        # Print summary
        print(f"  Audio files: {batch_result.audio_count}")
        print(f"  E2E latency (mean): {batch_result.e2e_latency.mean:.1f}ms")
        print(f"  E2E latency (p95): {batch_result.e2e_latency.p95:.1f}ms")
        if batch_result.stt_wer_mean is not None:
            print(f"  STT WER: {batch_result.stt_wer_mean:.2%}")
        if batch_result.translation_bleu_mean is not None:
            print(f"  Translation BLEU: {batch_result.translation_bleu_mean:.1f}")
        if batch_result.estimated_hourly_cost_usd is not None:
            print(f"  Est. hourly cost: ${batch_result.estimated_hourly_cost_usd:.2f}")

    return results
