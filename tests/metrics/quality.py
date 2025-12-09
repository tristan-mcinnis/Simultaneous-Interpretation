"""Quality metrics collection for STT and translation evaluation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from tests.utils.evaluation import (
    QualityEvaluator,
    calculate_bleu,
    calculate_cer,
    calculate_ter,
    calculate_wer,
)


@dataclass
class QualityMetrics:
    """Container for quality measurements from a single test."""

    # STT metrics
    wer: Optional[float] = None  # Word Error Rate (0-1, lower better)
    cer: Optional[float] = None  # Character Error Rate (0-1, lower better)

    # Translation metrics
    bleu: Optional[float] = None  # BLEU score (0-100, higher better)
    ter: Optional[float] = None  # Translation Edit Rate (0-1, lower better)
    comet: Optional[float] = None  # COMET score (0-1, higher better)

    # Human evaluation (if available)
    human_score: Optional[float] = None  # 1-5 scale
    naturalness: Optional[float] = None  # 1-5 scale for TTS

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    source_text: Optional[str] = None
    hypothesis: Optional[str] = None
    reference: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "wer": self.wer,
            "cer": self.cer,
            "bleu": self.bleu,
            "ter": self.ter,
            "comet": self.comet,
            "human_score": self.human_score,
            "naturalness": self.naturalness,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AggregatedQualityMetrics:
    """Aggregated quality metrics from multiple measurements."""

    count: int

    # STT aggregates
    wer_mean: Optional[float] = None
    wer_std: Optional[float] = None
    cer_mean: Optional[float] = None
    cer_std: Optional[float] = None

    # Translation aggregates
    bleu_mean: Optional[float] = None
    bleu_std: Optional[float] = None
    ter_mean: Optional[float] = None
    ter_std: Optional[float] = None
    comet_mean: Optional[float] = None
    comet_std: Optional[float] = None

    # Human evaluation aggregates
    human_score_mean: Optional[float] = None
    naturalness_mean: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "count": self.count,
            "wer_mean": self.wer_mean,
            "wer_std": self.wer_std,
            "cer_mean": self.cer_mean,
            "cer_std": self.cer_std,
            "bleu_mean": self.bleu_mean,
            "bleu_std": self.bleu_std,
            "ter_mean": self.ter_mean,
            "ter_std": self.ter_std,
            "comet_mean": self.comet_mean,
            "comet_std": self.comet_std,
            "human_score_mean": self.human_score_mean,
            "naturalness_mean": self.naturalness_mean,
        }


class QualityCollector:
    """Collects and aggregates quality metrics."""

    def __init__(
        self,
        name: str = "default",
        comet_model: Optional[str] = None,
    ):
        """
        Initialize quality collector.

        Args:
            name: Name/identifier for this collector
            comet_model: COMET model to use (None to skip COMET)
        """
        self.name = name
        self.measurements: list[QualityMetrics] = []
        self.evaluator = QualityEvaluator(comet_model=comet_model)

    def evaluate_stt(
        self,
        hypothesis: str,
        reference: str,
        record: bool = True,
    ) -> QualityMetrics:
        """
        Evaluate STT output quality.

        Args:
            hypothesis: Transcribed text
            reference: Ground truth text
            record: Whether to record this measurement

        Returns:
            QualityMetrics with WER and CER
        """
        wer = calculate_wer(hypothesis, reference)
        cer = calculate_cer(hypothesis, reference)

        metrics = QualityMetrics(
            wer=wer,
            cer=cer,
            hypothesis=hypothesis,
            reference=reference,
        )

        if record:
            self.measurements.append(metrics)

        return metrics

    def evaluate_translation(
        self,
        hypothesis: str,
        reference: str,
        source: Optional[str] = None,
        record: bool = True,
    ) -> QualityMetrics:
        """
        Evaluate translation quality.

        Args:
            hypothesis: Translated text
            reference: Reference translation
            source: Original source text (for COMET)
            record: Whether to record this measurement

        Returns:
            QualityMetrics with BLEU, TER, and optionally COMET
        """
        bleu = calculate_bleu(hypothesis, reference)
        ter = calculate_ter(hypothesis, reference)

        metrics = QualityMetrics(
            bleu=bleu,
            ter=ter,
            source_text=source,
            hypothesis=hypothesis,
            reference=reference,
        )

        # Try COMET if evaluator has it configured
        if self.evaluator.comet_model and source:
            try:
                from tests.utils.evaluation import calculate_comet

                metrics.comet = calculate_comet(
                    hypothesis, reference, source, self.evaluator.comet_model
                )
            except Exception:
                pass  # Skip COMET on error

        if record:
            self.measurements.append(metrics)

        return metrics

    def record_human_evaluation(
        self,
        score: float,
        naturalness: Optional[float] = None,
        hypothesis: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> QualityMetrics:
        """
        Record human evaluation scores.

        Args:
            score: Overall quality score (1-5)
            naturalness: TTS naturalness score (1-5)
            hypothesis: The output being evaluated
            reference: Reference (if applicable)

        Returns:
            QualityMetrics with human scores
        """
        metrics = QualityMetrics(
            human_score=score,
            naturalness=naturalness,
            hypothesis=hypothesis,
            reference=reference,
        )

        self.measurements.append(metrics)
        return metrics

    def get_aggregated(self) -> AggregatedQualityMetrics:
        """Get aggregated statistics for all measurements."""
        import statistics

        n = len(self.measurements)

        if n == 0:
            return AggregatedQualityMetrics(count=0)

        def safe_mean(values: list) -> Optional[float]:
            filtered = [v for v in values if v is not None]
            return statistics.mean(filtered) if filtered else None

        def safe_std(values: list) -> Optional[float]:
            filtered = [v for v in values if v is not None]
            return statistics.stdev(filtered) if len(filtered) > 1 else None

        wer_values = [m.wer for m in self.measurements]
        cer_values = [m.cer for m in self.measurements]
        bleu_values = [m.bleu for m in self.measurements]
        ter_values = [m.ter for m in self.measurements]
        comet_values = [m.comet for m in self.measurements]
        human_values = [m.human_score for m in self.measurements]
        nat_values = [m.naturalness for m in self.measurements]

        return AggregatedQualityMetrics(
            count=n,
            wer_mean=safe_mean(wer_values),
            wer_std=safe_std(wer_values),
            cer_mean=safe_mean(cer_values),
            cer_std=safe_std(cer_values),
            bleu_mean=safe_mean(bleu_values),
            bleu_std=safe_std(bleu_values),
            ter_mean=safe_mean(ter_values),
            ter_std=safe_std(ter_values),
            comet_mean=safe_mean(comet_values),
            comet_std=safe_std(comet_values),
            human_score_mean=safe_mean(human_values),
            naturalness_mean=safe_mean(nat_values),
        )

    def clear(self) -> None:
        """Clear all measurements."""
        self.measurements.clear()

    def to_dict(self) -> dict:
        """Export all data as dictionary."""
        return {
            "name": self.name,
            "count": len(self.measurements),
            "measurements": [m.to_dict() for m in self.measurements],
            "aggregated": self.get_aggregated().to_dict(),
        }


def quick_evaluate_stt(hypothesis: str, reference: str) -> dict:
    """Quick STT evaluation without collector."""
    return {
        "wer": calculate_wer(hypothesis, reference),
        "cer": calculate_cer(hypothesis, reference),
    }


def quick_evaluate_translation(
    hypothesis: str,
    reference: str,
    source: Optional[str] = None,
) -> dict:
    """Quick translation evaluation without collector."""
    result = {
        "bleu": calculate_bleu(hypothesis, reference),
        "ter": calculate_ter(hypothesis, reference),
    }

    if source:
        try:
            from tests.utils.evaluation import calculate_comet

            result["comet"] = calculate_comet(hypothesis, reference, source)
        except Exception:
            pass

    return result
