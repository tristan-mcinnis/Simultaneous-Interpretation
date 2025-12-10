"""Latency metrics collection and analysis."""

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Callable, Optional


@dataclass
class LatencyMetrics:
    """Container for latency measurements from a single test."""

    stt_ms: float = 0.0
    translation_ms: float = 0.0
    tts_first_ms: float = 0.0
    tts_complete_ms: float = 0.0
    e2e_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    # Optional additional context
    audio_duration_ms: Optional[float] = None
    text_length: Optional[int] = None

    @property
    def realtime_factor(self) -> Optional[float]:
        """
        Calculate realtime factor (processing time / audio duration).

        < 1.0 means faster than realtime
        = 1.0 means realtime
        > 1.0 means slower than realtime
        """
        if self.audio_duration_ms and self.audio_duration_ms > 0:
            return self.e2e_ms / self.audio_duration_ms
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "stt_ms": self.stt_ms,
            "translation_ms": self.translation_ms,
            "tts_first_ms": self.tts_first_ms,
            "tts_complete_ms": self.tts_complete_ms,
            "e2e_ms": self.e2e_ms,
            "timestamp": self.timestamp.isoformat(),
            "audio_duration_ms": self.audio_duration_ms,
            "text_length": self.text_length,
            "realtime_factor": self.realtime_factor,
        }


@dataclass
class LatencyStats:
    """Aggregated latency statistics from multiple measurements."""

    count: int
    mean: float
    std: float
    min: float
    max: float
    p50: float  # Median
    p90: float
    p95: float
    p99: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
        }


class LatencyCollector:
    """Collects and aggregates latency measurements."""

    def __init__(self, name: str = "default"):
        """
        Initialize latency collector.

        Args:
            name: Name/identifier for this collector
        """
        self.name = name
        self.measurements: list[LatencyMetrics] = []
        self._current_metrics: Optional[LatencyMetrics] = None
        self._start_times: dict[str, float] = {}

    def start_measurement(self) -> None:
        """Start a new measurement cycle."""
        self._current_metrics = LatencyMetrics()
        self._start_times = {}

    def start_component(self, component: str) -> None:
        """
        Mark the start of a component's processing.

        Args:
            component: Component name (stt, translation, tts)
        """
        self._start_times[component] = time.perf_counter()

    def end_component(self, component: str) -> float:
        """
        Mark the end of a component's processing.

        Args:
            component: Component name

        Returns:
            Latency in milliseconds
        """
        if component not in self._start_times:
            raise ValueError(f"Component {component} was not started")

        latency_ms = (time.perf_counter() - self._start_times[component]) * 1000

        if self._current_metrics:
            if component == "stt":
                self._current_metrics.stt_ms = latency_ms
            elif component == "translation":
                self._current_metrics.translation_ms = latency_ms
            elif component == "tts_first":
                self._current_metrics.tts_first_ms = latency_ms
            elif component == "tts_complete":
                self._current_metrics.tts_complete_ms = latency_ms
            elif component == "e2e":
                self._current_metrics.e2e_ms = latency_ms

        return latency_ms

    def record_tts_latencies(
        self,
        first_ms: float,
        complete_ms: float,
    ) -> None:
        """Record TTS latencies directly (for streaming)."""
        if self._current_metrics:
            self._current_metrics.tts_first_ms = first_ms
            self._current_metrics.tts_complete_ms = complete_ms

    def set_context(
        self,
        audio_duration_ms: Optional[float] = None,
        text_length: Optional[int] = None,
    ) -> None:
        """Set additional context for current measurement."""
        if self._current_metrics:
            if audio_duration_ms is not None:
                self._current_metrics.audio_duration_ms = audio_duration_ms
            if text_length is not None:
                self._current_metrics.text_length = text_length

    def finish_measurement(self) -> LatencyMetrics:
        """
        Finish current measurement and record it.

        Returns:
            The completed LatencyMetrics
        """
        if self._current_metrics is None:
            raise ValueError("No measurement in progress")

        # Calculate E2E if not set
        if self._current_metrics.e2e_ms == 0.0:
            self._current_metrics.e2e_ms = (
                self._current_metrics.stt_ms
                + self._current_metrics.translation_ms
                + self._current_metrics.tts_complete_ms
            )

        self.measurements.append(self._current_metrics)
        result = self._current_metrics
        self._current_metrics = None
        return result

    @contextmanager
    def measure(self, component: str):
        """
        Context manager for measuring component latency.

        Usage:
            with collector.measure("stt"):
                result = transcribe(audio)
        """
        self.start_component(component)
        try:
            yield
        finally:
            self.end_component(component)

    def timing(self, component: str) -> Callable:
        """
        Decorator for measuring function latency.

        Usage:
            @collector.timing("stt")
            def transcribe(audio):
                ...
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.start_component(component)
                try:
                    return func(*args, **kwargs)
                finally:
                    self.end_component(component)

            return wrapper

        return decorator

    def get_stats(self, metric: str = "e2e_ms") -> LatencyStats:
        """
        Calculate statistics for a specific metric.

        Args:
            metric: Metric name (stt_ms, translation_ms, tts_first_ms,
                    tts_complete_ms, e2e_ms)

        Returns:
            LatencyStats with aggregated statistics
        """
        values = [getattr(m, metric) for m in self.measurements]

        if not values:
            return LatencyStats(
                count=0,
                mean=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                p50=0.0,
                p90=0.0,
                p95=0.0,
                p99=0.0,
            )

        sorted_values = sorted(values)
        n = len(sorted_values)

        return LatencyStats(
            count=n,
            mean=statistics.mean(values),
            std=statistics.stdev(values) if n > 1 else 0.0,
            min=min(values),
            max=max(values),
            p50=sorted_values[int(n * 0.50)],
            p90=sorted_values[int(n * 0.90)] if n >= 10 else sorted_values[-1],
            p95=sorted_values[int(n * 0.95)] if n >= 20 else sorted_values[-1],
            p99=sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
        )

    def get_all_stats(self) -> dict[str, LatencyStats]:
        """Get statistics for all metrics."""
        metrics = [
            "stt_ms",
            "translation_ms",
            "tts_first_ms",
            "tts_complete_ms",
            "e2e_ms",
        ]
        return {metric: self.get_stats(metric) for metric in metrics}

    def clear(self) -> None:
        """Clear all measurements."""
        self.measurements.clear()
        self._current_metrics = None
        self._start_times.clear()

    def to_dict(self) -> dict:
        """Export all data as dictionary."""
        return {
            "name": self.name,
            "count": len(self.measurements),
            "measurements": [m.to_dict() for m in self.measurements],
            "stats": {k: v.to_dict() for k, v in self.get_all_stats().items()},
        }


def measure_function(func: Callable) -> tuple[any, float]:
    """
    Measure the execution time of a function call.

    Args:
        func: Function to call (no arguments)

    Returns:
        Tuple of (result, latency_ms)
    """
    start = time.perf_counter()
    result = func()
    latency_ms = (time.perf_counter() - start) * 1000
    return result, latency_ms
