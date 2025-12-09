"""
Stress test runner for sustained operation testing.

Tests reliability and performance over extended periods,
monitoring for memory leaks, thermal throttling, and degradation.
"""

import json
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from tests.configs import load_config
from tests.metrics.latency import LatencyCollector, LatencyStats
from tests.metrics.resources import ResourceCollector
from tests.runners.e2e_tests import E2ETestRunner, PipelineResult


@dataclass
class StressTestResult:
    """Result from a stress test run."""

    config_name: str
    duration_minutes: float
    actual_duration_sec: float

    # Test counts
    total_iterations: int
    successful_iterations: int
    failed_iterations: int
    failure_rate: float

    # Latency statistics
    e2e_latency: LatencyStats
    stt_latency: LatencyStats
    translation_latency: LatencyStats
    tts_latency: LatencyStats

    # Latency trend (to detect degradation)
    latency_trend_ms_per_min: float  # Positive = getting slower

    # Resource statistics
    cpu_mean: float
    cpu_max: float
    memory_mean_mb: float
    memory_max_mb: float
    memory_trend_mb_per_min: float  # Positive = memory leak

    # Stability metrics
    max_consecutive_failures: int
    recovery_time_ms: Optional[float]  # Time to recover from failure
    dropout_count: int  # Number of significant latency spikes

    # Thermal (if detectable)
    thermal_throttle_events: int

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "config_name": self.config_name,
            "duration_minutes": self.duration_minutes,
            "actual_duration_sec": self.actual_duration_sec,
            "total_iterations": self.total_iterations,
            "successful_iterations": self.successful_iterations,
            "failed_iterations": self.failed_iterations,
            "failure_rate": self.failure_rate,
            "e2e_latency": self.e2e_latency.to_dict(),
            "stt_latency": self.stt_latency.to_dict(),
            "translation_latency": self.translation_latency.to_dict(),
            "tts_latency": self.tts_latency.to_dict(),
            "latency_trend_ms_per_min": self.latency_trend_ms_per_min,
            "cpu_mean": self.cpu_mean,
            "cpu_max": self.cpu_max,
            "memory_mean_mb": self.memory_mean_mb,
            "memory_max_mb": self.memory_max_mb,
            "memory_trend_mb_per_min": self.memory_trend_mb_per_min,
            "max_consecutive_failures": self.max_consecutive_failures,
            "recovery_time_ms": self.recovery_time_ms,
            "dropout_count": self.dropout_count,
            "thermal_throttle_events": self.thermal_throttle_events,
            "timestamp": self.timestamp.isoformat(),
        }

    def is_stable(self) -> bool:
        """Check if the system passed stability criteria."""
        return (
            self.failure_rate < 0.05  # <5% failure rate
            and self.memory_trend_mb_per_min < 10  # <10MB/min leak
            and self.latency_trend_ms_per_min < 100  # <100ms/min degradation
            and self.max_consecutive_failures < 5  # No extended failures
        )


@dataclass
class IterationResult:
    """Result from a single iteration in stress test."""

    iteration: int
    timestamp: datetime
    success: bool
    error: Optional[str]
    e2e_latency_ms: Optional[float]
    stt_latency_ms: Optional[float]
    translation_latency_ms: Optional[float]
    tts_latency_ms: Optional[float]


class StressTestRunner:
    """Runner for stress/endurance tests."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        config: Optional[dict] = None,
    ):
        """
        Initialize stress test runner.

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

        # Initialize E2E runner for actual processing
        self.e2e_runner = E2ETestRunner(config=self.config)

        # Collectors
        self.latency_collector = LatencyCollector(self.config_name)
        self.resource_collector = ResourceCollector(
            self.config_name,
            interval_sec=5.0,  # Sample every 5 seconds
        )

        # State
        self.iterations: list[IterationResult] = []
        self._running = False

    def run_sustained(
        self,
        duration_minutes: float,
        audio_dir: Path,
        iteration_pause_sec: float = 1.0,
        verbose: bool = True,
    ) -> StressTestResult:
        """
        Run sustained stress test.

        Args:
            duration_minutes: How long to run the test
            audio_dir: Directory with test audio files
            iteration_pause_sec: Pause between iterations
            verbose: Print progress updates

        Returns:
            StressTestResult with all metrics
        """
        audio_dir = Path(audio_dir)
        audio_files = list(audio_dir.glob("*.wav"))

        if not audio_files:
            raise ValueError(f"No WAV files found in {audio_dir}")

        duration_sec = duration_minutes * 60
        start_time = time.time()

        self._running = True
        self.iterations.clear()
        self.latency_collector.clear()
        self.resource_collector.start_monitoring()

        consecutive_failures = 0
        max_consecutive_failures = 0
        context = []

        if verbose:
            print(f"Starting {duration_minutes} minute stress test...")
            print(f"Using {len(audio_files)} audio files")

        iteration = 0
        while self._running and (time.time() - start_time) < duration_sec:
            iteration += 1
            audio_path = random.choice(audio_files)

            try:
                result = self.e2e_runner.run_pipeline(
                    audio_path=audio_path,
                    context=context[-10:],
                )

                self.iterations.append(
                    IterationResult(
                        iteration=iteration,
                        timestamp=datetime.now(),
                        success=True,
                        error=None,
                        e2e_latency_ms=result.e2e_latency_ms,
                        stt_latency_ms=result.stt_latency_ms,
                        translation_latency_ms=result.translation_latency_ms,
                        tts_latency_ms=result.tts_complete_latency_ms,
                    )
                )

                # Record for latency stats
                self.latency_collector.start_measurement()
                self.latency_collector._current_metrics.stt_ms = result.stt_latency_ms
                self.latency_collector._current_metrics.translation_ms = result.translation_latency_ms
                self.latency_collector._current_metrics.tts_complete_ms = result.tts_complete_latency_ms
                self.latency_collector._current_metrics.e2e_ms = result.e2e_latency_ms
                self.latency_collector.finish_measurement()

                context.append(result.translation)
                consecutive_failures = 0

                if verbose and iteration % 10 == 0:
                    elapsed = (time.time() - start_time) / 60
                    print(
                        f"  Iteration {iteration}: "
                        f"E2E={result.e2e_latency_ms:.0f}ms, "
                        f"elapsed={elapsed:.1f}min"
                    )

            except Exception as e:
                consecutive_failures += 1
                max_consecutive_failures = max(
                    max_consecutive_failures, consecutive_failures
                )

                self.iterations.append(
                    IterationResult(
                        iteration=iteration,
                        timestamp=datetime.now(),
                        success=False,
                        error=str(e),
                        e2e_latency_ms=None,
                        stt_latency_ms=None,
                        translation_latency_ms=None,
                        tts_latency_ms=None,
                    )
                )

                if verbose:
                    print(f"  Iteration {iteration}: FAILED - {e}")

            time.sleep(iteration_pause_sec)

        actual_duration = time.time() - start_time
        self.resource_collector.stop_monitoring()

        if verbose:
            print(f"Stress test complete. {iteration} iterations in {actual_duration/60:.1f} minutes")

        return self._analyze_results(
            duration_minutes=duration_minutes,
            actual_duration_sec=actual_duration,
            max_consecutive_failures=max_consecutive_failures,
        )

    def _analyze_results(
        self,
        duration_minutes: float,
        actual_duration_sec: float,
        max_consecutive_failures: int,
    ) -> StressTestResult:
        """Analyze stress test results."""
        successful = [i for i in self.iterations if i.success]
        failed = [i for i in self.iterations if not i.success]

        # Calculate latency trend
        latency_trend = self._calculate_latency_trend()

        # Calculate memory trend
        memory_trend = self._calculate_memory_trend()

        # Count dropout events (latency > 2x mean)
        dropout_count = self._count_dropouts()

        # Get resource stats
        resource_agg = self.resource_collector.get_aggregated()

        return StressTestResult(
            config_name=self.config_name,
            duration_minutes=duration_minutes,
            actual_duration_sec=actual_duration_sec,
            total_iterations=len(self.iterations),
            successful_iterations=len(successful),
            failed_iterations=len(failed),
            failure_rate=len(failed) / len(self.iterations) if self.iterations else 0,
            e2e_latency=self.latency_collector.get_stats("e2e_ms"),
            stt_latency=self.latency_collector.get_stats("stt_ms"),
            translation_latency=self.latency_collector.get_stats("translation_ms"),
            tts_latency=self.latency_collector.get_stats("tts_complete_ms"),
            latency_trend_ms_per_min=latency_trend,
            cpu_mean=resource_agg.cpu_mean,
            cpu_max=resource_agg.cpu_max,
            memory_mean_mb=resource_agg.memory_mean_mb,
            memory_max_mb=resource_agg.memory_max_mb,
            memory_trend_mb_per_min=memory_trend,
            max_consecutive_failures=max_consecutive_failures,
            recovery_time_ms=self._calculate_recovery_time(),
            dropout_count=dropout_count,
            thermal_throttle_events=0,  # TODO: Implement thermal detection
        )

    def _calculate_latency_trend(self) -> float:
        """
        Calculate latency trend (ms/min).

        Positive = getting slower, Negative = getting faster
        """
        successful = [i for i in self.iterations if i.success and i.e2e_latency_ms]

        if len(successful) < 10:
            return 0.0

        # Split into first and last quarter
        n = len(successful)
        quarter = n // 4

        first_quarter = successful[:quarter]
        last_quarter = successful[-quarter:]

        first_avg = sum(i.e2e_latency_ms for i in first_quarter) / len(first_quarter)
        last_avg = sum(i.e2e_latency_ms for i in last_quarter) / len(last_quarter)

        # Calculate time span in minutes
        time_span_min = (
            last_quarter[-1].timestamp - first_quarter[0].timestamp
        ).total_seconds() / 60

        if time_span_min <= 0:
            return 0.0

        return (last_avg - first_avg) / time_span_min

    def _calculate_memory_trend(self) -> float:
        """
        Calculate memory trend (MB/min).

        Positive = growing (potential leak)
        """
        measurements = self.resource_collector.measurements

        if len(measurements) < 10:
            return 0.0

        n = len(measurements)
        quarter = n // 4

        first_quarter = measurements[:quarter]
        last_quarter = measurements[-quarter:]

        first_avg = sum(m.memory_mb for m in first_quarter) / len(first_quarter)
        last_avg = sum(m.memory_mb for m in last_quarter) / len(last_quarter)

        # Calculate time span in minutes
        time_span_min = (
            last_quarter[-1].timestamp - first_quarter[0].timestamp
        ).total_seconds() / 60

        if time_span_min <= 0:
            return 0.0

        return (last_avg - first_avg) / time_span_min

    def _count_dropouts(self) -> int:
        """Count iterations with latency > 2x mean (dropouts)."""
        successful = [i for i in self.iterations if i.success and i.e2e_latency_ms]

        if not successful:
            return 0

        mean_latency = sum(i.e2e_latency_ms for i in successful) / len(successful)
        threshold = mean_latency * 2

        return sum(1 for i in successful if i.e2e_latency_ms > threshold)

    def _calculate_recovery_time(self) -> Optional[float]:
        """Calculate average time to recover from failures."""
        recovery_times = []

        in_failure = False
        failure_start = None

        for i, result in enumerate(self.iterations):
            if not result.success:
                if not in_failure:
                    in_failure = True
                    failure_start = result.timestamp
            else:
                if in_failure:
                    recovery_time = (result.timestamp - failure_start).total_seconds() * 1000
                    recovery_times.append(recovery_time)
                    in_failure = False

        if recovery_times:
            return sum(recovery_times) / len(recovery_times)
        return None

    def stop(self) -> None:
        """Stop an ongoing stress test."""
        self._running = False

    def save_results(
        self,
        result: StressTestResult,
        output_dir: Path,
    ) -> Path:
        """
        Save stress test results to JSON.

        Args:
            result: StressTestResult to save
            output_dir: Directory to save results

        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stress_test_{self.config_name}_{timestamp}.json"
        output_path = output_dir / filename

        data = {
            "summary": result.to_dict(),
            "is_stable": result.is_stable(),
            "iterations": [
                {
                    "iteration": i.iteration,
                    "timestamp": i.timestamp.isoformat(),
                    "success": i.success,
                    "error": i.error,
                    "e2e_latency_ms": i.e2e_latency_ms,
                }
                for i in self.iterations
            ],
            "resource_samples": [
                m.to_dict() for m in self.resource_collector.measurements
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path


def run_stability_comparison(
    configs: list[Path],
    audio_dir: Path,
    duration_minutes: float = 30,
    output_dir: Optional[Path] = None,
) -> dict[str, StressTestResult]:
    """
    Run stress tests on multiple configurations for comparison.

    Args:
        configs: List of configuration file paths
        audio_dir: Directory with test audio
        duration_minutes: Duration for each test
        output_dir: Directory to save results

    Returns:
        Dictionary mapping config name to results
    """
    results = {}

    for config_path in configs:
        print(f"\n{'='*60}")
        print(f"Stress testing: {config_path.stem}")
        print(f"{'='*60}")

        runner = StressTestRunner(config_path=config_path)

        result = runner.run_sustained(
            duration_minutes=duration_minutes,
            audio_dir=audio_dir,
            verbose=True,
        )

        results[runner.config_name] = result

        if output_dir:
            runner.save_results(result, output_dir)

        # Print summary
        print(f"\nResults for {runner.config_name}:")
        print(f"  Total iterations: {result.total_iterations}")
        print(f"  Failure rate: {result.failure_rate:.2%}")
        print(f"  E2E latency mean: {result.e2e_latency.mean:.1f}ms")
        print(f"  E2E latency p95: {result.e2e_latency.p95:.1f}ms")
        print(f"  Latency trend: {result.latency_trend_ms_per_min:+.1f}ms/min")
        print(f"  Memory trend: {result.memory_trend_mb_per_min:+.1f}MB/min")
        print(f"  Stable: {'YES' if result.is_stable() else 'NO'}")

    return results
