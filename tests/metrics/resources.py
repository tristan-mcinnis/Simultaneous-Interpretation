"""Resource metrics collection (CPU, memory, GPU usage)."""

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ResourceMetrics:
    """Container for resource measurements at a point in time."""

    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    gpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Process-specific metrics
    process_cpu_percent: Optional[float] = None
    process_memory_mb: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "memory_percent": self.memory_percent,
            "gpu_percent": self.gpu_percent,
            "gpu_memory_mb": self.gpu_memory_mb,
            "tokens_per_sec": self.tokens_per_sec,
            "timestamp": self.timestamp.isoformat(),
            "process_cpu_percent": self.process_cpu_percent,
            "process_memory_mb": self.process_memory_mb,
        }


@dataclass
class AggregatedResourceMetrics:
    """Aggregated resource metrics from multiple measurements."""

    count: int
    duration_sec: float

    # CPU
    cpu_mean: float = 0.0
    cpu_max: float = 0.0
    process_cpu_mean: Optional[float] = None
    process_cpu_max: Optional[float] = None

    # Memory
    memory_mean_mb: float = 0.0
    memory_max_mb: float = 0.0
    process_memory_mean_mb: Optional[float] = None
    process_memory_max_mb: Optional[float] = None

    # GPU
    gpu_mean: Optional[float] = None
    gpu_max: Optional[float] = None
    gpu_memory_mean_mb: Optional[float] = None
    gpu_memory_max_mb: Optional[float] = None

    # Throughput
    tokens_per_sec_mean: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "count": self.count,
            "duration_sec": self.duration_sec,
            "cpu_mean": self.cpu_mean,
            "cpu_max": self.cpu_max,
            "process_cpu_mean": self.process_cpu_mean,
            "process_cpu_max": self.process_cpu_max,
            "memory_mean_mb": self.memory_mean_mb,
            "memory_max_mb": self.memory_max_mb,
            "process_memory_mean_mb": self.process_memory_mean_mb,
            "process_memory_max_mb": self.process_memory_max_mb,
            "gpu_mean": self.gpu_mean,
            "gpu_max": self.gpu_max,
            "gpu_memory_mean_mb": self.gpu_memory_mean_mb,
            "gpu_memory_max_mb": self.gpu_memory_max_mb,
            "tokens_per_sec_mean": self.tokens_per_sec_mean,
        }


class ResourceCollector:
    """Collects resource usage metrics over time."""

    def __init__(
        self,
        name: str = "default",
        interval_sec: float = 1.0,
        monitor_gpu: bool = True,
    ):
        """
        Initialize resource collector.

        Args:
            name: Name/identifier for this collector
            interval_sec: Sampling interval in seconds
            monitor_gpu: Whether to monitor GPU (if available)
        """
        self.name = name
        self.interval_sec = interval_sec
        self.monitor_gpu = monitor_gpu
        self.measurements: list[ResourceMetrics] = []

        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

        # Initialize psutil
        try:
            import psutil

            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
        except ImportError:
            raise ImportError("psutil is required. Install with: pip install psutil")

        # Check GPU availability
        self._gpu_available = False
        if monitor_gpu:
            self._init_gpu_monitoring()

    def _init_gpu_monitoring(self) -> None:
        """Initialize GPU monitoring if available."""
        # Try Apple Silicon (Metal)
        try:
            # Check for Apple Silicon
            import platform

            if platform.machine() == "arm64" and platform.system() == "Darwin":
                # Apple Silicon - use a simple activity monitor approach
                self._gpu_available = True
                self._gpu_type = "apple_silicon"
                return
        except Exception:
            pass

        # Try NVIDIA
        try:
            import pynvml

            pynvml.nvmlInit()
            self._gpu_available = True
            self._gpu_type = "nvidia"
            self._nvml = pynvml
            return
        except Exception:
            pass

        self._gpu_available = False

    def snapshot(self) -> ResourceMetrics:
        """
        Take a snapshot of current resource usage.

        Returns:
            ResourceMetrics with current measurements
        """
        # System-wide metrics
        cpu_percent = self._psutil.cpu_percent(interval=None)
        memory = self._psutil.virtual_memory()

        # Process-specific metrics
        try:
            process_cpu = self._process.cpu_percent(interval=None)
            process_memory = self._process.memory_info().rss / (1024 * 1024)
        except Exception:
            process_cpu = None
            process_memory = None

        metrics = ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_mb=memory.used / (1024 * 1024),
            memory_percent=memory.percent,
            process_cpu_percent=process_cpu,
            process_memory_mb=process_memory,
        )

        # GPU metrics
        if self._gpu_available:
            gpu_metrics = self._get_gpu_metrics()
            if gpu_metrics:
                metrics.gpu_percent = gpu_metrics.get("utilization")
                metrics.gpu_memory_mb = gpu_metrics.get("memory_mb")

        return metrics

    def _get_gpu_metrics(self) -> Optional[dict]:
        """Get GPU metrics based on available GPU type."""
        if not self._gpu_available:
            return None

        if self._gpu_type == "nvidia":
            try:
                handle = self._nvml.nvmlDeviceGetHandleByIndex(0)
                util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                memory = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                return {
                    "utilization": util.gpu,
                    "memory_mb": memory.used / (1024 * 1024),
                }
            except Exception:
                return None

        elif self._gpu_type == "apple_silicon":
            # Apple Silicon doesn't have easy GPU utilization API
            # Return None for now - could be enhanced with powermetrics
            return None

        return None

    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self._monitoring:
            return

        self._monitoring = True
        self._start_time = time.time()
        self.measurements.clear()

        # Initialize CPU percent
        self._psutil.cpu_percent(interval=None)
        self._process.cpu_percent(interval=None)

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            try:
                metrics = self.snapshot()
                self.measurements.append(metrics)
            except Exception:
                pass  # Ignore errors in monitoring

            time.sleep(self.interval_sec)

    def stop_monitoring(self) -> float:
        """
        Stop background monitoring.

        Returns:
            Duration of monitoring in seconds
        """
        self._monitoring = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

        duration = time.time() - self._start_time if self._start_time else 0.0
        return duration

    def record_tokens_per_sec(self, tokens_per_sec: float) -> None:
        """
        Record token generation speed (for LLM monitoring).

        Args:
            tokens_per_sec: Tokens generated per second
        """
        if self.measurements:
            self.measurements[-1].tokens_per_sec = tokens_per_sec
        else:
            metrics = self.snapshot()
            metrics.tokens_per_sec = tokens_per_sec
            self.measurements.append(metrics)

    def get_aggregated(self) -> AggregatedResourceMetrics:
        """Get aggregated statistics for all measurements."""
        n = len(self.measurements)

        if n == 0:
            return AggregatedResourceMetrics(count=0, duration_sec=0.0)

        def safe_mean(values: list) -> Optional[float]:
            filtered = [v for v in values if v is not None]
            return sum(filtered) / len(filtered) if filtered else None

        def safe_max(values: list) -> Optional[float]:
            filtered = [v for v in values if v is not None]
            return max(filtered) if filtered else None

        cpu_values = [m.cpu_percent for m in self.measurements]
        process_cpu_values = [m.process_cpu_percent for m in self.measurements]
        memory_values = [m.memory_mb for m in self.measurements]
        process_memory_values = [m.process_memory_mb for m in self.measurements]
        gpu_values = [m.gpu_percent for m in self.measurements]
        gpu_memory_values = [m.gpu_memory_mb for m in self.measurements]
        tokens_values = [m.tokens_per_sec for m in self.measurements]

        # Calculate duration
        if self.measurements:
            start = self.measurements[0].timestamp
            end = self.measurements[-1].timestamp
            duration = (end - start).total_seconds()
        else:
            duration = 0.0

        return AggregatedResourceMetrics(
            count=n,
            duration_sec=duration,
            cpu_mean=safe_mean(cpu_values) or 0.0,
            cpu_max=safe_max(cpu_values) or 0.0,
            process_cpu_mean=safe_mean(process_cpu_values),
            process_cpu_max=safe_max(process_cpu_values),
            memory_mean_mb=safe_mean(memory_values) or 0.0,
            memory_max_mb=safe_max(memory_values) or 0.0,
            process_memory_mean_mb=safe_mean(process_memory_values),
            process_memory_max_mb=safe_max(process_memory_values),
            gpu_mean=safe_mean(gpu_values),
            gpu_max=safe_max(gpu_values),
            gpu_memory_mean_mb=safe_mean(gpu_memory_values),
            gpu_memory_max_mb=safe_max(gpu_memory_values),
            tokens_per_sec_mean=safe_mean(tokens_values),
        )

    def clear(self) -> None:
        """Clear all measurements."""
        self.measurements.clear()
        self._start_time = None

    def to_dict(self) -> dict:
        """Export all data as dictionary."""
        return {
            "name": self.name,
            "count": len(self.measurements),
            "interval_sec": self.interval_sec,
            "measurements": [m.to_dict() for m in self.measurements],
            "aggregated": self.get_aggregated().to_dict(),
        }


class CostEstimator:
    """Estimates API costs for cloud components."""

    # Pricing as of 2024 (per 1K tokens or per request)
    PRICING = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # per 1K tokens
        "gpt-4o": {"input": 0.0025, "output": 0.01},  # per 1K tokens
        "gpt-5-mini": {"input": 0.0003, "output": 0.0012},  # estimated
        "gpt-4o-mini-tts": {"per_char": 0.000015},  # per character
        "gpt-4.1-tts": {"per_char": 0.00003},  # estimated
    }

    def __init__(self):
        """Initialize cost estimator."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tts_chars = 0
        self.api_calls = 0

    def record_translation(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Record translation API usage and return cost.

        Returns:
            Cost in USD for this call
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.api_calls += 1

        pricing = self.PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (
            (input_tokens / 1000) * pricing["input"]
            + (output_tokens / 1000) * pricing["output"]
        )
        return cost

    def record_tts(
        self,
        model: str,
        text: str,
    ) -> float:
        """
        Record TTS API usage and return cost.

        Returns:
            Cost in USD for this call
        """
        char_count = len(text)
        self.total_tts_chars += char_count
        self.api_calls += 1

        pricing = self.PRICING.get(model, {"per_char": 0.0})
        cost = char_count * pricing.get("per_char", 0.0)
        return cost

    def get_total_cost(self) -> float:
        """Get total estimated cost so far."""
        # Rough estimate using default models
        translation_cost = (
            (self.total_input_tokens / 1000) * 0.00015
            + (self.total_output_tokens / 1000) * 0.0006
        )
        tts_cost = self.total_tts_chars * 0.000015
        return translation_cost + tts_cost

    def estimate_hourly_cost(
        self,
        duration_sec: float,
    ) -> float:
        """
        Estimate hourly cost based on current usage.

        Args:
            duration_sec: Duration of monitoring in seconds

        Returns:
            Estimated hourly cost in USD
        """
        if duration_sec <= 0:
            return 0.0

        total_cost = self.get_total_cost()
        return (total_cost / duration_sec) * 3600

    def to_dict(self) -> dict:
        """Export cost data as dictionary."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tts_chars": self.total_tts_chars,
            "api_calls": self.api_calls,
            "estimated_total_cost_usd": self.get_total_cost(),
        }
