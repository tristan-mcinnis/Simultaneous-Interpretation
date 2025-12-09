"""
Results reporting and comparison for architecture testing.

Generates markdown reports, comparison tables, and visualizations
for test results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tests.runners.component_tests import ComponentTestResult
from tests.runners.e2e_tests import BatchResult
from tests.runners.stress_tests import StressTestResult


class ReportGenerator:
    """Generates comparison reports from test results."""

    def __init__(self, output_dir: Path):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_comparison_report(
        self,
        e2e_results: dict[str, BatchResult],
        stress_results: Optional[dict[str, StressTestResult]] = None,
        component_results: Optional[dict[str, dict[str, ComponentTestResult]]] = None,
    ) -> Path:
        """
        Generate a full comparison report in Markdown format.

        Args:
            e2e_results: Dictionary of config_name -> BatchResult
            stress_results: Dictionary of config_name -> StressTestResult
            component_results: Dictionary of config_name -> component -> ComponentTestResult

        Returns:
            Path to generated report
        """
        lines = []

        # Header
        lines.append("# Simultaneous Interpretation Architecture Comparison Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.extend(self._generate_summary(e2e_results, stress_results))
        lines.append("")

        # Architecture Overview
        lines.append("## Architectures Tested")
        lines.append("")
        lines.extend(self._generate_architecture_overview(e2e_results))
        lines.append("")

        # Latency Comparison
        lines.append("## Latency Comparison")
        lines.append("")
        lines.extend(self._generate_latency_table(e2e_results))
        lines.append("")

        # Quality Comparison
        lines.append("## Quality Comparison")
        lines.append("")
        lines.extend(self._generate_quality_table(e2e_results))
        lines.append("")

        # Cost Comparison
        lines.append("## Cost Analysis")
        lines.append("")
        lines.extend(self._generate_cost_table(e2e_results))
        lines.append("")

        # Stability Results
        if stress_results:
            lines.append("## Stability Analysis (Stress Test)")
            lines.append("")
            lines.extend(self._generate_stability_table(stress_results))
            lines.append("")

        # Detailed Results
        lines.append("## Detailed Results by Architecture")
        lines.append("")

        for config_name, result in e2e_results.items():
            lines.append(f"### {config_name}")
            lines.append("")
            lines.extend(self._generate_detailed_section(result))
            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        lines.extend(self._generate_recommendations(e2e_results, stress_results))
        lines.append("")

        # Write report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"comparison_report_{timestamp}.md"

        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        return report_path

    def _generate_summary(
        self,
        e2e_results: dict[str, BatchResult],
        stress_results: Optional[dict[str, StressTestResult]],
    ) -> list[str]:
        """Generate executive summary."""
        lines = []

        # Find best performers
        best_latency = min(e2e_results.items(), key=lambda x: x[1].e2e_latency.mean)
        best_quality = max(
            [(k, v) for k, v in e2e_results.items() if v.translation_bleu_mean],
            key=lambda x: x[1].translation_bleu_mean or 0,
            default=(None, None),
        )
        lowest_cost = min(
            [(k, v) for k, v in e2e_results.items() if v.estimated_hourly_cost_usd],
            key=lambda x: x[1].estimated_hourly_cost_usd or float("inf"),
            default=(None, None),
        )

        lines.append(f"**Best Latency:** {best_latency[0]} ({best_latency[1].e2e_latency.mean:.0f}ms mean E2E)")

        if best_quality[0]:
            lines.append(f"**Best Quality:** {best_quality[0]} ({best_quality[1].translation_bleu_mean:.1f} BLEU)")

        if lowest_cost[0] and lowest_cost[1].estimated_hourly_cost_usd:
            lines.append(f"**Lowest Cost:** {lowest_cost[0]} (${lowest_cost[1].estimated_hourly_cost_usd:.2f}/hour)")

        if stress_results:
            stable = [k for k, v in stress_results.items() if v.is_stable()]
            if stable:
                lines.append(f"**Stable Architectures:** {', '.join(stable)}")
            else:
                lines.append("**Stable Architectures:** None passed all stability criteria")

        return lines

    def _generate_architecture_overview(
        self,
        e2e_results: dict[str, BatchResult],
    ) -> list[str]:
        """Generate architecture overview table."""
        lines = []

        lines.append("| Architecture | Audio Files | Total Duration | Errors |")
        lines.append("|--------------|-------------|----------------|--------|")

        for name, result in e2e_results.items():
            duration_min = result.total_audio_duration_ms / 60000
            lines.append(
                f"| {name} | {result.audio_count} | {duration_min:.1f} min | {len(result.errors)} |"
            )

        return lines

    def _generate_latency_table(
        self,
        e2e_results: dict[str, BatchResult],
    ) -> list[str]:
        """Generate latency comparison table."""
        lines = []

        lines.append("### End-to-End Latency (milliseconds)")
        lines.append("")
        lines.append("| Architecture | Mean | P50 | P95 | P99 | Min | Max |")
        lines.append("|--------------|------|-----|-----|-----|-----|-----|")

        for name, result in e2e_results.items():
            lat = result.e2e_latency
            lines.append(
                f"| {name} | {lat.mean:.0f} | {lat.p50:.0f} | {lat.p95:.0f} | "
                f"{lat.p99:.0f} | {lat.min:.0f} | {lat.max:.0f} |"
            )

        lines.append("")
        lines.append("### Component Breakdown (mean milliseconds)")
        lines.append("")
        lines.append("| Architecture | STT | Translation | TTS |")
        lines.append("|--------------|-----|-------------|-----|")

        for name, result in e2e_results.items():
            lines.append(
                f"| {name} | {result.stt_latency.mean:.0f} | "
                f"{result.translation_latency.mean:.0f} | {result.tts_latency.mean:.0f} |"
            )

        return lines

    def _generate_quality_table(
        self,
        e2e_results: dict[str, BatchResult],
    ) -> list[str]:
        """Generate quality comparison table."""
        lines = []

        lines.append("| Architecture | STT WER | Translation BLEU |")
        lines.append("|--------------|---------|------------------|")

        for name, result in e2e_results.items():
            wer = f"{result.stt_wer_mean:.2%}" if result.stt_wer_mean else "N/A"
            bleu = f"{result.translation_bleu_mean:.1f}" if result.translation_bleu_mean else "N/A"
            lines.append(f"| {name} | {wer} | {bleu} |")

        lines.append("")
        lines.append("**Quality Thresholds:**")
        lines.append("- WER: <15% acceptable, <10% good, <5% excellent")
        lines.append("- BLEU: >20 acceptable, >30 good, >40 excellent")

        return lines

    def _generate_cost_table(
        self,
        e2e_results: dict[str, BatchResult],
    ) -> list[str]:
        """Generate cost comparison table."""
        lines = []

        lines.append("| Architecture | Est. Total Cost | Est. Hourly Cost | Cost Rating |")
        lines.append("|--------------|-----------------|------------------|-------------|")

        for name, result in e2e_results.items():
            total = f"${result.estimated_cost_usd:.4f}" if result.estimated_cost_usd else "N/A"
            hourly = f"${result.estimated_hourly_cost_usd:.2f}" if result.estimated_hourly_cost_usd else "N/A"

            # Rate cost
            if result.estimated_hourly_cost_usd is None:
                rating = "Free (local)"
            elif result.estimated_hourly_cost_usd < 0.50:
                rating = "Excellent"
            elif result.estimated_hourly_cost_usd < 2.00:
                rating = "Good"
            elif result.estimated_hourly_cost_usd < 5.00:
                rating = "Acceptable"
            else:
                rating = "Expensive"

            lines.append(f"| {name} | {total} | {hourly} | {rating} |")

        lines.append("")
        lines.append("*Note: Local-only architectures have no API costs but require hardware investment.*")

        return lines

    def _generate_stability_table(
        self,
        stress_results: dict[str, StressTestResult],
    ) -> list[str]:
        """Generate stability comparison table."""
        lines = []

        lines.append("| Architecture | Duration | Iterations | Failure Rate | Memory Trend | Latency Trend | Stable |")
        lines.append("|--------------|----------|------------|--------------|--------------|---------------|--------|")

        for name, result in stress_results.items():
            stable = "✓" if result.is_stable() else "✗"
            mem_trend = f"{result.memory_trend_mb_per_min:+.1f} MB/min"
            lat_trend = f"{result.latency_trend_ms_per_min:+.1f} ms/min"

            lines.append(
                f"| {name} | {result.duration_minutes:.0f} min | {result.total_iterations} | "
                f"{result.failure_rate:.2%} | {mem_trend} | {lat_trend} | {stable} |"
            )

        return lines

    def _generate_detailed_section(
        self,
        result: BatchResult,
    ) -> list[str]:
        """Generate detailed results for one architecture."""
        lines = []

        lines.append("**Performance:**")
        lines.append(f"- E2E Latency: {result.e2e_latency.mean:.0f}ms (mean), {result.e2e_latency.p95:.0f}ms (p95)")
        lines.append(f"- STT Latency: {result.stt_latency.mean:.0f}ms")
        lines.append(f"- Translation Latency: {result.translation_latency.mean:.0f}ms")
        lines.append(f"- TTS Latency: {result.tts_latency.mean:.0f}ms")
        lines.append("")

        if result.stt_wer_mean or result.translation_bleu_mean:
            lines.append("**Quality:**")
            if result.stt_wer_mean:
                lines.append(f"- STT WER: {result.stt_wer_mean:.2%}")
            if result.translation_bleu_mean:
                lines.append(f"- Translation BLEU: {result.translation_bleu_mean:.1f}")
            lines.append("")

        if result.cpu_mean or result.memory_max_mb:
            lines.append("**Resources:**")
            if result.cpu_mean:
                lines.append(f"- CPU Usage: {result.cpu_mean:.1f}%")
            if result.memory_max_mb:
                lines.append(f"- Peak Memory: {result.memory_max_mb:.0f} MB")
            lines.append("")

        if result.errors:
            lines.append(f"**Errors:** {len(result.errors)} failures during testing")
            lines.append("")

        return lines

    def _generate_recommendations(
        self,
        e2e_results: dict[str, BatchResult],
        stress_results: Optional[dict[str, StressTestResult]],
    ) -> list[str]:
        """Generate recommendations based on results."""
        lines = []

        lines.append("### Use Case Recommendations")
        lines.append("")

        # Analyze for different use cases
        results_list = list(e2e_results.items())

        # Best for low latency
        best_latency = min(results_list, key=lambda x: x[1].e2e_latency.mean)
        lines.append(f"**For lowest latency:** {best_latency[0]}")
        lines.append(f"  - E2E: {best_latency[1].e2e_latency.mean:.0f}ms mean")
        lines.append("")

        # Best for quality
        quality_results = [(k, v) for k, v in results_list if v.translation_bleu_mean]
        if quality_results:
            best_quality = max(quality_results, key=lambda x: x[1].translation_bleu_mean)
            lines.append(f"**For best quality:** {best_quality[0]}")
            lines.append(f"  - BLEU: {best_quality[1].translation_bleu_mean:.1f}")
            lines.append("")

        # Best for cost
        cost_results = [(k, v) for k, v in results_list if v.estimated_hourly_cost_usd is not None]
        if cost_results:
            lowest_cost = min(cost_results, key=lambda x: x[1].estimated_hourly_cost_usd)
            lines.append(f"**For lowest cost:** {lowest_cost[0]}")
            if lowest_cost[1].estimated_hourly_cost_usd:
                lines.append(f"  - Cost: ${lowest_cost[1].estimated_hourly_cost_usd:.2f}/hour")
            else:
                lines.append("  - Cost: Free (local processing)")
            lines.append("")

        # Most stable
        if stress_results:
            stable = [k for k, v in stress_results.items() if v.is_stable()]
            if stable:
                lines.append(f"**For sustained operation:** {', '.join(stable)}")
                lines.append("  - Passed all stability criteria")
            else:
                lines.append("**For sustained operation:** No architecture fully stable")
                lines.append("  - Consider further optimization")
            lines.append("")

        # Balanced recommendation
        lines.append("### Balanced Recommendation")
        lines.append("")
        lines.append("Consider your priorities:")
        lines.append("- **Real-time critical:** Prioritize latency (local architectures)")
        lines.append("- **Quality critical:** Prioritize BLEU/COMET (cloud GPT models)")
        lines.append("- **Cost sensitive:** Use local models where possible")
        lines.append("- **Reliability:** Ensure stability testing passes")

        return lines

    def export_results_json(
        self,
        e2e_results: dict[str, BatchResult],
        stress_results: Optional[dict[str, StressTestResult]] = None,
    ) -> Path:
        """
        Export all results to a JSON file.

        Returns:
            Path to JSON file
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "e2e_results": {k: v.to_dict() for k, v in e2e_results.items()},
        }

        if stress_results:
            data["stress_results"] = {k: v.to_dict() for k, v in stress_results.items()}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"results_{timestamp}.json"

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        return json_path


def generate_quick_summary(e2e_results: dict[str, BatchResult]) -> str:
    """Generate a quick console summary."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("ARCHITECTURE COMPARISON SUMMARY")
    lines.append("=" * 70)

    # Table header
    lines.append(f"{'Architecture':<20} {'E2E (ms)':<12} {'STT (ms)':<12} {'Trans (ms)':<12} {'BLEU':<8}")
    lines.append("-" * 70)

    for name, result in e2e_results.items():
        bleu = f"{result.translation_bleu_mean:.1f}" if result.translation_bleu_mean else "N/A"
        lines.append(
            f"{name:<20} {result.e2e_latency.mean:<12.0f} "
            f"{result.stt_latency.mean:<12.0f} {result.translation_latency.mean:<12.0f} "
            f"{bleu:<8}"
        )

    lines.append("=" * 70)

    return "\n".join(lines)
