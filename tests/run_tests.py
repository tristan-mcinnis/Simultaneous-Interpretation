#!/usr/bin/env python3
"""
Test runner CLI for the Simultaneous Interpretation testing framework.

Usage:
    python -m tests.run_tests component --config configs/arch_a_cloud.yaml --audio tests/corpus/audio/short
    python -m tests.run_tests e2e --config configs/arch_a_cloud.yaml --audio tests/corpus/audio
    python -m tests.run_tests stress --config configs/arch_a_cloud.yaml --audio tests/corpus/audio --duration 30
    python -m tests.run_tests compare --configs configs/ --audio tests/corpus/audio
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_parser() -> argparse.ArgumentParser:
    """Set up argument parser."""
    parser = argparse.ArgumentParser(
        description="Simultaneous Interpretation Architecture Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Test type to run")

    # Component tests
    component_parser = subparsers.add_parser(
        "component",
        help="Run component-level tests (STT, Translation, TTS)",
    )
    component_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to architecture config YAML",
    )
    component_parser.add_argument(
        "--audio",
        type=Path,
        help="Directory with test audio files",
    )
    component_parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Directory with ground truth transcriptions",
    )
    component_parser.add_argument(
        "--references",
        type=Path,
        help="Directory with reference translations",
    )
    component_parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/results"),
        help="Output directory for results",
    )
    component_parser.add_argument(
        "--component",
        choices=["stt", "translation", "tts", "all"],
        default="all",
        help="Which component to test",
    )

    # E2E tests
    e2e_parser = subparsers.add_parser(
        "e2e",
        help="Run end-to-end pipeline tests",
    )
    e2e_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to architecture config YAML",
    )
    e2e_parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Directory with test audio files",
    )
    e2e_parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Directory with ground truth transcriptions",
    )
    e2e_parser.add_argument(
        "--references",
        type=Path,
        help="Directory with reference translations",
    )
    e2e_parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/results"),
        help="Output directory for results",
    )
    e2e_parser.add_argument(
        "--no-resources",
        action="store_true",
        help="Disable resource monitoring",
    )

    # Stress tests
    stress_parser = subparsers.add_parser(
        "stress",
        help="Run stress/endurance tests",
    )
    stress_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to architecture config YAML",
    )
    stress_parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Directory with test audio files",
    )
    stress_parser.add_argument(
        "--duration",
        type=float,
        default=30,
        help="Test duration in minutes",
    )
    stress_parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/results"),
        help="Output directory for results",
    )
    stress_parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Pause between iterations (seconds)",
    )

    # Compare all architectures
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare all architecture configurations",
    )
    compare_parser.add_argument(
        "--configs",
        type=Path,
        default=Path("tests/configs"),
        help="Directory containing config files",
    )
    compare_parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Directory with test audio files",
    )
    compare_parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Directory with ground truth transcriptions",
    )
    compare_parser.add_argument(
        "--references",
        type=Path,
        help="Directory with reference translations",
    )
    compare_parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/results"),
        help="Output directory for results",
    )
    compare_parser.add_argument(
        "--include-stress",
        action="store_true",
        help="Include stress tests (takes longer)",
    )
    compare_parser.add_argument(
        "--stress-duration",
        type=float,
        default=10,
        help="Duration for stress tests (minutes)",
    )

    # Quick validation test
    validate_parser = subparsers.add_parser(
        "validate",
        help="Quick validation that components are working",
    )
    validate_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to architecture config YAML",
    )

    return parser


def run_component_tests(args: argparse.Namespace) -> int:
    """Run component-level tests."""
    from tests.runners.component_tests import ComponentTestRunner

    print(f"\n{'='*60}")
    print("COMPONENT TESTS")
    print(f"Config: {args.config}")
    print(f"{'='*60}\n")

    runner = ComponentTestRunner(config_path=args.config)

    results = {}

    # STT tests
    if args.component in ["stt", "all"] and args.audio:
        print("Running STT tests...")
        results["stt"] = runner.test_stt_batch(
            args.audio,
            args.ground_truth,
        )
        print(f"  Completed: {len(results['stt'].raw_results)} files")
        print(f"  Mean latency: {results['stt'].latency.mean:.0f}ms")
        if results["stt"].quality:
            wer = results["stt"].quality.get("wer_mean")
            if wer:
                print(f"  Mean WER: {wer:.2%}")

    # Translation tests (use sample texts if no audio)
    if args.component in ["translation", "all"]:
        print("\nRunning translation tests...")

        # Load sample texts
        import json

        samples_path = Path("tests/corpus/samples/sample_texts.json")
        if samples_path.exists():
            with open(samples_path) as f:
                samples = json.load(f)

            texts = [s["chinese"] for s in samples["short_utterances"]]
            refs = [s["english"] for s in samples["short_utterances"]]

            results["translation"] = runner.test_translation_batch(texts, refs)
            print(f"  Completed: {len(results['translation'].raw_results)} texts")
            print(f"  Mean latency: {results['translation'].latency.mean:.0f}ms")
            if results["translation"].quality:
                bleu = results["translation"].quality.get("bleu_mean")
                if bleu:
                    print(f"  Mean BLEU: {bleu:.1f}")

    # TTS tests
    if args.component in ["tts", "all"]:
        print("\nRunning TTS tests...")
        sample_texts = [
            "Hello, nice to meet you.",
            "The weather is really nice today.",
            "Thank you very much for your help.",
        ]
        results["tts"] = runner.test_tts_batch(sample_texts)
        print(f"  Completed: {len(results['tts'].raw_results)} texts")
        print(f"  Mean latency: {results['tts'].latency.mean:.0f}ms")

    # Save results
    if results:
        output_path = runner.save_results(results, args.output)
        print(f"\nResults saved to: {output_path}")

    return 0


def run_e2e_tests(args: argparse.Namespace) -> int:
    """Run end-to-end pipeline tests."""
    from tests.runners.e2e_tests import E2ETestRunner

    print(f"\n{'='*60}")
    print("END-TO-END PIPELINE TESTS")
    print(f"Config: {args.config}")
    print(f"Audio: {args.audio}")
    print(f"{'='*60}\n")

    runner = E2ETestRunner(config_path=args.config)

    result = runner.run_batch(
        audio_dir=args.audio,
        ground_truth_dir=args.ground_truth,
        reference_dir=args.references,
        monitor_resources=not args.no_resources,
    )

    # Print summary
    print(f"\nResults for {runner.config_name}:")
    print(f"  Audio files processed: {result.audio_count}")
    print(f"  Total audio duration: {result.total_audio_duration_ms/60000:.1f} minutes")
    print(f"  E2E latency (mean): {result.e2e_latency.mean:.0f}ms")
    print(f"  E2E latency (p95): {result.e2e_latency.p95:.0f}ms")

    if result.stt_wer_mean is not None:
        print(f"  STT WER: {result.stt_wer_mean:.2%}")
    if result.translation_bleu_mean is not None:
        print(f"  Translation BLEU: {result.translation_bleu_mean:.1f}")
    if result.estimated_hourly_cost_usd is not None:
        print(f"  Est. hourly cost: ${result.estimated_hourly_cost_usd:.2f}")

    if result.errors:
        print(f"  Errors: {len(result.errors)}")

    # Save results
    output_path = runner.save_results(result, args.output)
    print(f"\nResults saved to: {output_path}")

    return 0


def run_stress_tests(args: argparse.Namespace) -> int:
    """Run stress/endurance tests."""
    from tests.runners.stress_tests import StressTestRunner

    print(f"\n{'='*60}")
    print("STRESS TESTS")
    print(f"Config: {args.config}")
    print(f"Duration: {args.duration} minutes")
    print(f"{'='*60}\n")

    runner = StressTestRunner(config_path=args.config)

    result = runner.run_sustained(
        duration_minutes=args.duration,
        audio_dir=args.audio,
        iteration_pause_sec=args.pause,
        verbose=True,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"STRESS TEST RESULTS: {runner.config_name}")
    print(f"{'='*60}")
    print(f"  Total iterations: {result.total_iterations}")
    print(f"  Successful: {result.successful_iterations}")
    print(f"  Failed: {result.failed_iterations}")
    print(f"  Failure rate: {result.failure_rate:.2%}")
    print(f"  E2E latency (mean): {result.e2e_latency.mean:.0f}ms")
    print(f"  E2E latency (p95): {result.e2e_latency.p95:.0f}ms")
    print(f"  Memory trend: {result.memory_trend_mb_per_min:+.1f} MB/min")
    print(f"  Latency trend: {result.latency_trend_ms_per_min:+.1f} ms/min")
    print(f"  STABLE: {'YES' if result.is_stable() else 'NO'}")

    # Save results
    output_path = runner.save_results(result, args.output)
    print(f"\nResults saved to: {output_path}")

    return 0


def run_comparison(args: argparse.Namespace) -> int:
    """Run comparison across all architectures."""
    from tests.reporting import ReportGenerator, generate_quick_summary
    from tests.runners.e2e_tests import E2ETestRunner
    from tests.runners.stress_tests import StressTestRunner

    print(f"\n{'='*60}")
    print("ARCHITECTURE COMPARISON")
    print(f"{'='*60}\n")

    # Find all config files
    configs_dir = args.configs
    config_files = list(configs_dir.glob("arch_*.yaml"))

    if not config_files:
        print(f"No config files found in {configs_dir}")
        return 1

    print(f"Found {len(config_files)} configurations:")
    for cf in config_files:
        print(f"  - {cf.name}")

    # Run E2E tests for each
    e2e_results = {}
    for config_path in config_files:
        print(f"\n{'='*60}")
        print(f"Testing: {config_path.stem}")
        print(f"{'='*60}")

        try:
            runner = E2ETestRunner(config_path=config_path)
            result = runner.run_batch(
                audio_dir=args.audio,
                ground_truth_dir=args.ground_truth,
                reference_dir=args.references,
            )
            e2e_results[runner.config_name] = result
            runner.save_results(result, args.output / "e2e")

            print(f"  E2E latency: {result.e2e_latency.mean:.0f}ms mean")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Run stress tests if requested
    stress_results = {}
    if args.include_stress:
        print(f"\n{'='*60}")
        print("STRESS TESTS")
        print(f"{'='*60}")

        for config_path in config_files:
            print(f"\nStress testing: {config_path.stem}")

            try:
                runner = StressTestRunner(config_path=config_path)
                result = runner.run_sustained(
                    duration_minutes=args.stress_duration,
                    audio_dir=args.audio,
                    verbose=True,
                )
                stress_results[runner.config_name] = result
                runner.save_results(result, args.output / "stress")

            except Exception as e:
                print(f"  ERROR: {e}")

    # Generate comparison report
    print(f"\n{'='*60}")
    print("GENERATING REPORT")
    print(f"{'='*60}")

    reporter = ReportGenerator(args.output)
    report_path = reporter.generate_comparison_report(
        e2e_results=e2e_results,
        stress_results=stress_results if stress_results else None,
    )

    json_path = reporter.export_results_json(
        e2e_results=e2e_results,
        stress_results=stress_results if stress_results else None,
    )

    # Print quick summary
    print(generate_quick_summary(e2e_results))

    print(f"\nReport saved to: {report_path}")
    print(f"JSON data saved to: {json_path}")

    return 0


def run_validation(args: argparse.Namespace) -> int:
    """Quick validation that components are working."""
    from tests.configs import load_config

    print(f"\n{'='*60}")
    print("QUICK VALIDATION")
    print(f"{'='*60}\n")

    config = load_config(args.config)
    print(f"Config: {config.get('name', 'unknown')}")

    errors = []

    # Test STT client
    print("\n[1/3] Testing STT client...")
    try:
        from tests.utils.api_clients import WhisperCppClient

        stt_config = config.get("stt", {})
        client = WhisperCppClient(
            model=stt_config.get("model", "ggml-base.bin"),
            model_path=stt_config.get("model_path"),
        )
        print("  STT client initialized successfully")
    except Exception as e:
        errors.append(f"STT: {e}")
        print(f"  ERROR: {e}")

    # Test translation client
    print("\n[2/3] Testing translation client...")
    try:
        trans_config = config.get("translation", {})
        engine = trans_config.get("engine", "openai")

        if engine == "openai":
            from tests.utils.api_clients import OpenAITranslationClient

            client = OpenAITranslationClient(
                model=trans_config.get("model", "gpt-4o-mini"),
            )
            result = client.translate("你好")
            print(f"  Translation test: '你好' -> '{result.text}'")
            print(f"  Latency: {result.latency_ms:.0f}ms")

        elif engine == "lmstudio":
            from tests.utils.api_clients import LMStudioClient

            client = LMStudioClient(
                endpoint=trans_config.get("endpoint", "http://localhost:1234/v1/chat/completions"),
            )

            if client.check_health():
                result = client.translate("你好")
                print(f"  Translation test: '你好' -> '{result.text}'")
            else:
                errors.append("LM Studio server not running")
                print("  ERROR: LM Studio server not running")

    except Exception as e:
        errors.append(f"Translation: {e}")
        print(f"  ERROR: {e}")

    # Test TTS client
    print("\n[3/3] Testing TTS client...")
    try:
        tts_config = config.get("tts", {})
        engine = tts_config.get("engine", "openai")

        if engine == "openai":
            from tests.utils.api_clients import OpenAITTSClient

            client = OpenAITTSClient(
                model=tts_config.get("model", "gpt-4o-mini-tts"),
            )
            result = client.synthesize("Hello, world.")
            print(f"  TTS test: Generated {len(result.audio_data)} bytes")
            print(f"  First audio latency: {result.first_audio_latency_ms:.0f}ms")

        elif engine == "vibevoice":
            from tests.utils.api_clients import VibeVoiceClient

            client = VibeVoiceClient(
                endpoint=tts_config.get("endpoint", "ws://localhost:8765"),
            )

            if client.check_health():
                result = client.synthesize("Hello, world.")
                print(f"  TTS test: Generated {len(result.audio_data)} bytes")
            else:
                errors.append("VibeVoice server not running")
                print("  ERROR: VibeVoice server not running")

    except Exception as e:
        errors.append(f"TTS: {e}")
        print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'='*60}")
    if errors:
        print(f"VALIDATION FAILED: {len(errors)} error(s)")
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print("VALIDATION PASSED: All components working")
        return 0


def main() -> int:
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Create output directory
    if hasattr(args, "output"):
        args.output = Path(args.output)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = args.output / timestamp
        args.output.mkdir(parents=True, exist_ok=True)

    # Run appropriate test
    if args.command == "component":
        return run_component_tests(args)
    elif args.command == "e2e":
        return run_e2e_tests(args)
    elif args.command == "stress":
        return run_stress_tests(args)
    elif args.command == "compare":
        return run_comparison(args)
    elif args.command == "validate":
        return run_validation(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
