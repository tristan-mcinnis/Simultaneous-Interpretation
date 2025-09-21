from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simultaneous interpretation tool with modular speech recognition, translation, "
            "and text-to-speech components."
        )
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Only list input and output audio devices, then exit.",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        help="Index of the microphone to use. Defaults to the system default device.",
    )
    parser.add_argument(
        "--output-device",
        type=int,
        help="Index of the playback device to use for audio translation.",
    )
    parser.add_argument(
        "--input-language",
        default="en",
        help="Language code expected from the microphone input (e.g., en, zh).",
    )
    parser.add_argument(
        "--target-language",
        default="fr",
        help="Language code or name for translation output.",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Enable translation of transcripts using the OpenAI API.",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Enable audio playback of translations via OpenAI text-to-speech.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help=(
            "OpenAI model identifier to use for translation. "
            "Use available released models such as gpt-4o, gpt-4o-mini, etc."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for translation responses.",
    )
    parser.add_argument(
        "--tts-model",
        default="gpt-4o-mini-tts",
        help="OpenAI text-to-speech model to use when --tts is enabled.",
    )
    parser.add_argument(
        "--voice",
        default="alloy",
        help="Voice preset for text-to-speech playback.",
    )
    parser.add_argument(
        "--tts-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier for text-to-speech audio.",
    )
    parser.add_argument(
        "--transcriber",
        choices=["whispercpp", "faster-whisper"],
        default="whispercpp",
        help="Speech-to-text backend to use.",
    )
    parser.add_argument(
        "--whisper-model",
        default="base.en",
        help="Model name or path for the Whisper backend (e.g., base.en, small, custom path).",
    )
    parser.add_argument(
        "--whisper-threads",
        type=int,
        help="Number of CPU threads for whisper.cpp. Defaults to automatic selection.",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=10,
        help="Number of previous translated chunks to include as context.",
    )
    parser.add_argument(
        "--phrase-time-limit",
        type=int,
        default=8,
        help="Maximum length of captured phrases in seconds.",
    )
    parser.add_argument(
        "--ambient-duration",
        type=float,
        default=2.0,
        help="Seconds to sample ambient noise before listening starts.",
    )
    parser.add_argument(
        "--dictionary",
        help="Path to a custom dictionary mapping domain-specific terminology (term=translation).",
    )
    parser.add_argument(
        "--topic",
        default="",
        help="Optional description of the conversation topic to guide translation tone.",
    )
    parser.add_argument(
        "--log-file",
        default="logfile.txt",
        help="Path to the file where transcripts and translations are appended.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)
