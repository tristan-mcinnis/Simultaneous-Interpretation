"""
Simultaneous Interpretation Testing Framework

A comprehensive testing framework for comparing different architectural approaches
to real-time Chinese-to-English simultaneous interpretation.

Architecture Variants:
- Architecture A: Cloud-Based (whisper.cpp + OpenAI GPT + OpenAI TTS)
- Architecture B: Hybrid Local (whisper.cpp + Qwen2.5 via LM Studio + OpenAI TTS)
- Architecture C: Fully Local (whisper.cpp + Qwen2.5 + VibeVoice)
"""

__version__ = "0.1.0"
