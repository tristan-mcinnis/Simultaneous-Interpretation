from __future__ import annotations

from typing import Optional

import pyaudio
from openai import OpenAI


class OpenAITTSEngine:
    """Stream OpenAI text-to-speech audio through PyAudio."""

    def __init__(self, client: OpenAI, model: str, voice: str, speed: float) -> None:
        self.client = client
        self.model = model
        self.voice = voice
        self.speed = speed

    def speak(self, text: str, output_device_index: Optional[int]) -> None:
        if not text:
            return

        audio_interface = pyaudio.PyAudio()
        stream = None
        try:
            stream = audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True,
                output_device_index=output_device_index,
            )
            with self.client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="pcm",
                speed=self.speed,
            ) as response:
                for chunk in response.iter_bytes(chunk_size=1024):
                    stream.write(chunk)
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            audio_interface.terminate()
