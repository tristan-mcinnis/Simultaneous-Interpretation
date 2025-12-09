"""Audio utilities for loading, chunking, and analyzing audio files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from scipy.io import wavfile


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""

    data: np.ndarray
    sample_rate: int
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    index: int         # Chunk index

    @property
    def duration(self) -> float:
        """Duration of the chunk in seconds."""
        return self.end_time - self.start_time

    @property
    def duration_ms(self) -> float:
        """Duration of the chunk in milliseconds."""
        return self.duration * 1000


@dataclass
class AudioFile:
    """Represents a loaded audio file with metadata."""

    path: Path
    data: np.ndarray
    sample_rate: int
    channels: int = 1
    dtype: np.dtype = field(default_factory=lambda: np.dtype("int16"))

    @property
    def duration(self) -> float:
        """Duration of the audio in seconds."""
        return len(self.data) / self.sample_rate

    @property
    def duration_ms(self) -> float:
        """Duration of the audio in milliseconds."""
        return self.duration * 1000

    def to_mono(self) -> "AudioFile":
        """Convert stereo audio to mono."""
        if self.channels == 1:
            return self

        # Average channels for stereo to mono conversion
        mono_data = np.mean(self.data.reshape(-1, self.channels), axis=1)
        return AudioFile(
            path=self.path,
            data=mono_data.astype(self.dtype),
            sample_rate=self.sample_rate,
            channels=1,
            dtype=self.dtype,
        )

    def resample(self, target_rate: int) -> "AudioFile":
        """Resample audio to a target sample rate."""
        if self.sample_rate == target_rate:
            return self

        # Simple resampling using linear interpolation
        duration = len(self.data) / self.sample_rate
        target_length = int(duration * target_rate)

        indices = np.linspace(0, len(self.data) - 1, target_length)
        resampled = np.interp(indices, np.arange(len(self.data)), self.data)

        return AudioFile(
            path=self.path,
            data=resampled.astype(self.dtype),
            sample_rate=target_rate,
            channels=self.channels,
            dtype=self.dtype,
        )


def load_audio(path: Path | str, target_rate: Optional[int] = 16000) -> AudioFile:
    """
    Load an audio file and optionally resample to target rate.

    Args:
        path: Path to the audio file (WAV format)
        target_rate: Target sample rate (default 16kHz for whisper)

    Returns:
        AudioFile object with loaded data
    """
    path = Path(path)

    sample_rate, data = wavfile.read(path)

    # Determine number of channels
    if len(data.shape) == 1:
        channels = 1
    else:
        channels = data.shape[1]

    audio = AudioFile(
        path=path,
        data=data,
        sample_rate=sample_rate,
        channels=channels,
        dtype=data.dtype,
    )

    # Convert to mono if stereo
    if channels > 1:
        audio = audio.to_mono()

    # Resample if needed
    if target_rate and sample_rate != target_rate:
        audio = audio.resample(target_rate)

    return audio


def save_audio(audio: AudioFile, path: Path | str) -> Path:
    """
    Save audio data to a WAV file.

    Args:
        audio: AudioFile object to save
        path: Output path

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wavfile.write(path, audio.sample_rate, audio.data.astype(audio.dtype))
    return path


def get_audio_duration(path: Path | str) -> float:
    """Get the duration of an audio file in seconds without loading full data."""
    path = Path(path)

    sample_rate, data = wavfile.read(path)

    if len(data.shape) == 1:
        num_samples = len(data)
    else:
        num_samples = data.shape[0]

    return num_samples / sample_rate


def chunk_audio(
    audio: AudioFile,
    chunk_duration_sec: float = 3.0,
    overlap_sec: float = 0.0,
) -> Iterator[AudioChunk]:
    """
    Split audio into chunks of specified duration.

    Args:
        audio: AudioFile to chunk
        chunk_duration_sec: Duration of each chunk in seconds
        overlap_sec: Overlap between consecutive chunks in seconds

    Yields:
        AudioChunk objects
    """
    chunk_samples = int(chunk_duration_sec * audio.sample_rate)
    overlap_samples = int(overlap_sec * audio.sample_rate)
    step_samples = chunk_samples - overlap_samples

    total_samples = len(audio.data)
    index = 0
    start_sample = 0

    while start_sample < total_samples:
        end_sample = min(start_sample + chunk_samples, total_samples)

        chunk_data = audio.data[start_sample:end_sample]

        # Pad with zeros if this is the last chunk and it's too short
        if len(chunk_data) < chunk_samples:
            chunk_data = np.pad(
                chunk_data,
                (0, chunk_samples - len(chunk_data)),
                mode="constant",
            )

        yield AudioChunk(
            data=chunk_data,
            sample_rate=audio.sample_rate,
            start_time=start_sample / audio.sample_rate,
            end_time=end_sample / audio.sample_rate,
            index=index,
        )

        index += 1
        start_sample += step_samples


def calculate_rms(audio_data: np.ndarray) -> float:
    """Calculate Root Mean Square (RMS) energy of audio."""
    return float(np.sqrt(np.mean(audio_data.astype(np.float64) ** 2)))


def detect_voice_activity(
    audio: AudioFile,
    threshold: float = 0.02,
    frame_duration_ms: float = 30.0,
    min_speech_duration_ms: float = 100.0,
) -> list[tuple[float, float]]:
    """
    Simple VAD (Voice Activity Detection) based on energy threshold.

    Args:
        audio: AudioFile to analyze
        threshold: RMS threshold for voice detection (0-1 normalized)
        frame_duration_ms: Analysis frame duration in milliseconds
        min_speech_duration_ms: Minimum speech segment duration

    Returns:
        List of (start_time, end_time) tuples for detected speech segments
    """
    frame_samples = int(frame_duration_ms * audio.sample_rate / 1000)
    min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)

    # Normalize audio to 0-1 range
    max_val = np.abs(audio.data).max()
    if max_val == 0:
        return []

    normalized = audio.data.astype(np.float64) / max_val

    # Calculate RMS for each frame
    num_frames = len(normalized) // frame_samples
    is_speech = []

    for i in range(num_frames):
        frame = normalized[i * frame_samples : (i + 1) * frame_samples]
        rms = np.sqrt(np.mean(frame ** 2))
        is_speech.append(rms > threshold)

    # Find speech segments
    segments = []
    in_speech = False
    speech_start = 0
    speech_frames = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            in_speech = True
            speech_start = i
            speech_frames = 1
        elif speech and in_speech:
            speech_frames += 1
        elif not speech and in_speech:
            if speech_frames >= min_speech_frames:
                start_time = speech_start * frame_duration_ms / 1000
                end_time = i * frame_duration_ms / 1000
                segments.append((start_time, end_time))
            in_speech = False
            speech_frames = 0

    # Handle case where speech extends to end
    if in_speech and speech_frames >= min_speech_frames:
        start_time = speech_start * frame_duration_ms / 1000
        end_time = len(is_speech) * frame_duration_ms / 1000
        segments.append((start_time, end_time))

    return segments


def extract_speech_segments(
    audio: AudioFile,
    segments: list[tuple[float, float]],
) -> list[AudioChunk]:
    """
    Extract speech segments from audio based on VAD output.

    Args:
        audio: Source audio file
        segments: List of (start_time, end_time) tuples from VAD

    Returns:
        List of AudioChunk objects containing speech
    """
    chunks = []

    for i, (start_time, end_time) in enumerate(segments):
        start_sample = int(start_time * audio.sample_rate)
        end_sample = int(end_time * audio.sample_rate)

        chunks.append(
            AudioChunk(
                data=audio.data[start_sample:end_sample],
                sample_rate=audio.sample_rate,
                start_time=start_time,
                end_time=end_time,
                index=i,
            )
        )

    return chunks
