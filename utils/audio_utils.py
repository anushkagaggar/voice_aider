"""
utils/audio_utils.py
====================
Audio buffer helpers. The pipeline captures raw PCM samples via sounddevice
(int16 numpy arrays at SAMPLE_RATE Hz mono). Groq's Whisper endpoint wants a
file-like object containing a real WAV header, so we encode in-memory here.

Two functions:
  - pcm_to_wav_bytes: int16 numpy -> bytes with WAV header
  - wav_bytes_to_file_tuple: bytes -> (filename, BytesIO, mime) tuple for the
    groq SDK's `file=` argument
"""

import io
import wave
from typing import Final

import numpy as np

from config.settings import settings

_SAMPLE_WIDTH_BYTES: Final[int] = 2  # int16
_CHANNELS: Final[int] = 1            # mono


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int | None = None) -> bytes:
    """
    Encode an int16 mono PCM buffer as a WAV byte string.

    Args:
        pcm: 1-D numpy array, dtype int16.
        sample_rate: Hz. Defaults to settings.SAMPLE_RATE.

    Returns:
        Bytes containing a complete RIFF/WAVE file.
    """
    if pcm.dtype != np.int16:
        # If we ever get float32 from sounddevice, clip + cast.
        pcm = np.clip(pcm * 32767, -32768, 32767).astype(np.int16)

    sr = sample_rate or settings.SAMPLE_RATE
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_SAMPLE_WIDTH_BYTES)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_bytes_to_file_tuple(
    wav_bytes: bytes, filename: str = "utterance.wav"
) -> tuple[str, io.BytesIO, str]:
    """
    Wrap WAV bytes in the (name, fileobj, mime) tuple shape that the
    Groq Python SDK expects for `client.audio.transcriptions.create(file=...)`.
    """
    return (filename, io.BytesIO(wav_bytes), "audio/wav")


def duration_seconds(pcm: np.ndarray, sample_rate: int | None = None) -> float:
    """Length of an int16 PCM buffer in seconds. Useful for log lines."""
    sr = sample_rate or settings.SAMPLE_RATE
    return len(pcm) / sr