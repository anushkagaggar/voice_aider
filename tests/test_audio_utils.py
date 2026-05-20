"""
tests/test_audio_utils.py
=========================
Unit tests for the WAV encoder. We don't have a real mic in CI, so we generate
synthetic PCM (silence + a sine tone) and verify:

  - WAV header is correct (RIFF magic, sample rate, channels, bit depth)
  - byte count matches the input sample count
  - duration_seconds returns the expected wall-clock duration
"""

from __future__ import annotations

import io
import wave

import numpy as np

from config.settings import settings
from utils.audio_utils import duration_seconds, pcm_to_wav_bytes, wav_bytes_to_file_tuple


def _silence(seconds: float, sample_rate: int | None = None) -> np.ndarray:
    sr = sample_rate or settings.SAMPLE_RATE
    return np.zeros(int(sr * seconds), dtype=np.int16)


def _sine(seconds: float, freq_hz: float = 440.0, sample_rate: int | None = None) -> np.ndarray:
    sr = sample_rate or settings.SAMPLE_RATE
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (np.sin(2 * np.pi * freq_hz * t) * 16000).astype(np.int16)


def test_pcm_to_wav_bytes_starts_with_riff_header():
    pcm = _silence(0.5)
    wav = pcm_to_wav_bytes(pcm)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


def test_pcm_to_wav_bytes_round_trips_via_wave_module():
    """Encode → decode → compare sample-for-sample."""
    pcm = _sine(0.25)
    wav = pcm_to_wav_bytes(pcm)

    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2          # int16 = 2 bytes
        assert wf.getframerate() == settings.SAMPLE_RATE
        frames = wf.readframes(wf.getnframes())

    decoded = np.frombuffer(frames, dtype=np.int16)
    assert np.array_equal(decoded, pcm)


def test_pcm_to_wav_bytes_handles_float32_input():
    """sounddevice can return float32; the encoder should clip + cast."""
    pcm_float = np.array([0.5, -0.5, 0.0, 1.0, -1.0], dtype=np.float32)
    wav = pcm_to_wav_bytes(pcm_float)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getsampwidth() == 2


def test_duration_seconds_matches_input_length():
    pcm = _silence(1.5)
    assert abs(duration_seconds(pcm) - 1.5) < 0.001


def test_wav_bytes_to_file_tuple_shape():
    wav = pcm_to_wav_bytes(_silence(0.1))
    name, fileobj, mime = wav_bytes_to_file_tuple(wav)
    assert name.endswith(".wav")
    assert mime == "audio/wav"
    # The fileobj must be seekable so the Groq SDK can replay it on retry.
    fileobj.seek(0)
    assert fileobj.read(4) == b"RIFF"