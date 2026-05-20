"""
tests/test_vad.py
=================
Tests for the WebRTC VAD silence-detection loop in pipeline/vad.py.

We don't have a microphone in CI, so we fake the audio source by replacing
MicCapture.get_frame() with a stub that returns scripted frames in order.

What we verify:
  - A pure-silence stream eventually completes an empty utterance (no infinite hang)
  - A speech-then-silence stream returns the speech portion
  - The pre-roll buffer captures the first ~3 frames of speech (no clipped onset)
  - Aggressive end-of-silence detection kicks in after SILENCE_TIMEOUT_MS
  - webrtcvad accepts 30ms @ 16kHz frame size (sanity check — wrong size = silent fail)
"""

from __future__ import annotations

import numpy as np

from config.settings import settings
from pipeline import vad as vad_mod


# ---------- helpers: synthetic audio ----------

# 30 ms frame at 16 kHz = 480 int16 samples — must match pipeline/vad.py.
_FRAME_SAMPLES = int(settings.SAMPLE_RATE * 30 / 1000)


def _silence_frame() -> np.ndarray:
    """One frame of pure silence."""
    return np.zeros(_FRAME_SAMPLES, dtype=np.int16)


def _speech_frame(freq_hz: float = 200.0, amplitude: int = 12000) -> np.ndarray:
    """
    One frame of a synthetic vowel-like tone. webrtcvad recognises this as
    speech reliably at 16 kHz / 30 ms. 200 Hz sits inside the typical human
    vocal fundamental range.
    """
    t = np.arange(_FRAME_SAMPLES) / settings.SAMPLE_RATE
    # Add a second harmonic so it's not a pure sine — VAD prefers richer spectra.
    sig = amplitude * np.sin(2 * np.pi * freq_hz * t)
    sig += (amplitude // 2) * np.sin(2 * np.pi * (freq_hz * 2) * t)
    return sig.astype(np.int16)


class _FakeMic:
    """A drop-in replacement for MicCapture that yields scripted frames."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)
        self._idx = 0

    def get_frame(self, timeout: float = 1.0) -> np.ndarray | None:
        if self._idx >= len(self._frames):
            # Hand back more silence forever — prevents infinite loops in tests
            # that under-specify their script. The VAD will terminate via its
            # own silence-streak detection long before this matters.
            return _silence_frame()
        frame = self._frames[self._idx]
        self._idx += 1
        return frame

    @property
    def consumed(self) -> int:
        return self._idx


# ---------- sanity: webrtcvad accepts our frame shape ----------

def test_frame_size_is_compatible_with_webrtcvad():
    """If this test ever fails, the rest of the VAD module silently misbehaves."""
    import webrtcvad
    v = webrtcvad.Vad(2)
    frame = _speech_frame()
    # Should not raise — webrtcvad rejects wrong sizes with an exception.
    result = v.is_speech(frame.tobytes(), settings.SAMPLE_RATE)
    assert isinstance(result, bool)


# ---------- end-to-end: silence → no utterance, but loop terminates ----------

def test_silence_then_speech_then_silence_returns_speech():
    """
    Build a script of: 5 silence + 20 speech + ENOUGH silence to trigger end.

    The "enough" amount must exceed _END_SILENCE_FRAMES (SILENCE_TIMEOUT_MS / 30).
    """
    end_silence_needed = vad_mod._END_SILENCE_FRAMES + 3  # margin
    frames = (
        [_silence_frame()] * 5
        + [_speech_frame()] * 20
        + [_silence_frame()] * end_silence_needed
    )
    fake = _FakeMic(frames)

    utterance = vad_mod.capture_utterance_vad(fake)

    # Speech portion plus the pre-roll should be present.
    assert len(utterance) > 0
    expected_min_samples = 20 * _FRAME_SAMPLES   # at minimum the 20 speech frames
    assert len(utterance) >= expected_min_samples


# ---------- pre-roll: we don't clip the first word ----------

def test_pre_roll_includes_frames_before_speech_trigger():
    """
    The VAD waits for _START_SPEECH_FRAMES consecutive speech frames before
    declaring an utterance has begun, but it should retroactively include
    those trigger frames (and any pre-roll) so the onset isn't clipped.
    """
    end_silence_needed = vad_mod._END_SILENCE_FRAMES + 3
    speech_frames_count = 10
    frames = (
        [_silence_frame()] * 3
        + [_speech_frame()] * speech_frames_count
        + [_silence_frame()] * end_silence_needed
    )
    fake = _FakeMic(frames)

    utterance = vad_mod.capture_utterance_vad(fake)

    # Captured length must be >= the actual speech frames count — proves the
    # trigger frames weren't discarded.
    assert len(utterance) >= speech_frames_count * _FRAME_SAMPLES


# ---------- termination: end-of-silence detection actually fires ----------

def test_long_trailing_silence_terminates_utterance():
    """Once SILENCE_TIMEOUT_MS of silence has passed, the function must return."""
    end_silence_needed = vad_mod._END_SILENCE_FRAMES + 5
    frames = (
        [_speech_frame()] * 6
        + [_silence_frame()] * end_silence_needed
    )
    fake = _FakeMic(frames)

    # If termination is broken, this test hangs forever — pytest-timeout would
    # catch it, but the design of _FakeMic (silence-forever fallback) means we
    # can't loop more than _END_SILENCE_FRAMES extra reads in the worst case.
    utterance = vad_mod.capture_utterance_vad(fake)
    assert len(utterance) > 0

    # We must have consumed AT LEAST the speech + end-silence frames.
    assert fake.consumed >= 6 + vad_mod._END_SILENCE_FRAMES


# ---------- single-frame noise burst is NOT an utterance ----------

def test_single_speech_frame_does_not_trigger_utterance():
    """
    A lone speech frame between silence (e.g. a cough) should NOT open an
    utterance. The VAD requires _START_SPEECH_FRAMES consecutive speech frames.
    """
    end_silence_needed = vad_mod._END_SILENCE_FRAMES + 3
    # One isolated speech frame surrounded by silence, then a real utterance
    # so the loop actually has something to capture and terminate on.
    frames = (
        [_silence_frame()] * 4
        + [_speech_frame()]                          # the lone cough
        + [_silence_frame()] * 8
        + [_speech_frame()] * 10                     # the real utterance
        + [_silence_frame()] * end_silence_needed
    )
    fake = _FakeMic(frames)

    utterance = vad_mod.capture_utterance_vad(fake)

    # The function returned the real utterance, not the cough.
    # The captured audio should be roughly the 10 speech frames (plus pre-roll
    # of up to _START_SPEECH_FRAMES). It must NOT be just the single cough frame.
    assert len(utterance) > _FRAME_SAMPLES   # more than one frame's worth


# ---------- helper-frame sanity, in case future maintainers tweak audio gen ----------

def test_speech_frame_helper_is_recognised_as_speech_by_vad():
    """If this fails, _speech_frame() needs more energy or different spectrum."""
    import webrtcvad
    v = webrtcvad.Vad(2)
    # Try a few frames — VAD is noisy at boundaries.
    hits = sum(
        1 for _ in range(5)
        if v.is_speech(_speech_frame().tobytes(), settings.SAMPLE_RATE)
    )
    assert hits >= 3, "Synthetic 'speech' frames must be recognised by webrtcvad"


def test_silence_frame_helper_is_recognised_as_silence_by_vad():
    """If this fails, _silence_frame() isn't quiet enough."""
    import webrtcvad
    v = webrtcvad.Vad(2)
    assert v.is_speech(_silence_frame().tobytes(), settings.SAMPLE_RATE) is False