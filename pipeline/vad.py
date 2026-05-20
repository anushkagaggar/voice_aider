"""
pipeline/vad.py
===============
WebRTC voice-activity detection for hands-free mode.

WebRTC's VAD is a tiny GMM-based classifier that returns is_speech() per frame
(10/20/30 ms at 8/16/32/48 kHz). It's not as accurate as Silero, but it has
zero deep-learning dependencies and runs in microseconds — perfect for a
real-time pipeline.

Capture loop in hands-free mode:
  1. Start streaming frames from MicCapture.
  2. Wait until we see >= START_SPEECH_FRAMES consecutive speech frames
     (avoids triggering on a single cough).
  3. From then on, collect frames into the utterance buffer.
  4. End the utterance when we see SILENCE_TIMEOUT_MS of continuous silence.

Returns one int16 numpy array per utterance, same shape as capture_ptt().
"""

from __future__ import annotations

import numpy as np
import webrtcvad

from config.settings import settings
from pipeline.capture import MicCapture
from utils.logger import get_logger

log = get_logger(__name__)

# 30 ms frames at SAMPLE_RATE — matches MicCapture's blocksize.
_FRAME_MS = 30
_FRAME_SAMPLES = int(settings.SAMPLE_RATE * _FRAME_MS / 1000)

# How many consecutive speech frames before we open an utterance.
# 3 frames @ 30ms = 90ms — long enough to skip a click or breath pop.
_START_SPEECH_FRAMES = 3

# How many consecutive silence frames before we close an utterance.
# Derived from settings.SILENCE_TIMEOUT_MS so it's tunable from .env.
_END_SILENCE_FRAMES = max(1, settings.SILENCE_TIMEOUT_MS // _FRAME_MS)


def _is_speech(vad: webrtcvad.Vad, frame: np.ndarray) -> bool:
    """webrtcvad expects raw bytes, not numpy. One conversion per frame."""
    if len(frame) != _FRAME_SAMPLES:
        # Tail end of a stream may be shorter — treat as non-speech to avoid IndexError.
        return False
    return vad.is_speech(frame.tobytes(), settings.SAMPLE_RATE)


def capture_utterance_vad(mic: MicCapture) -> np.ndarray:
    """
    Block until a complete utterance is detected, then return it.

    A complete utterance starts when speech is detected and ends after
    settings.SILENCE_TIMEOUT_MS of continuous silence.

    Caller is responsible for starting `mic` first.
    """
    vad = webrtcvad.Vad(settings.VAD_AGGRESSIVENESS)
    log.info(
        "VAD armed: aggressiveness=%d, end-silence=%dms (%d frames)",
        settings.VAD_AGGRESSIVENESS,
        settings.SILENCE_TIMEOUT_MS,
        _END_SILENCE_FRAMES,
    )

    speech_streak = 0       # consecutive speech frames seen so far (pre-trigger)
    silence_streak = 0      # consecutive silence frames seen so far (post-trigger)
    in_utterance = False
    buffer: list[np.ndarray] = []
    # A small pre-roll buffer so we don't chop off the first ~100ms of the utterance.
    pre_roll: list[np.ndarray] = []
    _PRE_ROLL_FRAMES = _START_SPEECH_FRAMES

    while True:
        frame = mic.get_frame(timeout=1.0)
        if frame is None:
            # Mic stalled — keep trying.
            continue

        is_speech = _is_speech(vad, frame)

        if not in_utterance:
            # ----- pre-trigger: looking for start of speech -----
            pre_roll.append(frame)
            if len(pre_roll) > _PRE_ROLL_FRAMES:
                pre_roll.pop(0)

            if is_speech:
                speech_streak += 1
                if speech_streak >= _START_SPEECH_FRAMES:
                    log.info("Utterance start detected")
                    in_utterance = True
                    buffer.extend(pre_roll)   # include the pre-roll
                    pre_roll.clear()
                    silence_streak = 0
            else:
                speech_streak = 0

        else:
            # ----- in-utterance: collecting until silence -----
            buffer.append(frame)
            if is_speech:
                silence_streak = 0
            else:
                silence_streak += 1
                if silence_streak >= _END_SILENCE_FRAMES:
                    log.info("Utterance end detected (silence)")
                    break

    if not buffer:
        return np.zeros(0, dtype=np.int16)

    utterance = np.concatenate(buffer).astype(np.int16, copy=False)
    log.info("VAD captured %.2fs of audio", len(utterance) / settings.SAMPLE_RATE)
    return utterance