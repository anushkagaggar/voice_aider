"""
pipeline/capture.py
===================
Microphone capture with push-to-talk (PTT) semantics.

Two capture modes, both yielding int16 mono PCM at SAMPLE_RATE:

  - capture_ptt()  — holds-down PTT. Press the configured key (default SPACE)
                     to start, release to stop. Returns one utterance per call.
  - capture_vad()  — hands-free. Pulls audio continuously and hands frames to
                     vad.py for silence detection. Implemented in vad.py to
                     keep this file focused on raw I/O.

Design notes:
  - sounddevice's InputStream uses a PortAudio callback. We push frames into
    a queue.Queue so the main thread can pull them deterministically.
  - pynput's keyboard listener runs in its own thread; we use threading.Event
    for the PTT signal rather than polling.
  - PTT key is configurable via settings.PUSH_TO_TALK_KEY (space/ctrl/alt).
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Final

import numpy as np
import sounddevice as sd
from pynput import keyboard

from config.settings import settings
from utils.logger import get_logger

log = get_logger(__name__)

# Frame size in samples. 30ms at 16kHz = 480 samples. Matches webrtcvad's
# expected frame durations (10/20/30ms) so vad.py can consume the same stream.
_FRAME_MS: Final[int] = 30
_FRAME_SAMPLES: Final[int] = int(settings.SAMPLE_RATE * _FRAME_MS / 1000)

# Map of friendly name → pynput Key enum.
_PTT_KEYS = {
    "space": keyboard.Key.space,
    "ctrl":  keyboard.Key.ctrl,
    "alt":   keyboard.Key.alt,
}


class MicCapture:
    """
    Holds an open InputStream and a frame queue. One instance per pipeline.

    Why a class? sounddevice's InputStream is a context manager but we want
    to keep the stream open for the lifetime of the pipeline — opening/closing
    the device for each utterance adds ~100ms latency and risks driver flakes.
    """

    def __init__(self) -> None:
        self._frame_q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None

    def _on_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """PortAudio callback — runs in the audio thread. Keep it fast!"""
        if status:
            log.debug("sounddevice status: %s", status)
        # indata is shape (frames, 1) float32 by default. We requested int16
        # via dtype="int16" below, so it'll be int16. Copy() because PortAudio
        # reuses the buffer.
        self._frame_q.put(indata[:, 0].copy())

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=settings.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=_FRAME_SAMPLES,
            callback=self._on_audio,
        )
        self._stream.start()
        log.info("MicCapture started: %d Hz, %d-sample frames", settings.SAMPLE_RATE, _FRAME_SAMPLES)

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        # Drain any leftover frames so the next start() begins clean.
        while not self._frame_q.empty():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                break

    def get_frame(self, timeout: float = 1.0) -> np.ndarray | None:
        """Block until one frame is available, or return None on timeout."""
        try:
            return self._frame_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Discard any queued frames — useful right before a new utterance."""
        while not self._frame_q.empty():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                return


# -----------------------------------------------------------------------------
# Push-to-talk via pynput
# -----------------------------------------------------------------------------

class PTTListener:
    """
    Listens for the configured PTT key. While held: `is_active` is True.
    On release: sets `released_event`.

    Designed to be used as a context manager so the keyboard listener thread
    is always cleaned up.
    """

    def __init__(self) -> None:
        self._key = _PTT_KEYS[settings.PUSH_TO_TALK_KEY]
        self.is_active = False
        self.pressed_event = threading.Event()
        self.released_event = threading.Event()
        self._listener: keyboard.Listener | None = None

    def _on_press(self, key) -> None:
        if key == self._key and not self.is_active:
            self.is_active = True
            self.released_event.clear()
            self.pressed_event.set()
            log.debug("PTT pressed")

    def _on_release(self, key) -> None:
        if key == self._key and self.is_active:
            self.is_active = False
            self.pressed_event.clear()
            self.released_event.set()
            log.debug("PTT released")

    def __enter__(self) -> "PTTListener":
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,  # don't eat the key from other apps
        )
        self._listener.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


# -----------------------------------------------------------------------------
# High-level capture function
# -----------------------------------------------------------------------------

def capture_ptt(mic: MicCapture, ptt: PTTListener) -> np.ndarray:
    """
    Block until the PTT key is pressed, then collect frames until it's released.

    Returns a single int16 numpy array (the full utterance).

    Caller is responsible for starting `mic` and entering `ptt` as a context
    manager before invoking this.
    """
    log.info("Waiting for PTT (%s key)…", settings.PUSH_TO_TALK_KEY)
    ptt.pressed_event.wait()

    # Clear any stale frames from before the user pressed — otherwise the
    # utterance starts with random ambient noise.
    mic.drain()

    log.info("Recording…")
    chunks: list[np.ndarray] = []
    while ptt.is_active:
        frame = mic.get_frame(timeout=0.1)
        if frame is not None:
            chunks.append(frame)

    # PTT released. Grab any frames the callback queued in the last few ms
    # (avoids cutting off the last word).
    deadline = time.monotonic() + 0.15
    while time.monotonic() < deadline:
        frame = mic.get_frame(timeout=0.05)
        if frame is None:
            break
        chunks.append(frame)

    if not chunks:
        log.warning("PTT released with no audio captured")
        return np.zeros(0, dtype=np.int16)

    utterance = np.concatenate(chunks).astype(np.int16, copy=False)
    log.info("Captured %.2fs of audio (%d samples)", len(utterance) / settings.SAMPLE_RATE, len(utterance))
    return utterance