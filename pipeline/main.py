"""
pipeline/main.py
================
Entry point for the voice process.

What it does, in order, forever:

  1. Open the mic (MicCapture) and optionally arm PTT or VAD.
  2. Capture one utterance (PTT release, or VAD silence trigger).
  3. Encode PCM → WAV bytes.
  4. Build initial VoiceState and invoke the compiled LangGraph.
  5. write_state_node inside the graph publishes results to tmp/state.json.
  6. Loop. Until aider's /exit slash command takes the subprocess down,
     or the user Ctrl-C's.

Run with:
    python -m pipeline.main
or:
    python pipeline/main.py
"""

from __future__ import annotations

import signal
import sys

import numpy as np

from config.settings import settings
from graph import build_graph, new_state
from graph.nodes import aider_session
from pipeline.capture import MicCapture, PTTListener, capture_ptt
from pipeline.vad import capture_utterance_vad
from utils.audio_utils import pcm_to_wav_bytes
from utils.logger import get_logger
from utils.state_bridge import write_state

log = get_logger(__name__)


# Minimum utterance length we'll even send to Whisper.
# 0.3s of audio is ~5000 samples at 16kHz — anything shorter is almost
# certainly a key-bounce or mic blip, not real speech.
_MIN_UTTERANCE_SAMPLES = int(0.3 * settings.SAMPLE_RATE)


def _publish_status(message: str) -> None:
    """
    Update state.json with an out-of-band status line so the UI shows
    `🎙️  listening…` even before the first graph run completes.
    """
    try:
        write_state({"status": message, "last_update": _now()})
    except Exception:
        log.exception("Could not publish status")


def _now() -> float:
    import time
    return time.time()


def _shutdown(signum: int, frame) -> None:
    """SIGINT / SIGTERM handler — close aider gracefully before exit."""
    log.info("Signal %d received — shutting down…", signum)
    try:
        aider_session.stop()
    except Exception:
        log.exception("Error stopping aider_session")
    sys.exit(0)


def run() -> None:
    """Main pipeline loop."""
    log.info("=" * 60)
    log.info("voice-aider pipeline starting")
    log.info("  STT model     : %s", settings.GROQ_STT_MODEL)
    log.info("  LLM model     : %s", settings.GROQ_LLM_MODEL)
    log.info("  Aider model   : %s", settings.AIDER_MODEL)
    log.info("  Mode          : %s", "hands-free (VAD)" if settings.HANDS_FREE_MODE else f"PTT ({settings.PUSH_TO_TALK_KEY})")
    log.info("  Confidence    : %.2f (≤%d retries)", settings.CONFIDENCE_THRESHOLD, settings.MAX_STT_RETRIES)
    log.info("=" * 60)

    # Wire signal handlers up-front so Ctrl-C cleanly stops aider.
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    graph = build_graph()
    mic = MicCapture()

    # Pre-warm aider so the first command (clear/undo/exit) works.
    # Without this, command-as-first-utterance crashes because execute_cmd
    # tries to send /clear to a subprocess that hasn't been started.
    log.info("Pre-warming aider subprocess (this takes ~3-5s)…")
    _publish_status("⏳ starting aider…")
    try:
        aider_session.start()
        log.info("Aider ready.")
    except FileNotFoundError:
        log.error("aider executable not found — install with `pip install aider-chat`")
        return
    except Exception as e:
        log.exception("Aider failed to start: %s", e)
        return

    try:
        mic.start()
        _publish_status("🎙️  ready — speak now" if settings.HANDS_FREE_MODE
                        else f"🎙️  hold {settings.PUSH_TO_TALK_KEY.upper()} to talk")

        if settings.HANDS_FREE_MODE:
            _run_vad_loop(mic, graph)
        else:
            _run_ptt_loop(mic, graph)

    finally:
        mic.stop()
        aider_session.stop()
        log.info("pipeline stopped")


def _run_ptt_loop(mic: MicCapture, graph) -> None:
    with PTTListener() as ptt:
        while True:
            utterance = capture_ptt(mic, ptt)
            _handle_utterance(utterance, graph)


def _run_vad_loop(mic: MicCapture, graph) -> None:
    while True:
        utterance = capture_utterance_vad(mic)
        _handle_utterance(utterance, graph)


def _handle_utterance(pcm: np.ndarray, graph) -> None:
    """Encode + invoke + log one utterance worth of audio."""
    if len(pcm) < _MIN_UTTERANCE_SAMPLES:
        log.info("Utterance too short (%d samples) — skipping", len(pcm))
        _publish_status("⚠️  utterance too short — try again")
        return

    wav_bytes = pcm_to_wav_bytes(pcm)
    state_in = new_state(wav_bytes)

    log.info("▶ invoking graph (%d audio bytes)", len(wav_bytes))
    try:
        final_state = graph.invoke(state_in)
    except Exception:
        log.exception("Graph invocation failed")
        _publish_status("❌ pipeline error — see terminal log")
        return

    transcript = final_state.get("transcript", "")
    intent = final_state.get("intent", "?")
    action = final_state.get("action") or ""
    log.info("◀ done: intent=%s action=%s transcript=%r", intent, action, transcript)

    # Honour the voice "exit"/"stop" command: tear down the whole session.
    if action in ("exit", "stop"):
        log.info("Voice exit command received — shutting down session.")
        _publish_status("👋 exiting — goodbye")
        # The signal handler does the cleanup; SIGINT triggers it gracefully.
        import os, signal as _signal
        os.kill(os.getpid(), _signal.SIGINT)


if __name__ == "__main__":
    run()