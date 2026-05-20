"""
graph/nodes/stt_node.py
=======================
First node in the graph. Takes raw WAV bytes from state and calls Groq Whisper
to fill `transcript`, `confidence`, and `audio_duration`.

The node itself is intentionally thin — all the API quirks (confidence proxy
math, error swallowing, SDK normalization) live in groq_clients/whisper_client.py.
This file just adapts the (input dict) -> (partial dict) contract LangGraph wants.
"""

from __future__ import annotations

from graph.state import VoiceState
from groq_clients.whisper_client import transcribe
from utils.logger import get_logger

log = get_logger(__name__)


def stt_node(state: VoiceState) -> VoiceState:
    """
    Transcribe the audio in state.audio_bytes. Returns a partial state update.

    LangGraph merges this partial dict into the running state — we don't have
    to copy the input fields through.
    """
    audio = state.get("audio_bytes") or b""
    if not audio:
        log.warning("stt_node received empty audio")
        return VoiceState(
            transcript="",
            confidence=0.0,
            audio_duration=0.0,
            error="stt_node: no audio in state",
            ui_log=["⚠️  stt_node: no audio received"],
        )

    result = transcribe(audio)

    log_line = (
        f"📝 transcript={result.text!r}  "
        f"conf={result.confidence:.2f}  "
        f"dur={result.duration:.1f}s"
    )

    return VoiceState(
        transcript=result.text,
        confidence=result.confidence,
        audio_duration=result.duration,
        error=None,
        ui_log=[log_line],
    )