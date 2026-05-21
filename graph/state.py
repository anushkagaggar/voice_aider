"""
graph/state.py
==============
The shared state schema for the LangGraph state machine. Every node receives
this dict, mutates a subset of fields, and returns the partial update — that's
LangGraph's reducer pattern.

Fields are grouped by which node populates them:
  - audio_bytes       : pipeline/vad.py (entry)
  - transcript, ...   : stt_node
  - retry_count       : confidence_node
  - intent, action    : classify_node
  - aider_response    : aider_node
  - error             : any node on failure
  - ui_log            : every node appends a status line for the dashboard
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages  # noqa: F401  (reserved for future)

# Use the built-in list reducer pattern: every node returning `{"ui_log": ["x"]}`
# appends, not replaces. We implement this via Annotated + a plain `+` reducer
# below.


def _append(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Reducer for list-typed state fields: appends right to left."""
    return (left or []) + (right or [])


class VoiceState(TypedDict, total=False):
    """All fields are optional (`total=False`) so partial updates work cleanly."""

    # ---- Audio input ----
    audio_bytes: bytes              # raw WAV from pipeline/vad.py

    # ---- STT outputs ----
    transcript: str                 # Groq Whisper transcribed text
    confidence: float               # 0.0 … 1.0 confidence proxy
    audio_duration: float           # seconds — for logging

    # ---- Retry control ----
    retry_count: int                # incremented by confidence_node; max = MAX_STT_RETRIES

    # ---- Retry control ----
    retry_count: int                # incremented by confidence_node; max = MAX_STT_RETRIES
    should_retry: bool              # set by confidence_node, read by route_after_confidence

    # ---- Classification outputs ----
    intent: Literal["cmd", "prompt", "unknown"]
    action: Optional[str]           # resolved command name, or None

    # ---- Action outputs ----
    aider_response: Optional[str]   # last stdout chunk from aider
    cmd_result: Optional[str]       # "session cleared", "exited", etc.

    # ---- Diagnostics ----
    error: Optional[str]            # node name + message if anything failed
    ui_log: Annotated[list[str], _append]   # append-only audit trail


def new_state(audio_bytes: bytes) -> VoiceState:
    """Factory used by pipeline/main.py to seed each graph invocation."""
    return VoiceState(
        audio_bytes=audio_bytes,
        transcript="",
        confidence=0.0,
        audio_duration=0.0,
        retry_count=0,
        should_retry=False,
        intent="unknown",
        action=None,
        aider_response=None,
        cmd_result=None,
        error=None,
        ui_log=[],
    )