"""
graph/nodes/write_state_node.py
===============================
Terminal node of the graph. Serialises the relevant slice of state to
tmp/state.json so the Streamlit UI process can render it.

We deliberately strip a few fields from the payload:
  - audio_bytes : binary, not JSON-serialisable and not useful to the UI
  - retry_count : internal control, no UI surface

Everything else goes through, including ui_log (rendered as the live feed).
"""

from __future__ import annotations

import time

from graph.state import VoiceState
from utils.logger import get_logger
from utils.state_bridge import read_state, write_state

log = get_logger(__name__)


# Fields that should NEVER appear in state.json.
_EXCLUDE = {"audio_bytes", "retry_count"}


def write_state_node(state: VoiceState) -> VoiceState:
    """
    Merge this run's outputs with the previous state.json contents (so the
    history accumulates across utterances), then atomically rewrite the file.
    """
    # Pull the slice we want to publish. dict() comprehension over a TypedDict
    # is safe — TypedDict is a dict at runtime.
    publishable = {k: v for k, v in state.items() if k not in _EXCLUDE}

    # Append the latest turn into a "history" list across runs.
    previous = read_state() or {}
    history = list(previous.get("history", []))

    turn = {
        "timestamp": time.time(),
        "transcript": state.get("transcript", ""),
        "intent": state.get("intent"),
        "action": state.get("action"),
        "cmd_result": state.get("cmd_result"),
        "aider_response": state.get("aider_response"),
        "confidence": state.get("confidence", 0.0),
        "error": state.get("error"),
    }
    history.append(turn)
    # Cap history to last 50 turns — keeps state.json under ~200 KB.
    history = history[-50:]

    payload = {
        **publishable,
        "history": history,
        "last_update": time.time(),
    }

    try:
        write_state(payload)
    except Exception as e:
        log.exception("write_state_node: failed to persist state.json")
        return VoiceState(
            error=f"write_state_node: {e}",
            ui_log=[f"❌ failed to write state.json: {e}"],
        )

    log.debug("write_state_node: state.json updated (history=%d)", len(history))
    return VoiceState(ui_log=["💾 state.json flushed"])