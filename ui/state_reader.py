"""
ui/state_reader.py
==================
Reads tmp/state.json on every Streamlit rerun.

We deliberately do NOT use @st.cache_data here — caching would defeat the whole
point of polling. The file is small (~tens of KB) and the read is fast (<1ms),
so re-reading on every refresh is fine.

Returns a normalised dict with safe defaults so the components don't have to
defensively-check every field.
"""

from __future__ import annotations

from typing import Any

from utils.state_bridge import read_state


_DEFAULT_STATE: dict[str, Any] = {
    "status": "⏳ waiting for pipeline…",
    "transcript": "",
    "confidence": 0.0,
    "intent": None,
    "action": None,
    "cmd_result": None,
    "aider_response": None,
    "error": None,
    "ui_log": [],
    "history": [],
    "last_update": 0.0,
}


def load_state() -> dict[str, Any]:
    """
    Read state.json and fill in defaults for any missing keys. Always returns
    a usable dict — the UI never has to handle None.
    """
    raw = read_state()
    if raw is None:
        return dict(_DEFAULT_STATE)

    merged = dict(_DEFAULT_STATE)
    merged.update(raw)
    return merged