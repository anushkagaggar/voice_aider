"""
ui/components/history_panel.py
==============================
Renders the rolling history list built by write_state_node.

Each entry is one full voice turn:
  - timestamp
  - transcript
  - intent + action
  - one-line summary of the result

Most recent first. Capped to 20 entries on-screen even though the JSON may
hold up to 50, to keep the UI scannable.
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st


_VISIBLE_LIMIT = 20


def _short(text: str | None, n: int = 80) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ⏎ ")
    return text if len(text) <= n else text[:n] + "…"


def _intent_glyph(intent: str | None) -> str:
    return {
        "cmd": "🛠️",
        "prompt": "💬",
        "unknown": "❓",
    }.get(intent or "unknown", "•")


def render(state: dict[str, Any]) -> None:
    st.subheader("History")

    history = list(state.get("history") or [])
    if not history:
        st.caption("No turns yet.")
        return

    # Newest first, capped.
    history = list(reversed(history))[:_VISIBLE_LIMIT]

    for turn in history:
        ts = turn.get("timestamp") or 0
        when = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "—"
        glyph = _intent_glyph(turn.get("intent"))
        transcript = _short(turn.get("transcript"))
        action = turn.get("action")
        cmd_result = _short(turn.get("cmd_result"), 60)
        aider_response = _short(turn.get("aider_response"), 100)
        confidence = float(turn.get("confidence") or 0.0)

        with st.container(border=True):
            top = f"`{when}`  {glyph}  *{transcript or '(empty)'}*"
            st.markdown(top)
            meta_bits: list[str] = [f"conf {confidence:.0%}"]
            if action:
                meta_bits.append(f"action: `{action}`")
            if cmd_result:
                meta_bits.append(f"→ {cmd_result}")
            elif aider_response:
                meta_bits.append(f"→ {aider_response}")
            if turn.get("error"):
                meta_bits.append(f"⚠️ {turn['error']}")
            st.caption(" · ".join(meta_bits))