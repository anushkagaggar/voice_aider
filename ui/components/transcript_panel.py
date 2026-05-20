"""
ui/components/transcript_panel.py
=================================
Renders the most recent voice turn:
  - transcript text in a quote block
  - confidence score with a coloured progress bar
  - resolved intent + action badge
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def _confidence_color(score: float) -> str:
    """Pick a colour name based on the confidence bucket."""
    if score >= 0.85:
        return "green"
    if score >= 0.65:
        return "orange"
    return "red"


def _intent_badge(intent: str | None, action: str | None) -> str:
    if not intent or intent == "unknown":
        return ":gray-badge[unknown]"
    if intent == "cmd":
        label = f"cmd · {action}" if action else "cmd"
        return f":blue-badge[{label}]"
    if intent == "prompt":
        return ":green-badge[prompt → aider]"
    return f":gray-badge[{intent}]"


def render(state: dict[str, Any]) -> None:
    st.subheader("Latest turn")

    transcript = state.get("transcript") or ""
    confidence = float(state.get("confidence") or 0.0)
    intent = state.get("intent")
    action = state.get("action")

    if not transcript:
        st.caption("No utterance captured yet — speak to begin.")
        return

    # Transcript
    st.markdown(f"> {transcript}")

    # Confidence + intent on one row
    col_conf, col_intent = st.columns([3, 2])
    with col_conf:
        color = _confidence_color(confidence)
        st.markdown(
            f"**Confidence:** :{color}[{confidence:.0%}]"
        )
        st.progress(min(max(confidence, 0.0), 1.0))
    with col_intent:
        st.markdown(f"**Intent:** {_intent_badge(intent, action)}")