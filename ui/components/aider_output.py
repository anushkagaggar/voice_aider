"""
ui/components/aider_output.py
=============================
Renders the most recent aider reply or cmd result.

Aider's output is largely markdown — code fences, headers, file paths — so we
just pass it through st.markdown. For control commands, we show a small
confirmation block instead.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render(state: dict[str, Any]) -> None:
    st.subheader("Agent output")

    intent = state.get("intent")
    aider_response = state.get("aider_response")
    cmd_result = state.get("cmd_result")

    if intent == "cmd" and cmd_result:
        # Show the command's effect as a one-liner.
        st.info(f"✅ {cmd_result}", icon="🛠️")
        return

    if intent == "prompt" and aider_response:
        # aider output is markdown — render natively.
        with st.container(border=True):
            st.markdown(aider_response)
        return

    st.caption("Waiting for the first response from aider…")