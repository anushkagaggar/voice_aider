"""
ui/components/status_bar.py
===========================
Top status strip. Shows:
  - current mode (PTT / hands-free)
  - pipeline status message
  - last update timestamp
  - error banner (if any)
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from config.settings import settings


def _format_age(ts: float) -> str:
    """Render a 'how long ago' string for the last-update timestamp."""
    if ts <= 0:
        return "never"
    age = time.time() - ts
    if age < 2:
        return "just now"
    if age < 60:
        return f"{int(age)}s ago"
    if age < 3600:
        return f"{int(age // 60)}m ago"
    return f"{int(age // 3600)}h ago"


def render(state: dict[str, Any]) -> None:
    mode = "🎙️ hands-free (VAD)" if settings.HANDS_FREE_MODE else f"🎙️ PTT — hold {settings.PUSH_TO_TALK_KEY.upper()}"
    status = state.get("status") or "ready"
    last_update = _format_age(state.get("last_update", 0.0))

    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        st.markdown(f"**{mode}**")
    with col2:
        st.markdown(f"**Status:** {status}")
    with col3:
        st.caption(f"updated {last_update}")

    error = state.get("error")
    if error:
        st.error(f"⚠️ {error}", icon="🚨")