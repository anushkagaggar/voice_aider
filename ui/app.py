"""
ui/app.py
=========
Streamlit entrypoint for the dashboard process.

Run with:
    streamlit run ui/app.py

Layout (top → bottom):
  ┌─────────────────────────────────────────────────────────────────┐
  │  status_bar      (mode · status · last-update · error)          │
  ├──────────────────────────────────┬──────────────────────────────┤
  │  transcript_panel                │  aider_output                │
  │  (latest turn + confidence)      │  (markdown reply / cmd ack)  │
  ├──────────────────────────────────┴──────────────────────────────┤
  │  history_panel   (rolling list of past turns)                   │
  └─────────────────────────────────────────────────────────────────┘

Refresh model:
  - st_autorefresh polls state.json every 1 s.
  - The pipeline process writes state.json atomically (utils/state_bridge).
  - No direct shared memory — these are two completely independent processes.
"""

from __future__ import annotations

# ---- sys.path bootstrap ----
# Streamlit launches this script with its own working directory and may strip
# the project root from sys.path. We restore it explicitly so the project's
# packages always resolve, no matter how the user launched the app.
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from ui.components import aider_output, history_panel, status_bar, transcript_panel
from ui.state_reader import load_state


# -------- page config (must be the first Streamlit call) --------
st.set_page_config(
    page_title="voice-aider",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------- polling --------
# 1 s interval. Quick enough to feel live; slow enough that the JSON reader
# never collides with the writer. Key is required so the component preserves
# its counter across reruns.
st_autorefresh(interval=1000, key="state_poll")


def main() -> None:
    state = load_state()

    # Top: status bar
    status_bar.render(state)
    st.divider()

    # Middle: transcript + agent output side-by-side
    col_left, col_right = st.columns([5, 7], gap="large")
    with col_left:
        transcript_panel.render(state)
    with col_right:
        aider_output.render(state)

    st.divider()

    # Bottom: rolling history
    history_panel.render(state)

    # Sidebar: command cheat-sheet so the user knows what to say.
    with st.sidebar:
        st.markdown("### Voice commands")
        st.markdown(
            """
            **Session control**
            - "clear" — clear the chat
            - "undo" — revert last edit
            - "save" — commit changes
            - "exit" — stop the session

            **UI**
            - "show history"
            - "scroll up" / "scroll down"
            - "help"

            **Coding (free-form)**
            - "Write a Python function that…"
            - "Refactor this to use async"
            - "Add tests for…"
            - "Explain what this file does"
            """
        )
        st.divider()
        st.caption(
            "Pipeline process writes to `tmp/state.json`; "
            "this page polls every second."
        )


if __name__ == "__main__":
    main()