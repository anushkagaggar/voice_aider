"""
graph/nodes/execute_cmd.py
==========================
Handles the CMD branch of the conditional edge. Every command in
config/commands.py routes here.

What's tricky: some commands need to affect the aider subprocess
(clear, undo, exit), and some only affect the UI dashboard (history,
scroll_up, scroll_down). The node signals BOTH by:

  - returning cmd_result with a human-readable message for ui_log
  - setting state.action so write_state_node writes the action to state.json,
    where the Streamlit dashboard picks it up and acts on UI-only commands.

For aider-affecting commands (clear, undo, exit, save, stop) we send the
appropriate string into the aider subprocess via aider_node's session helper.
"""

from __future__ import annotations

from graph.state import VoiceState
from utils.logger import get_logger

log = get_logger(__name__)


# Mapping: action name → message piped to aider's stdin.
# Aider accepts /-prefixed slash commands directly.
_AIDER_SLASH_COMMANDS: dict[str, str] = {
    "clear":   "/clear",
    "undo":    "/undo",
    "save":    "/commit",
    "exit":    "/exit",
    "stop":    "/exit",
    "help":    "/help",
}

# UI-only commands — handled by the Streamlit dashboard reading state.json.
_UI_ONLY_COMMANDS: set[str] = {
    "history",
    "scroll_up",
    "scroll_down",
}


def execute_cmd(state: VoiceState) -> VoiceState:
    """
    Resolve state.action into either an aider slash-command (relayed via
    aider_session) or a UI-only signal.
    """
    action = state.get("action")
    if not action:
        return VoiceState(
            cmd_result="(no action)",
            error="execute_cmd: missing action",
            ui_log=["⚠️  execute_cmd called without an action"],
        )

    # Aider-affecting commands: send the slash command via the aider session.
    if action in _AIDER_SLASH_COMMANDS:
        slash = _AIDER_SLASH_COMMANDS[action]
        # Import here, not at module load, to avoid spawning aider just because
        # this module was imported (e.g. by tests).
        from graph.nodes.aider_node import aider_session

        try:
            aider_session.send_command(slash)
            msg = f"sent {slash} to aider"
        except Exception as e:
            log.exception("Failed to send %s to aider", slash)
            return VoiceState(
                cmd_result=None,
                error=f"execute_cmd: {e}",
                ui_log=[f"❌ execute_cmd: failed to send {slash}"],
            )

        log_line = f"🛠️  cmd: {action} → {msg}"
        log.info(log_line)
        return VoiceState(cmd_result=msg, ui_log=[log_line])

    # UI-only commands: just announce. The dashboard reads state.action.
    if action in _UI_ONLY_COMMANDS:
        log_line = f"🖥️  cmd: {action} (UI-only)"
        log.info(log_line)
        return VoiceState(cmd_result=f"ui:{action}", ui_log=[log_line])

    # Anything else slipped past _validate_action — defensive fallback.
    log.warning("execute_cmd: unhandled action=%r", action)
    return VoiceState(
        cmd_result=None,
        error=f"unhandled action: {action}",
        ui_log=[f"⚠️  unhandled action: {action}"],
    )