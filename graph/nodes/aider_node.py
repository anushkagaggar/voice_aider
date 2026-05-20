"""
graph/nodes/aider_node.py
=========================
Owns the persistent aider subprocess.

Why persistent?
  Aider takes ~3-5s to boot (loads the model adapter, scans the repo, builds
  the tags cache). If we spawned a new process per utterance the demo would
  feel sluggish. Instead, we spawn aider ONCE at pipeline startup, hold open
  pipes to its stdin/stdout, and stream prompts in line-by-line.

Public API:
  aider_session : module-level _AiderSession singleton
    .start()                  -> spawn the subprocess
    .send_prompt(text)        -> write to stdin, capture next stdout chunk
    .send_command(slash)      -> same, but for /clear, /undo etc.
    .stop()                   -> graceful shutdown

  aider_node(state)           -> the LangGraph node fn (uses the session)

The node returns aider_response in state, which write_state_node serialises
into state.json and the Streamlit dashboard renders.
"""

from __future__ import annotations

import queue
import shlex
import subprocess
import threading
import time
from typing import IO

from config.settings import settings
from graph.state import VoiceState
from utils.logger import get_logger

log = get_logger(__name__)


# How long we wait for aider to produce output after we send a prompt.
# Aider streams tokens; we collect for up to RESPONSE_WAIT_SECS, then stop if
# the output has been quiet for QUIET_AFTER_SECS.
_RESPONSE_WAIT_SECS = 60.0
_QUIET_AFTER_SECS = 1.5


class _AiderSession:
    """
    Thin wrapper around a long-lived `aider` subprocess.

    Threading model:
      - Main thread writes prompts to stdin.
      - A background reader thread continuously drains stdout into a Queue,
        because subprocess pipes block — we can't `read()` on-demand without
        risking a deadlock if aider hasn't flushed yet.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._stdout_q: queue.Queue[str] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ---------- lifecycle ----------

    def start(self) -> None:
        """Launch aider. Idempotent — calling twice is a no-op."""
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return  # already running

            cmd = [
                "aider",
                "--model", settings.AIDER_MODEL,
                *settings.aider_args_list,
            ]
            log.info("Starting aider: %s", " ".join(shlex.quote(c) for c in cmd))

            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # merge into stdout for simpler reading
                text=True,
                bufsize=1,                  # line-buffered
            )

            self._reader_thread = threading.Thread(
                target=self._drain_stdout,
                name="aider-stdout-reader",
                daemon=True,
            )
            self._reader_thread.start()

            # Give aider a moment to print its startup banner so the first
            # prompt doesn't race with init output.
            self._wait_for_quiet(timeout=8.0, quiet_window=0.8)

    def _drain_stdout(self) -> None:
        """Background thread: pump aider stdout into the queue, line by line."""
        assert self._proc is not None and self._proc.stdout is not None
        stdout: IO[str] = self._proc.stdout
        try:
            for line in iter(stdout.readline, ""):
                self._stdout_q.put(line)
        except Exception:
            log.exception("aider stdout reader crashed")
        finally:
            self._stdout_q.put("")  # sentinel — signals EOF

    def _wait_for_quiet(self, timeout: float, quiet_window: float) -> str:
        """
        Drain output until `quiet_window` seconds pass with no new lines,
        or `timeout` total elapses. Returns the accumulated text.
        """
        out_lines: list[str] = []
        deadline = time.monotonic() + timeout
        last_line_at = time.monotonic()

        while time.monotonic() < deadline:
            try:
                line = self._stdout_q.get(timeout=0.1)
                if line == "":
                    break  # EOF
                out_lines.append(line)
                last_line_at = time.monotonic()
            except queue.Empty:
                if time.monotonic() - last_line_at >= quiet_window:
                    break

        return "".join(out_lines).strip()

    # ---------- I/O ----------

    def _send_raw(self, text: str) -> str:
        """Write a line to aider stdin, then collect stdout until quiet."""
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("aider subprocess is not running — call start() first")
        if self._proc.stdin is None:
            raise RuntimeError("aider stdin is not available")

        if not text.endswith("\n"):
            text += "\n"

        log.info("→ aider: %s", text.strip())
        self._proc.stdin.write(text)
        self._proc.stdin.flush()

        return self._wait_for_quiet(
            timeout=_RESPONSE_WAIT_SECS,
            quiet_window=_QUIET_AFTER_SECS,
        )

    def send_prompt(self, prompt: str) -> str:
        """Send a natural-language prompt and return aider's reply."""
        return self._send_raw(prompt)

    def send_command(self, slash_cmd: str) -> str:
        """Send a slash-command like /clear or /undo and return aider's reply."""
        if not slash_cmd.startswith("/"):
            slash_cmd = "/" + slash_cmd
        return self._send_raw(slash_cmd)

    # ---------- shutdown ----------

    def stop(self) -> None:
        """Send /exit, then SIGTERM/kill if aider doesn't shut down."""
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            return  # already dead

        try:
            self._send_raw("/exit")
        except Exception:
            pass

        try:
            self._proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            log.warning("aider didn't exit cleanly — terminating")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# Module-level singleton. Imported lazily by execute_cmd to avoid spawning
# aider just because graph.nodes was imported.
aider_session = _AiderSession()


# -----------------------------------------------------------------------------
# Node function
# -----------------------------------------------------------------------------

def aider_node(state: VoiceState) -> VoiceState:
    """
    LangGraph node: feed state.transcript into aider as a prompt, capture
    the reply into state.aider_response.
    """
    prompt = (state.get("transcript") or "").strip()
    if not prompt:
        return VoiceState(
            aider_response=None,
            error="aider_node: empty prompt",
            ui_log=["⚠️  aider_node: empty prompt — skipping"],
        )

    # Lazy-start: first time we hit this node in a session, spawn aider.
    if aider_session._proc is None:
        log.info("aider_node: first call — starting subprocess")
        try:
            aider_session.start()
        except FileNotFoundError:
            msg = "aider executable not found — install with `pip install aider-chat`"
            log.error(msg)
            return VoiceState(
                aider_response=None,
                error=msg,
                ui_log=[f"❌ {msg}"],
            )
        except Exception as e:
            log.exception("aider failed to start")
            return VoiceState(
                aider_response=None,
                error=f"aider_node: start failed: {e}",
                ui_log=[f"❌ aider failed to start: {e}"],
            )

    try:
        reply = aider_session.send_prompt(prompt)
    except Exception as e:
        log.exception("aider send_prompt failed")
        return VoiceState(
            aider_response=None,
            error=f"aider_node: {e}",
            ui_log=[f"❌ aider error: {e}"],
        )

    preview = reply.replace("\n", " ⏎ ")
    preview = preview[:80] + ("…" if len(preview) > 80 else "")
    log_line = f"🤖 aider replied ({len(reply)} chars): {preview}"
    log.info(log_line)

    return VoiceState(aider_response=reply, ui_log=[log_line])