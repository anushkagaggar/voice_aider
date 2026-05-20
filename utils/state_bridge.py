"""
utils/state_bridge.py
=====================
The IPC bridge between the voice pipeline process and the Streamlit UI process.

Why atomic writes?
  Streamlit polls state.json every ~1s. If we wrote with a plain `open("w")`
  the reader could catch us mid-write and parse a half-flushed JSON, throwing
  a JSONDecodeError on every other reload. The fix is a write-to-temp + rename:
  rename is atomic on POSIX (and on Windows when target doesn't exist or via
  os.replace), so a reader only ever sees the old file or the complete new file.

Public API:
  write_state(payload: dict) -> None
  read_state() -> dict | None
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from config.settings import settings
from utils.logger import get_logger

log = get_logger(__name__)


def _ensure_parent(path: Path) -> None:
    """Create the parent directory for the state file if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_state(payload: dict[str, Any]) -> None:
    """
    Atomically write the given dict to settings.state_file as JSON.

    Steps:
      1. Write to a temp file in the same directory (same filesystem -> rename is atomic).
      2. os.replace() onto the final path.
    """
    target = settings.state_file
    _ensure_parent(target)

    # NamedTemporaryFile with delete=False so we control the lifecycle;
    # dir=target.parent guarantees same filesystem for atomic rename.
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    )
    try:
        json.dump(payload, tmp, indent=2, default=str)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, target)
    except Exception:
        # Clean up the orphan temp file if anything went wrong.
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        log.exception("Failed to write state to %s", target)
        raise


def read_state() -> dict[str, Any] | None:
    """
    Read the current state file. Returns None if the file does not exist yet
    or if it could not be parsed (treat as 'no state available').
    """
    target = settings.state_file
    if not target.exists():
        return None
    try:
        with target.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        # Race with a writer or empty file on first boot — non-fatal for the UI.
        log.debug("read_state transient error: %s", e)
        return None