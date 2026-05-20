"""
tests/test_state_bridge.py
==========================
Covers the atomic JSON IPC layer. The contract write_state/read_state has to
honour:

  1. read on a missing file returns None (not raise).
  2. write creates the file and parent dirs.
  3. write is round-trippable via read.
  4. write is atomic — no partial files left behind on the disk.
  5. read swallows corrupt JSON and returns None (a writer race during the
     reader's open() shouldn't crash the UI).
"""

from __future__ import annotations

import json
import os

from utils.state_bridge import read_state, write_state


def test_read_missing_file_returns_none(tmp_state_file):
    assert not tmp_state_file.exists()
    assert read_state() is None


def test_write_creates_file_and_parent_dirs(tmp_state_file):
    write_state({"hello": "world"})
    assert tmp_state_file.exists()
    assert tmp_state_file.parent.is_dir()


def test_write_then_read_round_trip(tmp_state_file):
    payload = {
        "transcript": "write a fibonacci function",
        "confidence": 0.91,
        "intent": "prompt",
        "action": None,
        "history": [{"timestamp": 1, "intent": "prompt"}],
    }
    write_state(payload)
    loaded = read_state()
    assert loaded == payload


def test_write_overwrites_atomically(tmp_state_file):
    """Subsequent writes must not leave .tmp turds beside the target."""
    for i in range(5):
        write_state({"i": i})

    # Final content should be the last write.
    assert read_state() == {"i": 4}

    # Parent dir should contain exactly the state file + nothing else.
    siblings = list(tmp_state_file.parent.iterdir())
    assert siblings == [tmp_state_file], f"orphan files left: {siblings}"


def test_read_returns_none_on_corrupt_json(tmp_state_file):
    """If a reader catches the writer mid-write somehow, we treat it as 'no state'."""
    tmp_state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_state_file.write_text("{not valid json")
    assert read_state() is None


def test_write_accepts_non_serializable_via_default(tmp_state_file):
    """write_state uses default=str — Path objects shouldn't crash."""
    from pathlib import Path
    write_state({"path": Path("/tmp/foo")})
    loaded = read_state()
    assert loaded is not None
    assert isinstance(loaded["path"], str)