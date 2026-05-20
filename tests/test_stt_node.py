"""
tests/test_stt_node.py
======================
Tests the stt_node adapter itself. The math behind `confidence` lives in
groq_clients/whisper_client.py and is covered by test_confidence_proxy.py;
this file covers the *node contract*:

  - Empty audio_bytes → returns a defensive partial state with error set.
  - Non-empty audio → returns transcript/confidence/duration mapped 1:1 from
    the transcribe() result, plus a ui_log line.
  - The node calls transcribe() exactly once, with the audio bytes from state.

We mock `groq_clients.whisper_client.transcribe` so no network is touched.
"""

from __future__ import annotations

from unittest.mock import patch

from graph.nodes.stt_node import stt_node
from groq_clients.whisper_client import TranscriptionResult


# ---------- empty audio defensive path ----------

def test_empty_audio_bytes_returns_error_state():
    """No audio in state → don't call Whisper, return an error marker."""
    with patch("graph.nodes.stt_node.transcribe") as mock_t:
        out = stt_node({"audio_bytes": b""})

    mock_t.assert_not_called()
    assert out["transcript"] == ""
    assert out["confidence"] == 0.0
    assert out["audio_duration"] == 0.0
    assert "no audio" in (out.get("error") or "").lower()


def test_missing_audio_bytes_key_treated_as_empty():
    """A state dict without audio_bytes at all should behave like empty audio."""
    with patch("graph.nodes.stt_node.transcribe") as mock_t:
        out = stt_node({})

    mock_t.assert_not_called()
    assert out["transcript"] == ""
    assert out["error"] is not None


# ---------- happy path ----------

def test_transcribe_result_mapped_into_state():
    """Whisper output must land in transcript/confidence/audio_duration fields."""
    fake_result = TranscriptionResult(
        text="write a fibonacci function",
        confidence=0.91,
        duration=2.3,
    )
    with patch("graph.nodes.stt_node.transcribe", return_value=fake_result) as mock_t:
        out = stt_node({"audio_bytes": b"\x00\x00fake-wav-bytes"})

    mock_t.assert_called_once()
    # The argument to transcribe should be the same bytes we passed in.
    called_with = mock_t.call_args.args[0]
    assert called_with == b"\x00\x00fake-wav-bytes"

    assert out["transcript"] == "write a fibonacci function"
    assert out["confidence"] == 0.91
    assert out["audio_duration"] == 2.3
    assert out["error"] is None


# ---------- ui_log emission ----------

def test_ui_log_line_emitted_on_success():
    """The dashboard's live feed needs one line per node fire."""
    fake_result = TranscriptionResult(text="undo", confidence=0.88, duration=0.5)
    with patch("graph.nodes.stt_node.transcribe", return_value=fake_result):
        out = stt_node({"audio_bytes": b"x" * 100})

    assert isinstance(out["ui_log"], list)
    assert len(out["ui_log"]) == 1
    log_line = out["ui_log"][0]
    # The line should mention the transcript and confidence so the UI shows context.
    assert "undo" in log_line
    assert "0.88" in log_line


def test_ui_log_emitted_on_empty_audio():
    """Even on the error path the UI gets a status line — silence is worse than red."""
    out = stt_node({"audio_bytes": b""})
    assert isinstance(out["ui_log"], list)
    assert len(out["ui_log"]) == 1


# ---------- ui_log preserves full transcript ----------

def test_ui_log_contains_full_transcript_for_dashboard():
    """
    The dashboard renders state.transcript directly; ui_log carries the
    one-line summary for the live feed. We don't truncate either — the
    UI's transcript_panel handles wrap/scroll for long text.
    """
    long_text = "refactor this entire module to use async generators " * 5
    fake_result = TranscriptionResult(text=long_text, confidence=0.9, duration=4.0)
    with patch("graph.nodes.stt_node.transcribe", return_value=fake_result):
        out = stt_node({"audio_bytes": b"x"})

    assert out["transcript"] == long_text
    assert isinstance(out["ui_log"], list)
    assert len(out["ui_log"]) == 1