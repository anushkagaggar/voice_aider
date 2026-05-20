"""
tests/test_graph.py
===================
End-to-end smoke test of the compiled LangGraph. Mocks Groq calls and the
aider subprocess so the graph runs entirely on locally-computed values.

What this proves:
  - The graph compiles.
  - STT → confidence → classify → branch → write_state edges fire in order.
  - cmd branch reaches execute_cmd (and aider_session.send_command).
  - prompt branch reaches aider_node (and aider_session.send_prompt).
  - write_state_node writes a valid JSON file under tmp/.
  - The retry loop fires when confidence is below threshold.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from graph import build_graph, new_state
from groq_clients.llm_client import ClassificationResult
from groq_clients.whisper_client import TranscriptionResult


def _stub_transcribe_high_conf(text: str):
    """Factory: returns a transcribe stub that always yields a confident result."""
    def _f(*_args, **_kwargs):
        return TranscriptionResult(text=text, confidence=0.92, duration=1.0)
    return _f


def _stub_transcribe_sequence(*results: TranscriptionResult):
    """Factory: returns a stub that returns each result in order, then sticks on the last."""
    iterator = iter(results)
    last = [results[-1]]

    def _f(*_args, **_kwargs):
        try:
            value = next(iterator)
            last[0] = value
            return value
        except StopIteration:
            return last[0]
    return _f


# ---------- cmd path ----------

def test_graph_cmd_path_routes_to_execute_cmd(tmp_state_file):
    """Saying 'undo' should hit the fuzzy fast path, run execute_cmd, write state."""
    mock_session = MagicMock()
    mock_session.send_command.return_value = "Undid last edit."

    # The aider_session singleton is imported by both aider_node and (lazily) by
    # execute_cmd. Patch it in both module namespaces so either lookup sees the mock.
    with patch("graph.nodes.stt_node.transcribe", _stub_transcribe_high_conf("undo")), \
         patch("graph.nodes.aider_node.aider_session", mock_session), \
         patch("graph.nodes.execute_cmd.aider_session", mock_session, create=True):

        graph = build_graph()
        final = graph.invoke(new_state(b"fake-wav-bytes"))

    assert final["transcript"] == "undo"
    assert final["intent"] == "cmd"
    assert final["action"] == "undo"
    mock_session.send_command.assert_called_once_with("/undo")

    persisted = json.loads(tmp_state_file.read_text())
    assert persisted["intent"] == "cmd"
    assert persisted["action"] == "undo"
    assert len(persisted["history"]) == 1


# ---------- prompt path ----------

def test_graph_prompt_path_routes_to_aider(tmp_state_file):
    """A free-form coding request should reach aider_node.send_prompt."""
    transcript = "write a fibonacci function in python"

    with patch("graph.nodes.stt_node.transcribe", _stub_transcribe_high_conf(transcript)), \
         patch("graph.nodes.classify_node.classify",
               return_value=ClassificationResult(intent="prompt", action=None)), \
         patch("graph.nodes.aider_node.aider_session") as mock_session:
        # Need _proc to look initialised so aider_node skips the start() call.
        mock_session._proc = MagicMock()
        mock_session.send_prompt.return_value = "def fib(n): ..."

        graph = build_graph()
        final = graph.invoke(new_state(b"fake-wav-bytes"))

    assert final["intent"] == "prompt"
    assert final["aider_response"] == "def fib(n): ..."
    mock_session.send_prompt.assert_called_once_with(transcript)

    persisted = json.loads(tmp_state_file.read_text())
    assert persisted["aider_response"] == "def fib(n): ..."


# ---------- retry loop ----------

def test_graph_retries_on_low_confidence(tmp_state_file):
    """
    First STT call returns low confidence → confidence_node should bump
    retry_count and the conditional edge should route back to stt_node.
    Second call returns confident result → proceed to classify.
    """
    bad = TranscriptionResult(text="", confidence=0.1, duration=0.5)
    good = TranscriptionResult(text="clear", confidence=0.95, duration=0.5)
    stub = _stub_transcribe_sequence(bad, good)

    call_counter = MagicMock(side_effect=stub)
    with patch("graph.nodes.stt_node.transcribe", call_counter), \
         patch("graph.nodes.aider_node.aider_session") as mock_session:
        mock_session.send_command.return_value = "cleared"

        graph = build_graph()
        final = graph.invoke(new_state(b"fake-wav-bytes"))

    # STT must have been called at least twice (initial + retry).
    assert call_counter.call_count >= 2
    assert final["transcript"] == "clear"
    assert final["intent"] == "cmd"
    assert final["action"] == "clear"


# ---------- unknown intent path ----------

def test_graph_unknown_intent_skips_action_nodes(tmp_state_file):
    """If classify returns 'unknown', neither execute_cmd nor aider_node should run."""
    with patch("graph.nodes.stt_node.transcribe",
               _stub_transcribe_high_conf("uhhhh ummmm")), \
         patch("graph.nodes.classify_node.classify",
               return_value=ClassificationResult(intent="unknown", action=None)), \
         patch("graph.nodes.aider_node.aider_session") as mock_session:

        graph = build_graph()
        final = graph.invoke(new_state(b"fake-wav-bytes"))

    assert final["intent"] == "unknown"
    mock_session.send_command.assert_not_called()
    mock_session.send_prompt.assert_not_called()

    # state.json should still be written for the UI to render the failure.
    assert tmp_state_file.exists()
    persisted = json.loads(tmp_state_file.read_text())
    assert persisted["intent"] == "unknown"