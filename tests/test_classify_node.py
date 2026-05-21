"""
tests/test_classify_node.py
===========================
Exercises the two-stage classifier in graph/nodes/classify_node.py.

The fuzzy fast-path should NOT call the LLM. The LLM fallback path should be
invoked only when fuzzy match fails. We patch groq_clients.llm_client.classify
to avoid any real network calls.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from graph.nodes.classify_node import classify_node
from groq_clients.llm_client import ClassificationResult


# ---------- fuzzy fast-path ----------

@pytest.mark.parametrize("transcript,expected_action", [
    ("undo",                "undo"),
    ("undo that",           "undo"),
    ("undo that last edit", "undo"),     # token_set_ratio allows extra words
    ("clear",               "clear"),
    ("clear the screen",    "clear"),
    ("exit",                "exit"),
    ("please refactor this", None),      # NOT in fuzzy allowlist — must fall to LLM
])
def test_fuzzy_match_resolves_known_commands(transcript, expected_action):
    """If fuzzy matches, classify_node must NOT call the LLM at all."""
    with patch("graph.nodes.classify_node.classify") as mock_llm:
        # Provide a default LLM stub in case the fallback is reached.
        mock_llm.return_value = ClassificationResult(intent="prompt", action=None)
        out = classify_node({"transcript": transcript})

    if expected_action is not None:
        assert out["intent"] == "cmd"
        assert out["action"] == expected_action
        assert mock_llm.call_count == 0, "fuzzy hit must skip the LLM"
    else:
        # Fell through to LLM; we patched it to return "prompt".
        assert mock_llm.call_count == 1


# ---------- LLM fallback path ----------

def test_llm_fallback_for_unusual_phrasing():
    """Phrases that miss the fuzzy threshold must hit the LLM."""
    # "please refactor this code" shares no significant tokens with any
    # allowlist phrasing — guaranteed fuzzy miss across rapidfuzz versions.
    # The LLM stub then classifies it as a prompt (which is correct anyway).
    with patch("graph.nodes.classify_node.classify") as mock_llm:
        mock_llm.return_value = ClassificationResult(intent="prompt", action=None)
        out = classify_node({"transcript": "please refactor this code"})

    assert mock_llm.call_count == 1
    assert out["intent"] == "prompt"

def test_llm_prompt_route():
    """A real coding request should route to aider, not to the cmd branch."""
    with patch("graph.nodes.classify_node.classify") as mock_llm:
        mock_llm.return_value = ClassificationResult(intent="prompt", action=None)
        out = classify_node({"transcript": "write a fibonacci function in rust"})

    assert out["intent"] == "prompt"
    assert out["action"] is None


# ---------- defensive: empty and broken inputs ----------

def test_empty_transcript_routes_to_unknown():
    out = classify_node({"transcript": ""})
    assert out["intent"] == "unknown"


def test_whitespace_only_transcript_routes_to_unknown():
    out = classify_node({"transcript": "   \n\t  "})
    assert out["intent"] == "unknown"


def test_missing_transcript_key_routes_to_unknown():
    out = classify_node({})
    assert out["intent"] == "unknown"


# ---------- defensive: llm_client coerces unknown actions ----------

def test_llm_client_rejects_unknown_action():
    """If the LLM hallucinates an action not in COMMANDS, llm_client coerces it to None."""
    from groq_clients.llm_client import _validate_action

    assert _validate_action("cmd", "undo") == "undo"
    assert _validate_action("cmd", "log_out") is None          # not in COMMANDS
    assert _validate_action("prompt", "undo") is None          # wrong intent
    assert _validate_action("unknown", "undo") is None