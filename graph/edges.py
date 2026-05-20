"""
graph/edges.py
==============
Conditional-edge routing functions. Each function takes the current state and
returns the *name* of the next node (or END). LangGraph wires these in via
`graph.add_conditional_edges(source_node, fn, mapping)`.

We keep these functions tiny and pure (no side effects, no logging) so the
graph's control flow is easy to trace.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END

from config.settings import settings
from graph.state import VoiceState

# Node-name string literals — kept here so voice_graph.py and edges.py
# can't drift out of sync.
NODE_STT = "stt_node"
NODE_CONFIDENCE = "confidence_node"
NODE_CLASSIFY = "classify_node"
NODE_EXECUTE = "execute_cmd"
NODE_AIDER = "aider_node"
NODE_WRITE_STATE = "write_state_node"


def route_after_confidence(state: VoiceState) -> Literal["stt_node", "classify_node"]:
    """
    After confidence_node:
      - if retry_count was bumped (and is now still within budget), loop back to stt_node
      - else proceed to classify_node
    """
    confidence = state.get("confidence", 0.0)
    retry_count = state.get("retry_count", 0)
    transcript = (state.get("transcript") or "").strip()

    needs_retry = (not transcript) or confidence < settings.CONFIDENCE_THRESHOLD
    if needs_retry and retry_count <= settings.MAX_STT_RETRIES and retry_count > 0:
        # retry_count > 0 means confidence_node decided to retry on this pass
        return NODE_STT
    return NODE_CLASSIFY


def route_after_classify(state: VoiceState) -> Literal["execute_cmd", "aider_node", "write_state_node"]:
    """
    After classify_node:
      - intent='cmd'     → execute_cmd
      - intent='prompt'  → aider_node
      - intent='unknown' → write_state_node (record the failure; don't loop)
    """
    intent = state.get("intent", "unknown")
    if intent == "cmd":
        return NODE_EXECUTE
    if intent == "prompt":
        return NODE_AIDER
    return NODE_WRITE_STATE