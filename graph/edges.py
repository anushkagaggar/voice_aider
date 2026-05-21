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
      - if confidence_node set should_retry=True, loop back to stt_node
      - else proceed to classify_node

    We use an explicit `should_retry` flag in state rather than inferring
    from retry_count, because retry_count is monotonic across the whole
    graph run — once it hits MAX_STT_RETRIES, every subsequent low-confidence
    pass would loop forever if we routed on retry_count alone.
    """
    return NODE_STT if state.get("should_retry") else NODE_CLASSIFY


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