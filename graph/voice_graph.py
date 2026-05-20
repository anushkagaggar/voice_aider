"""
graph/voice_graph.py
====================
Compiles the LangGraph StateGraph for the voice pipeline.

Topology (matches the project README diagram):

        ┌──────────────┐
        │   stt_node   │ ◀───── (retry loop)
        └──────┬───────┘                  │
               ▼                          │
       ┌───────────────────┐              │
       │ confidence_node   │──── low conf ┘
       └──────┬────────────┘
              ▼
       ┌───────────────┐
       │ classify_node │
       └──────┬────────┘
              ▼
       ┌──────────────┐
       │  cond. edge  │
       └─┬────────┬───┘
   cmd   │        │  prompt
         ▼        ▼
  ┌──────────┐ ┌──────────┐
  │execute_  │ │aider_node│
  │  cmd     │ │          │
  └────┬─────┘ └────┬─────┘
       └────┬───────┘
            ▼
    ┌─────────────────┐
    │ write_state_    │
    │     node        │
    └────────┬────────┘
             ▼
            END

Public API:
    build_graph()   -> compiled CompiledGraph ready to .invoke({...})
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from graph.edges import (
    NODE_AIDER,
    NODE_CLASSIFY,
    NODE_CONFIDENCE,
    NODE_EXECUTE,
    NODE_STT,
    NODE_WRITE_STATE,
    route_after_classify,
    route_after_confidence,
)
from graph.nodes import (
    aider_node,
    classify_node,
    confidence_node,
    execute_cmd,
    stt_node,
    write_state_node,
)
from graph.state import VoiceState
from utils.logger import get_logger

log = get_logger(__name__)


def build_graph():
    """
    Construct + compile the LangGraph. Returns a CompiledGraph whose .invoke()
    runs one full pass: audio → stt → confidence → classify → branch → write.
    """
    g = StateGraph(VoiceState)

    # ---- register nodes ----
    g.add_node(NODE_STT, stt_node)
    g.add_node(NODE_CONFIDENCE, confidence_node)
    g.add_node(NODE_CLASSIFY, classify_node)
    g.add_node(NODE_EXECUTE, execute_cmd)
    g.add_node(NODE_AIDER, aider_node)
    g.add_node(NODE_WRITE_STATE, write_state_node)

    # ---- linear edges ----
    g.set_entry_point(NODE_STT)
    g.add_edge(NODE_STT, NODE_CONFIDENCE)

    # ---- conditional edges ----
    # After confidence: either loop back to stt or proceed to classify.
    g.add_conditional_edges(
        NODE_CONFIDENCE,
        route_after_confidence,
        {
            NODE_STT: NODE_STT,
            NODE_CLASSIFY: NODE_CLASSIFY,
        },
    )

    # After classify: branch into cmd / prompt / unknown.
    g.add_conditional_edges(
        NODE_CLASSIFY,
        route_after_classify,
        {
            NODE_EXECUTE: NODE_EXECUTE,
            NODE_AIDER: NODE_AIDER,
            NODE_WRITE_STATE: NODE_WRITE_STATE,
        },
    )

    # Both action branches converge on write_state.
    g.add_edge(NODE_EXECUTE, NODE_WRITE_STATE)
    g.add_edge(NODE_AIDER, NODE_WRITE_STATE)

    # Final node terminates.
    g.add_edge(NODE_WRITE_STATE, END)

    compiled = g.compile()
    log.info("Voice graph compiled.")
    return compiled