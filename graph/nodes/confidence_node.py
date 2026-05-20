"""
graph/nodes/confidence_node.py
==============================
Pure gating node. Decides whether the transcript from stt_node is good enough
to act on, or whether we should retry STT.

The node itself just inspects state and increments retry_count. The actual
branching (loop back to stt_node vs. proceed to classify_node) lives in
graph/edges.py as a `conditional_edge`. Keeping logic-vs-routing separate is
the LangGraph convention.
"""

from __future__ import annotations

from config.settings import settings
from graph.state import VoiceState
from utils.logger import get_logger

log = get_logger(__name__)


def confidence_node(state: VoiceState) -> VoiceState:
    """
    Decide if we should retry. If confidence is below threshold AND we
    haven't exhausted retries, bump retry_count. The edge function reads
    the result and routes back to stt_node.
    """
    confidence = state.get("confidence", 0.0)
    retry_count = state.get("retry_count", 0)
    transcript = state.get("transcript", "")

    # Empty transcript counts as "definitely not good" regardless of score.
    is_empty = not transcript.strip()
    below_threshold = confidence < settings.CONFIDENCE_THRESHOLD
    retries_left = retry_count < settings.MAX_STT_RETRIES

    should_retry = (is_empty or below_threshold) and retries_left

    if should_retry:
        new_count = retry_count + 1
        log_line = (
            f"🔁 confidence={confidence:.2f} < {settings.CONFIDENCE_THRESHOLD:.2f}, "
            f"retry {new_count}/{settings.MAX_STT_RETRIES}"
        )
        log.info(log_line)
        return VoiceState(retry_count=new_count, ui_log=[log_line])

    # Either confident enough, or out of retries. Proceed.
    if is_empty or below_threshold:
        log_line = f"⏭️  retries exhausted (count={retry_count}), proceeding with low-confidence transcript"
    else:
        log_line = f"✅ confidence={confidence:.2f} ≥ {settings.CONFIDENCE_THRESHOLD:.2f}"

    log.info(log_line)
    return VoiceState(ui_log=[log_line])