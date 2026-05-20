"""
graph/nodes/classify_node.py
============================
Decides whether the transcript is a control command (cmd) or a prompt for aider.

Two-stage classification — fast path first, LLM fallback second:

  1. Fuzzy-match the transcript against config.commands.ALL_PHRASINGS.
     If similarity ≥ FUZZY_THRESHOLD → intent="cmd", skip the LLM entirely.
     This catches ~95% of commands at ~0.5ms latency.

  2. Otherwise, call Groq Llama-3.1-8b-instant for full intent classification.
     The LLM is only consulted for ambiguous / unusual phrasings like
     "kill the session" → exit.

If the LLM returns intent="cmd" with an unknown action, llm_client.classify()
already coerces it to None — we don't have to re-validate here.
"""

from __future__ import annotations

from rapidfuzz import process, fuzz

from config.commands import ALL_PHRASINGS, FUZZY_THRESHOLD, PHRASE_TO_ACTION
from graph.state import VoiceState
from groq_clients.llm_client import classify
from utils.logger import get_logger

log = get_logger(__name__)


def _fuzzy_match(transcript: str) -> str | None:
    """
    Return the matched action name if the transcript fuzzy-matches a known
    command phrasing, else None.

    Uses token_set_ratio so word order doesn't matter and stray filler words
    don't kill the match. "yeah undo that thing" should still match "undo that".
    """
    candidate = process.extractOne(
        transcript.lower().strip(),
        ALL_PHRASINGS,
        scorer=fuzz.token_set_ratio,
        score_cutoff=FUZZY_THRESHOLD,
    )
    if candidate is None:
        return None
    matched_phrase, score, _ = candidate
    log.debug("Fuzzy match: %r → %r (score=%.1f)", transcript, matched_phrase, score)
    return PHRASE_TO_ACTION[matched_phrase]


def classify_node(state: VoiceState) -> VoiceState:
    """
    Fill state.intent and state.action. Returns a partial state update.
    """
    transcript = (state.get("transcript") or "").strip()

    if not transcript:
        return VoiceState(
            intent="unknown",
            action=None,
            ui_log=["❓ classify: empty transcript → unknown"],
        )

    # ----- Fast path: fuzzy allowlist match -----
    action = _fuzzy_match(transcript)
    if action is not None:
        log_line = f"⚡ classify (fuzzy): action={action}"
        log.info(log_line)
        return VoiceState(intent="cmd", action=action, ui_log=[log_line])

    # ----- Fallback: LLM classification -----
    result = classify(transcript)

    log_line = (
        f"🧠 classify (LLM): intent={result.intent}"
        + (f" action={result.action}" if result.action else "")
    )
    log.info(log_line)

    return VoiceState(
        intent=result.intent,
        action=result.action,
        ui_log=[log_line],
    )