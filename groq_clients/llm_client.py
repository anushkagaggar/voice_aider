"""
groq_clients/llm_client.py
==========================
Thin wrapper around Groq's chat.completions endpoint, scoped to the single
job of intent classification for classify_node.

We use Groq's JSON mode (`response_format={"type": "json_object"}`) plus
temperature=0.0 so the output is deterministic structured data. The LLM call
is the *fallback* path — the fast path is fuzzy matching against the CMD
allowlist, which lives in config/commands.py. Only transcripts that miss the
allowlist hit this client.

Returned shape:
    ClassificationResult(intent="cmd"|"prompt"|"unknown", action=str|None)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from groq import Groq
from pydantic import BaseModel, Field, ValidationError

from config.prompts import CLASSIFY_SYSTEM_PROMPT, CLASSIFY_USER_TEMPLATE
from config.settings import settings
from config.commands import COMMANDS
from utils.logger import get_logger

log = get_logger(__name__)

Intent = Literal["cmd", "prompt", "unknown"]


@dataclass(frozen=True)
class ClassificationResult:
    """Output of a single classify call."""

    intent: Intent
    action: str | None


class _ClassifySchema(BaseModel):
    """Strict schema used to validate the LLM's JSON output."""

    intent: Intent
    action: str | None = Field(default=None)


_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


def _validate_action(intent: Intent, action: str | None) -> str | None:
    """
    Defensive check: if intent is "cmd", action must be one of the known
    command keys. Otherwise force it to None. This guards against the LLM
    inventing actions like "log_out" that execute_cmd cannot handle.
    """
    if intent != "cmd":
        return None
    if action in COMMANDS:
        return action
    log.warning("LLM returned cmd intent with unknown action=%r — coercing to None", action)
    return None


def classify(transcript: str) -> ClassificationResult:
    """
    Classify a transcript into (intent, action). Falls back to "unknown" on
    any error so the graph can decide whether to retry STT.
    """
    transcript = transcript.strip()
    if not transcript:
        return ClassificationResult(intent="unknown", action=None)

    client = _get_client()
    log.info("Classify call: transcript=%r", transcript)

    try:
        resp = client.chat.completions.create(
            model=settings.GROQ_LLM_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": CLASSIFY_USER_TEMPLATE.format(transcript=transcript)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=64,  # JSON is tiny; cap to avoid runaway
        )
    except Exception:
        log.exception("Classify API call failed")
        return ClassificationResult(intent="unknown", action=None)

    raw = (resp.choices[0].message.content or "").strip()

    try:
        data = json.loads(raw)
        parsed = _ClassifySchema(**data)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        log.warning("Classify output failed to parse: %s | raw=%r", e, raw)
        return ClassificationResult(intent="unknown", action=None)

    action = _validate_action(parsed.intent, parsed.action)
    log.info("Classify result: intent=%s action=%s", parsed.intent, action)
    return ClassificationResult(intent=parsed.intent, action=action)