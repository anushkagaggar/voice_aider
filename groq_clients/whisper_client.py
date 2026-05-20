"""
groq_clients/whisper_client.py
==============================
Thin wrapper around Groq's audio.transcriptions endpoint.

The Groq Whisper API does not return a single "confidence" number out of the
box. We compute a proxy by averaging the `avg_logprob` field across every
segment in the `verbose_json` response, then mapping that into a 0..1 score
via the sigmoid-like transformation `exp(avg_logprob)`. We also incorporate
`no_speech_prob` so that mostly-silent utterances score low even if the few
recognized tokens were high-confidence.

Returned shape:
    TranscriptionResult(text=str, confidence=float, duration=float)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from groq import Groq

from config.settings import settings
from utils.audio_utils import wav_bytes_to_file_tuple
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class TranscriptionResult:
    """Output of a single Whisper call."""

    text: str
    confidence: float   # 0.0 (no idea) … 1.0 (very confident)
    duration: float     # length of the audio in seconds (from Whisper)


# Module-level client. Re-used across calls so we keep the HTTP connection pool warm.
_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


def _score_from_segments(segments: list[dict[str, Any]]) -> float:
    """
    Collapse the segment-level avg_logprob and no_speech_prob into one score in [0,1].

    avg_logprob is a negative number (closer to 0 = more confident). exp() maps
    it into (0, 1]. We then dampen by (1 - mean(no_speech_prob)) so utterances
    that Whisper thinks were silence get penalised.
    """
    if not segments:
        return 0.0

    logprobs = [s.get("avg_logprob", -10.0) for s in segments]
    no_speech = [s.get("no_speech_prob", 0.0) for s in segments]

    mean_logprob = sum(logprobs) / len(logprobs)
    mean_no_speech = sum(no_speech) / len(no_speech)

    # exp(avg_logprob) is in (0, 1]. Clamp for safety.
    raw = math.exp(min(0.0, mean_logprob))
    score = raw * (1.0 - mean_no_speech)
    return max(0.0, min(1.0, score))


def transcribe(wav_bytes: bytes, language: str = "en") -> TranscriptionResult:
    """
    Send WAV bytes to Groq Whisper and return text + confidence proxy.

    Args:
        wav_bytes: Output of utils.audio_utils.pcm_to_wav_bytes().
        language: ISO-639-1 code. "en" is the default; pass None to auto-detect
                  (slower, less accurate for short utterances).

    Returns:
        TranscriptionResult. On empty / silent audio, returns text="" with
        confidence=0.0 instead of raising — the confidence_node will route this
        to a retry.
    """
    client = _get_client()
    file_tuple = wav_bytes_to_file_tuple(wav_bytes)

    log.info("Whisper call: %d bytes", len(wav_bytes))

    try:
        resp = client.audio.transcriptions.create(
            file=file_tuple,
            model=settings.GROQ_STT_MODEL,
            response_format="verbose_json",
            language=language,
            temperature=0.0,
        )
    except Exception as e:
        log.exception("Whisper API call failed")
        # Surface as an empty transcription so the graph can decide to retry.
        return TranscriptionResult(text="", confidence=0.0, duration=0.0)

    # The SDK returns a typed object; we tolerate both attribute and dict access
    # because the response model wording has shifted across SDK versions.
    text = (getattr(resp, "text", None) or "").strip()
    duration = float(getattr(resp, "duration", 0.0) or 0.0)
    segments_raw = getattr(resp, "segments", None) or []

    # Normalise segments to dicts (some SDK versions return pydantic models).
    segments: list[dict[str, Any]] = [
        s if isinstance(s, dict) else s.model_dump() for s in segments_raw
    ]

    confidence = _score_from_segments(segments)

    log.info(
        "Whisper result: text=%r confidence=%.2f duration=%.2fs",
        text[:60] + ("…" if len(text) > 60 else ""),
        confidence,
        duration,
    )
    return TranscriptionResult(text=text, confidence=confidence, duration=duration)