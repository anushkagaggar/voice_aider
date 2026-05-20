"""
tests/test_confidence_proxy.py
==============================
Verifies the confidence proxy in groq_clients/whisper_client.py.

The formula is:
    score = exp(mean(avg_logprob)) * (1 - mean(no_speech_prob))

Edge cases:
  - empty segments list → 0.0
  - clean confident speech → high score
  - high no_speech_prob (silence) → low score
  - score is always clamped to [0, 1]
"""

from __future__ import annotations

import math

from groq_clients.whisper_client import _score_from_segments


def test_empty_segments_returns_zero():
    assert _score_from_segments([]) == 0.0


def test_high_confidence_clear_speech():
    segments = [
        {"avg_logprob": -0.1, "no_speech_prob": 0.01},
        {"avg_logprob": -0.15, "no_speech_prob": 0.02},
    ]
    score = _score_from_segments(segments)
    # exp(-0.125) ≈ 0.882, dampened by (1 - 0.015) ≈ 0.985 → ~0.87
    assert 0.80 < score < 0.95


def test_low_confidence_muffled_speech():
    segments = [{"avg_logprob": -1.5, "no_speech_prob": 0.1}]
    score = _score_from_segments(segments)
    # exp(-1.5) ≈ 0.223, dampened by 0.9 → ~0.20
    assert 0.15 < score < 0.30


def test_silence_segment_scores_near_zero():
    """High no_speech_prob should crush the score even if avg_logprob looks fine."""
    segments = [{"avg_logprob": -0.2, "no_speech_prob": 0.95}]
    score = _score_from_segments(segments)
    assert score < 0.10


def test_score_is_clamped_to_unit_interval():
    """Pathological inputs (positive avg_logprob, weird no_speech_prob) must stay in [0,1]."""
    # avg_logprob shouldn't be positive in practice but the code clamps it.
    segments = [{"avg_logprob": 2.0, "no_speech_prob": -0.5}]
    score = _score_from_segments(segments)
    assert 0.0 <= score <= 1.0


def test_missing_fields_use_safe_defaults():
    """Some SDK versions omit fields on edge-case segments."""
    segments = [{}]
    score = _score_from_segments(segments)
    # Default avg_logprob=-10 → exp(-10) ≈ 4.5e-5; default no_speech_prob=0
    assert 0.0 <= score < 0.001


def test_score_is_monotonic_in_logprob():
    """Higher avg_logprob (closer to 0) → higher score, all else equal."""
    low = _score_from_segments([{"avg_logprob": -2.0, "no_speech_prob": 0.0}])
    mid = _score_from_segments([{"avg_logprob": -1.0, "no_speech_prob": 0.0}])
    high = _score_from_segments([{"avg_logprob": -0.1, "no_speech_prob": 0.0}])
    assert low < mid < high


def test_score_is_monotonic_in_no_speech_prob():
    """Higher no_speech_prob → lower score, all else equal."""
    quiet = _score_from_segments([{"avg_logprob": -0.3, "no_speech_prob": 0.05}])
    noisy = _score_from_segments([{"avg_logprob": -0.3, "no_speech_prob": 0.5}])
    silent = _score_from_segments([{"avg_logprob": -0.3, "no_speech_prob": 0.9}])
    assert silent < noisy < quiet