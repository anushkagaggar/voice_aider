"""
config/commands.py
==================
Strict allowlist of voice-driven commands. Anything matching here (exact or
fuzzy ≥ FUZZY_THRESHOLD) bypasses the LLM classifier and routes straight to
execute_cmd. Anything else falls through to aider_node as a prompt.

Why an allowlist?
- Deterministic: no LLM hallucination for safety-critical actions (exit, undo).
- Fast: rapidfuzz match is sub-millisecond vs ~300ms for a Groq call.
- Auditable: the full set of voice-triggerable side effects lives in one file.
"""

from typing import Final

# Each command maps action_name -> list of accepted phrasings.
# The first phrasing is the canonical form (used in logs + UI).
COMMANDS: Final[dict[str, list[str]]] = {
    "clear":        ["clear", "clear screen", "clear chat"],
    "undo":         ["undo", "undo that", "revert", "go back"],
    "exit":         ["exit", "quit", "stop session", "shut down"],
    "help":         ["help", "show help", "commands"],
    "save":         ["save", "save changes", "commit"],
    "history":      ["show history", "history", "what did i say"],
    "stop":         ["stop", "cancel", "abort"],
    "scroll_up":    ["scroll up", "page up", "go up"],
    "scroll_down":  ["scroll down", "page down", "go down"],
}

# Flat list of every phrasing — used by rapidfuzz for matching.
ALL_PHRASINGS: Final[list[str]] = [
    phrase for phrases in COMMANDS.values() for phrase in phrases
]

# Reverse lookup: phrasing -> action_name
PHRASE_TO_ACTION: Final[dict[str, str]] = {
    phrase: action
    for action, phrases in COMMANDS.items()
    for phrase in phrases
}

# Fuzzy match threshold (0-100). Below this, the transcript is treated as a prompt.
# 85 is strict enough to reject "I want to undo my last refactor" (that's a prompt)
# but loose enough to accept "undue" → "undo" (whisper misspelling).
FUZZY_THRESHOLD: Final[int] = 85