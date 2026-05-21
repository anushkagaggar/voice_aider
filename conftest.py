"""
Root-level conftest.py
======================
pytest loads this BEFORE collecting any test module. We use it to:

  1. Put the project root on sys.path so `from graph import build_graph`
     works without needing `pip install -e .` (which gets flaky on Windows
     when the path contains spaces).
  2. Inject a dummy GROQ_API_KEY into the env so `config.settings` doesn't
     raise during test collection on machines without a real .env file.

tests/conftest.py also does (2), but doing it here too is harmless and
guarantees the env is set even if pytest's collection order shifts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root = the folder this file lives in.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Stub Groq key for test collection.
os.environ.setdefault("GROQ_API_KEY", "gsk_test_dummy_key_for_pytest")
os.environ.setdefault("STATE_FILE_PATH", str(PROJECT_ROOT / "tmp" / "test_state.json"))