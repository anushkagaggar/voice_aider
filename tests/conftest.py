"""
tests/conftest.py
=================
Shared pytest fixtures. Most importantly, this stubs GROQ_API_KEY into the env
*before* any project module is imported, so `config.settings` doesn't fail
during test collection on a machine that hasn't configured .env.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make sure the project root is on sys.path so `import graph`, `import utils`,
# etc. work regardless of where pytest is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Inject a dummy Groq key BEFORE config.settings is imported. settings.py would
# otherwise raise pydantic.ValidationError because GROQ_API_KEY is required.
os.environ.setdefault("GROQ_API_KEY", "gsk_test_dummy_key_for_pytest")
# Point the state file at a path inside the test tmpdir, never the real tmp/.
os.environ.setdefault("STATE_FILE_PATH", str(PROJECT_ROOT / "tmp" / "test_state.json"))

import pytest  # noqa: E402


@pytest.fixture
def tmp_state_file(tmp_path, monkeypatch):
    """
    Redirect settings.state_file to a per-test temp path. Use this in any
    test that calls write_state / read_state so tests can't trample each
    other's state.
    """
    target = tmp_path / "state.json"
    # Patch the settings instance directly — it's already imported.
    from config.settings import settings
    monkeypatch.setattr(settings, "STATE_FILE_PATH", target)
    return target