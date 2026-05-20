"""
config/settings.py
==================
Central settings loader. Reads .env via python-dotenv, then validates
through pydantic-settings. No raw os.getenv calls anywhere in the project —
every other module imports `settings` from here.
"""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from the project root BEFORE pydantic-settings reads env vars.
# This is the only place in the codebase that touches dotenv directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """All runtime configuration. Pulled from .env, validated at import."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ---- Groq API ----
    GROQ_API_KEY: str = Field(..., description="Groq API key — required")
    GROQ_STT_MODEL: str = "whisper-large-v3"
    GROQ_LLM_MODEL: str = "llama-3.1-8b-instant"

    # ---- Confidence + retry ----
    CONFIDENCE_THRESHOLD: float = Field(0.75, ge=0.0, le=1.0)
    MAX_STT_RETRIES: int = Field(2, ge=0, le=5)

    # ---- VAD / capture ----
    SAMPLE_RATE: int = 16000
    VAD_AGGRESSIVENESS: int = Field(2, ge=0, le=3)
    SILENCE_TIMEOUT_MS: int = Field(800, ge=200, le=3000)

    # ---- Push-to-talk ----
    PUSH_TO_TALK_KEY: Literal["space", "ctrl", "alt"] = "space"
    HANDS_FREE_MODE: bool = False

    # ---- IPC bridge ----
    STATE_FILE_PATH: Path = PROJECT_ROOT / "tmp" / "state.json"

    # ---- Aider subprocess ----
    AIDER_MODEL: str = "groq/llama-3.1-8b-instant"
    AIDER_EXTRA_ARGS: str = "--no-auto-commit --yes"

    @property
    def state_file(self) -> Path:
        """Resolved absolute path to the IPC state file."""
        p = Path(self.STATE_FILE_PATH)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def aider_args_list(self) -> list[str]:
        """AIDER_EXTRA_ARGS split into a list for subprocess.Popen."""
        return self.AIDER_EXTRA_ARGS.split()


# Single shared instance. Import this everywhere:
#     from config.settings import settings
settings = Settings()