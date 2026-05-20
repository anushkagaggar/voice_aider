"""
utils/logger.py
===============
One logger factory used by every module in the project. Output is human-readable
on stderr (for the terminal that runs the pipeline) and includes the module name
so you can see at a glance which node logged what.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("stt finished, confidence=%.2f", score)
"""

import logging
import sys
from typing import Final

_LEVEL: Final[int] = logging.INFO
_FMT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATEFMT: Final[str] = "%H:%M:%S"

_configured: bool = False


def _configure_root() -> None:
    """Set up the root logger once. Idempotent."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.setLevel(_LEVEL)
    # Clear any default handlers (Streamlit / Jupyter may add their own)
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy third-party libs
    for noisy in ("httpx", "httpcore", "urllib3", "groq._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Configures root on first call."""
    _configure_root()
    # Trim the leading package name for readability:
    # "graph.nodes.stt_node" -> "nodes.stt_node"
    short = name.split(".", 1)[-1] if "." in name else name
    return logging.getLogger(short)