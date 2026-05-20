"""Utils package — logging, audio encoding, IPC bridge."""
from utils.logger import get_logger
from utils.state_bridge import read_state, write_state

__all__ = ["get_logger", "read_state", "write_state"]