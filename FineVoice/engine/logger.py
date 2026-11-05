"""Logging shared by all engine modules: terminal + logs/finevoice.log + an in-memory buffer for the UI."""
import logging
import sys
import threading
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

_buffer: deque = deque(maxlen=3000)
_lock = threading.Lock()


class _BufferHandler(logging.Handler):
    """Keeps the latest log lines in memory so the Streamlit UI can show them."""

    def emit(self, record: logging.LogRecord) -> None:
        with _lock:
            _buffer.append(self.format(record))


def get_logger(name: str) -> logging.Logger:
    root = logging.getLogger("engine")
    if not root.handlers:  # configure once, even when Streamlit reruns the script
        formatter = logging.Formatter(LOG_FORMAT, "%H:%M:%S")
        handlers = [logging.StreamHandler(sys.stdout), _BufferHandler()]
        try:
            LOG_DIR.mkdir(exist_ok=True)
            handlers.append(RotatingFileHandler(LOG_DIR / "finevoice.log", maxBytes=2_000_000,
                                                backupCount=2, encoding="utf-8"))
        except OSError:
            pass  # read-only folder: terminal + UI logs still work
        for handler in handlers:
            handler.setFormatter(formatter)
            root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logging.getLogger(f"engine.{name}")


def get_log_text(last_n: int = 300) -> str:
    with _lock:
        lines = list(_buffer)[-last_n:]
    return "\n".join(lines) if lines else "(no logs yet)"


def clear_logs() -> None:
    with _lock:
        _buffer.clear()