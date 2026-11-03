"""Terminal logging shared by all engine modules."""
import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    root = logging.getLogger("engine")
    if not root.handlers:  # configure once, even when Streamlit reruns the script
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, "%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logging.getLogger(f"engine.{name}")
