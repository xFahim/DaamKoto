"""Centralized logging configuration for Railway-friendly structured logs.

All modules should use:
    from app.core.logging_config import get_logger
    logger = get_logger(__name__)
"""

import logging
import sys
from datetime import datetime, timezone, timedelta


# Bangladesh Standard Time (UTC+6)
BST = timezone(timedelta(hours=6))


class RailwayFormatter(logging.Formatter):
    """Concise, structured formatter optimized for Railway log viewer."""

    LEVEL_ICONS = {
        "DEBUG": "🔍",
        "INFO": "ℹ️",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🔥",
    }

    def format(self, record: logging.LogRecord) -> str:
        # Timestamp in BST
        dt = datetime.fromtimestamp(record.created, tz=BST)
        ts = dt.strftime("%Y-%m-%d %H:%M:%S")

        icon = self.LEVEL_ICONS.get(record.levelname, "")
        module = record.name.replace("app.", "")

        base = f"[{ts}] {icon} {record.levelname:<7} | {module} | {record.getMessage()}"

        if record.exc_info and record.exc_info[0]:
            base += "\n" + self.formatException(record.exc_info)

        return base


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging. Call once at startup."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear any existing handlers to avoid duplicates on reload
    root.handlers.clear()

    handler = logging.StreamHandler(
        stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1, closefd=False)
    )
    handler.setFormatter(RailwayFormatter())
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Use __name__ as the argument."""
    return logging.getLogger(name)
