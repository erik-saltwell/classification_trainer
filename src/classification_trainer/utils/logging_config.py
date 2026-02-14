from __future__ import annotations

import logging
import os

from rich.logging import RichHandler


def configure_logging() -> None:
    """
    Configure root logging.

    Reads LOG_LEVEL from the environment (default: INFO).
    Uses RichHandler for nicer console output.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    # Avoid double-configuring if something already set handlers (common in tests/interactive).
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
