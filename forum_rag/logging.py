"""Central logging setup. Call setup_logging() once at process start; every
module then does `log = logging.getLogger(__name__)` as usual.

Console output uses rich for readability; a rotating file (./logs/forum_rag.log)
keeps history for post-mortem debugging. Controlled by LOG_LEVEL (default INFO).
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

_configured = False

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    root.setLevel(level)

    try:
        from rich.logging import RichHandler

        console = RichHandler(show_path=False, rich_tracebacks=True)
        console.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    except Exception:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    root.addHandler(console)

    try:
        LOG_DIR.mkdir(exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / "forum_rag.log", maxBytes=5_000_000, backupCount=3
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        root.addHandler(file_handler)
    except OSError:
        # Read-only / ephemeral filesystem (e.g. some PaaS dynos) — console-only is fine.
        root.warning("Could not open %s for writing; file logging disabled.", LOG_DIR)

    # Quiet noisy third-party loggers unless explicitly debugging.
    for noisy in ("httpx", "httpcore", "urllib3", "google", "googleapiclient"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))
