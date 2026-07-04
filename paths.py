"""Filesystem locations for the cache and logs, resolved for dev vs frozen builds.

Dev runs keep everything in the project's data/ directory. A PyInstaller build
must not write next to the exe (Program Files is read-only), so frozen runs use
%LOCALAPPDATA%\\nasdaq-advisor instead.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

APP_NAME = "nasdaq-advisor"


def data_dir() -> Path:
    if getattr(sys, "frozen", False):
        import platformdirs

        d = Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))
    else:
        d = Path(__file__).resolve().parent / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path() -> Path:
    return data_dir() / "cache.db"


def log_path() -> Path:
    return data_dir() / "advisor.log"


def setup_logging(level: int = logging.INFO) -> None:
    """File logging always (a --windowed exe has no stderr), console in dev."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler = logging.handlers.RotatingFileHandler(
        log_path(), maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    if not getattr(sys, "frozen", False):
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)

    def _excepthook(exc_type, exc, tb):
        logging.getLogger("uncaught").critical(
            "Uncaught exception", exc_info=(exc_type, exc, tb)
        )
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook
