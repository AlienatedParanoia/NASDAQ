"""Filesystem locations for the cache, logs and bundled data files, resolved
for dev vs frozen builds.

Two different roots, and conflating them is a bug that shows up only in the
packaged exe:

  * **writable** state (cache.db, logs, investments.json) - dev runs keep it in
    the project's data/ directory; a PyInstaller build must not write next to
    the exe (Program Files is read-only), so frozen runs use the per-user
    %LOCALAPPDATA% app directory instead. That is `data_dir()`.
  * **read-only resources** shipped with the app (the index constituent CSVs) -
    PyInstaller unpacks these into a temporary directory it advertises as
    `sys._MEIPASS`, nowhere near %LOCALAPPDATA%. That is `resource_dir()`.
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


def resource_dir() -> Path:
    """Root for read-only files shipped with the app.

    Deliberately NOT data_dir(): a frozen build unpacks bundled resources into
    sys._MEIPASS, while data_dir() points at the writable %LOCALAPPDATA% cache.
    Reading a shipped file through data_dir() works in dev and fails only once
    packaged, which is the worst time to find out.
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def constituents_dir() -> Path:
    """Where the index membership CSVs live (see data/universe.py)."""
    return resource_dir() / "data" / "constituents"


def db_path() -> Path:
    return data_dir() / "cache.db"


def log_path() -> Path:
    return data_dir() / "advisor.log"


def investments_path() -> Path:
    # user data, deliberately separate from cache.db (which is wiped on
    # CACHE_VERSION bumps)
    return data_dir() / "investments.json"


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
