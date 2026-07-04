"""Shared JSON persistence helpers for user data files (portfolio,
investments): atomic writes and corrupt-file recovery with numbered backups."""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def write_json_atomic(path: Path, payload: dict) -> None:
    """Temp file + atomic replace: a crash mid-write can never leave a
    truncated file behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def backup_file(path: Path) -> None:
    """Move a damaged file aside to a numbered .bak, never clobbering an
    earlier — possibly good — backup. Callers MUST do this before any code
    path that would overwrite the file, or user data is silently lost."""
    try:
        backup = path.with_suffix(path.suffix + ".bak")
        n = 1
        while backup.exists():
            backup = path.with_suffix(f"{path.suffix}.bak{n}")
            n += 1
        path.replace(backup)
    except OSError:
        pass


def read_json_or_backup(path: Path) -> dict | None:
    """Parse JSON; on corruption back the file up and return None."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.exception("%s unreadable; starting empty (backup kept)", path.name)
        backup_file(path)
        return None
