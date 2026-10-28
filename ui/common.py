"""Bits shared by the Watchlist and My Investments tabs.

Both colour their gains green and their losses red, and format money the same
way; this module is the one definition of each.
"""
from __future__ import annotations

from PyQt6.QtGui import QColor

GREEN = QColor("#1e8e3e")
GREY = QColor("#7a7a7a")
RED = QColor("#d93025")


def money(v: float) -> str:
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):,.0f}"
