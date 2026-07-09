"""Bits shared by the portfolio tabs (Paper Trading and My Investments).

Both panels offer the same strategy dropdown, colour their P&L and Signal
cells the same way, and format money and dates identically; this module is
the one definition of each.
"""
from __future__ import annotations

import datetime as _dt

from PyQt6.QtGui import QColor

from classifier.classify import Bucket
from signals.signal_generator import Signal

GREEN = QColor("#1e8e3e")
GREY = QColor("#7a7a7a")
RED = QColor("#d93025")

SIGNAL_COLORS = {
    Signal.BUY: GREEN,
    Signal.HOLD: GREY,
    Signal.AVOID: RED,
}

SIDE_COLORS = {"Buy": GREEN, "Sell": RED}

# The strategy a position is deployed under. None = let classify() pick.
STRATEGY_CHOICES = (
    ("Recommended (auto)", None),
    ("Trend", Bucket.TREND.value),
    ("Mean-reversion", Bucket.MEAN_REVERSION.value),
    ("Long-term hold", Bucket.LONG_TERM.value),
)
_STRATEGY_LABELS = {value: label for label, value in STRATEGY_CHOICES}


def strategy_choice_label(strategy: str | None) -> str:
    return _STRATEGY_LABELS.get(strategy, "Recommended (auto)")


def strategy_choice_index(strategy: str | None) -> int:
    """Index of `strategy` in STRATEGY_CHOICES; 0 ("Recommended") if unknown."""
    return next(
        (i for i, (_, value) in enumerate(STRATEGY_CHOICES) if value == strategy), 0
    )


def money(v: float) -> str:
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):,.0f}"


def day_month(iso: str) -> str:
    """ISO YYYY-MM-DD -> DD/MM. Anything unparseable renders as-is."""
    try:
        return _dt.date.fromisoformat(iso).strftime("%d/%m")
    except (ValueError, TypeError):
        return iso
