"""Ticker normalisation — the single definition of what a symbol looks like.

Yahoo writes class shares with a hyphen (BRK-B, BF-B) where most index sources,
and most humans, write a dot (BRK.B). Left alone, the same company enters the
cache, the universe and the ledger under two spellings and silently becomes two
positions.

This lives in its own module rather than in fetcher.py because the universe, the
constituents refresh script, the portfolio tracker and the trade ledger all need
it, and none of them should have to import the fetcher to get it.
"""
from __future__ import annotations

from typing import Iterable

__all__ = ["normalize", "normalize_all"]


def normalize(ticker: str) -> str:
    """'brk.b' -> 'BRK-B'. Upper-cased, trimmed, dots folded to hyphens.

    Raises ValueError on a symbol with no alphanumeric character: a blank (or
    punctuation-only) ticker reaching the fetcher becomes a confusing Yahoo 404
    several layers away from the real mistake.
    """
    cleaned = str(ticker).strip().upper().replace(".", "-")
    if not any(c.isalnum() for c in cleaned):
        raise ValueError(f"not a usable ticker symbol: {ticker!r}")
    return cleaned


def normalize_all(tickers: Iterable[str]) -> list[str]:
    """Normalise a sequence, dropping duplicates and preserving first-seen
    order (so a universe stays in its source order rather than being sorted)."""
    seen: dict[str, None] = {}
    for t in tickers:
        seen.setdefault(normalize(t), None)
    return list(seen)
