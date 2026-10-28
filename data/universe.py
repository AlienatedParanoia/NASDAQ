"""Index membership, read from data/constituents/*.csv.

The lists are data files rather than code so a reconstitution is a reviewable
diff instead of a hand-edited tuple, and so one maintenance script
(scripts/refresh_constituents.py) can rebuild them from Wikipedia.

**Survivorship bias — the thing to remember about every backtest run on this
universe.** These files hold *today's* members. A company that was in the S&P 500
in 2018 and then went bankrupt, got acquired, or was simply demoted is not here,
and yfinance mostly cannot serve its prices anyway. So any historical replay runs
on a pre-filtered list of survivors and will read optimistically — the losers
were removed before the test began. The honest measurement is forward paper
trading, where the universe is not yet known to have survived.

The two indexes do not describe themselves identically: `sector` is GICS for the
S&P 500 and ICB for the NASDAQ-100, so it is comparable within an index but not
across them, and `date_added` is empty for every NASDAQ-100 row because the
source table has no such column.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import paths
from data.symbols import normalize, normalize_all

log = logging.getLogger(__name__)

INDEXES: dict[str, str] = {
    "sp500": "S&P 500",
    "nasdaq100": "NASDAQ-100",
}

DEFAULT_INDEXES: tuple[str, ...] = ("sp500", "nasdaq100")

SURVIVORSHIP_NOTE = (
    "Universe is today's index members — delisted companies are gone, so "
    "backtests read optimistic. Forward paper trading is the honest test."
)


@dataclass(frozen=True, slots=True)
class Constituent:
    ticker: str
    name: str
    sector: str       # GICS for the S&P 500, ICB for the NASDAQ-100
    date_added: str   # ISO date, or "" when the source supplies none


class UnknownIndex(KeyError):
    """Asked for an index that does not exist."""


def _csv_path(index: str) -> Path:
    return paths.constituents_dir() / f"{index}.csv"


@lru_cache(maxsize=8)
def get_constituents(index: str) -> tuple[Constituent, ...]:
    """Every member of one index. Cached — the watchlist re-reads the universe
    on each refresh, and these files cannot change while the app is running."""
    if index not in INDEXES:
        raise UnknownIndex(f"unknown index {index!r}; expected one of {sorted(INDEXES)}")
    path = _csv_path(index)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — run scripts/refresh_constituents.py "
            f"(and check that the build bundles data/constituents/)"
        )

    members: list[Constituent] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("ticker") or "").strip()
            if not raw:
                continue
            ticker = normalize(raw)
            if ticker in seen:
                continue
            seen.add(ticker)
            members.append(
                Constituent(
                    ticker=ticker,
                    name=(row.get("name") or "").strip(),
                    sector=(row.get("sector") or "").strip(),
                    date_added=(row.get("date_added") or "").strip(),
                )
            )
    if not members:
        raise ValueError(f"{path} has no usable rows")
    return tuple(members)


def get_universe(indexes: Sequence[str] | None = None) -> list[str]:
    """De-duplicated union of the named indexes, in index-then-file order.

    Defaults to both. An unknown name raises rather than quietly returning a
    short list — a typo in a UI selector should be loud, not a silently smaller
    universe for the bots to trade.
    """
    chosen = DEFAULT_INDEXES if indexes is None else tuple(indexes)
    if not chosen:
        raise UnknownIndex("no indexes selected")
    tickers: list[str] = []
    for index in chosen:
        tickers.extend(c.ticker for c in get_constituents(index))
    return normalize_all(tickers)


def index_label(index: str) -> str:
    if index not in INDEXES:
        raise UnknownIndex(f"unknown index {index!r}")
    return INDEXES[index]


def describe(indexes: Sequence[str] | None = None) -> str:
    """'S&P 500 + NASDAQ-100 (518 symbols)' — for status lines."""
    chosen = DEFAULT_INDEXES if indexes is None else tuple(indexes)
    names = " + ".join(index_label(i) for i in chosen)
    return f"{names} ({len(get_universe(chosen))} symbols)"
