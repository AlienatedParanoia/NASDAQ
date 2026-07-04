"""Long-term hold: a weighted 0-100 composite over fundamentals.

Each component maps linearly onto [0, 100] with clipping. A missing (None)
component scores a neutral 50 and is reported as missing so the reasoning
string can say so; too many missing components should make the caller
refuse to recommend (handled in the signal generator).

The backtest for this bucket is buy-and-hold — that IS the strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.fetcher import Fundamentals
from strategies import indicators as ind

WEIGHTS = {
    "revenue growth": 20.0,
    "earnings growth": 20.0,
    "profit margins": 20.0,
    "debt": 15.0,
    "valuation": 15.0,
    "beta stability": 10.0,
}
NEUTRAL = 50.0


def _linear(value: float | None, at_zero: float, at_hundred: float) -> float | None:
    """Map value onto 0-100 so that at_zero -> 0 and at_hundred -> 100
    (works in either direction), clipped."""
    if value is None:
        return None
    t = (value - at_zero) / (at_hundred - at_zero)
    return float(np.clip(t, 0.0, 1.0) * 100.0)


def _valuation(pe: float | None, peg: float | None) -> float | None:
    pe_score = _linear(pe, 40.0, 10.0)
    peg_score = _linear(peg, 3.0, 0.5)
    if pe_score is None and peg_score is None:
        return None
    if peg_score is None:
        return pe_score
    if pe_score is None:
        return peg_score
    return 0.6 * pe_score + 0.4 * peg_score


def composite_score(f: Fundamentals) -> tuple[float, dict[str, float | None], int]:
    """Returns (score 0-100, per-component scores with None = missing,
    number of missing components)."""
    components: dict[str, float | None] = {
        "revenue growth": _linear(f.revenue_growth, -0.05, 0.25),
        "earnings growth": _linear(f.earnings_growth, -0.10, 0.30),
        "profit margins": _linear(f.profit_margins, 0.0, 0.30),
        "debt": _linear(f.debt_to_equity, 2.0, 0.0),
        "valuation": _valuation(f.pe, f.peg),
        "beta stability": None if f.beta is None
        else float(np.clip(1.0 - abs(f.beta - 1.0), 0.0, 1.0) * 100.0),
    }
    n_missing = sum(1 for v in components.values() if v is None)
    score = sum(
        WEIGHTS[k] * (v if v is not None else NEUTRAL)
        for k, v in components.items()
    ) / sum(WEIGHTS.values())
    return score, components, n_missing


class LongTermStrategy:
    name = "Buy-and-hold (fundamentals composite)"

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=prices.index)

    def overlays(self, prices: pd.DataFrame) -> dict[str, pd.Series]:
        return {"SMA200": ind.sma(prices["close"], 200)}
