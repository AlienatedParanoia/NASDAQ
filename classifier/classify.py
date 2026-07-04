"""Three-bucket classification: which strategy family fits each stock.

Priority order is fixed: long-term hold, then trend, then mean-reversion,
else unclassified. A metric that is unavailable (missing fundamentals or
fewer than 200 bars of history) makes any rule requiring it evaluate False,
so incomplete data falls through to Unclassified instead of crashing.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

import pandas as pd

from data.fetcher import StockData
from strategies import indicators as ind


class Bucket(enum.StrEnum):
    LONG_TERM = "Long-term"
    TREND = "Trend"
    MEAN_REVERSION = "Mean-reversion"
    UNCLASSIFIED = "Unclassified"


@dataclass(frozen=True, slots=True)
class ClassifierMetrics:
    vol_ann: float | None          # 200-day annualized volatility, decimal
    adx: float | None              # ADX(14), latest
    beta: float | None
    revenue_growth: float | None   # decimal fraction, MRQ YoY
    debt_to_equity: float | None   # ratio
    pct_from_200ma: float | None   # close / SMA200 - 1


def _last(series: pd.Series | None) -> float | None:
    if series is None or len(series) == 0:
        return None
    v = series.iloc[-1]
    return float(v) if pd.notna(v) else None


def compute_metrics(data: StockData) -> ClassifierMetrics:
    px = data.prices
    close = px["close"]
    ma200 = _last(ind.sma(close, 200))
    pct_from_200ma = None
    if ma200 is not None and ma200 > 0:
        pct_from_200ma = float(close.iloc[-1] / ma200 - 1.0)
    f = data.fundamentals
    return ClassifierMetrics(
        vol_ann=_last(ind.annualized_vol(close, 200)),
        adx=_last(ind.adx_wilder(px["high"], px["low"], close, 14)),
        beta=f.beta if f else None,
        revenue_growth=f.revenue_growth if f else None,
        debt_to_equity=f.debt_to_equity if f else None,
        pct_from_200ma=pct_from_200ma,
    )


def _fmt(v: float | None, spec: str = "{:.2f}") -> str:
    return spec.format(v) if v is not None else "n/a"


def classify(m: ClassifierMetrics) -> tuple[Bucket, str]:
    """Returns (bucket, one-line rule trace for the reasoning string)."""
    if (
        m.vol_ann is not None and m.vol_ann < 0.30
        and m.debt_to_equity is not None and m.debt_to_equity < 1.5
        and m.revenue_growth is not None and m.revenue_growth > 0.05
        and m.beta is not None and 0.7 <= m.beta <= 1.3
    ):
        return Bucket.LONG_TERM, (
            f"vol {m.vol_ann:.0%} < 30%, D/E {m.debt_to_equity:.2f} < 1.5, "
            f"revenue growth {m.revenue_growth:.1%} > 5%, beta {m.beta:.2f} in [0.7, 1.3]"
        )
    if (
        m.adx is not None and m.adx > 25
        and m.pct_from_200ma is not None and abs(m.pct_from_200ma) > 0.05
    ):
        return Bucket.TREND, (
            f"ADX {m.adx:.1f} > 25, price {m.pct_from_200ma:+.1%} vs 200MA (>5% away)"
        )
    if (
        m.adx is not None and m.adx < 20
        and m.vol_ann is not None and m.vol_ann > 0.25
    ):
        return Bucket.MEAN_REVERSION, (
            f"ADX {m.adx:.1f} < 20 (rangebound), vol {m.vol_ann:.0%} > 25%"
        )
    return Bucket.UNCLASSIFIED, (
        "no bucket matched — "
        f"vol {_fmt(m.vol_ann, '{:.0%}')}, ADX {_fmt(m.adx, '{:.1f}')}, "
        f"beta {_fmt(m.beta)}, revenue growth {_fmt(m.revenue_growth, '{:.1%}')}, "
        f"D/E {_fmt(m.debt_to_equity)}, 200MA distance {_fmt(m.pct_from_200ma, '{:+.1%}')}"
    )
