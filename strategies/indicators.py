"""Shared technical indicators (pure pandas/numpy, Wilder conventions).

Both the classifier and the strategies import from here so every consumer
sees identical values. All functions return series aligned to the input
index, with NaN during their warmup window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.backtest_engine import time_base


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return rsi


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Columns: mid (SMA), upper, lower."""
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return pd.DataFrame(
        {"mid": mid, "upper": mid + num_std * std, "lower": mid - num_std * std}
    )


def adx_wilder(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    alpha = 1.0 / period
    sm_tr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    sm_plus = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    sm_minus = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * sm_plus / sm_tr
    minus_di = 100.0 * sm_minus / sm_tr
    di_sum = plus_di + minus_di
    dx = (100.0 * (plus_di - minus_di).abs() / di_sum).where(di_sum != 0, 0.0)
    return dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()


def annualized_vol(close: pd.Series, window: int = 200) -> pd.Series:
    """Rolling annualized volatility of daily returns, as a decimal fraction
    (0.30 == 30%).

    Annualized with the series' own observed bars-per-year (via time_base)
    rather than a hardcoded 252, so this matches the calendar-aware
    annualization used by the backtest engine and portfolio math. For real
    US-equity data (~252 trading days/yr) the factor is √252; a crypto or
    foreign listing with a different calendar is scaled correctly instead of
    understated — which matters because the classifier gates buckets on this
    value."""
    _, ppy = time_base(close)
    return close.pct_change().rolling(window, min_periods=window).std(ddof=1) * np.sqrt(ppy)


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on bars where `a` crosses above `b`."""
    crossed = (a > b) & (a.shift(1) <= b.shift(1))
    return crossed.fillna(False)
