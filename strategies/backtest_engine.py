"""Shared backtest engine: a 0/1 position series -> performance vs buy-and-hold.

Assumptions (also surfaced in the UI footnote): long-only, all-or-nothing
allocation, fills at the daily close, no transaction costs or slippage,
risk-free rate 0. Closes are auto-adjusted, so dividends are implicitly
included for both the strategy and the benchmark.

Position convention: strategies emit positions already shifted by one bar
(the decision made at bar t's close is held from bar t+1), with NaN over
their warmup window. The engine evaluates strategy and buy-and-hold on the
identical post-warmup window so the comparison is fair.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp | None   # None = still open
    exit_price: float | None


@dataclass(slots=True)
class BacktestResult:
    total_return: float
    sharpe: float
    max_drawdown: float              # <= 0
    bh_total_return: float
    bh_sharpe: float
    bh_max_drawdown: float
    n_trades: int
    positions: pd.Series             # 0/1 over the evaluation window
    equity: pd.Series
    trades: list[Trade]


def _perf(returns: pd.Series) -> tuple[float, float, float, pd.Series]:
    equity = (1.0 + returns).cumprod()
    if len(equity) == 0:
        return 0.0, 0.0, 0.0, equity
    total = float(equity.iloc[-1] - 1.0)
    std = returns.std(ddof=1)
    sharpe = 0.0
    if pd.notna(std) and std > 0:
        sharpe = float(returns.mean() / std * np.sqrt(TRADING_DAYS))
    drawdown = float((equity / equity.cummax() - 1.0).min())
    return total, sharpe, drawdown, equity


def _extract_trades(close: pd.Series, pos: pd.Series) -> list[Trade]:
    trades: list[Trade] = []
    entry_date = None
    entry_price = 0.0
    prev = 0.0
    for date, p in pos.items():
        if p > prev:
            entry_date = date
            entry_price = float(close.loc[date])
        elif p < prev and entry_date is not None:
            trades.append(Trade(entry_date, entry_price, date, float(close.loc[date])))
            entry_date = None
        prev = p
    if entry_date is not None:
        trades.append(Trade(entry_date, entry_price, None, None))
    return trades


def run_backtest(prices: pd.DataFrame, positions: pd.Series) -> BacktestResult:
    close = prices["close"]
    pos = positions.reindex(close.index)
    window = pos.dropna().index
    if len(window) < 2:
        empty = pd.Series(dtype=float)
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, pos.dropna(), empty, [])

    pos = pos.loc[window].astype(float)
    if not pos.isin([0.0, 1.0]).all():
        raise ValueError("positions must be 0 or 1 (long-only, all-or-nothing)")

    close_w = close.loc[window]
    r = close_w.pct_change().iloc[1:]
    strat_returns = pos.iloc[1:] * r

    total, sharpe, dd, equity = _perf(strat_returns)
    bh_total, bh_sharpe, bh_dd, _ = _perf(r)
    trades = _extract_trades(close_w, pos)

    return BacktestResult(
        total_return=total,
        sharpe=sharpe,
        max_drawdown=dd,
        bh_total_return=bh_total,
        bh_sharpe=bh_sharpe,
        bh_max_drawdown=bh_dd,
        n_trades=len(trades),
        positions=pos,
        equity=equity,
        trades=trades,
    )
