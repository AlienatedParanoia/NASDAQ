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
    years: float                     # evaluation window span in trading years
    cagr: float                      # annualized total_return over that span
    bh_cagr: float
    returns: pd.Series | None = None  # daily strategy returns (portfolio math)


def time_base(returns: pd.Series) -> tuple[float, float]:
    """(span in calendar years, observed bars per year) for a return series.

    Calendar-based rather than a hardcoded 252 so non-equity calendars
    (crypto trades ~365 days/yr, foreign listings differ too) annualize
    correctly; falls back to 252 for non-datetime indexes."""
    n = len(returns)
    if n > 1 and isinstance(returns.index, pd.DatetimeIndex):
        years = max((returns.index[-1] - returns.index[0]).days, 1) / 365.25
        return years, n / years
    return n / TRADING_DAYS, float(TRADING_DAYS)


def perf_stats(returns: pd.Series) -> tuple[float, float, float, float, pd.Series]:
    """(total_return, sharpe, max_drawdown, cagr, equity) for a daily return
    series. Shared by the backtest engine and the portfolio aggregation so
    both use identical formulas."""
    equity = (1.0 + returns).cumprod()
    if len(equity) == 0:
        return 0.0, 0.0, 0.0, 0.0, equity
    total = float(equity.iloc[-1] - 1.0)
    years, ppy = time_base(returns)
    std = returns.std(ddof=1)
    sharpe = 0.0
    if pd.notna(std) and std > 0:
        sharpe = float(returns.mean() / std * np.sqrt(ppy))
    # the peak includes the starting capital of 1.0, else a window that opens
    # with a decline understates its drawdown by the first bar's loss
    peak = equity.cummax().clip(lower=1.0)
    drawdown = float(min((equity / peak - 1.0).min(), 0.0))
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    return total, sharpe, drawdown, cagr, equity


def _extract_trades(close: pd.Series, pos: pd.Series) -> list[Trade]:
    """Fills are at the decision bar's close — the bar BEFORE the (already
    shifted) position series changes — so recorded trades reconcile exactly
    with the return accounting. At the window's first bar there is no prior
    bar and the first return is dropped, so that bar is its own fill."""
    trades: list[Trade] = []
    entry_date = None
    entry_price = 0.0
    prev = 0.0
    prev_date = None
    for date, p in pos.items():
        if p > prev:
            fill = prev_date if prev_date is not None else date
            entry_date = fill
            entry_price = float(close.loc[fill])
        elif p < prev and entry_date is not None:
            trades.append(
                Trade(entry_date, entry_price, prev_date, float(close.loc[prev_date]))
            )
            entry_date = None
        prev = p
        prev_date = date
    if entry_date is not None:
        trades.append(Trade(entry_date, entry_price, None, None))
    return trades


def run_backtest(prices: pd.DataFrame, positions: pd.Series) -> BacktestResult:
    close = prices["close"]
    pos = positions.reindex(close.index)
    window = pos.dropna().index
    if len(window) < 2:
        empty = pd.Series(dtype=float)
        return BacktestResult(
            total_return=0.0, sharpe=0.0, max_drawdown=0.0,
            bh_total_return=0.0, bh_sharpe=0.0, bh_max_drawdown=0.0,
            n_trades=0, positions=pos.dropna(), equity=empty, trades=[],
            years=0.0, cagr=0.0, bh_cagr=0.0, returns=empty,
        )

    pos = pos.loc[window].astype(float)
    if not pos.isin([0.0, 1.0]).all():
        raise ValueError("positions must be 0 or 1 (long-only, all-or-nothing)")

    close_w = close.loc[window]
    r = close_w.pct_change().iloc[1:]
    strat_returns = pos.iloc[1:] * r

    total, sharpe, dd, cagr, equity = perf_stats(strat_returns)
    bh_total, bh_sharpe, bh_dd, bh_cagr, _ = perf_stats(r)
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
        years=time_base(r)[0],
        cagr=cagr,
        bh_cagr=bh_cagr,
        returns=strat_returns,
    )
