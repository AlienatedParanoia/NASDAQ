"""Regression checks for the defects confirmed by the pre-packaging
adversarial review (trade-fill offset, drawdown peak seeding, calendar-aware
CAGR, uninterruptible rate-limit backoff). Offline.

The negative-PEG polarity check went with strategies/long_term.py when the
signal system was removed in phase 0; the fundamentals scoring it guarded no
longer exists.

Run from the project root: python -m tests.regression_review_fixes
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import paths
from data.fetcher import DataFetcher, FetchError
from strategies.backtest_engine import run_backtest


def _df(closes) -> pd.DataFrame:
    c = np.asarray(closes, dtype=float)
    idx = pd.bdate_range("2024-01-01", periods=len(c))
    return pd.DataFrame(
        {"open": c, "high": c, "low": c, "close": c, "volume": np.ones(len(c))},
        index=idx,
    )


def trade_fills_reconcile() -> None:
    """Recorded trades must reconcile exactly with the return accounting."""
    px = _df([100, 100, 100, 97, 103, 104, 105, 105])
    pos = pd.Series([0, 0, 0, 0, 1, 1, 0, 0], index=px.index, dtype=float)
    bt = run_backtest(px, pos)
    assert bt.n_trades == 1
    trade = bt.trades[0]
    assert trade.entry_price == 97.0 and trade.exit_price == 104.0, (
        trade.entry_price, trade.exit_price,
    )
    trade_return = trade.exit_price / trade.entry_price - 1.0
    assert abs(trade_return - bt.total_return) < 1e-12, (trade_return, bt.total_return)
    print(f"trade fills reconcile: entry 97 -> exit 104 == total {bt.total_return:+.3%}")


def drawdown_includes_starting_capital() -> None:
    px = _df([100, 90, 81])
    bt = run_backtest(px, pd.Series(1.0, index=px.index))
    assert abs(bt.total_return - (-0.19)) < 1e-9
    assert abs(bt.max_drawdown - (-0.19)) < 1e-9, bt.max_drawdown
    assert abs(bt.bh_max_drawdown - (-0.19)) < 1e-9
    print(f"drawdown measured from starting capital: {bt.max_drawdown:.1%}")


def cagr_annualizes_correctly() -> None:
    # +100% over 504 bars; CAGR must annualize by the CALENDAR span so that
    # non-252-day calendars (crypto, foreign listings) come out right too
    daily = 2.0 ** (1.0 / 504)
    closes = 100.0 * daily ** np.arange(505)
    px = _df(closes)
    bt = run_backtest(px, pd.Series(1.0, index=px.index))
    years = (px.index[-1] - px.index[1]).days / 365.25  # returns start at bar 2
    assert abs(bt.total_return - 1.0) < 1e-6, bt.total_return
    assert abs(bt.years - years) < 1e-9, (bt.years, years)
    assert abs(bt.cagr - (2.0 ** (1.0 / years) - 1.0)) < 1e-6, bt.cagr
    print(f"CAGR sanity: +100% over {bt.years:.2f} calendar years -> {bt.cagr:+.2%}/yr")


def backoff_interruptible_and_breaker_fails_fast() -> None:
    cancelled = DataFetcher(paths.db_path(), should_stop=lambda: True)
    t0 = time.perf_counter()
    try:
        cancelled._backoff(60.0)
        raise AssertionError("backoff should raise once cancelled")
    except FetchError:
        pass
    assert time.perf_counter() - t0 < 2.0, "cancel did not interrupt the backoff"
    cancelled.close()

    cooled = DataFetcher(paths.db_path())
    DataFetcher._shared_cooldown_until = time.monotonic() + 300.0  # shared breaker
    try:
        try:
            cooled._network_call(
                lambda: (_ for _ in ()).throw(RuntimeError("no network!"))
            )
            raise AssertionError("cooldown should fail fast without a network call")
        except FetchError as e:
            assert "cooldown" in str(e)
        assert cooled.network_fetches == 0
    finally:
        DataFetcher._shared_cooldown_until = 0.0
        cooled.close()
    print("interruptible backoff + rate-limit circuit breaker OK")


def main() -> None:
    trade_fills_reconcile()
    drawdown_includes_starting_capital()
    cagr_annualizes_correctly()
    backoff_interruptible_and_breaker_fails_fast()
    print("REVIEW-FIX REGRESSION CHECKS PASSED")


if __name__ == "__main__":
    main()
