"""Phase 3 sanity check (offline once the cache is warm): synthetic-series
truths, a look-ahead test, and invariants across every cached ticker.

Run from the project root: python -m tests.phase3_backtest_check
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import paths
from data.fetcher import DataFetcher, FetchError
from data.universe import get_universe
from strategies.backtest_engine import run_backtest
from strategies.mean_reversion import MeanReversionStrategy
from strategies.trend import TrendStrategy


def _synthetic(closes: np.ndarray) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": np.full(len(closes), 1000.0),
        },
        index=idx,
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    # 1. linear up-ramp: trend should be fully invested and track buy-and-hold
    ramp = _synthetic(np.linspace(100.0, 200.0, 300))
    trend = TrendStrategy()
    bt = run_backtest(ramp, trend.positions(ramp))
    assert bt.n_trades == 1 and bt.trades[0].exit_date is None, "ramp should be one open trade"
    assert abs(bt.total_return - bt.bh_total_return) < 1e-12, (
        f"fully-invested trend must equal B&H: {bt.total_return} vs {bt.bh_total_return}"
    )
    assert bt.max_drawdown == 0.0
    print(f"up-ramp: trend return {bt.total_return:+.1%} == B&H, 1 open trade OK")

    # mean reversion never triggers on a clean ramp
    mr = MeanReversionStrategy()
    bt_mr = run_backtest(ramp, mr.positions(ramp))
    assert bt_mr.n_trades == 0 and bt_mr.total_return == 0.0 and bt_mr.sharpe == 0.0
    print("up-ramp: mean-reversion stays flat OK")

    # 2. flat series: everything is zero
    flat = _synthetic(np.full(300, 100.0))
    bt_flat = run_backtest(flat, trend.positions(flat))
    assert bt_flat.total_return == 0.0 and bt_flat.sharpe == 0.0 and bt_flat.max_drawdown == 0.0
    print("flat series: 0/0/0 OK")

    # 3. look-ahead test on real data: dropping the last bar must not change
    # any earlier position
    fetcher = DataFetcher(paths.db_path())
    aapl = fetcher.get_stock("AAPL").prices
    for strat in (TrendStrategy(), MeanReversionStrategy()):
        full = strat.positions(aapl)
        trunc = strat.positions(aapl.iloc[:-1])
        pd.testing.assert_series_equal(full.loc[trunc.index], trunc, check_names=False)
    print("look-ahead test: truncating history changes no earlier positions OK")

    # 4. invariants + Sharpe tripwire across every cached ticker
    strategies = {
        "trend": TrendStrategy(),
        "mean_reversion": MeanReversionStrategy(),
    }
    suspicious: list[tuple[str, str, float, int]] = []
    checked = 0
    for t in get_universe(("nasdaq100",)):
        try:
            px = fetcher.get_stock(t).prices
        except FetchError:
            continue
        if len(px) < 260:
            continue
        for name, strat in strategies.items():
            bt = run_backtest(px, strat.positions(px))
            assert bt.max_drawdown <= 0.0, f"{t}/{name}: positive drawdown"
            assert bt.max_drawdown >= -1.0, f"{t}/{name}: drawdown beyond -100%"
            assert bt.n_trades >= 0
            if bt.sharpe > 3.0:
                suspicious.append((t, name, bt.sharpe, bt.n_trades))
        checked += 1
    fetcher.close()

    print(f"invariants OK across {checked} tickers x {len(strategies)} strategies")
    if suspicious:
        print("SHARPE > 3 OCCURRENCES (investigate each):")
        for t, name, s, n in suspicious:
            print(f"  {t} {name}: Sharpe {s:.2f} over {n} trades")
    else:
        print("no Sharpe > 3 occurrences")

    print("PHASE 3 CHECKS PASSED")


if __name__ == "__main__":
    main()
