"""Phase 6 checks: paper-portfolio math, persistence, and strategy overrides.

Offline once the cache is warm (the bogus-ticker case makes one quick failed
network call). Run from the project root: python -m tests.phase6_portfolio_check
"""
from __future__ import annotations

import json
import logging
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import paths
from classifier.classify import Bucket
from data.fetcher import DataFetcher, StockData
from portfolio.manager import (
    Holding,
    analyze_portfolio,
    joint_metrics,
    load_portfolio,
    save_portfolio,
)
from signals.signal_generator import Signal, generate_signal
from strategies.backtest_engine import TRADING_DAYS


def joint_math_hand_check() -> None:
    idx = pd.bdate_range("2024-01-01", periods=2)
    a = pd.Series([0.10, 0.00], index=idx)  # +10% then flat
    b = pd.Series([0.00, 0.10], index=idx)  # flat then +10%
    jm = joint_metrics({"A": a, "B": b}, {"A": 100.0, "B": 100.0})
    assert jm is not None
    assert abs(jm.total_return - 0.10) < 1e-12, jm.total_return
    profits = 100.0 * jm.common_returns["A"] + 100.0 * jm.common_returns["B"]
    assert abs(profits - 200.0 * jm.total_return) < 1e-9, (profits, jm.total_return)
    print(f"hand check: portfolio +{jm.total_return:.0%}, sum of profits == total profit")


def diversification_shows_up() -> None:
    idx = pd.bdate_range("2024-01-01", periods=200)
    swing = np.tile([0.01, -0.01], 100)
    a = pd.Series(swing, index=idx)
    b = pd.Series(-swing, index=idx)
    vol_single = float(a.std(ddof=1) * np.sqrt(TRADING_DAYS))
    jm = joint_metrics({"A": a, "B": b}, {"A": 100.0, "B": 100.0})
    assert jm is not None
    assert jm.ann_vol < 0.2 * vol_single, (jm.ann_vol, vol_single)
    print(
        f"diversification: single-stock vol {vol_single:.1%}/yr vs "
        f"anti-correlated portfolio {jm.ann_vol:.2%}/yr"
    )


def persistence_roundtrip_and_corruption() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "portfolio.json"
        holdings = [Holding("AAPL", 1500.0, None), Holding("SHOP", 800.0, "Trend")]
        save_portfolio(holdings, p)
        loaded = load_portfolio(p)
        assert loaded == holdings, loaded
        assert json.loads(p.read_text())["version"] == 1

        p.write_text("{ this is not json", encoding="utf-8")
        recovered = load_portfolio(p)
        assert recovered == []
        assert p.with_suffix(".json.bak").exists(), "corrupt file should be backed up"
    print("persistence round-trip + corruption backup OK")


def overrides_change_the_analysis(fetcher: DataFetcher) -> None:
    data = fetcher.get_stock("AAPL")
    auto = generate_signal(data, 10)
    trend = generate_signal(data, 10, strategy_override=Bucket.TREND)
    meanrev = generate_signal(data, 10, strategy_override=Bucket.MEAN_REVERSION)

    assert trend.bucket is Bucket.TREND
    assert meanrev.bucket is Bucket.MEAN_REVERSION
    assert len(trend.trades) > 1, "EMA crossover over 10y should trade repeatedly"
    if auto.bucket is not Bucket.TREND:
        assert "recommends" in trend.reasoning, trend.reasoning[:120]
    assert trend.total_return != meanrev.total_return, "different strategies, same numbers?"
    print(
        f"override OK: AAPL auto={auto.bucket}, forced trend "
        f"{trend.total_return:+.0%} ({len(trend.trades)} trades) vs forced mean-rev "
        f"{meanrev.total_return:+.0%}"
    )

    bare = StockData(ticker="AAPL", prices=data.prices, fundamentals=None, stale=False)
    forced_lt = generate_signal(bare, 10, strategy_override=Bucket.LONG_TERM)
    assert forced_lt.signal is Signal.AVOID
    assert "insufficient" in forced_lt.reasoning.lower()
    print("forced long-term without fundamentals -> Avoid, no crash")


def real_portfolio_report(fetcher: DataFetcher) -> None:
    holdings = [
        Holding("AAPL", 4000.0),
        Holding("NVDA", 3000.0),
        Holding("KHC", 3000.0),
    ]
    rep = analyze_portfolio(holdings, fetcher, 10)
    assert rep.total_invested == 10_000.0
    assert abs(sum(r.weight for r in rep.results) - 1.0) < 1e-9
    for name in ("total_return", "profit", "cagr", "ann_vol", "sharpe", "max_drawdown", "years"):
        v = getattr(rep, name)
        assert v is not None and math.isfinite(v), (name, v)
    per_holding = sum(r.profit for r in rep.results if r.profit is not None)
    assert abs(per_holding - rep.profit) < 1e-6
    assert rep.risk in ("Low", "Moderate", "High", "Very high")
    print(
        f"AAPL/NVDA/KHC $10k over {rep.years:.1f}y: return {rep.total_return:+.0%}, "
        f"profit {rep.profit:+,.0f}, risk {rep.ann_vol:.0%}/yr ({rep.risk}), "
        f"Sharpe {rep.sharpe:.2f}, MaxDD {rep.max_drawdown:.1%}"
    )

    rep2 = analyze_portfolio(holdings + [Holding("SNDK", 1000.0)], fetcher, 10)
    assert rep2.limiting_ticker == "SNDK", rep2.limiting_ticker
    assert rep2.years is not None and rep2.years < 2.0, rep2.years
    print(f"young holding: SNDK limits the joint window to {rep2.years:.1f}y")

    rep3 = analyze_portfolio(
        holdings + [Holding("ZZZINVALIDZZZ", 500.0)], fetcher, 10
    )
    assert "ZZZINVALIDZZZ" in rep3.skipped
    assert rep3.total_return is not None, "others must still be analyzed"
    print("bogus holding skipped without killing the report")


def unclassified_falls_back_to_buy_and_hold(fetcher: DataFetcher) -> None:
    from classifier.classify import classify, compute_metrics

    candidate = None
    for t in ("PEP", "WMT", "XEL", "AMD", "AMZN", "INTC", "ORLY", "MAR"):
        data = fetcher.get_stock(t)
        if classify(compute_metrics(data))[0] is Bucket.UNCLASSIFIED:
            candidate = t
            break
    if candidate is None:
        print("NOTE: no currently-unclassified candidate found; fallback check skipped")
        return
    rep = analyze_portfolio([Holding(candidate, 1000.0)], fetcher, 10)
    res = rep.results[0]
    assert "Buy & hold" in res.strategy_label, res.strategy_label
    assert rep.total_return is not None
    print(f"unclassified fallback OK on {candidate}: {res.strategy_label}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    joint_math_hand_check()
    diversification_shows_up()
    persistence_roundtrip_and_corruption()
    fetcher = DataFetcher(paths.db_path())
    try:
        overrides_change_the_analysis(fetcher)
        real_portfolio_report(fetcher)
        unclassified_falls_back_to_buy_and_hold(fetcher)
    finally:
        fetcher.close()
    print("PHASE 6 PORTFOLIO CHECKS PASSED")


if __name__ == "__main__":
    main()
