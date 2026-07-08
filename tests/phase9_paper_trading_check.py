"""Phase 9 checks: forward-simulated paper trading (portfolio/paper.py).

The pure slice/compound/persistence helpers run fully offline; the end-to-end
analyze_paper check reuses the warm price cache (plus one quick failed network
call for the bogus-ticker case, like phase 7).

Run from the project root: python -m tests.phase9_paper_trading_check
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pandas as pd

import paths
from classifier.classify import Bucket
from data.fetcher import DataFetcher
from portfolio.paper import (
    PaperHolding,
    PaperSession,
    _fills_since,
    _holding_pnl,
    _position_of,
    analyze_paper,
    load_paper,
    save_paper,
)
from signals.signal_generator import ScanRow, Signal
from strategies.backtest_engine import Trade


def _scan_row(bucket: Bucket, trades: list[Trade]) -> ScanRow:
    return ScanRow(
        ticker="TEST", bucket=bucket, signal=Signal.BUY, total_return=0.0, cagr=0.0,
        sharpe=0.0, max_dd=0.0, bh_return=0.0, bh_cagr=0.0, reasoning="", trades=trades,
    )


def holding_pnl_slices_and_compounds() -> None:
    idx = pd.date_range("2026-01-01", periods=10, freq="D")
    returns = pd.Series([0.01] * 10, index=idx)
    start = idx[4]  # keep 6 bars (idx[4..9])

    equity, value, pnl, ret = _holding_pnl(returns, 1000.0, start)
    assert len(equity) == 6, len(equity)
    expected = 1000.0 * (1.01 ** 6)
    assert abs(value - expected) < 1e-6, (value, expected)
    assert abs(pnl - (expected - 1000.0)) < 1e-6
    assert abs(ret - pnl / 1000.0) < 1e-12

    # a session that started today (no bars yet) is all cash: value == budget
    equity0, value0, pnl0, ret0 = _holding_pnl(returns, 1000.0, idx[-1] + pd.Timedelta(days=5))
    assert len(equity0) == 0 and value0 == 1000.0 and pnl0 == 0.0 and ret0 == 0.0
    print(f"_holding_pnl OK: 6 bars @1% -> ${value:,.2f}; empty slice stays at budget")


def fills_and_position_respect_start() -> None:
    start = pd.Timestamp("2026-03-01")
    before = Trade(pd.Timestamp("2026-02-01"), 100.0, pd.Timestamp("2026-02-15"), 110.0)
    across = Trade(pd.Timestamp("2026-02-20"), 100.0, pd.Timestamp("2026-03-10"), 120.0)
    after_open = Trade(pd.Timestamp("2026-03-15"), 130.0, None, None)

    fills = _fills_since("AAA", [before, across, after_open], start)
    # 'before' is entirely pre-start (dropped); 'across' contributes only its
    # sell (>= start); 'after_open' contributes its buy
    got = [(f.side, f.date.date().isoformat()) for f in fills]
    assert got == [("Sell", "2026-03-10"), ("Buy", "2026-03-15")], got

    assert _position_of(_scan_row(Bucket.TREND, [after_open])) == "In"
    assert _position_of(_scan_row(Bucket.TREND, [before])) == "Flat"
    assert _position_of(_scan_row(Bucket.TREND, [])) == "Flat"
    assert _position_of(_scan_row(Bucket.UNCLASSIFIED, [])) == "In"  # plain buy & hold
    print(f"fills sliced to >= start OK: {got}; position derivation OK")


def persistence_roundtrip_and_corruption() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "paper_trading.json"
        session = PaperSession(
            "My session", "2026-06-01",
            [
                PaperHolding("AAPL", 1000.0, None),
                PaperHolding("SHOP", 2500.0, Bucket.TREND.value),
            ],
        )
        save_paper(session, p)
        loaded = load_paper(p)
        assert loaded == session, loaded
        assert json.loads(p.read_text())["version"] == 1
        p.write_text("{ nope", encoding="utf-8")
        empty = load_paper(p)
        assert empty.holdings == [] and empty.name and empty.start_date
        assert p.with_suffix(".json.bak").exists()
    print("paper persistence round-trip + corruption backup OK")


def structurally_bad_files_are_backed_up() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "paper_trading.json"
        p.write_text(json.dumps({"session": {}, "holdings": "nope"}), encoding="utf-8")
        assert load_paper(p).holdings == []
        assert p.with_suffix(".json.bak").exists(), "structural damage not backed up"
        assert not p.exists()
    print("structurally-bad paper files are backed up, not silently overwritten")


def analyze_paper_end_to_end(fetcher: DataFetcher) -> None:
    px = fetcher.get_stock("AAPL").prices
    start = px.index[-250]
    session = PaperSession(
        "e2e", str(start.date()),
        [
            PaperHolding("AAPL", 1000.0, None),                    # recommended/auto
            PaperHolding("MSFT", 2000.0, Bucket.TREND.value),      # forced trend
            PaperHolding("ZZINVALID77", 500.0, None),              # bad symbol
        ],
    )
    report = analyze_paper(session, fetcher, window_years=10)
    assert len(report.results) == 3
    assert report.skipped == ["ZZINVALID77"], report.skipped
    assert report.total_budget == 3500.0

    usable = [r for r in report.results if r.error is None]
    assert len(usable) == 2
    # overall P&L equals the sum of per-holding P&L exactly
    assert abs(sum(r.pnl for r in usable) - report.pnl) < 1e-6
    assert abs(report.current_value - sum(r.value for r in usable)) < 1e-6
    # every executed fill lands on/after the session start
    assert all(f.date >= start for f in report.blotter)
    assert report.value_curve is not None and len(report.value_curve) >= 2
    assert report.cagr is not None and report.risk is not None
    print(
        f"analyze_paper OK: value ${report.current_value:,.0f}, P&L {report.pnl:+,.0f} "
        f"== sum of holdings, {len(report.blotter)} fills all >= {start.date()}, "
        f"bogus ticker skipped"
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    holding_pnl_slices_and_compounds()
    fills_and_position_respect_start()
    persistence_roundtrip_and_corruption()
    structurally_bad_files_are_backed_up()
    fetcher = DataFetcher(paths.db_path())
    try:
        analyze_paper_end_to_end(fetcher)
    finally:
        fetcher.close()
    print("PHASE 9 PAPER TRADING CHECKS PASSED")


if __name__ == "__main__":
    main()
