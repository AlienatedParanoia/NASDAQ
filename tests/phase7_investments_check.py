"""Phase 7 checks: real-money investment tracking (portfolio/tracker.py).

Offline once the cache is warm (the bogus-ticker case makes one quick failed
network call). Run from the project root: python -m tests.phase7_investments_check
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pandas as pd

import paths
from data.fetcher import DataFetcher
from portfolio.tracker import (
    Lot,
    analyze_investments,
    load_investments,
    save_investments,
)


def shares_and_worth_math(fetcher: DataFetcher) -> None:
    px = fetcher.get_stock("AAPL").prices
    buy_idx = px.index[-250]
    lot = Lot("AAPL", 1000.0, str(buy_idx.date()))
    rep = analyze_investments([lot], fetcher)
    res = rep.results[0]
    assert res.error is None, res.error
    expected_shares = 1000.0 / float(px["close"].loc[buy_idx])
    assert abs(res.shares - expected_shares) < 1e-12
    expected_worth = expected_shares * float(px["close"].iloc[-1])
    assert abs(res.worth - expected_worth) < 1e-9
    assert abs(res.pnl - (expected_worth - 1000.0)) < 1e-9
    assert rep.current_value == res.worth and rep.pnl == res.pnl
    assert rep.since == str(buy_idx.date())
    assert rep.max_drawdown is not None and rep.max_drawdown <= 0.0
    print(
        f"shares math OK: {res.shares:.4f} AAPL shares, worth ${res.worth:,.0f}, "
        f"P&L {res.pnl:+,.0f} over {rep.since}..today"
    )


def weekend_buy_snaps_forward(fetcher: DataFetcher) -> None:
    px = fetcher.get_stock("AAPL").prices
    anchor = px.index[-100]
    saturday = anchor + pd.Timedelta(days=int((5 - anchor.weekday()) % 7))
    if saturday.weekday() != 5:  # anchor was itself a Saturday-adjacent edge
        saturday += pd.Timedelta(days=7)
    assert saturday not in px.index
    expected_eff = px.index[px.index >= saturday][0]
    rep = analyze_investments([Lot("AAPL", 500.0, str(saturday.date()))], fetcher)
    res = rep.results[0]
    assert res.buy_eff == expected_eff, (res.buy_eff, expected_eff)
    assert abs(res.shares - 500.0 / float(px["close"].loc[expected_eff])) < 1e-12
    print(f"weekend buy {saturday.date()} snapped to {expected_eff.date()} OK")


def sold_lot_uses_entered_amounts(fetcher: DataFetcher) -> None:
    px = fetcher.get_stock("MSFT").prices
    lot = Lot(
        "MSFT", 1500.0, str(px.index[-250].date()),
        money_out=2000.0, sell_date=str(px.index[-10].date()),
    )
    rep = analyze_investments([lot], fetcher)
    res = rep.results[0]
    assert res.error is None
    assert abs(res.pnl - 500.0) < 1e-9, "sold P&L must use the entered amounts"
    assert abs(res.ret - 500.0 / 1500.0) < 1e-12
    assert res.ann_vol is not None and res.max_dd is not None
    assert res.signal is None, "sold lots get no live signal"
    print(f"sold lot OK: entered 1500->2000 gives P&L +500, risk {res.ann_vol:.0%}/yr")


def overall_report_identities(fetcher: DataFetcher) -> None:
    px = fetcher.get_stock("AAPL").prices
    old = str(px.index[-400].date())
    lots = [
        Lot("AAPL", 1000.0, old),
        Lot("MSFT", 1500.0, old, money_out=2000.0, sell_date=str(px.index[-10].date())),
        Lot("ZZINVALID77", 500.0, old),
    ]
    rep = analyze_investments(lots, fetcher)
    assert len(rep.results) == 3
    assert rep.skipped == ["ZZINVALID77"]
    assert rep.total_in == 3000.0
    usable_pnl = sum(r.pnl for r in rep.results if r.error is None)
    assert abs(usable_pnl - rep.pnl) < 1e-6, "overall P&L must equal sum of lots"
    assert rep.annualized is not None and rep.risk is not None
    print(
        f"overall OK: P&L {rep.pnl:+,.0f} == sum of lots, risk "
        f"{rep.ann_vol:.0%}/yr ({rep.risk}), bogus lot skipped"
    )


def clamped_and_short_holds(fetcher: DataFetcher) -> None:
    px = fetcher.get_stock("AAPL").prices
    rep = analyze_investments([Lot("AAPL", 1000.0, "2001-01-01")], fetcher)
    res = rep.results[0]
    assert res.note is not None and "predates" in res.note
    assert rep.notes and "predates" in rep.notes[0]

    rep2 = analyze_investments(
        [Lot("AAPL", 1000.0, str(px.index[-5].date()))], fetcher
    )
    assert rep2.results[0].annualized is None, "short holds must not annualize"
    assert rep2.annualized is None
    print("history clamp noted; <30-day holds not annualized")


def persistence_roundtrip_and_corruption() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "investments.json"
        lots = [
            Lot("AAPL", 1000.0, "2024-05-01"),
            Lot("SHOP", 750.0, "2023-11-20", money_out=900.0, sell_date="2025-02-14"),
        ]
        save_investments(lots, p)
        assert load_investments(p) == lots
        assert json.loads(p.read_text())["version"] == 1
        p.write_text("{ nope", encoding="utf-8")
        assert load_investments(p) == []
        assert p.with_suffix(".json.bak").exists()
    print("investments persistence round-trip + corruption backup OK")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    persistence_roundtrip_and_corruption()
    fetcher = DataFetcher(paths.db_path())
    try:
        shares_and_worth_math(fetcher)
        weekend_buy_snaps_forward(fetcher)
        sold_lot_uses_entered_amounts(fetcher)
        overall_report_identities(fetcher)
        clamped_and_short_holds(fetcher)
    finally:
        fetcher.close()
    print("PHASE 7 INVESTMENT CHECKS PASSED")


if __name__ == "__main__":
    main()
