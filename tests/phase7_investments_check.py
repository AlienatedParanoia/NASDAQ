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
    assert rep.overall.current_value == res.worth and rep.overall.pnl == res.pnl
    assert rep.overall.since == str(buy_idx.date())
    assert rep.overall.max_drawdown is not None and rep.overall.max_drawdown <= 0.0
    # nothing sold: the open-only view is the whole portfolio
    assert rep.open_only.count == rep.overall.count == 1
    assert rep.open_only.pnl == rep.overall.pnl
    print(
        f"shares math OK: {res.shares:.4f} AAPL shares, worth ${res.worth:,.0f}, "
        f"P&L {res.pnl:+,.0f} over {rep.overall.since}..today"
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
    """An explicit money_out (a real fill, or a v1 record) overrides the close."""
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


def sold_lot_freezes_at_the_sell_close(fetcher: DataFetcher) -> None:
    """With no money_out, a sale is valued at the close on its sell date and
    keeps that value however many bars come afterwards."""
    px = fetcher.get_stock("AAPL").prices
    close = px["close"]
    buy_idx, sell_idx = px.index[-250], px.index[-10]
    lot = Lot("AAPL", 1000.0, str(buy_idx.date()), sell_date=str(sell_idx.date()))
    assert lot.is_sold and Lot("AAPL", 1000.0, str(buy_idx.date())).is_sold is False
    assert Lot("AAPL", 1000.0, str(buy_idx.date()), money_out=1.0).is_sold, "v1 record"

    res = analyze_investments([lot], fetcher).results[0]
    shares = 1000.0 / float(close.loc[buy_idx])
    expected = shares * float(close.loc[sell_idx])
    assert abs(res.worth - expected) < 1e-9, "sale must be valued at the sell close"
    assert abs(res.pnl - (expected - 1000.0)) < 1e-9
    assert abs(res.worth - shares * float(close.iloc[-1])) > 1e-9, (
        "a sold lot must not track the latest close"
    )

    # a sell date on a non-trading day falls back to the prior close, with a note
    sunday = sell_idx + pd.Timedelta(days=int((6 - sell_idx.weekday()) % 7) or 7)
    if sunday in px.index:
        sunday += pd.Timedelta(days=7)
    res_wk = analyze_investments(
        [Lot("AAPL", 1000.0, str(buy_idx.date()), sell_date=str(sunday.date()))],
        fetcher,
    ).results[0]
    last_bar = px.index[px.index <= sunday][-1]
    assert abs(res_wk.worth - shares * float(close.loc[last_bar])) < 1e-9
    assert res_wk.note is not None and "no trading on" in res_wk.note
    print(
        f"sold lot frozen at {sell_idx.date()} close: worth ${res.worth:,.0f} "
        f"(latest close would be ${shares * float(close.iloc[-1]):,.0f})"
    )


def open_only_view_excludes_sold(fetcher: DataFetcher) -> None:
    px = fetcher.get_stock("AAPL").prices
    old = str(px.index[-400].date())
    open_lot = Lot("AAPL", 1000.0, old)
    sold_lot = Lot("MSFT", 1500.0, old, sell_date=str(px.index[-10].date()))
    rep = analyze_investments([open_lot, sold_lot], fetcher)

    assert rep.overall.count == 2 and rep.open_only.count == 1
    assert rep.overall.total_in == 2500.0
    assert rep.open_only.total_in == 1000.0, "open-only must ignore the sold lot's cash"
    open_res = rep.results[0]
    assert abs(rep.open_only.pnl - open_res.pnl) < 1e-6
    assert abs(rep.open_only.current_value - open_res.worth) < 1e-6
    assert rep.open_only.risk is not None
    print(
        f"dual view OK: all {rep.overall.count} lots P&L {rep.overall.pnl:+,.0f}, "
        f"open-only P&L {rep.open_only.pnl:+,.0f}"
    )


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
    assert rep.total_in == 3000.0, "total_in counts the errored lot too"
    assert rep.overall.total_in == 2500.0, "aggregates cover analyzable lots only"
    usable_pnl = sum(r.pnl for r in rep.results if r.error is None)
    assert abs(usable_pnl - rep.overall.pnl) < 1e-6, "overall P&L must equal sum of lots"
    assert rep.overall.annualized is not None and rep.overall.risk is not None
    print(
        f"overall OK: P&L {rep.overall.pnl:+,.0f} == sum of lots, risk "
        f"{rep.overall.ann_vol:.0%}/yr ({rep.overall.risk}), bogus lot skipped"
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
    assert rep2.overall.annualized is None
    print("history clamp noted; <30-day holds not annualized")


def sale_day_curve_ends_at_proceeds(fetcher: DataFetcher) -> None:
    """Review regressions: the V(t) contribution must switch to the entered
    proceeds ON the sale day (including same-day sales on the latest bar),
    and stale price data must be flagged."""
    from portfolio.tracker import _analyze_lot

    data = fetcher.get_stock("AAPL")
    px = data.prices
    old = str(px.index[-250].date())

    # sale on the latest bar: curve endpoint must equal the entered proceeds
    same_day = Lot("AAPL", 1000.0, old, money_out=5000.0,
                   sell_date=str(px.index[-1].date()))
    _, curve = _analyze_lot(same_day, data)
    assert curve.iloc[-1] == 5000.0, curve.iloc[-1]

    # past sale: the sale DAY itself carries the proceeds, not the market close
    past = Lot("AAPL", 1000.0, old, money_out=1234.0,
               sell_date=str(px.index[-10].date()))
    res, curve2 = _analyze_lot(past, data)
    assert curve2.loc[px.index[-10]] == 1234.0
    assert curve2.iloc[-1] == 1234.0

    from data.fetcher import StockData

    stale_data = StockData(ticker="AAPL", prices=px, fundamentals=None, stale=True)
    res_stale, _ = _analyze_lot(Lot("AAPL", 1000.0, old), stale_data)
    assert res_stale.note is not None and "out of date" in res_stale.note
    print("sale-day proceeds in V(t) + stale-price note OK")


def structurally_bad_files_are_backed_up() -> None:
    """Review regression: valid JSON with the wrong structure must be backed
    up before the app can overwrite it."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "investments.json"
        p.write_text(json.dumps({"lots": "not-a-list"}), encoding="utf-8")
        assert load_investments(p) == []
        assert p.with_suffix(".json.bak").exists(), "structural damage not backed up"
        assert not p.exists()
    print("structurally-bad user files are backed up, not silently overwritten")


def persistence_roundtrip_and_corruption() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "investments.json"
        lots = [
            Lot("AAPL", 1000.0, "2024-05-01", strategy="Trend"),
            Lot("SHOP", 750.0, "2023-11-20", sell_date="2025-02-14",
                strategy="Mean-reversion"),
            Lot("VOO", 500.0, "2024-01-02"),  # strategy None = recommended
        ]
        save_investments(lots, p)
        assert load_investments(p) == lots, "round-trip must preserve strategy"
        assert json.loads(p.read_text())["version"] == 2
        p.write_text("{ nope", encoding="utf-8")
        assert load_investments(p) == []
        assert p.with_suffix(".json.bak").exists()
    print("investments persistence round-trip + corruption backup OK")


def v1_files_still_load() -> None:
    """A file written before trends existed must load, with strategy None and
    its entered proceeds intact."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "investments.json"
        p.write_text(json.dumps({
            "version": 1,
            "lots": [
                {"ticker": "aapl", "money_in": 1000.0, "buy_date": "2024-05-01",
                 "money_out": None, "sell_date": None},
                {"ticker": "SHOP", "money_in": 750.0, "buy_date": "2023-11-20",
                 "money_out": 900.0, "sell_date": "2025-02-14"},
            ],
        }), encoding="utf-8")
        lots = load_investments(p)
        assert [l.strategy for l in lots] == [None, None]
        assert lots[0].ticker == "AAPL" and not lots[0].is_sold
        assert lots[1].is_sold and lots[1].money_out == 900.0
    print("v1 investments files load with strategy=None and proceeds preserved")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    persistence_roundtrip_and_corruption()
    v1_files_still_load()
    structurally_bad_files_are_backed_up()
    fetcher = DataFetcher(paths.db_path())
    try:
        shares_and_worth_math(fetcher)
        weekend_buy_snaps_forward(fetcher)
        sold_lot_uses_entered_amounts(fetcher)
        sold_lot_freezes_at_the_sell_close(fetcher)
        open_only_view_excludes_sold(fetcher)
        overall_report_identities(fetcher)
        clamped_and_short_holds(fetcher)
        sale_day_curve_ends_at_proceeds(fetcher)
    finally:
        fetcher.close()
    print("PHASE 7 INVESTMENT CHECKS PASSED")


if __name__ == "__main__":
    main()
