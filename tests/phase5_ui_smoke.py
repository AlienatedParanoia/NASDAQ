"""Phase 5 smoke test on the offscreen Qt platform: drives a watchlist refresh
over a small cached subset, verifies live row insertion, numeric sorting,
filtering, and the detail chart, then exercises the My Investments tab and
saves a screenshot for visual inspection.

Run from the project root: python -m tests.phase5_ui_smoke
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import paths  # noqa: E402
from data.fetcher import configure_yfinance  # noqa: E402

SUBSET = ["AAPL", "MSFT", "NVDA", "KHC", "PEP", "TSLA", "PLTR", "XEL"]
NEG_INF = float("-inf")

# isolate the smoke test's user data from the real investments file
_TMP_PORTFOLIO_DIR = tempfile.mkdtemp()
paths.investments_path = lambda: Path(_TMP_PORTFOLIO_DIR) / "investments.json"


def _wait_panel_idle(app, panel, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while (panel._thread is not None or panel._dirty) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)
    assert panel._thread is None, "panel worker did not finish in time"
    app.processEvents()


def main() -> None:
    configure_yfinance(paths.data_dir())

    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    import ui.main_window as mw

    mw.get_universe = lambda: list(SUBSET)  # patch in a small, cached universe

    app = QApplication([])
    win = mw.MainWindow()
    win.show()

    win._start_refresh()
    assert not win._refresh_btn.isEnabled(), "Refresh should disable during a refresh"
    deadline = time.monotonic() + 180
    while win._thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)
    assert win._thread is None, "refresh did not finish within 180s"
    assert win._refresh_btn.isEnabled(), "Refresh should re-enable afterwards"
    assert win._model.rowCount() == len(SUBSET), (
        f"expected {len(SUBSET)} rows, got {win._model.rowCount()}"
    )
    assert all(
        win._model.row_at(r).error is None for r in range(win._model.rowCount())
    ), "every cached subset ticker should load"
    print(f"refresh finished with {win._model.rowCount()} rows; status: {win._status.text()}")

    # numeric (not lexicographic) sort by Day change %, descending
    win._table.sortByColumn(mw.C_CHANGE, Qt.SortOrder.DescendingOrder)
    app.processEvents()
    keys = []
    for r in range(win._proxy.rowCount()):
        source = win._proxy.mapToSource(win._proxy.index(r, mw.C_CHANGE))
        change = win._model.row_at(source.row()).day_change
        keys.append(change if change is not None else NEG_INF)
    assert keys == sorted(keys, reverse=True), f"day-change sort not numeric: {keys}"
    print(f"day-change sort OK: {['—' if k == NEG_INF else f'{k:+.2%}' for k in keys]}")

    # ticker substring filter
    win._proxy.set_ticker_filter("AA")
    app.processEvents()
    visible = [
        win._model.row_at(win._proxy.mapToSource(win._proxy.index(r, 0)).row()).ticker
        for r in range(win._proxy.rowCount())
    ]
    assert visible == ["AAPL"], f"ticker filter wrong: {visible}"
    win._proxy.set_ticker_filter("")
    app.processEvents()
    assert win._proxy.rowCount() == len(SUBSET)
    print("ticker filter OK")

    # row selection renders the detail chart
    win._table.selectRow(0)
    app.processEvents()
    assert win._detail._plot.listDataItems(), "detail chart empty after selection"
    assert win._detail._title.text() in SUBSET, win._detail._title.text()
    print(f"detail chart OK for {win._detail._title.text()}")

    # ---- My Investments tab (real-position tracker) ----
    import json as _json
    from datetime import date as _date, timedelta as _timedelta

    import ui.investments_panel as ip

    def _stub_add(ticker: str, money_in: float, buy: _date, sell=None):
        """Stand in for the modal InvestmentDialog: exec() returns Accepted and
        values() returns exactly what the user would have typed."""
        ip.InvestmentDialog = lambda parent, edit=None: type(
            "D", (), {
                "exec": lambda self: ip.QDialog.DialogCode.Accepted,
                "values": lambda self: (ticker, money_in, buy, sell, False),
            },
        )()

    def _stub_sell(when: _date):
        ip.SellDialog = lambda parent, lot: type(
            "D", (), {
                "exec": lambda self: ip.QDialog.DialogCode.Accepted,
                "sell_date": lambda self: when,
            },
        )()

    tabs = win.centralWidget()
    ip.QMessageBox = type("MB", (), {"warning": staticmethod(lambda *a, **k: None)})
    tabs.setCurrentIndex(1)  # Watchlist | My Investments
    inv = win._investments
    assert inv._table.rowCount() == 0

    a_year_ago = _date.today() - _timedelta(days=365)
    _stub_add("AAPL", 1000.0, a_year_ago)
    inv._on_new_investment()
    _wait_panel_idle(app, inv)
    _stub_add("MSFT", 1500.0, a_year_ago, sell=_date.today() - _timedelta(days=30))
    inv._on_new_investment()
    _wait_panel_idle(app, inv)

    assert inv._table.rowCount() == 2, inv._table.rowCount()
    inv_summary = inv._summary.text()
    assert "Put in $2,500" in inv_summary and "since" in inv_summary, inv_summary
    msft = next(r for r in inv._report.results if r.lot.ticker == "MSFT")
    assert msft.error is None and msft.worth > 0, msft
    assert msft.prices is None, "a sold lot carries no chart data"
    aapl = next(r for r in inv._report.results if r.lot.ticker == "AAPL")
    assert aapl.prices is not None, "an open lot carries its prices for the chart"
    print(f"investments summary OK: {inv_summary[:110]}…")

    # the cash ledger rendered: a buy with no cash on hand infers its own
    # deposit, so each lot contributes a deposit leg and a buy leg. The maths
    # itself is covered offline and exhaustively by phase11_real_pnl_check.
    assert inv._ledger.rowCount() >= 4, inv._ledger.rowCount()
    assert inv._cash_label.text(), "cash label empty after two buys"
    print(f"cash ledger OK: {inv._ledger.rowCount()} rows, {inv._cash_label.text()}")

    # sell flow on the open AAPL lot: SellDialog takes only a date, and the
    # worker resolves that day's close into money_out
    _stub_sell(_date.today() - _timedelta(days=7))
    aapl_lot = next(l for l in inv._lots if l.ticker == "AAPL")
    inv._on_sell(aapl_lot)
    _wait_panel_idle(app, inv)
    assert aapl_lot.sell_date, "sell date not recorded"
    assert aapl_lot.money_out and aapl_lot.money_out > 0, (
        f"worker did not resolve the sell close into money_out: {aapl_lot.money_out}"
    )
    aapl_row = next(
        i for i in range(inv._table.rowCount())
        if inv._table.item(i, 0).text() == "AAPL"
    )
    # read the column off the panel rather than hardcoding it, so the next
    # column change cannot silently turn this into a no-op check
    sell_cell = inv._table.cellWidget(aapl_row, ip.C_SELL)
    assert isinstance(sell_cell, ip.QPushButton) and sell_cell.text() == "Undo sale", (
        "a sold lot should offer Undo sale, not a stale Sell button"
    )
    print(
        f"sell flow OK: AAPL closed on {aapl_lot.sell_date} at "
        f"${aapl_lot.money_out:,.0f}, Sell button replaced by Undo sale"
    )

    # bogus ticker never persisted
    _stub_add("QQFAKE55", 500.0, a_year_ago)
    inv._on_new_investment()
    for _ in range(4):
        _wait_panel_idle(app, inv)
        if inv._thread is None and not inv._dirty and not inv._pending:
            break
    assert "QQFAKE55" not in {l.ticker for l in inv._lots}
    inv_saved = _json.loads(paths.investments_path().read_text(encoding="utf-8"))
    assert "QQFAKE55" not in {l["ticker"] for l in inv_saved["lots"]}
    print("investments bogus-ticker rejection OK")
    tabs.setCurrentIndex(0)

    shot = paths.data_dir() / "ui_smoke.png"
    win.grab().save(str(shot))
    print(f"screenshot saved to {shot}")

    win.close()
    app.processEvents()
    print("PHASE 5 UI SMOKE PASSED")


if __name__ == "__main__":
    main()
