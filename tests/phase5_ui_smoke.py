"""Phase 5 smoke test on the offscreen Qt platform: drives a scan over a small
cached subset, verifies live row insertion, numeric sorting, filtering, and
the detail panel, then saves a screenshot for visual inspection.

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

    assert win._window_box.currentData() == 10, "default backtest window should be 10y"

    win._start_scan()
    assert not win._scan_btn.isEnabled(), "Scan button should disable during a scan"
    deadline = time.monotonic() + 180
    while win._thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)
    assert win._thread is None, "scan did not finish within 180s"
    assert win._scan_btn.isEnabled(), "Scan button should re-enable after the scan"
    assert win._model.rowCount() == len(SUBSET), (
        f"expected {len(SUBSET)} rows, got {win._model.rowCount()}"
    )
    print(f"scan finished with {win._model.rowCount()} rows; status: {win._status.text()}")

    # numeric (not lexicographic) sort by Sharpe, descending
    win._table.sortByColumn(4, Qt.SortOrder.DescendingOrder)
    app.processEvents()
    keys = []
    for r in range(win._proxy.rowCount()):
        source = win._proxy.mapToSource(win._proxy.index(r, 4))
        s = win._model.row_at(source.row()).sharpe
        keys.append(s if s is not None else NEG_INF)
    assert keys == sorted(keys, reverse=True), f"Sharpe sort not numeric: {keys}"
    print(f"Sharpe sort OK: {['—' if k == NEG_INF else f'{k:.2f}' for k in keys]}")

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

    # row selection renders the detail panel
    win._table.selectRow(0)
    app.processEvents()
    assert win._detail._text.toPlainText(), "detail reasoning empty after selection"
    print("detail panel OK")

    # ---- Portfolio tab (real-position tracker) ----
    import json as _json

    import ui.investments_panel as ip
    from PyQt6.QtCore import QDate

    tabs = win.centralWidget()
    ip.QMessageBox = type("MB", (), {"warning": staticmethod(lambda *a, **k: None)})
    tabs.setCurrentIndex(3)  # Scanner | Daily Report | Paper Trading | My Investments
    inv = win._investments
    assert inv._table.rowCount() == 0
    inv._ticker_edit.setText("AAPL")
    inv._in_spin.setValue(1000)
    inv._buy_date.setDate(QDate.currentDate().addYears(-1))
    inv._on_add()
    _wait_panel_idle(app, inv)
    inv._ticker_edit.setText("MSFT")
    inv._in_spin.setValue(1500)
    inv._buy_date.setDate(QDate.currentDate().addYears(-1))
    inv._sold_check.setChecked(True)
    inv._out_spin.setValue(2000)
    inv._on_add()
    _wait_panel_idle(app, inv)
    assert inv._table.rowCount() == 2, inv._table.rowCount()
    inv_summary = inv._summary.text()
    assert "Put in $2,500" in inv_summary and "since" in inv_summary, inv_summary
    msft = next(r for r in inv._report.results if r.lot.ticker == "MSFT")
    assert abs(msft.pnl - 500.0) < 1e-9, msft.pnl
    print(f"investments summary OK: {inv_summary[:110]}…")

    # sell flow on the open AAPL lot
    ip.QInputDialog = type(
        "QID", (), {"getDouble": staticmethod(lambda *a, **k: (1750.0, True))}
    )
    aapl_lot = next(l for l in inv._lots if l.ticker == "AAPL")
    inv._on_sell(aapl_lot)
    _wait_panel_idle(app, inv)
    assert aapl_lot.money_out == 1750.0 and aapl_lot.sell_date
    aapl_res = next(r for r in inv._report.results if r.lot.ticker == "AAPL")
    assert abs(aapl_res.pnl - 750.0) < 1e-9
    aapl_row = next(
        i for i in range(inv._table.rowCount())
        if inv._table.item(i, 0).text() == "AAPL"
    )
    assert inv._table.cellWidget(aapl_row, 10) is None, (
        "stale Sell button survived the open->sold re-render"
    )
    print("sell flow OK: AAPL closed at entered proceeds, Sell button removed")

    # bogus ticker never persisted
    inv._sold_check.setChecked(False)
    inv._ticker_edit.setText("QQFAKE55")
    inv._in_spin.setValue(500)
    inv._on_add()
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
