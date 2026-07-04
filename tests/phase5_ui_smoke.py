"""Phase 5 smoke test on the offscreen Qt platform: drives a scan over a small
cached subset, verifies live row insertion, numeric sorting, filtering, and
the detail panel, then saves a screenshot for visual inspection.

Run from the project root: python -m tests.phase5_ui_smoke
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import paths  # noqa: E402
from data.fetcher import configure_yfinance  # noqa: E402

SUBSET = ["AAPL", "MSFT", "NVDA", "KHC", "PEP", "TSLA", "PLTR", "XEL"]
NEG_INF = float("-inf")


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

    shot = paths.data_dir() / "ui_smoke.png"
    win.grab().save(str(shot))
    print(f"screenshot saved to {shot}")

    win.close()
    app.processEvents()
    print("PHASE 5 UI SMOKE PASSED")


if __name__ == "__main__":
    main()
