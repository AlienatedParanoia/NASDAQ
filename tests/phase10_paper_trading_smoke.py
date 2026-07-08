"""Phase 10 smoke test on the offscreen Qt platform: drives the Paper Trading
tab — deploy an algorithm on a fake budget, validate a new ticker, reject a
bogus one, and start a fresh session — then saves a screenshot.

Reuses the warm price cache. Run from the project root:
    python -m tests.phase10_paper_trading_smoke
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import paths  # noqa: E402
from data.fetcher import configure_yfinance  # noqa: E402

SUBSET = ["AAPL", "MSFT", "NVDA", "KHC", "PEP", "TSLA", "PLTR", "XEL"]

# isolate the smoke test's session file from the real one
_TMP_DIR = tempfile.mkdtemp()
paths.paper_trading_path = lambda: Path(_TMP_DIR) / "paper_trading.json"


def _wait_panel_idle(app, panel, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while (panel._thread is not None or panel._dirty) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)
    assert panel._thread is None, "panel worker did not finish in time"
    app.processEvents()


def main() -> None:
    configure_yfinance(paths.data_dir())

    from PyQt6.QtWidgets import QApplication, QMessageBox as _RealMB

    from classifier.classify import Bucket
    import ui.main_window as mw
    import ui.paper_trading_panel as pp

    mw.get_universe = lambda: list(SUBSET)  # patch in a small, cached universe
    # modal dialogs must never block the headless run; "New session" auto-confirms
    pp.QMessageBox = type(
        "MB", (), {
            "warning": staticmethod(lambda *a, **k: None),
            "question": staticmethod(lambda *a, **k: _RealMB.StandardButton.Yes),
            "StandardButton": _RealMB.StandardButton,
        },
    )

    app = QApplication([])
    win = mw.MainWindow()
    win.show()

    tabs = win.centralWidget()
    assert tabs.tabText(2) == "Paper Trading", tabs.tabText(2)
    tabs.setCurrentIndex(2)
    paper = win._paper
    assert paper._table.rowCount() == 0, "session should start empty"

    # the "New trade…" options dialog builds and reads back its fields (create
    # mode), and pre-fills in edit mode — exercised without a blocking exec()
    dlg = pp.PaperTradeDialog(paper)
    dlg._ticker.setText("aapl")
    dlg._budget.setValue(1000)
    assert dlg.values() == ("AAPL", 1000.0, None), dlg.values()
    edit_dlg = pp.PaperTradeDialog(
        paper, edit=pp.PaperHolding("MSFT", 2000.0, Bucket.TREND.value)
    )
    assert edit_dlg._ticker.isReadOnly()
    assert edit_dlg.values() == ("MSFT", 2000.0, Bucket.TREND.value), edit_dlg.values()
    print("options dialog OK: create reads back, edit pre-fills with ticker locked")

    # deploy the recommended (auto) algorithm on a fake $1,000 AAPL position
    paper._add_holding("AAPL", 1000.0, None)
    _wait_panel_idle(app, paper)
    assert paper._table.rowCount() == 1, paper._table.rowCount()
    assert "AAPL" in {h.ticker for h in paper._session.holdings}
    assert paper._report is not None
    res = paper._report.results[0]
    assert res.error is None, res.error
    assert res.value is not None and res.strategy_label
    summary = paper._summary.text()
    assert summary.startswith("Started $1,000"), summary
    print(f"deploy OK: AAPL -> {res.strategy_label}, value ${res.value:,.0f}; {summary[:90]}…")

    # a second forced-strategy holding
    paper._add_holding("MSFT", 2000.0, Bucket.TREND.value)
    _wait_panel_idle(app, paper)
    assert paper._table.rowCount() == 2, paper._table.rowCount()

    # edit an existing holding: SET (not add to) budget + switch algorithm
    paper._edit_holding("AAPL", 1500.0, Bucket.MEAN_REVERSION.value)
    _wait_panel_idle(app, paper)
    aapl = next(h for h in paper._session.holdings if h.ticker == "AAPL")
    assert aapl.budget == 1500.0 and aapl.strategy == Bucket.MEAN_REVERSION.value
    print("edit OK: AAPL budget set to $1,500, algorithm switched to mean-reversion")

    # selecting a row renders the detail chart
    paper._table.selectRow(0)
    app.processEvents()
    assert paper._detail._text.toPlainText(), "detail reasoning empty after selection"
    print("detail panel OK")

    # bogus ticker is validated away and never persisted
    paper._add_holding("QQFAKE55", 500.0, None)
    for _ in range(4):
        _wait_panel_idle(app, paper)
        if paper._thread is None and not paper._dirty and not paper._pending:
            break
    assert "QQFAKE55" not in {h.ticker for h in paper._session.holdings}
    saved = json.loads(paths.paper_trading_path().read_text(encoding="utf-8"))
    assert "QQFAKE55" not in {h["ticker"] for h in saved["holdings"]}
    assert {h["ticker"] for h in saved["holdings"]} == {"AAPL", "MSFT"}
    print("bogus-ticker rejection + persistence OK")

    # start a fresh session: holdings cleared, file emptied
    paper._on_new_session()
    _wait_panel_idle(app, paper)
    assert paper._session.holdings == []
    assert paper._table.rowCount() == 0
    saved2 = json.loads(paths.paper_trading_path().read_text(encoding="utf-8"))
    assert saved2["holdings"] == []
    print("new-session reset OK")

    shot = paths.data_dir() / "paper_smoke.png"
    tabs.setCurrentIndex(2)
    win.grab().save(str(shot))
    print(f"screenshot saved to {shot}")

    win.close()
    app.processEvents()
    print("PHASE 10 PAPER TRADING SMOKE PASSED")


if __name__ == "__main__":
    main()
