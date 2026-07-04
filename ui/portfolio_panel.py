"""Portfolio tab: a persistent paper portfolio.

The user adds any Yahoo ticker with a dollar amount and (optionally) a
strategy; by default each holding runs the classifier-recommended strategy.
All fetching and math happen in a PortfolioWorker on its own QThread (same
lifecycle pattern as the scanner's ScanWorker); the UI only renders reports.
New tickers are validated by the worker — a symbol Yahoo doesn't recognize is
reported and NOT saved.
"""
from __future__ import annotations

import logging
import threading

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import paths
from classifier.classify import Bucket
from data.fetcher import DataFetcher
from portfolio.manager import (
    Holding,
    PortfolioReport,
    analyze_portfolio,
    load_portfolio,
    save_portfolio,
)
from signals.signal_generator import Signal
from ui.detail_panel import DetailPanel

log = logging.getLogger(__name__)

STRATEGY_CHOICES = (
    ("Recommended (auto)", None),
    ("Trend", Bucket.TREND.value),
    ("Mean-reversion", Bucket.MEAN_REVERSION.value),
    ("Long-term hold", Bucket.LONG_TERM.value),
)

COLS = ("Ticker", "Invested", "Weight", "Strategy", "Signal", "Return", "Profit", "Max DD", "")

SIGNAL_COLORS = {
    Signal.BUY: QColor("#1e8e3e"),
    Signal.HOLD: QColor("#7a7a7a"),
    Signal.AVOID: QColor("#d93025"),
}

DISCLAIMER = (
    "Paper portfolio — numbers are a historical backtest over the selected "
    "window, not live P&L. No costs/slippage; past performance guarantees nothing."
)

# error fragments that mean "network hiccup", not "bad symbol" — a new ticker
# failing this way stays pending instead of being rejected
_TRANSIENT_MARKERS = ("cooldown", "cancelled", "rate limit", "timed out", "curl")


def _looks_transient(error: str) -> bool:
    e = error.lower()
    return any(marker in e for marker in _TRANSIENT_MARKERS)


def _money(v: float) -> str:
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):,.0f}"


class PortfolioWorker(QObject):
    finished = pyqtSignal(object)  # PortfolioReport
    failed = pyqtSignal(str)

    def __init__(self, holdings: list[Holding], window_years: int) -> None:
        super().__init__()
        self._holdings = holdings
        self._window_years = window_years
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            fetcher = DataFetcher(paths.db_path(), should_stop=self._stop.is_set)
            try:
                report = analyze_portfolio(
                    self._holdings, fetcher, self._window_years,
                    should_stop=self._stop.is_set,
                )
            finally:
                fetcher.close()
            self.finished.emit(report)
        except Exception as e:  # pragma: no cover - surfaced to the UI
            log.exception("portfolio worker crashed")
            self.failed.emit(str(e))


class PortfolioPanel(QWidget):
    def __init__(self, window_years_provider) -> None:
        """window_years_provider: zero-arg callable returning the current
        backtest window (shared with the scanner's toolbar combo)."""
        super().__init__()
        self._window_years = window_years_provider
        self._holdings: list[Holding] = load_portfolio()
        self._report: PortfolioReport | None = None
        self._thread: QThread | None = None
        self._worker: PortfolioWorker | None = None
        self._dirty = False
        # tickers added but not yet validated by a worker report; never saved
        self._pending: set[str] = set()

        # -- add-holding row
        self._ticker_edit = QLineEdit()
        self._ticker_edit.setPlaceholderText("Ticker (e.g. AAPL, SHOP, VOO)…")
        self._ticker_edit.setMaxLength(12)
        self._ticker_edit.returnPressed.connect(self._on_add)
        self._amount_spin = QDoubleSpinBox()
        self._amount_spin.setRange(1.0, 100_000_000.0)
        self._amount_spin.setDecimals(0)
        self._amount_spin.setValue(1000.0)
        self._amount_spin.setPrefix("$ ")
        self._amount_spin.setGroupSeparatorShown(True)
        self._strategy_combo = QComboBox()
        for label, value in STRATEGY_CHOICES:
            self._strategy_combo.addItem(label, value)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add)

        add_row = QHBoxLayout()
        add_row.addWidget(self._ticker_edit, 1)
        add_row.addWidget(self._amount_spin)
        add_row.addWidget(self._strategy_combo)
        add_row.addWidget(add_btn)

        # -- holdings table
        self._table = QTableWidget(0, len(COLS))
        self._table.setHorizontalHeaderLabels(COLS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(len(COLS) - 1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_selection)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("font-weight: bold;")
        self._note = QLabel(DISCLAIMER)
        self._note.setWordWrap(True)
        self._status = QLabel("")

        left_layout = QVBoxLayout()
        left_layout.addLayout(add_row)
        left_layout.addWidget(self._table, 1)
        left_layout.addWidget(self._summary)
        left_layout.addWidget(self._status)
        left_layout.addWidget(self._note)
        left = QWidget()
        left.setLayout(left_layout)

        self._detail = DetailPanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer = QVBoxLayout(self)
        outer.addWidget(splitter)

        self._render_empty()
        if self._holdings:
            self.recompute()

    # ------------------------------------------------------------ lifecycle

    def recompute(self) -> None:
        if self._thread is not None:
            self._dirty = True
            return
        if not self._holdings:
            self._report = None
            self._render_empty()
            return
        self._status.setText("Computing portfolio…")
        self._worker = PortfolioWorker(list(self._holdings), self._window_years())
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_report)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _teardown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

    def shutdown(self) -> None:
        """Called from MainWindow.closeEvent."""
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            if not self._thread.wait(10_000):
                self._thread.wait()

    # ------------------------------------------------------------ mutations

    def _persist(self) -> None:
        """Save everything except still-unvalidated pending tickers."""
        keep = [h for h in self._holdings if h.ticker not in self._pending]
        try:
            save_portfolio(keep)
        except OSError as e:
            log.exception("portfolio save failed")
            self._status.setText(f"WARNING: portfolio not saved: {e}")

    def _on_add(self) -> None:
        ticker = self._ticker_edit.text().strip().upper()
        if not ticker:
            return
        amount = float(self._amount_spin.value())
        strategy = self._strategy_combo.currentData()
        existing = next((h for h in self._holdings if h.ticker == ticker), None)
        if existing is not None:
            existing.amount += amount
            existing.strategy = strategy
            self._persist()  # already validated previously
        else:
            self._holdings.append(Holding(ticker, amount, strategy))
            self._pending.add(ticker)  # persisted only after validation
        self._ticker_edit.clear()
        self.recompute()

    def _on_remove(self, ticker: str) -> None:
        self._holdings = [h for h in self._holdings if h.ticker != ticker]
        self._pending.discard(ticker)
        self._persist()
        self.recompute()

    def _on_row_strategy_changed(self, ticker: str, combo: QComboBox) -> None:
        value = combo.currentData()
        for h in self._holdings:
            if h.ticker == ticker:
                h.strategy = value
        self._persist()
        self.recompute()

    # -------------------------------------------------------------- results

    def _on_report(self, report: PortfolioReport) -> None:
        self._teardown()
        rejected: list[tuple[str, str]] = []
        for pending in sorted(self._pending):
            res = next(
                (r for r in report.results if r.holding.ticker == pending), None
            )
            if res is None:
                # report predates this add — validate against the next one
                # (one is already queued: _on_add always calls recompute)
                continue
            if res.error is None:
                self._pending.discard(pending)
            elif _looks_transient(res.error):
                # can't tell a bad symbol from a network hiccup; keep it
                # pending (and unsaved) so a later recompute decides
                self._status.setText(
                    f"{pending}: not validated yet ({res.error}) — will retry."
                )
            else:
                self._holdings = [h for h in self._holdings if h.ticker != pending]
                self._pending.discard(pending)
                rejected.append((pending, res.error))
        self._persist()
        if rejected:
            QMessageBox.warning(
                self,
                "Ticker not added",
                "\n".join(f"{t}: Yahoo returned no data ({e})" for t, e in rejected)
                + "\nCheck the symbol and try again.",
            )
            self._dirty = False
            self.recompute()
            return
        self._report = report
        self._render(report)
        if self._dirty:
            self._dirty = False
            self.recompute()

    def _on_failed(self, err: str) -> None:
        self._teardown()
        if self._pending:
            # never keep holdings that were never validated
            self._holdings = [
                h for h in self._holdings if h.ticker not in self._pending
            ]
            self._pending.clear()
        self._status.setText(f"Portfolio analysis failed: {err}")
        if self._dirty:
            self._dirty = False
            self.recompute()

    # ------------------------------------------------------------ rendering

    def _render_empty(self) -> None:
        self._table.setRowCount(0)
        self._detail.clear()
        self._summary.setText("No holdings yet — add a stock above to start a paper portfolio.")
        self._status.setText("")

    def _render(self, report: PortfolioReport) -> None:
        self._status.setText("")
        self._table.blockSignals(True)
        self._table.setRowCount(len(report.results))
        for i, res in enumerate(report.results):
            h = res.holding
            self._set_item(i, 0, h.ticker)
            self._set_item(i, 1, f"${h.amount:,.0f}")
            self._set_item(i, 2, f"{res.weight:.0%}")

            combo = QComboBox()
            for label, value in STRATEGY_CHOICES:
                combo.addItem(label, value)
            combo.setCurrentIndex(
                next(
                    (k for k, (_, v) in enumerate(STRATEGY_CHOICES) if v == h.strategy),
                    0,
                )
            )
            combo.currentIndexChanged.connect(
                lambda _idx, t=h.ticker, c=combo: self._on_row_strategy_changed(t, c)
            )
            self._table.setCellWidget(i, 3, combo)

            if res.error is not None:
                self._set_item(i, 4, "no data")
                self._set_item(i, 5, "—")
                self._set_item(i, 6, "—")
                self._set_item(i, 7, "—")
            else:
                row = res.row
                sig_item = QTableWidgetItem(
                    str(row.signal) if row and row.signal else "—"
                )
                if row is not None and row.signal is not None:
                    sig_item.setForeground(SIGNAL_COLORS[row.signal])
                self._table.setItem(i, 4, sig_item)
                self._set_item(
                    i, 5,
                    f"{res.common_return:+.1%}" if res.common_return is not None else "—",
                )
                self._set_item(
                    i, 6, _money(res.profit) if res.profit is not None else "—"
                )
                self._set_item(
                    i, 7,
                    f"{row.max_dd:.1%}" if row is not None and row.max_dd is not None else "—",
                )

            remove = QPushButton("Remove")
            remove.clicked.connect(lambda _c, t=h.ticker: self._on_remove(t))
            self._table.setCellWidget(i, len(COLS) - 1, remove)
        self._table.blockSignals(False)

        if report.total_return is None:
            text = (
                f"Invested ${report.total_invested:,.0f} — nothing analyzable yet"
                + (f" (skipped: {', '.join(report.skipped)})" if report.skipped else "")
                + "."
            )
        else:
            text = (
                f"Invested ${report.total_invested:,.0f}  |  "
                f"Return {report.total_return:+.1%}  |  "
                f"Profit {_money(report.profit)}  |  "
                f"CAGR {report.cagr:+.1%}/yr  |  "
                f"Risk {report.ann_vol:.0%}/yr ({report.risk})  |  "
                f"Sharpe {report.sharpe:.2f}  |  "
                f"MaxDD {report.max_drawdown:.1%}  |  "
                f"over {report.years:.1f}y"
            )
            if report.limiting_ticker:
                text += f"  —  window limited by {report.limiting_ticker}"
            if report.short_window:
                text += "  —  SHORT JOINT WINDOW: metrics unreliable"
            if report.skipped:
                text += f"  —  skipped: {', '.join(report.skipped)}"
        self._summary.setText(text)

    def _set_item(self, row: int, col: int, text: str) -> None:
        self._table.setItem(row, col, QTableWidgetItem(text))

    def _on_selection(self) -> None:
        if self._report is None:
            return
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            return
        i = indexes[0].row()
        if 0 <= i < len(self._report.results):
            res = self._report.results[i]
            if res.row is not None:
                self._detail.show_row(res.row)
