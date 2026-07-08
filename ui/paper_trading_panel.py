"""Paper Trading tab: a forward-simulated fake portfolio.

The user starts a session (anchored to today), adds any Yahoo ticker with a
fake dollar budget and (optionally) a strategy; by default each holding runs
the classifier-recommended — "suitable" — algorithm. The deployed algorithm's
trades are executed for the user and shown in the blotter, and the session
reports what it has made SINCE the start date, marked to market at the latest
close. All fetching and math happen in a PaperTradingWorker on its own QThread
(same lifecycle pattern as the scanner's ScanWorker); the UI only renders.
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
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
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
from portfolio.paper import (
    PaperHolding,
    PaperReport,
    PaperSession,
    analyze_paper,
    load_paper,
    looks_transient,
    save_paper,
    today_iso,
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
_STRATEGY_LABELS = {value: label for label, value in STRATEGY_CHOICES}


def strategy_choice_label(strategy: str | None) -> str:
    return _STRATEGY_LABELS.get(strategy, "Recommended (auto)")


# columns: ... plus an Edit and a Remove action button at the end
COLS = (
    "Ticker", "Budget", "Strategy", "Signal", "Position",
    "Trades", "Value", "P&L", "Return", "", "",
)

SIGNAL_COLORS = {
    Signal.BUY: QColor("#1e8e3e"),
    Signal.HOLD: QColor("#7a7a7a"),
    Signal.AVOID: QColor("#d93025"),
}

BLOTTER_COLS = ("Date", "Ticker", "Side", "Price")
SIDE_COLORS = {"Buy": QColor("#1e8e3e"), "Sell": QColor("#d93025")}

DISCLAIMER = (
    "Paper trading — a forward simulation from the session start date, valued at "
    "the latest close. Fake money, no costs/slippage; P&L begins near $0 and "
    "builds as new trading days pass. Past performance guarantees nothing."
)


def _money(v: float) -> str:
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):,.0f}"


class PaperTradeDialog(QDialog):
    """Pop-up panel for entering a paper trade's options (ticker, fake budget,
    algorithm). Opened by "New trade…" to create one, or a row's "Edit…" button
    to change an existing one (the ticker is fixed in edit mode)."""

    def __init__(self, parent=None, *, edit: PaperHolding | None = None) -> None:
        super().__init__(parent)
        self._editing = edit is not None
        self.setWindowTitle("Edit paper trade" if self._editing else "New paper trade")
        self.setModal(True)

        self._ticker = QLineEdit()
        self._ticker.setMaxLength(12)
        self._ticker.setPlaceholderText("e.g. AAPL, SHOP, VOO")
        self._budget = QDoubleSpinBox()
        self._budget.setRange(1.0, 100_000_000.0)
        self._budget.setDecimals(0)
        self._budget.setPrefix("$ ")
        self._budget.setGroupSeparatorShown(True)
        self._budget.setValue(1000.0)
        self._strategy = QComboBox()
        for label, value in STRATEGY_CHOICES:
            self._strategy.addItem(label, value)

        if edit is not None:
            self._ticker.setText(edit.ticker)
            self._ticker.setReadOnly(True)
            self._budget.setValue(edit.budget)
            self._strategy.setCurrentIndex(
                next(
                    (k for k, (_, v) in enumerate(STRATEGY_CHOICES) if v == edit.strategy),
                    0,
                )
            )

        form = QFormLayout()
        form.addRow("Ticker", self._ticker)
        form.addRow("Fake budget", self._budget)
        form.addRow("Algorithm", self._strategy)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        if not self._ticker.text().strip():
            QMessageBox.warning(self, "Ticker required", "Enter a ticker symbol.")
            return
        self.accept()

    def values(self) -> tuple[str, float, str | None]:
        return (
            self._ticker.text().strip().upper(),
            float(self._budget.value()),
            self._strategy.currentData(),
        )


class PaperTradingWorker(QObject):
    finished = pyqtSignal(object)  # PaperReport
    failed = pyqtSignal(str)

    def __init__(self, session: PaperSession, window_years: int) -> None:
        super().__init__()
        self._session = session
        self._window_years = window_years
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            from data.fetcher import DataFetcher

            fetcher = DataFetcher(paths.db_path(), should_stop=self._stop.is_set)
            try:
                report = analyze_paper(
                    self._session, fetcher, self._window_years,
                    should_stop=self._stop.is_set,
                )
            finally:
                fetcher.close()
            self.finished.emit(report)
        except Exception as e:  # pragma: no cover - surfaced to the UI
            log.exception("paper trading worker crashed")
            self.failed.emit(str(e))


class PaperTradingPanel(QWidget):
    def __init__(self, window_years_provider) -> None:
        """window_years_provider: zero-arg callable returning the current
        backtest window (shared with the scanner's toolbar combo); it sizes the
        history the deployed algorithm sees when computing the live signal."""
        super().__init__()
        self._window_years = window_years_provider
        self._session: PaperSession = load_paper()
        self._report: PaperReport | None = None
        self._thread: QThread | None = None
        self._worker: PaperTradingWorker | None = None
        self._dirty = False
        # tickers added but not yet validated by a worker report; never saved
        self._pending: set[str] = set()
        # bumped on every mutation; reports computed under an older generation
        # are discarded so mid-flight edits never render mixed state
        self._generation = 0
        self._running_gen = -1

        # -- session bar: label + the two actions ("New trade…" opens the
        # options panel; "New session" resets the whole paper portfolio)
        self._session_label = QLabel("")
        self._session_label.setStyleSheet("font-weight: bold;")
        add_btn = QPushButton("New trade…")
        add_btn.clicked.connect(self._on_new_trade)
        new_btn = QPushButton("New session")
        new_btn.clicked.connect(self._on_new_session)
        session_row = QHBoxLayout()
        session_row.addWidget(self._session_label, 1)
        session_row.addWidget(add_btn)
        session_row.addWidget(new_btn)

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
        header.setSectionResizeMode(len(COLS) - 2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_selection)

        # -- trade blotter (fills the deployed algorithm executed)
        self._blotter = QTableWidget(0, len(BLOTTER_COLS))
        self._blotter.setHorizontalHeaderLabels(BLOTTER_COLS)
        self._blotter.verticalHeader().setVisible(False)
        self._blotter.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._blotter.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._blotter.setMaximumHeight(160)
        self._blotter.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._blotter_label = QLabel("Trade blotter (executed by the deployed algorithm)")

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("font-weight: bold;")
        self._note = QLabel(DISCLAIMER)
        self._note.setWordWrap(True)
        self._status = QLabel("")

        left_layout = QVBoxLayout()
        left_layout.addLayout(session_row)
        left_layout.addWidget(self._table, 1)
        left_layout.addWidget(self._summary)
        left_layout.addWidget(self._blotter_label)
        left_layout.addWidget(self._blotter)
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
        if self._session.holdings:
            self.recompute()

    # ------------------------------------------------------------ lifecycle

    def window_changed(self) -> None:
        """Backtest window switched: any in-flight report is now stale."""
        self._generation += 1
        self.recompute()

    def recompute(self) -> None:
        if self._thread is not None:
            self._dirty = True
            return
        if not self._session.holdings:
            self._report = None
            self._render_empty()
            return
        self._status.setText("Running the paper session…")
        self._running_gen = self._generation
        snapshot = PaperSession(
            self._session.name, self._session.start_date, list(self._session.holdings)
        )
        self._worker = PaperTradingWorker(snapshot, self._window_years())
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
        keep = [h for h in self._session.holdings if h.ticker not in self._pending]
        try:
            save_paper(PaperSession(self._session.name, self._session.start_date, keep))
        except OSError as e:
            log.exception("paper session save failed")
            self._status.setText(f"WARNING: session not saved: {e}")

    def _on_new_session(self) -> None:
        if self._session.holdings:
            reply = QMessageBox.question(
                self, "New paper session",
                "Start a fresh paper session? This clears all current holdings "
                "and resets the start date to today.",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._generation += 1
        self._session = PaperSession(self._session.name, today_iso(), [])
        self._pending.clear()
        self._persist()
        self._report = None
        self.recompute()

    def _on_new_trade(self) -> None:
        """Open the options panel to create a new paper trade."""
        dialog = PaperTradeDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            ticker, amount, strategy = dialog.values()
            if ticker:
                self._add_holding(ticker, amount, strategy)

    def _on_edit(self, ticker: str) -> None:
        """Open the options panel pre-filled to edit an existing holding."""
        holding = next(
            (h for h in self._session.holdings if h.ticker == ticker), None
        )
        if holding is None:
            return
        dialog = PaperTradeDialog(self, edit=holding)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            _, budget, strategy = dialog.values()
            self._edit_holding(ticker, budget, strategy)

    def _add_holding(self, ticker: str, amount: float, strategy: str | None) -> None:
        self._generation += 1
        existing = next((h for h in self._session.holdings if h.ticker == ticker), None)
        if existing is not None:
            existing.budget += amount
            existing.strategy = strategy
            self._persist()  # already validated previously
        else:
            self._session.holdings.append(PaperHolding(ticker, amount, strategy))
            self._pending.add(ticker)  # persisted only after validation
        self.recompute()

    def _edit_holding(self, ticker: str, budget: float, strategy: str | None) -> None:
        """Set (not add to) an existing holding's budget and algorithm."""
        self._generation += 1
        for h in self._session.holdings:
            if h.ticker == ticker:
                h.budget = budget
                h.strategy = strategy
        self._persist()  # ticker already validated
        self.recompute()

    def _on_remove(self, ticker: str) -> None:
        self._generation += 1
        self._session.holdings = [
            h for h in self._session.holdings if h.ticker != ticker
        ]
        self._pending.discard(ticker)
        self._persist()
        self.recompute()

    # -------------------------------------------------------------- results

    def _on_report(self, report: PaperReport) -> None:
        self._teardown()
        if self._running_gen != self._generation:
            # holdings changed while this report was computing; discard it
            if self._dirty:
                self._dirty = False
                self.recompute()
            return
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
            elif looks_transient(res.error):
                # can't tell a bad symbol from a network hiccup; keep it pending
                # (and unsaved) so a later recompute decides
                self._status.setText(
                    f"{pending}: not validated yet ({res.error}) — will retry."
                )
            else:
                self._session.holdings = [
                    h for h in self._session.holdings if h.ticker != pending
                ]
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
            self._session.holdings = [
                h for h in self._session.holdings if h.ticker not in self._pending
            ]
            self._pending.clear()
        self._status.setText(f"Paper session failed: {err}")
        if self._dirty:
            self._dirty = False
            self.recompute()

    # ------------------------------------------------------------ rendering

    def _session_text(self, total_budget: float) -> str:
        return (
            f"{self._session.name} — since {self._session.start_date} · "
            f"fake capital ${total_budget:,.0f}"
        )

    def _render_empty(self) -> None:
        self._table.setRowCount(0)
        self._blotter.setRowCount(0)
        self._detail.clear()
        total = float(sum(h.budget for h in self._session.holdings))
        self._session_label.setText(self._session_text(total))
        self._summary.setText(
            "No holdings yet — click “New trade…” to deploy an algorithm and start "
            "the paper session."
        )
        self._status.setText("")

    def _render(self, report: PaperReport) -> None:
        self._status.setText("")
        self._session_label.setText(self._session_text(report.total_budget))
        self._table.blockSignals(True)
        self._table.setRowCount(len(report.results))
        for i, res in enumerate(report.results):
            h = res.holding
            self._set_item(i, 0, h.ticker)
            self._set_item(i, 1, f"${h.budget:,.0f}")
            # the chosen algorithm actually run (recommended shows what auto picked)
            strategy_text = (
                res.strategy_label if res.error is None
                else strategy_choice_label(h.strategy)
            )
            self._set_item(i, 2, strategy_text)

            if res.error is not None:
                for col in (3, 4, 5, 6, 7, 8):
                    self._set_item(i, col, "no data" if col == 3 else "—")
            else:
                sig_item = QTableWidgetItem(str(res.signal) if res.signal else "—")
                if res.signal is not None:
                    sig_item.setForeground(SIGNAL_COLORS[res.signal])
                self._table.setItem(i, 3, sig_item)
                self._set_item(i, 4, res.position)
                self._set_item(i, 5, str(res.n_trades))
                self._set_item(
                    i, 6, f"${res.value:,.0f}" if res.value is not None else "—"
                )
                self._set_item(i, 7, _money(res.pnl) if res.pnl is not None else "—")
                self._set_item(
                    i, 8, f"{res.ret:+.1%}" if res.ret is not None else "—"
                )

            edit = QPushButton("Edit…")
            edit.clicked.connect(lambda _c, t=h.ticker: self._on_edit(t))
            self._table.setCellWidget(i, len(COLS) - 2, edit)
            remove = QPushButton("Remove")
            remove.clicked.connect(lambda _c, t=h.ticker: self._on_remove(t))
            self._table.setCellWidget(i, len(COLS) - 1, remove)
        self._table.blockSignals(False)

        self._render_blotter(report)

        if report.total_return is None:
            text = (
                f"Started ${report.total_budget:,.0f} on {report.start_date} — "
                "nothing analyzable yet"
                + (f" (skipped: {', '.join(report.skipped)})" if report.skipped else "")
                + "."
            )
        else:
            text = (
                f"Started ${report.total_budget:,.0f} on {report.start_date}  |  "
                f"Value ${report.current_value:,.0f}  |  "
                f"P&L {_money(report.pnl)}  |  "
                f"Return {report.total_return:+.1%}"
            )
            if report.cagr is not None:
                text += f"  |  CAGR {report.cagr:+.1%}/yr"
            if report.ann_vol is not None:
                text += f"  |  Risk {report.ann_vol:.0%}/yr ({report.risk})"
            if report.sharpe is not None:
                text += f"  |  Sharpe {report.sharpe:.2f}"
            if report.max_drawdown is not None:
                text += f"  |  MaxDD {report.max_drawdown:.1%}"
            if report.years is not None:
                text += f"  |  over {report.years:.2f}y"
            if report.skipped:
                text += f"  —  skipped: {', '.join(report.skipped)}"
        self._summary.setText(text)

    def _render_blotter(self, report: PaperReport) -> None:
        fills = report.blotter[-50:]  # most recent 50, oldest first
        self._blotter.setRowCount(len(fills))
        for i, f in enumerate(fills):
            self._blotter.setItem(i, 0, QTableWidgetItem(str(f.date.date())))
            self._blotter.setItem(i, 1, QTableWidgetItem(f.ticker))
            side_item = QTableWidgetItem(f.side)
            if f.side in SIDE_COLORS:
                side_item.setForeground(SIDE_COLORS[f.side])
            self._blotter.setItem(i, 2, side_item)
            self._blotter.setItem(i, 3, QTableWidgetItem(f"${f.price:,.2f}"))
        if not fills:
            self._blotter_label.setText(
                "Trade blotter — no fills yet (algorithm is flat or the session "
                "just started)"
            )
        else:
            self._blotter_label.setText(
                f"Trade blotter — {len(report.blotter)} fill(s) executed by the "
                "deployed algorithm"
            )

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
