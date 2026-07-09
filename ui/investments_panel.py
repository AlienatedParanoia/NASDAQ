"""My Investments tab: real-money lot tracking.

Each row is an actual investment the user made: money in on a buy date, under
a chosen trend, and — once sold — a sell date. Open lots are valued live from
the latest cached close; sold lots freeze at the close on the day they were
sold, keeping the return they had earned and staying on the books. Same
worker/pending-validation architecture as the paper-portfolio panel: all
fetching happens on a QThread, and a new ticker is only persisted after a
worker report proves Yahoo knows it.
"""
from __future__ import annotations

import logging
import threading
from datetime import date

from PyQt6.QtCore import QDate, QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
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
from data.fetcher import DataFetcher
from portfolio.tracker import (
    InvestmentReport,
    Lot,
    Totals,
    analyze_investments,
    load_investments,
    looks_transient,
    save_investments,
)
from ui.common import (
    GREEN,
    RED,
    SIGNAL_COLORS,
    STRATEGY_CHOICES,
    money,
    strategy_choice_label,
)
from ui.detail_panel import DetailPanel

log = logging.getLogger(__name__)

COLS = (
    "Ticker", "Put in", "Bought", "Trend", "Worth / Out", "P&L", "Return",
    "/yr", "Risk", "Max DD", "Signal", "Sell", "",
)
(
    C_TICKER, C_IN, C_BOUGHT, C_TREND, C_WORTH, C_PNL, C_RETURN,
    C_YR, C_RISK, C_MAXDD, C_SIGNAL, C_SELL, C_REMOVE,
) = range(len(COLS))

# every stat column, blanked out when a lot has no data
_STAT_COLS = (C_PNL, C_RETURN, C_YR, C_RISK, C_MAXDD, C_SIGNAL)

DISCLAIMER = (
    "Real-position tracker — open lots are valued from ~15-min-delayed closes; "
    "sold lots are frozen at the close on the day you sold, so their return "
    "never moves again."
)


class InvestmentDialog(QDialog):
    """Pop-up panel for entering an investment: which stock, how much money
    went in, when, under which trend, and whether it has already been sold."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New investment")
        self.setModal(True)

        today = QDate.currentDate()

        self._ticker = QLineEdit()
        self._ticker.setMaxLength(12)
        self._ticker.setPlaceholderText("e.g. AAPL, SHOP, VOO")
        self._money_in = QDoubleSpinBox()
        self._money_in.setRange(1.0, 100_000_000.0)
        self._money_in.setDecimals(2)
        self._money_in.setPrefix("$ ")
        self._money_in.setGroupSeparatorShown(True)
        self._money_in.setValue(1000.0)
        self._buy_date = QDateEdit(today)
        self._buy_date.setCalendarPopup(True)
        self._buy_date.setMaximumDate(today)
        self._strategy = QComboBox()
        for label, value in STRATEGY_CHOICES:
            self._strategy.addItem(label, value)
        self._sold_check = QCheckBox("I have already sold this")
        self._sold_check.toggled.connect(self._on_sold_toggled)
        self._sell_date = QDateEdit(today)
        self._sell_date.setCalendarPopup(True)
        self._sell_date.setMaximumDate(today)
        self._sell_date.setEnabled(False)

        form = QFormLayout()
        form.addRow("Ticker", self._ticker)
        form.addRow("Money in", self._money_in)
        form.addRow("Bought on", self._buy_date)
        form.addRow("Trend", self._strategy)
        form.addRow("", self._sold_check)
        form.addRow("Sold on", self._sell_date)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _on_sold_toggled(self, checked: bool) -> None:
        self._sell_date.setEnabled(checked)

    def _on_accept(self) -> None:
        if not self._ticker.text().strip():
            QMessageBox.warning(self, "Ticker required", "Enter a ticker symbol.")
            return
        if (
            self._sold_check.isChecked()
            and self._sell_date.date() < self._buy_date.date()
        ):
            QMessageBox.warning(
                self,
                "Invalid sell date",
                "The sell date can't be before the buy date.",
            )
            return
        self.accept()

    def values(self) -> tuple[str, float, date, str | None, date | None]:
        sold = self._sold_check.isChecked()
        return (
            self._ticker.text().strip().upper(),
            float(self._money_in.value()),
            self._buy_date.date().toPyDate(),
            self._strategy.currentData(),
            self._sell_date.date().toPyDate() if sold else None,
        )


class SellDialog(QDialog):
    """Asks only when the position was sold — the proceeds are the market
    close on that day, so there is nothing to type."""

    def __init__(self, parent, lot: Lot) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Sell {lot.ticker}")
        self.setModal(True)

        today = QDate.currentDate()
        self._sell_date = QDateEdit(today)
        self._sell_date.setCalendarPopup(True)
        self._sell_date.setMaximumDate(today)
        self._sell_date.setMinimumDate(QDate.fromString(lot.buy_date, Qt.DateFormat.ISODate))

        form = QFormLayout()
        form.addRow("Sold on", self._sell_date)

        note = QLabel(
            f"Your {lot.ticker} position will be valued at that day's close and "
            f"kept on the books; its return stops there."
        )
        note.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def sell_date(self) -> date:
        return self._sell_date.date().toPyDate()


class InvestmentsWorker(QObject):
    finished = pyqtSignal(object)  # InvestmentReport
    failed = pyqtSignal(str)

    def __init__(self, lots: list[Lot]) -> None:
        super().__init__()
        self._lots = lots
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            fetcher = DataFetcher(paths.db_path(), should_stop=self._stop.is_set)
            try:
                report = analyze_investments(
                    self._lots, fetcher, should_stop=self._stop.is_set
                )
            finally:
                fetcher.close()
            self.finished.emit(report)
        except Exception as e:  # pragma: no cover - surfaced to the UI
            log.exception("investments worker crashed")
            self.failed.emit(str(e))


class InvestmentsPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._lots: list[Lot] = load_investments()
        self._report: InvestmentReport | None = None
        self._thread: QThread | None = None
        self._worker: InvestmentsWorker | None = None
        self._dirty = False
        self._pending: set[str] = set()  # tickers never yet validated; unsaved
        # bumped on every mutation; reports computed under an older generation
        # are discarded so a mid-flight Sell/Remove never renders mixed state
        self._generation = 0
        self._running_gen = -1

        add_btn = QPushButton("Add investment…")
        add_btn.clicked.connect(self._on_new_investment)
        add_row = QHBoxLayout()
        add_row.addWidget(add_btn)
        add_row.addStretch(1)

        self._table = QTableWidget(0, len(COLS))
        self._table.setHorizontalHeaderLabels(COLS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for col in (len(COLS) - 2, len(COLS) - 1):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_selection)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("font-weight: bold;")
        self._summary_open = QLabel("")
        self._summary_open.setWordWrap(True)
        self._status = QLabel("")
        self._note = QLabel(DISCLAIMER)
        self._note.setWordWrap(True)

        left_layout = QVBoxLayout()
        left_layout.addLayout(add_row)
        left_layout.addWidget(self._table, 1)
        left_layout.addWidget(self._summary)
        left_layout.addWidget(self._summary_open)
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
        if self._lots:
            self.recompute()

    # ------------------------------------------------------------ lifecycle

    def recompute(self) -> None:
        if self._thread is not None:
            self._dirty = True
            return
        if not self._lots:
            self._report = None
            self._render_empty()
            return
        self._status.setText("Valuing investments…")
        self._running_gen = self._generation
        self._worker = InvestmentsWorker(list(self._lots))
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
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            if not self._thread.wait(10_000):
                self._thread.wait()

    # ------------------------------------------------------------ mutations

    def _persist(self) -> None:
        keep = [lot for lot in self._lots if lot.ticker not in self._pending]
        try:
            save_investments(keep)
        except OSError as e:
            log.exception("investments save failed")
            self._status.setText(f"WARNING: investments not saved: {e}")

    def _on_new_investment(self) -> None:
        dialog = InvestmentDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        ticker, money_in, buy, strategy, sell = dialog.values()
        if not ticker:
            return
        lot = Lot(
            ticker=ticker,
            money_in=money_in,
            buy_date=buy.isoformat(),
            sell_date=sell.isoformat() if sell is not None else None,
            strategy=strategy,
        )
        already_validated = any(
            l.ticker == ticker and l.ticker not in self._pending for l in self._lots
        )
        self._generation += 1
        self._lots.append(lot)
        if not already_validated:
            self._pending.add(ticker)  # saved only once a report validates it
        else:
            self._persist()
        self.recompute()

    def _on_remove(self, lot: Lot) -> None:
        self._generation += 1
        self._lots = [l for l in self._lots if l is not lot]
        if not any(l.ticker == lot.ticker for l in self._lots):
            self._pending.discard(lot.ticker)
        self._persist()
        self.recompute()

    def _on_sell(self, lot: Lot) -> None:
        dialog = SellDialog(self, lot)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._generation += 1
        # the lot stays on the books; leaving money_out None makes the tracker
        # value it at the close on this date, freezing the return it earned
        lot.sell_date = dialog.sell_date().isoformat()
        self._persist()
        self.recompute()

    # -------------------------------------------------------------- results

    def _on_report(self, report: InvestmentReport) -> None:
        self._teardown()
        if self._running_gen != self._generation:
            # lots changed while this report was computing; discard it — the
            # mutation already queued a fresh recompute via the dirty flag
            if self._dirty:
                self._dirty = False
                self.recompute()
            return
        rejected: list[tuple[str, str]] = []
        for pending in sorted(self._pending):
            results = [r for r in report.results if r.lot.ticker == pending]
            if not results:
                continue  # stale report from before this add; next one decides
            errors = [r.error for r in results if r.error is not None]
            if not errors:
                self._pending.discard(pending)
            elif all(looks_transient(e) for e in errors):
                self._status.setText(
                    f"{pending}: not validated yet ({errors[0]}) — will retry."
                )
            else:
                self._lots = [l for l in self._lots if l.ticker != pending]
                self._pending.discard(pending)
                rejected.append((pending, errors[0]))
        self._persist()
        if rejected:
            QMessageBox.warning(
                self,
                "Investment not added",
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
            self._lots = [l for l in self._lots if l.ticker not in self._pending]
            self._pending.clear()
        self._status.setText(f"Investment analysis failed: {err}")
        if self._dirty:
            self._dirty = False
            self.recompute()

    # ------------------------------------------------------------ rendering

    def _render_empty(self) -> None:
        self._table.setRowCount(0)
        self._detail.clear()
        self._summary.setText(
            "No investments recorded — add what you actually bought above."
        )
        self._summary_open.setVisible(False)
        self._status.setText("")

    def _render(self, report: InvestmentReport) -> None:
        self._status.setText("")
        self._table.setRowCount(len(report.results))
        for i, res in enumerate(report.results):
            lot = res.lot
            self._set_item(i, C_TICKER, lot.ticker)
            self._set_item(i, C_IN, f"${lot.money_in:,.0f}")
            bought = lot.buy_date
            if lot.is_sold:
                bought += f" → {lot.sell_date or '?'}"
            self._set_item(i, C_BOUGHT, bought)
            self._set_item(i, C_TREND, strategy_choice_label(lot.strategy))
            if res.error is not None:
                self._set_item(i, C_WORTH, "no data")
                for col in _STAT_COLS:
                    self._set_item(i, col, "—")
            else:
                self._set_item(i, C_WORTH, f"${res.worth:,.0f}")
                pnl_item = QTableWidgetItem(money(res.pnl))
                pnl_item.setForeground(GREEN if res.pnl >= 0 else RED)
                self._table.setItem(i, C_PNL, pnl_item)
                self._set_item(i, C_RETURN, f"{res.ret:+.1%}")
                self._set_item(
                    i, C_YR,
                    f"{res.annualized:+.1%}" if res.annualized is not None else "—",
                )
                self._set_item(
                    i, C_RISK, f"{res.ann_vol:.0%}" if res.ann_vol is not None else "—"
                )
                self._set_item(
                    i, C_MAXDD, f"{res.max_dd:.1%}" if res.max_dd is not None else "—"
                )
                sig_item = QTableWidgetItem(str(res.signal) if res.signal else "—")
                if res.signal is not None:
                    sig_item.setForeground(SIGNAL_COLORS[res.signal])
                self._table.setItem(i, C_SIGNAL, sig_item)
            if not lot.is_sold and res.error is None:
                sell_btn = QPushButton("Sell…")
                sell_btn.clicked.connect(lambda _c, l=lot: self._on_sell(l))
                self._table.setCellWidget(i, C_SELL, sell_btn)
            else:
                # setItem alone never removes an existing cell widget — a
                # leftover Sell button would silently rewrite recorded sales
                self._table.removeCellWidget(i, C_SELL)
                self._set_item(i, C_SELL, "Sold" if lot.is_sold else "")
            remove_btn = QPushButton("Remove")
            remove_btn.clicked.connect(lambda _c, l=lot: self._on_remove(l))
            self._table.setCellWidget(i, C_REMOVE, remove_btn)

        if report.overall is None:
            text = (
                f"Put in ${report.total_in:,.0f} — nothing analyzable yet"
                + (f" (skipped: {', '.join(report.skipped)})" if report.skipped else "")
                + "."
            )
            self._summary_open.setVisible(False)
        else:
            text = self._totals_line("All positions (incl. sold)", report.overall)
            if report.skipped:
                text += f"  —  skipped: {', '.join(report.skipped)}"
            # the open-only line would just repeat the overall one when
            # nothing has been sold, and says nothing when nothing is open
            show_open = (
                report.open_only is not None
                and report.open_only.count != report.overall.count
            )
            self._summary_open.setVisible(show_open)
            if show_open:
                self._summary_open.setText(
                    self._totals_line("Open positions only", report.open_only)
                )
        for note in report.notes:
            text += f"\nNote: {note}"
        self._summary.setText(text)

    def _totals_line(self, prefix: str, totals: Totals) -> str:
        text = (
            f"{prefix} ({totals.count})  |  "
            f"Put in ${totals.total_in:,.0f}  |  "
            f"Now worth {money(totals.current_value).lstrip('+')}  |  "
            f"P&L {money(totals.pnl)}"
        )
        if totals.total_return is not None:
            text += f"  |  Return {totals.total_return:+.1%}"
        if totals.annualized is not None:
            text += f"  |  {totals.annualized:+.1%}/yr"
        if totals.ann_vol is not None:
            text += (
                f"  |  Risk {totals.ann_vol:.0%}/yr ({totals.risk})"
                f"  |  Sharpe {totals.sharpe:.2f}"
                f"  |  MaxDD {totals.max_drawdown:.1%}"
            )
        text += f"  |  since {totals.since}"
        return text

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
            if res.scan_row is not None:
                self._detail.show_row(res.scan_row)
            else:
                self._detail.clear()
