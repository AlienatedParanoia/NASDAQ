"""My Investments tab: real-money lot tracking.

Each row is an actual investment the user made: money in on a buy date, and
— once sold — money out on a sell date. Open lots are valued live from the
latest cached close; sold lots use the user's entered proceeds as the truth
for P&L. Same worker/pending-validation architecture as the paper-portfolio
panel: all fetching happens on a QThread, and a new ticker is only persisted
after a worker report proves Yahoo knows it.
"""
from __future__ import annotations

import logging
import threading
from datetime import date

from PyQt6.QtCore import QDate, QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QDateEdit,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
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
from portfolio.manager import looks_transient
from portfolio.tracker import (
    InvestmentReport,
    Lot,
    analyze_investments,
    load_investments,
    save_investments,
)
from ui.detail_panel import DetailPanel
from ui.portfolio_panel import SIGNAL_COLORS, _money

log = logging.getLogger(__name__)

COLS = (
    "Ticker", "Put in", "Bought", "Worth / Out", "P&L", "Return",
    "/yr", "Risk", "Max DD", "Signal", "Sell", "",
)

GREEN = QColor("#1e8e3e")
RED = QColor("#d93025")

DISCLAIMER = (
    "Real-position tracker — open lots are valued from ~15-min-delayed closes; "
    "sold lots use the amounts you entered (your real fills and fees)."
)


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

        today = QDate.currentDate()

        self._ticker_edit = QLineEdit()
        self._ticker_edit.setPlaceholderText("Ticker (e.g. AAPL, SHOP, VOO)…")
        self._ticker_edit.setMaxLength(12)
        self._ticker_edit.returnPressed.connect(self._on_add)
        self._in_spin = QDoubleSpinBox()
        self._in_spin.setRange(1.0, 100_000_000.0)
        self._in_spin.setDecimals(2)
        self._in_spin.setValue(1000.0)
        self._in_spin.setPrefix("in $ ")
        self._in_spin.setGroupSeparatorShown(True)
        self._buy_date = QDateEdit(today)
        self._buy_date.setCalendarPopup(True)
        self._buy_date.setMaximumDate(today)
        self._sold_check = QCheckBox("Sold")
        self._sold_check.toggled.connect(self._on_sold_toggled)
        self._out_spin = QDoubleSpinBox()
        self._out_spin.setRange(0.0, 1_000_000_000.0)
        self._out_spin.setDecimals(2)
        self._out_spin.setValue(1000.0)
        self._out_spin.setPrefix("out $ ")
        self._out_spin.setGroupSeparatorShown(True)
        self._out_spin.setEnabled(False)
        self._sell_date = QDateEdit(today)
        self._sell_date.setCalendarPopup(True)
        self._sell_date.setMaximumDate(today)
        self._sell_date.setEnabled(False)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add)

        add_row = QHBoxLayout()
        add_row.addWidget(self._ticker_edit, 1)
        add_row.addWidget(self._in_spin)
        add_row.addWidget(self._buy_date)
        add_row.addWidget(self._sold_check)
        add_row.addWidget(self._out_spin)
        add_row.addWidget(self._sell_date)
        add_row.addWidget(add_btn)

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
        self._status = QLabel("")
        self._note = QLabel(DISCLAIMER)
        self._note.setWordWrap(True)

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

    def _on_sold_toggled(self, checked: bool) -> None:
        self._out_spin.setEnabled(checked)
        self._sell_date.setEnabled(checked)

    def _on_add(self) -> None:
        ticker = self._ticker_edit.text().strip().upper()
        if not ticker:
            return
        buy = self._buy_date.date().toPyDate()
        sold = self._sold_check.isChecked()
        sell = self._sell_date.date().toPyDate() if sold else None
        if sold and sell < buy:
            self._status.setText("Sell date can't be before the buy date.")
            return
        lot = Lot(
            ticker=ticker,
            money_in=float(self._in_spin.value()),
            buy_date=buy.isoformat(),
            money_out=float(self._out_spin.value()) if sold else None,
            sell_date=sell.isoformat() if sold else None,
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
        self._ticker_edit.clear()
        self.recompute()

    def _on_remove(self, lot: Lot) -> None:
        self._generation += 1
        self._lots = [l for l in self._lots if l is not lot]
        if not any(l.ticker == lot.ticker for l in self._lots):
            self._pending.discard(lot.ticker)
        self._persist()
        self.recompute()

    def _on_sell(self, lot: Lot) -> None:
        proceeds, ok = QInputDialog.getDouble(
            self,
            "Sell position",
            f"Proceeds from selling your {lot.ticker} position (what you got out):",
            value=lot.money_in,
            min=0.01,
            max=1_000_000_000.0,
            decimals=2,
        )
        if not ok:
            return
        self._generation += 1
        lot.money_out = float(proceeds)
        lot.sell_date = date.today().isoformat()
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
        self._status.setText("")

    def _render(self, report: InvestmentReport) -> None:
        self._status.setText("")
        self._table.setRowCount(len(report.results))
        for i, res in enumerate(report.results):
            lot = res.lot
            self._set_item(i, 0, lot.ticker)
            self._set_item(i, 1, f"${lot.money_in:,.0f}")
            bought = lot.buy_date
            if lot.money_out is not None:
                bought += f" → {lot.sell_date or '?'}"
            self._set_item(i, 2, bought)
            if res.error is not None:
                self._set_item(i, 3, "no data")
                for col in (4, 5, 6, 7, 8, 9):
                    self._set_item(i, col, "—")
            else:
                self._set_item(i, 3, f"${res.worth:,.0f}")
                pnl_item = QTableWidgetItem(_money(res.pnl))
                pnl_item.setForeground(GREEN if res.pnl >= 0 else RED)
                self._table.setItem(i, 4, pnl_item)
                self._set_item(i, 5, f"{res.ret:+.1%}")
                self._set_item(
                    i, 6,
                    f"{res.annualized:+.1%}" if res.annualized is not None else "—",
                )
                self._set_item(
                    i, 7, f"{res.ann_vol:.0%}" if res.ann_vol is not None else "—"
                )
                self._set_item(
                    i, 8, f"{res.max_dd:.1%}" if res.max_dd is not None else "—"
                )
                sig_item = QTableWidgetItem(str(res.signal) if res.signal else "—")
                if res.signal is not None:
                    sig_item.setForeground(SIGNAL_COLORS[res.signal])
                self._table.setItem(i, 9, sig_item)
            if lot.money_out is None and res.error is None:
                sell_btn = QPushButton("Sell…")
                sell_btn.clicked.connect(lambda _c, l=lot: self._on_sell(l))
                self._table.setCellWidget(i, 10, sell_btn)
            else:
                # setItem alone never removes an existing cell widget — a
                # leftover Sell button would silently rewrite recorded sales
                self._table.removeCellWidget(i, 10)
                self._set_item(i, 10, "")
            remove_btn = QPushButton("Remove")
            remove_btn.clicked.connect(lambda _c, l=lot: self._on_remove(l))
            self._table.setCellWidget(i, 11, remove_btn)

        if report.current_value is None:
            text = (
                f"Put in ${report.total_in:,.0f} — nothing analyzable yet"
                + (f" (skipped: {', '.join(report.skipped)})" if report.skipped else "")
                + "."
            )
        else:
            text = (
                f"Put in ${report.total_in:,.0f}  |  "
                f"Now worth {_money(report.current_value).lstrip('+')}  |  "
                f"P&L {_money(report.pnl)}  |  "
                f"Return {report.total_return:+.1%}"
            )
            if report.annualized is not None:
                text += f"  |  {report.annualized:+.1%}/yr"
            if report.ann_vol is not None:
                text += (
                    f"  |  Risk {report.ann_vol:.0%}/yr ({report.risk})"
                    f"  |  Sharpe {report.sharpe:.2f}"
                    f"  |  MaxDD {report.max_drawdown:.1%}"
                )
            if report.since:
                text += f"  |  since {report.since}"
            if report.skipped:
                text += f"  —  skipped: {', '.join(report.skipped)}"
        for note in report.notes:
            text += f"\nNote: {note}"
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
            if res.scan_row is not None:
                self._detail.show_row(res.scan_row)
            else:
                self._detail.clear()
