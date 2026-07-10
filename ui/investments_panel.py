"""My Investments tab: real-money lot tracking.

Each row is an actual investment the user made: money in on a buy date, under
a chosen trend, and — once sold — a sell date. Open lots are valued live from
the latest cached close; sold lots freeze at the close on the day they were
sold, keeping the return they had earned and staying on the books. Same
worker/pending-validation architecture as the paper-portfolio panel: all
fetching happens on a QThread, and a new ticker is only persisted after a
worker report proves Yahoo knows it.

The account also holds uninvested cash. Buying spends it, selling refills it,
and it only grows from outside via an explicit Deposit — so a position funded
by reinvested profits can never be mistaken for fresh money from the user's
bank. That distinction is what the Real Investment panel reports on; the maths
lives in portfolio/cash.py.
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
    QGroupBox,
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
from portfolio.cash import (
    DEPOSIT,
    WITHDRAWAL,
    CashFlow,
    compute_real_pnl,
    next_seq,
    validate,
)
from portfolio.tracker import (
    InvestmentReport,
    Lot,
    Totals,
    analyze_investments,
    load_investments_doc,
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
    "never moves again. Buying spends cash, so deposit before you buy."
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


class CashFlowDialog(QDialog):
    """Money moving between the user's bank and the account. This is the only
    place external capital is recorded, so it is the only thing Real Net Profit
    measures the account against."""

    def __init__(self, parent, direction: str) -> None:
        super().__init__(parent)
        self._direction = direction
        depositing = direction == DEPOSIT
        verb = "Deposit" if depositing else "Withdraw"
        self.setWindowTitle(verb)
        self.setModal(True)

        today = QDate.currentDate()
        self._amount = QDoubleSpinBox()
        self._amount.setRange(0.01, 100_000_000.0)
        self._amount.setDecimals(2)
        self._amount.setPrefix("$ ")
        self._amount.setGroupSeparatorShown(True)
        self._amount.setValue(1000.0)
        self._date = QDateEdit(today)
        self._date.setCalendarPopup(True)
        self._date.setMaximumDate(today)

        form = QFormLayout()
        form.addRow("Amount", self._amount)
        form.addRow("On", self._date)

        note = QLabel(
            "Money in from your bank. It sits as cash until you buy something."
            if depositing else
            "Money out to your bank. It leaves the account for good."
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

    def values(self) -> tuple[float, date]:
        return float(self._amount.value()), self._date.date().toPyDate()


class RealInvestmentPanel(QGroupBox):
    """How much the user is up on the money they actually deposited.

    Deliberately separate from the summary strip above it: that one measures
    the positions, this one measures the account. They report the same profit
    (see portfolio/cash.py) but different denominators, and the difference is
    the whole point.
    """

    _ROWS = (
        ("deposited", "Total deposited"),
        ("withdrawn", "Total withdrawn"),
        ("net_external", "Net external capital"),
        ("account_value", "Current account value"),
        ("profit", "Real net profit"),
        ("return_pct", "Real return"),
    )

    def __init__(self) -> None:
        super().__init__("Real Investment")
        self._values: dict[str, QLabel] = {}
        self._labels: dict[str, QLabel] = {}
        form = QFormLayout(self)
        for key, caption in self._ROWS:
            label = QLabel(caption)
            value = QLabel("—")
            self._labels[key] = label
            self._values[key] = value
            form.addRow(label, value)
        self._values["profit"].setStyleSheet("font-weight: bold;")

    def clear(self) -> None:
        for value in self._values.values():
            value.setText("—")
            value.setStyleSheet("")
        self._values["profit"].setStyleSheet("font-weight: bold;")
        self._set_return_visible(False)

    def _set_return_visible(self, visible: bool) -> None:
        self._labels["return_pct"].setVisible(visible)
        self._values["return_pct"].setVisible(visible)

    def show_pnl(self, r, priced: bool) -> None:
        """`priced` is False while any open position lacks a market value — a
        sale still awaiting its close, or a ticker Yahoo would not return. The
        external-capital rows are exact regardless; the rows derived from
        account value are withheld rather than shown short."""
        plain = lambda v: f"${v:,.2f}"  # noqa: E731 — these can't be negative
        self._values["deposited"].setText(plain(r.total_deposited))
        self._values["withdrawn"].setText(plain(r.total_withdrawn))
        self._values["net_external"].setText(plain(r.net_external))

        if not priced:
            for key in ("account_value", "profit", "return_pct"):
                self._values[key].setText("valuing…")
                self._values[key].setStyleSheet("")
            self._values["profit"].setStyleSheet("font-weight: bold;")
            self._set_return_visible(True)
            return

        self._values["account_value"].setText(plain(r.account_value))

        profit = self._values["profit"]
        profit.setText(money(r.real_net_profit))
        colour = GREEN if r.real_net_profit >= 0 else RED
        # QTableWidgetItem.setForeground doesn't exist on QLabel
        profit.setStyleSheet(f"font-weight: bold; color: {colour.name()};")

        # a return on zero or negative capital is meaningless, not infinite
        show_return = r.real_return_pct is not None
        self._set_return_visible(show_return)
        if show_return:
            value = self._values["return_pct"]
            value.setText(f"{r.real_return_pct:+.1f}%")
            value.setStyleSheet(f"color: {colour.name()};")


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
        self._lots, self._flows = load_investments_doc()
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
        deposit_btn = QPushButton("Deposit…")
        deposit_btn.clicked.connect(lambda: self._on_cash_flow(DEPOSIT))
        withdraw_btn = QPushButton("Withdraw…")
        withdraw_btn.clicked.connect(lambda: self._on_cash_flow(WITHDRAWAL))
        # cash is now a first-class concept: a buy spends it, a sale refills it
        self._cash_label = QLabel("")
        self._cash_label.setStyleSheet("font-weight: bold;")
        add_row = QHBoxLayout()
        add_row.addWidget(add_btn)
        add_row.addWidget(deposit_btn)
        add_row.addWidget(withdraw_btn)
        add_row.addStretch(1)
        add_row.addWidget(self._cash_label)

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
        self._real = RealInvestmentPanel()
        self._note = QLabel(DISCLAIMER)
        self._note.setWordWrap(True)

        left_layout = QVBoxLayout()
        left_layout.addLayout(add_row)
        left_layout.addWidget(self._table, 1)
        left_layout.addWidget(self._summary)
        left_layout.addWidget(self._summary_open)
        left_layout.addWidget(self._real)
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
            save_investments(keep, cash_flows=self._flows)
        except OSError as e:
            log.exception("investments save failed")
            self._status.setText(f"WARNING: investments not saved: {e}")

    def _refuse(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def _on_new_investment(self) -> None:
        dialog = InvestmentDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        ticker, money_in, buy, strategy, sell = dialog.values()
        if not ticker:
            return
        seq = next_seq(self._lots, self._flows)
        lot = Lot(
            ticker=ticker,
            money_in=money_in,
            buy_date=buy.isoformat(),
            sell_date=sell.isoformat() if sell is not None else None,
            strategy=strategy,
            buy_seq=seq,
            sell_seq=seq + 1 if sell is not None else None,
        )
        # a buy spends cash that must already be there; the user deposits first.
        # This is what forces reinvested profits through the cash balance
        # instead of being miscounted as fresh external capital.
        error = validate(self._lots + [lot], self._flows)
        if error is not None:
            self._refuse("Not enough cash", error)
            return
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

    def _on_cash_flow(self, direction: str) -> None:
        dialog = CashFlowDialog(self, direction)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        amount, when = dialog.values()
        flow = CashFlow(
            amount=amount,
            direction=direction,
            date=when.isoformat(),
            seq=next_seq(self._lots, self._flows),
        )
        error = validate(self._lots, self._flows + [flow])
        if error is not None:
            self._refuse("Not enough cash", error)
            return
        self._flows.append(flow)
        self._persist()
        # external cash moves no position, so nothing needs revaluing and the
        # generation must not bump — that would discard an in-flight report
        self._refresh_real()

    def _on_remove(self, lot: Lot) -> None:
        if lot.is_sold:
            # realized P&L is locked in: allowing a closed lot to be deleted
            # would let a losing trade be erased to inflate Real Net Profit
            self._refuse(
                "Position already sold",
                f"{lot.ticker} has been sold, so its result is on the books. "
                f"Undo the sale first if you need to remove it.",
            )
            return
        remaining = [l for l in self._lots if l is not lot]
        # dropping an open lot only credits its money_in back, which raises
        # every later balance — this can never actually reject. Kept as a
        # defensive assertion; see tests/phase11_real_pnl_check.py.
        error = validate(remaining, self._flows)
        if error is not None:  # pragma: no cover - unreachable by construction
            self._refuse("Cannot remove", error)
            return
        self._generation += 1
        self._lots = remaining
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
        # value it at the close on this date, freezing the return it earned.
        # The worker resolves that close into money_out (see _on_report), which
        # is the proceeds credited to cash — a sale only ever adds cash, so it
        # needs no overdraft check.
        lot.sell_date = dialog.sell_date().isoformat()
        lot.sell_seq = next_seq(self._lots, self._flows)
        self._persist()
        self.recompute()

    def _on_undo_sell(self, lot: Lot) -> None:
        """Reopen a sold lot. This withdraws the sale's credit, lowering every
        later balance, so unlike Remove it genuinely can overdraw — a later buy
        may have been funded by these proceeds."""
        reopened = Lot(
            ticker=lot.ticker, money_in=lot.money_in, buy_date=lot.buy_date,
            strategy=lot.strategy, buy_seq=lot.buy_seq,
        )
        prospective = [reopened if l is lot else l for l in self._lots]
        error = validate(prospective, self._flows)
        if error is not None:
            self._refuse("Cannot undo this sale", error)
            return
        self._generation += 1
        lot.sell_date = None
        lot.sell_seq = None
        lot.money_out = None
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
        self._resolve_proceeds(report)
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

    def _resolve_proceeds(self, report: InvestmentReport) -> None:
        """Freeze a fresh sale's proceeds into `money_out`.

        `SellDialog` runs on the UI thread and has no DataFetcher, so it cannot
        price the sale itself. The worker already does: for a sold lot with no
        money_out, `_analyze_lot` values it at the close on the sell date, and
        that number *is* the proceeds. Writing it back makes the cash balance a
        ledger — fixed once, immune to later price-cache refreshes — and leaves
        tracker.py's valuation untouched, since it now takes its existing
        `money_out is not None` branch and returns the identical figure.
        """
        for res in report.results:
            lot = res.lot
            if (
                lot.sell_date is not None
                and lot.money_out is None
                and res.error is None
                and res.worth is not None
            ):
                lot.money_out = float(res.worth)

    def _on_failed(self, err: str) -> None:
        self._teardown()
        if self._pending:
            self._lots = [l for l in self._lots if l.ticker not in self._pending]
            self._pending.clear()
        self._status.setText(f"Investment analysis failed: {err}")
        self._refresh_real()
        if self._dirty:
            self._dirty = False
            self.recompute()

    # ------------------------------------------------------------ rendering

    def _refresh_real(self) -> None:
        """Cash balance and the Real Investment panel. Both read the ledger, so
        they are correct the moment a deposit lands — no worker round-trip."""
        worth_by_lot = {
            id(res.lot): res.worth
            for res in (self._report.results if self._report else ())
            if res.error is None and res.worth is not None
        }
        open_worth = {
            i: worth_by_lot[id(lot)]
            for i, lot in enumerate(self._lots)
            if id(lot) in worth_by_lot
        }
        r = compute_real_pnl(self._lots, self._flows, open_worth)

        # cash needs no prices; only an unpriced sale can leave it unknown
        if r.unresolved_sales:
            self._cash_label.setText("Cash: valuing sale…")
        else:
            self._cash_label.setText(f"Cash: ${r.cash:,.2f}")

        priced = not r.unresolved_sales and all(
            id(lot) in worth_by_lot for lot in self._lots if not lot.is_sold
        )
        self._real.show_pnl(r, priced)

    def _render_empty(self) -> None:
        self._table.setRowCount(0)
        self._detail.clear()
        self._summary.setText(
            "No investments recorded — add what you actually bought above."
        )
        self._summary_open.setVisible(False)
        self._status.setText("")
        self._refresh_real()

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
            elif lot.is_sold:
                # the sale credited cash and froze the lot's return; undoing it
                # is the only way back to an open position (and thence Remove)
                undo_btn = QPushButton("Undo sale")
                undo_btn.clicked.connect(lambda _c, l=lot: self._on_undo_sell(l))
                self._table.setCellWidget(i, C_SELL, undo_btn)
            else:
                # setItem alone never removes an existing cell widget — a
                # leftover Sell button would silently rewrite recorded sales
                self._table.removeCellWidget(i, C_SELL)
                self._set_item(i, C_SELL, "")
            remove_btn = QPushButton("Remove")
            remove_btn.setEnabled(not lot.is_sold)
            if lot.is_sold:
                # realized P&L stays on the books; see _on_remove
                remove_btn.setToolTip("Undo the sale first.")
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
        self._refresh_real()

    def _totals_line(self, prefix: str, totals: Totals) -> str:
        # FLAG (real-investment-pnl pass): "Put in" is sum(money_in), which
        # counts every reinvestment of a previous sale. The user reads it as
        # money deposited from their bank; it is not. Left as-is by request —
        # the Real Investment panel below carries the honest figures.
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
