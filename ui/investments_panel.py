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

import copy
import logging
import threading
from dataclasses import dataclass
from datetime import date
from typing import Sequence

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
    funding_breakdown,
    ledger_events,
    next_seq,
    replay,
    validate,
)
from portfolio.storage import backup_file
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
    GREY,
    RED,
    SIGNAL_COLORS,
    STRATEGY_CHOICES,
    money,
    strategy_choice_index,
    strategy_choice_label,
)
from ui.detail_panel import DetailPanel

log = logging.getLogger(__name__)

COLS = (
    "Ticker", "Put in", "Funded", "Bought", "Trend", "Worth / Out", "P&L",
    "Return", "/yr", "Risk", "Max DD", "Signal", "Sell", "", "",
)
(
    C_TICKER, C_IN, C_FUNDED, C_BOUGHT, C_TREND, C_WORTH, C_PNL, C_RETURN,
    C_YR, C_RISK, C_MAXDD, C_SIGNAL, C_SELL, C_EDIT, C_REMOVE,
) = range(len(COLS))

# the cash ledger, exactly as portfolio/cash.py replays it
LEDGER_COLS = ("Date", "Transaction", "Amount", "Cash balance")

# how many actions back Undo can walk. The stack lives in memory only: closing
# the app is not an undo, and restarting it starts fresh.
MAX_UNDO = 20


@dataclass(slots=True, eq=False)
class Snapshot:
    """The whole mutable state of the tab before one user action.

    Deep-copied on the way in, because a Lot is mutated in place — a sale
    writes sell_date, and the worker later writes money_out into the very same
    object. A shallow copy would let those writes reach back and rewrite
    history that Undo is holding.
    """
    label: str                  # names the action being undone, for the tooltip
    lots: list[Lot]
    flows: list[CashFlow]
    pending: set[str]

# every stat column, blanked out when a lot has no data
_STAT_COLS = (C_PNL, C_RETURN, C_YR, C_RISK, C_MAXDD, C_SIGNAL)

DISCLAIMER = (
    "Real-position tracker — open lots are valued from ~15-min-delayed closes; "
    "sold lots are frozen at the close on the day you sold, so their return "
    "never moves again. Buying spends any cash you hold; whatever it falls "
    "short by is recorded as money you put in. Deposit only to record cash you "
    "transferred in but have not invested."
)


class InvestmentDialog(QDialog):
    """Pop-up panel for entering an investment: which stock, how much money
    went in, when, under which trend, and whether it has already been sold.

    Opened by "Add investment…" to create one, or a row's "Edit…" button to
    change an existing one — the same shape as PaperTradeDialog. In edit mode
    the ticker is fixed (it has already been validated against Yahoo, and
    changing it would make this a different position), and the already-sold
    fields are gone: Edit is offered on open lots only, because rewriting a
    closed lot's amount would rewrite its realized P&L.
    """

    def __init__(self, parent=None, *, edit: Lot | None = None) -> None:
        super().__init__(parent)
        self._editing = edit is not None
        self.setWindowTitle("Edit investment" if self._editing else "New investment")
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

        if edit is not None:
            self._ticker.setText(edit.ticker)
            self._ticker.setReadOnly(True)
            self._money_in.setValue(edit.money_in)
            self._buy_date.setDate(
                QDate.fromString(edit.buy_date, Qt.DateFormat.ISODate)
            )
            self._strategy.setCurrentIndex(strategy_choice_index(edit.strategy))

        form = QFormLayout()
        form.addRow("Ticker", self._ticker)
        form.addRow("Money in", self._money_in)
        form.addRow("Bought on", self._buy_date)
        form.addRow("Trend", self._strategy)
        if edit is None:
            form.addRow("", self._sold_check)
            form.addRow("Sold on", self._sell_date)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        note = QLabel(
            "Any part of this the cash in your account can't cover is recorded "
            "as money you put in, dated on the buy."
            if edit is None else
            "Changing the amount or the buy date moves this transaction in the "
            "cash ledger, so what counts as money you put in may change."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
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
        # the sold fields are not shown when editing, so an edit never closes
        # a position — that is what the Sell button is for
        sold = self._sold_check.isChecked() and not self._editing
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

    Three display states, and keeping them apart is the difference between a
    useful panel and a lying one:

    WAITING   a worker report is on its way. The numbers will arrive; say so.
    PARTIAL   some position has no price data and never will. An unknown worth
              is still non-negative, so what we can compute is a genuine lower
              bound — shown with a "≥" and the excluded tickers named.
    EXACT     everything is priced.
    """

    WAITING, PARTIAL, EXACT = "waiting", "partial", "exact"

    _ROWS = (
        ("deposited", "Total deposited"),
        ("withdrawn", "Total withdrawn"),
        ("net_external", "Net external capital"),
        ("account_value", "Current account value"),
        ("profit", "Real net profit"),
        ("return_pct", "Real return"),
    )
    _DERIVED = ("account_value", "profit", "return_pct")

    def __init__(self) -> None:
        super().__init__("Real Investment")
        self._values: dict[str, QLabel] = {}
        self._labels: dict[str, QLabel] = {}
        form = QFormLayout()
        for key, caption in self._ROWS:
            label = QLabel(caption)
            value = QLabel("—")
            self._labels[key] = label
            self._values[key] = value
            form.addRow(label, value)
        self._values["profit"].setStyleSheet("font-weight: bold;")
        self._caveat = QLabel("")
        self._caveat.setWordWrap(True)
        self._caveat.setStyleSheet(f"color: {GREY.name()};")
        self._caveat.setVisible(False)
        outer = QVBoxLayout(self)
        outer.addLayout(form)
        outer.addWidget(self._caveat)

    def _set_return_visible(self, visible: bool) -> None:
        self._labels["return_pct"].setVisible(visible)
        self._values["return_pct"].setVisible(visible)

    def show_pnl(self, r, state: str, missing: Sequence[str] = ()) -> None:
        """The external-capital rows are exact in every state: they read the
        cash-flow log, which needs no prices. Only the rows derived from
        account value depend on the market."""
        plain = lambda v: f"${v:,.2f}"  # noqa: E731 — these can't be negative
        deposited = plain(r.total_deposited)
        if r.auto_funded > 0:
            # never let an inferred figure masquerade as one the user typed
            deposited += f"  (incl. {plain(r.auto_funded)} from your buys)"
        self._values["deposited"].setText(deposited)
        self._values["withdrawn"].setText(plain(r.total_withdrawn))
        self._values["net_external"].setText(plain(r.net_external))

        if state == self.WAITING:
            for key in self._DERIVED:
                self._values[key].setText("valuing…")
                self._values[key].setStyleSheet("")
            self._values["profit"].setStyleSheet("font-weight: bold;")
            self._set_return_visible(True)
            self._caveat.setVisible(False)
            return

        bound = "≥ " if state == self.PARTIAL else ""
        self._values["account_value"].setText(bound + plain(r.account_value))

        # In PARTIAL the profit is a lower bound, so a negative figure does not
        # establish a loss — the missing position could cover it. Colouring it
        # red would assert something we have not shown. Green is safe: the gain
        # can only be larger.
        if state == self.PARTIAL and r.real_net_profit < 0:
            colour = GREY
        else:
            colour = GREEN if r.real_net_profit >= 0 else RED

        profit = self._values["profit"]
        profit.setText(bound + money(r.real_net_profit))
        # QTableWidgetItem.setForeground doesn't exist on QLabel
        profit.setStyleSheet(f"font-weight: bold; color: {colour.name()};")

        # a return on zero or negative capital is meaningless, not infinite
        show_return = r.real_return_pct is not None
        self._set_return_visible(show_return)
        if show_return:
            value = self._values["return_pct"]
            value.setText(f"{bound}{r.real_return_pct:+.1f}%")
            value.setStyleSheet(f"color: {colour.name()};")

        self._caveat.setVisible(state == self.PARTIAL)
        if state == self.PARTIAL:
            self._caveat.setText(
                f"No price data for {', '.join(missing)} — excluded, so these "
                f"are lower bounds: the true figures can only be higher."
            )


class ProfitReportDialog(QDialog):
    """Two truths about the same portfolio, kept deliberately apart.

    The top line is the overall profit — account value minus the money that
    actually came from your bank. The table below breaks it down per position:
    what each cost, whether it was bought with your money or with proceeds
    recycled from an earlier sale, what it is worth now, and its profit or loss.

    The footer shows reinvested proceeds and net profit on their own lines and
    never adds them. Reinvestment is a flow of money through the account, not a
    gain; the profit total already reflects every reinvested dollar that was
    later won or lost, so summing the two would count the same money twice.
    """

    _COLS = ("Position", "Cost", "Funded by", "Value now", "P&L")

    def __init__(self, parent, results, funding, pnl) -> None:
        super().__init__(parent)
        self.setWindowTitle("Profit report")
        self.setModal(True)
        self.resize(560, 420)
        layout = QVBoxLayout(self)

        # -- overall (the same figure the Real Investment panel shows live)
        headline = QLabel(f"Real net profit: {money(pnl.real_net_profit)}")
        colour = GREEN if pnl.real_net_profit >= 0 else RED
        headline.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {colour.name()};"
        )
        layout.addWidget(headline)
        sub = QLabel(
            f"Account value ${pnl.account_value:,.2f}  −  money you put in "
            f"${pnl.net_external:,.2f}"
            + (f"   ({pnl.real_return_pct:+.1f}%)"
               if pnl.real_return_pct is not None else "")
        )
        sub.setStyleSheet(f"color: {GREY.name()};")
        layout.addWidget(sub)

        # -- per-position breakdown
        table = QTableWidget(len(results), len(self._COLS))
        table.setHorizontalHeaderLabels(self._COLS)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        reinvested = 0.0
        pnl_total = 0.0
        priced = True
        for r, (res, fund) in enumerate(zip(results, funding)):
            lot = res.lot
            table.setItem(r, 0, QTableWidgetItem(lot.ticker))
            table.setItem(r, 1, QTableWidgetItem(f"${lot.money_in:,.0f}"))

            if fund is None:
                funded = "—"
            elif fund.source == "Own":
                funded = "your money"
            elif fund.source == "Proceeds":
                funded = "from proceeds"
                reinvested += fund.from_proceeds
            else:
                funded = f"${fund.from_own:,.0f} own + ${fund.from_proceeds:,.0f} proc."
                reinvested += fund.from_proceeds
            table.setItem(r, 2, QTableWidgetItem(funded))

            if res.error is not None or res.worth is None:
                table.setItem(r, 3, QTableWidgetItem("no data"))
                table.setItem(r, 4, QTableWidgetItem("—"))
                priced = False
                continue
            table.setItem(r, 3, QTableWidgetItem(f"${res.worth:,.0f}"))
            pnl_item = QTableWidgetItem(money(res.pnl))
            pnl_item.setForeground(GREEN if res.pnl >= 0 else RED)
            table.setItem(r, 4, pnl_item)
            pnl_total += res.pnl
        layout.addWidget(table, 1)

        # -- footer: two separate lines, never summed
        reinv = QLabel(f"Reinvested proceeds (money recycled through positions): "
                       f"${reinvested:,.0f}")
        reinv.setStyleSheet(f"color: {GREY.name()};")
        reinv.setWordWrap(True)
        layout.addWidget(reinv)

        net = QLabel(
            f"Net profit across positions: {money(pnl_total)}"
            + ("" if priced else "   (excludes positions with no price data)")
        )
        net.setStyleSheet("font-weight: bold;")
        layout.addWidget(net)

        note = QLabel(
            "Reinvested proceeds is how much money has cycled through your "
            "positions, not a gain — it is not added to your profit. The profit "
            "already includes every reinvested dollar that was later won or lost."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {GREY.name()}; font-size: 11px;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.accept)
        layout.addWidget(buttons)


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
        self._undo: list[Snapshot] = []  # in-memory only; see MAX_UNDO
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
        self._undo_btn = QPushButton("Undo")
        self._undo_btn.clicked.connect(self._on_undo)
        self._reset_btn = QPushButton("Reset…")
        self._reset_btn.clicked.connect(self._on_reset)
        report_btn = QPushButton("Profit report…")
        report_btn.clicked.connect(self._on_report_dialog)
        # cash is now a first-class concept: a buy spends it, a sale refills it
        self._cash_label = QLabel("")
        self._cash_label.setStyleSheet("font-weight: bold;")
        add_row = QHBoxLayout()
        add_row.addWidget(add_btn)
        add_row.addWidget(deposit_btn)
        add_row.addWidget(withdraw_btn)
        add_row.addWidget(self._undo_btn)
        add_row.addWidget(self._reset_btn)
        add_row.addWidget(report_btn)
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
        for col in (C_SELL, C_EDIT, C_REMOVE):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_selection)

        # -- cash ledger (every external flow and every buy/sell leg, in order)
        self._ledger = QTableWidget(0, len(LEDGER_COLS))
        self._ledger.setHorizontalHeaderLabels(LEDGER_COLS)
        self._ledger.verticalHeader().setVisible(False)
        self._ledger.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._ledger.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._ledger.setMaximumHeight(160)
        self._ledger.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._ledger_label = QLabel("Transactions")

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
        left_layout.addWidget(self._ledger_label)
        left_layout.addWidget(self._ledger)
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
        self._update_undo_button()
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

    # ----------------------------------------------------------------- undo

    def _checkpoint(self, label: str) -> None:
        """Record the state *before* an action. Call it only once the action is
        certain to happen — after any validation or dialog cancel — or Undo
        fills up with steps that changed nothing."""
        self._undo.append(Snapshot(
            label=label,
            lots=copy.deepcopy(self._lots),
            flows=copy.deepcopy(self._flows),
            pending=set(self._pending),
        ))
        del self._undo[:-MAX_UNDO]   # keep the newest MAX_UNDO entries
        self._update_undo_button()

    def _drop_checkpoint(self, label: str) -> None:
        """Forget the top checkpoint if it belongs to `label`. Used when an
        action is undone by the app itself — a rejected ticker — so Undo does
        not offer to reverse something that never took effect."""
        if self._undo and self._undo[-1].label == label:
            self._undo.pop()
            self._update_undo_button()

    def _update_undo_button(self) -> None:
        can = bool(self._undo)
        self._undo_btn.setEnabled(can)
        self._undo_btn.setToolTip(
            f"Undo: {self._undo[-1].label}" if can else "Nothing to undo"
        )

    def _on_undo(self) -> None:
        if not self._undo:
            return
        snapshot = self._undo.pop()
        self._generation += 1
        self._lots = snapshot.lots
        self._flows = snapshot.flows
        self._pending = snapshot.pending
        self._persist()
        self._update_undo_button()
        self._status.setText(f"Undid: {snapshot.label}")
        # the restored lots are fresh objects, so the current report describes
        # positions that no longer exist — drop it rather than render it
        self._report = None
        self._table.setRowCount(0)
        self._detail.clear()
        self.recompute()
        if not self._lots:
            self._render_empty()
            self._status.setText(f"Undid: {snapshot.label}")

    def _on_report_dialog(self) -> None:
        if not self._lots:
            self._refuse("Nothing to report", "Add an investment first.")
            return
        if self._report is None:
            self._refuse(
                "Still valuing",
                "The positions are still being priced — try again in a moment.",
            )
            return
        # align each result with its funding by lot identity; the report's rows
        # are in self._lots order, but never assume it
        fb = {
            id(self._lots[f.index]): f
            for f in funding_breakdown(self._lots, self._flows).values()
            if 0 <= f.index < len(self._lots)
        }
        results = self._report.results
        funding = [fb.get(id(res.lot)) for res in results]
        worth = {
            i: res.worth
            for i, res in enumerate(results)
            if res.error is None and res.worth is not None
        }
        # recompute against exactly these results, so the headline matches
        open_worth = {
            i: worth[j]
            for i, lot in enumerate(self._lots)
            for j, res in enumerate(results)
            if res.lot is lot and j in worth
        }
        pnl = compute_real_pnl(self._lots, self._flows, open_worth)
        ProfitReportDialog(self, results, funding, pnl).exec()

    def _on_reset(self) -> None:
        if not self._lots and not self._flows:
            return
        answer = QMessageBox.question(
            self,
            "Reset My Investments",
            f"Delete all {len(self._lots)} position(s) and {len(self._flows)} "
            f"cash flow(s)?\n\nA backup of investments.json is kept, and this "
            f"can be undone while the app stays open.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        # the in-memory undo dies with the process; the file is the real safety
        # net, so move it aside before anything overwrites it
        backup_file(paths.investments_path())
        self._checkpoint("Reset")
        self._generation += 1
        self._lots = []
        self._flows = []
        self._pending.clear()
        self._report = None
        self._persist()
        self._detail.clear()
        self._render_empty()
        self._status.setText("Reset — a backup of your data was kept.")

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
        # a buy spends cash on hand first and infers a deposit for the rest, so
        # it cannot fail — reinvested profits are consumed before any new
        # external capital is invented. The check stays for withdrawals, which
        # are the only event that can still overdraw.
        error = validate(self._lots + [lot], self._flows)
        if error is not None:  # pragma: no cover - buys autofund
            self._refuse("Not enough cash", error)
            return
        already_validated = any(
            l.ticker == ticker and l.ticker not in self._pending for l in self._lots
        )
        self._checkpoint(f"Add {ticker}")
        self._generation += 1
        self._lots.append(lot)
        if not already_validated:
            self._pending.add(ticker)  # saved only once a report validates it
        else:
            self._persist()
        self.recompute()

    def _on_edit(self, lot: Lot) -> None:
        """Reopen the investment dialog pre-filled, the way Paper Trading's
        Edit… does. Offered on open lots only: a closed lot's realized P&L is
        on the books, and editing its cost basis would rewrite it.

        Raising `money_in`, or moving `buy_date` earlier, debits the ledger
        harder or sooner. That cannot fail — a buy funds itself — but it does
        change how much capital is inferred, so the panel is recomputed.
        """
        if lot.is_sold:  # pragma: no cover - the button is disabled
            self._refuse(
                "Position already sold",
                f"{lot.ticker} has been sold. Undo the sale before editing it.",
            )
            return
        dialog = InvestmentDialog(self, edit=lot)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        _, money_in, buy, strategy, _ = dialog.values()
        updated = Lot(
            ticker=lot.ticker,          # fixed: already validated against Yahoo
            money_in=money_in,
            buy_date=buy.isoformat(),
            strategy=strategy,
            buy_seq=lot.buy_seq,        # keeps its place among same-date events
        )
        error = validate([updated if l is lot else l for l in self._lots], self._flows)
        if error is not None:
            self._refuse("Not enough cash", error)
            return
        self._checkpoint(f"Edit {lot.ticker}")
        self._generation += 1
        lot.money_in = money_in
        lot.buy_date = updated.buy_date
        lot.strategy = strategy
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
        self._checkpoint("Deposit" if direction == DEPOSIT else "Withdrawal")
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
        self._checkpoint(f"Remove {lot.ticker}")
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
        self._checkpoint(f"Sell {lot.ticker}")
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
        self._checkpoint(f"Undo sale of {lot.ticker}")
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
                # the app just reversed the add itself; offering to undo it
                # again would walk the user back past an action they never saw
                self._drop_checkpoint(f"Add {pending}")
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
        they are correct the moment a deposit lands — no worker round-trip.

        A lot without a value is either *waiting* (no report has covered it
        yet, one is coming) or *broken* (the worker priced it and Yahoo had
        nothing). Conflating the two strands the panel on "valuing…" forever
        for a delisted ticker, so they are separated here.
        """
        results = {
            id(res.lot): res
            for res in (self._report.results if self._report else ())
        }
        worth_by_lot = {
            lot_id: res.worth
            for lot_id, res in results.items()
            if res.error is None and res.worth is not None
        }
        open_worth = {
            i: worth_by_lot[id(lot)]
            for i, lot in enumerate(self._lots)
            if id(lot) in worth_by_lot
        }
        r = compute_real_pnl(self._lots, self._flows, open_worth)

        waiting: set[int] = set()
        broken: set[int] = set()
        for lot in self._lots:
            # a sold lot needs its proceeds; an open one needs a market value
            known = (
                lot.money_out is not None if lot.is_sold
                else id(lot) in worth_by_lot
            )
            if known:
                continue
            res = results.get(id(lot))
            failed = res is not None and res.error is not None
            (broken if failed else waiting).add(id(lot))

        # Cash needs no prices — only a sale of unknown proceeds dents it, and
        # an unresolved sale credits nothing (portfolio/cash.py), so the figure
        # is understated: a lower bound, never an overstatement.
        unpriced_sales = [
            lot for lot in self._lots if lot.is_sold and lot.money_out is None
        ]
        if not unpriced_sales:
            self._cash_label.setText(f"Cash: ${r.cash:,.2f}")
        elif any(id(lot) in waiting for lot in unpriced_sales):
            self._cash_label.setText("Cash: valuing sale…")
        else:
            self._cash_label.setText(f"Cash: ≥ ${r.cash:,.2f}")

        tickers = lambda ids: sorted({  # noqa: E731
            lot.ticker for lot in self._lots if id(lot) in ids
        })
        if waiting:
            state, missing = RealInvestmentPanel.WAITING, ()
        elif broken:
            state, missing = RealInvestmentPanel.PARTIAL, tickers(broken)
        else:
            state, missing = RealInvestmentPanel.EXACT, ()
        self._real.show_pnl(r, state, missing)
        self._render_ledger(r.unresolved_sales)

    def _render_ledger(self, unresolved_sales: int) -> None:
        """Every cash movement in replay order, with the running balance — the
        same fold `validate` walks, so when it refuses a transaction the user
        can see exactly which row it meant."""
        trail = replay(ledger_events(self._lots, self._flows))
        self._ledger.setRowCount(len(trail))
        for i, (event, balance) in enumerate(trail):
            self._ledger.setItem(i, 0, QTableWidgetItem(event.date))
            self._ledger.setItem(i, 1, QTableWidgetItem(event.title))
            amount = QTableWidgetItem(money(event.delta))
            amount.setForeground(GREEN if event.delta >= 0 else RED)
            self._ledger.setItem(i, 2, amount)
            self._ledger.setItem(i, 3, QTableWidgetItem(f"${balance:,.2f}"))
        self._ledger.scrollToBottom()

        if not trail:
            self._ledger_label.setText(
                "Transactions — none yet. Deposit cash before you buy."
            )
            return
        text = f"Transactions — {len(trail)} in the cash ledger"
        if unresolved_sales:
            # its credit is missing from the rows above, so say so rather than
            # let the closing balance look wrong
            text += (
                f", plus {unresolved_sales} sale(s) still being valued "
                f"(their proceeds are not yet in the balance)"
            )
        self._ledger_label.setText(text)

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
        # how each lot was paid for, keyed by the lot object so it survives the
        # report's ordering (funding_breakdown keys by index into self._lots)
        funding = {
            id(self._lots[f.index]): f
            for f in funding_breakdown(self._lots, self._flows).values()
            if 0 <= f.index < len(self._lots)
        }
        self._table.setRowCount(len(report.results))
        for i, res in enumerate(report.results):
            lot = res.lot
            self._set_item(i, C_TICKER, lot.ticker)
            self._set_item(i, C_IN, f"${lot.money_in:,.0f}")
            self._set_funded_item(i, funding.get(id(lot)))
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
            # Edit and Remove are gated identically: a closed lot's realized
            # P&L is on the books and must not be rewritten from the table
            edit_btn = QPushButton("Edit…")
            edit_btn.setEnabled(not lot.is_sold)
            edit_btn.clicked.connect(lambda _c, l=lot: self._on_edit(l))
            remove_btn = QPushButton("Remove")
            remove_btn.setEnabled(not lot.is_sold)
            if lot.is_sold:
                edit_btn.setToolTip("Undo the sale first.")
                remove_btn.setToolTip("Undo the sale first.")
            remove_btn.clicked.connect(lambda _c, l=lot: self._on_remove(l))
            self._table.setCellWidget(i, C_EDIT, edit_btn)
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

    def _set_funded_item(self, row: int, f) -> None:
        """Own / Proceeds / Mixed, with the exact dollar split on hover."""
        if f is None:
            self._set_item(row, C_FUNDED, "—")
            return
        item = QTableWidgetItem(f.source)
        if f.source == "Proceeds":
            item.setForeground(GREEN)   # bought entirely with gains realized earlier
        item.setToolTip(
            f"${f.from_own:,.0f} your money  +  ${f.from_proceeds:,.0f} from proceeds"
        )
        self._table.setItem(row, C_FUNDED, item)

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
