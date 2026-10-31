"""Bots tab: what each fake-money bot holds and has done, plus its controls.

Read-only apart from four buttons, and those go through bots/admin.py, never
straight to the ledger: pause, resume, kill and reset have rules (kill cancels
resting orders in the same transaction; reset only after a kill) that belong
next to the ledger, not in a widget. The UI never places an order or moves
cash. Running the bots stays headless (scheduled task / `bots.run_daily`).

Threading: unlike the Watchlist, nothing here touches the network. Every
number comes from two local SQLite files (the ledger and the price cache),
opened for the length of one refresh and closed again, so reads happen on the
UI thread. The ledger is WAL, so reading never blocks the 06:30 run's writes
and the run never blocks a read.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import paths
from bots import admin, report
from bots.registry import BotSpec, ConfigError, load_specs
from data import calendar
from data.fetcher import DataFetcher
from engine.ledger import ACTIVE, KILLED, PAUSED, SOURCE_UI, Ledger
from ui.common import GREEN, GREY, RED

log = logging.getLogger(__name__)

DASH = "—"

OVERVIEW_COLS = [
    "Bot", "Status", "Equity", "Cash", "Return", f"vs {report.BENCHMARK}",
    "Max DD", "Positions", "Last session",
]


def _pct(v: float | None) -> str:
    return DASH if v is None else f"{v:+.2%}"


def _usd(v: float | None) -> str:
    return DASH if v is None else f"${v:,.2f}"


def _day(ts) -> str:
    return DASH if ts is None else str(pd.Timestamp(ts).date())


def _signed(item: QTableWidgetItem, v: float | None) -> QTableWidgetItem:
    if v is not None:
        item.setForeground(GREEN if v >= 0 else RED)
    return item


def _num(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item


def _table(cols: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(cols))
    table.setHorizontalHeaderLabels(cols)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    return table


def _epoch_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    return index.astype("int64").to_numpy() / 1e9


class BotsPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._specs: list[BotSpec] = []
        self._summaries: list[report.BotSummary] = []
        self._order_trails: dict[int, list[dict]] = {}

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        top = QHBoxLayout()
        top.addWidget(self._refresh_btn)
        top.addWidget(self._status, 1)

        self._overview = _table(OVERVIEW_COLS)
        self._overview.itemSelectionChanged.connect(self._on_selection)

        self._pause_btn = QPushButton("Pause")
        self._resume_btn = QPushButton("Resume")
        self._kill_btn = QPushButton("Kill…")
        self._reset_btn = QPushButton("Reset…")
        self._pause_btn.clicked.connect(self._on_pause)
        self._resume_btn.clicked.connect(self._on_resume)
        self._kill_btn.clicked.connect(self._on_kill)
        self._reset_btn.clicked.connect(self._on_reset)
        self._pause_btn.setToolTip(
            "Stop making new decisions from the next session. Orders already "
            "placed still fill."
        )
        self._kill_btn.setToolTip(
            "Permanent. Cancels resting orders; sells everything if the bot's "
            "liquidate_on_kill is set."
        )
        self._reset_btn.setToolTip(
            "Killed bots only: start a fresh account at the starting cash in "
            "bots.yaml. The old account's history is kept."
        )
        controls = QHBoxLayout()
        for b in (self._pause_btn, self._resume_btn, self._kill_btn, self._reset_btn):
            controls.addWidget(b)
        controls.addStretch(1)

        upper = QWidget()
        upper_layout = QVBoxLayout(upper)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        upper_layout.addLayout(top)
        upper_layout.addWidget(self._overview, 1)
        upper_layout.addLayout(controls)

        # detail ----------------------------------------------------------
        self._detail_title = QLabel("Select a bot.")
        self._detail_title.setStyleSheet("font-weight: bold; font-size: 14px;")

        # white like the Watchlist chart, set here rather than relying on
        # detail_panel having called pg.setConfigOptions first
        self._plot = pg.PlotWidget(
            background="w", axisItems={"bottom": pg.DateAxisItem()}
        )
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._legend = self._plot.addLegend(offset=(10, 10))

        self._positions = _table(
            ["Ticker", "Shares", "Avg cost", "Last close", "Value", "Unrealised", "%"]
        )
        self._orders = _table(
            ["#", "Created", "Ticker", "Side", "Type", "Qty", "Status", "Reason"]
        )
        self._orders.itemSelectionChanged.connect(self._on_order_selected)
        self._order_trail = _table(["Session", "Status", "Reject reason", "At"])
        orders_split = QSplitter(Qt.Orientation.Vertical)
        orders_split.addWidget(self._orders)
        orders_split.addWidget(self._order_trail)
        orders_split.setStretchFactor(0, 3)
        orders_split.setStretchFactor(1, 1)
        self._activity = _table(["Session", "Kind", "Detail", "Amount"])

        self._detail_tabs = QTabWidget()
        self._detail_tabs.addTab(self._plot, "Equity")
        self._detail_tabs.addTab(self._positions, "Positions")
        self._detail_tabs.addTab(orders_split, "Orders")
        self._detail_tabs.addTab(self._activity, "Activity")

        lower = QWidget()
        lower_layout = QVBoxLayout(lower)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        lower_layout.addWidget(self._detail_title)
        lower_layout.addWidget(self._detail_tabs, 1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(upper)
        splitter.addWidget(lower)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        self._update_buttons()

    # ------------------------------------------------------------- data access

    def _open(self) -> tuple[Ledger, DataFetcher]:
        return Ledger(paths.ledger_path()), DataFetcher(paths.db_path())

    def showEvent(self, event) -> None:  # noqa: N802 — Qt override
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        selected = self._selected_name()
        try:
            self._specs = load_specs()
        except ConfigError as e:
            self._specs, self._summaries = [], []
            self._overview.setRowCount(0)
            self._status.setText(f"bots.yaml: {e}")
            self._update_buttons()
            return

        latest = calendar.latest_complete_session()
        ledger, prices = self._open()
        try:
            self._summaries = [
                report.summarise(ledger, spec, latest, prices) for spec in self._specs
            ]
        finally:
            ledger.close()
            prices.close()

        self._fill_overview()
        behind = [s.name for s in self._summaries if s.behind]
        msg = f"{len(self._summaries)} bot(s); latest complete session {latest.date()}."
        if behind:
            msg += (
                f" Behind: {', '.join(behind)}. The daily run has not reached "
                f"that session yet (or is waiting for price data; see advisor.log)."
            )
        self._status.setText(msg)

        row = next(
            (i for i, s in enumerate(self._summaries) if s.name == selected), 0
        )
        if self._summaries:
            self._overview.blockSignals(True)
            self._overview.selectRow(row)
            self._overview.blockSignals(False)
        self._show_detail()

    def _fill_overview(self) -> None:
        self._overview.blockSignals(True)
        self._overview.setRowCount(len(self._summaries))
        for r, s in enumerate(self._summaries):
            name = QTableWidgetItem(s.name)
            if s.previous_accounts:
                name.setToolTip(
                    f"trading as {s.account.name}; {s.previous_accounts} earlier "
                    f"account(s) kept as history"
                )
            status = s.status if s.enabled else f"{s.status} (disabled)"
            status_item = QTableWidgetItem(status)
            if s.status == KILLED:
                status_item.setForeground(RED)
            elif s.status == PAUSED:
                status_item.setForeground(GREY)
            last = QTableWidgetItem(_day(s.last_session))
            if s.behind:
                last.setForeground(RED)
                last.setToolTip("behind the latest complete session")
            items = [
                name,
                status_item,
                _num(_usd(s.equity)),
                _num(_usd(s.cash)),
                _signed(_num(_pct(s.total_return)), s.total_return),
                _signed(_num(_pct(s.excess_return)), s.excess_return),
                _num(DASH if s.max_drawdown is None else f"{s.max_drawdown:.2%}"),
                _num(str(s.positions)),
                last,
            ]
            items[5].setToolTip(
                DASH if s.benchmark_return is None
                else f"{report.BENCHMARK} buy-and-hold over the same sessions: "
                     f"{_pct(s.benchmark_return)}"
            )
            for c, item in enumerate(items):
                if not s.enabled:
                    item.setForeground(GREY)
                self._overview.setItem(r, c, item)
        self._overview.blockSignals(False)

    # --------------------------------------------------------------- selection

    def _selected(self) -> report.BotSummary | None:
        rows = self._overview.selectionModel().selectedRows()
        if not rows or rows[0].row() >= len(self._summaries):
            return None
        return self._summaries[rows[0].row()]

    def _selected_name(self) -> str | None:
        s = self._selected()
        return s.name if s else None

    def _spec(self, name: str) -> BotSpec:
        return next(s for s in self._specs if s.name == name)

    def _on_selection(self) -> None:
        self._show_detail()

    def _update_buttons(self) -> None:
        s = self._selected()
        status = s.status if s else None
        self._pause_btn.setEnabled(status == ACTIVE)
        self._resume_btn.setEnabled(status == PAUSED)
        self._kill_btn.setEnabled(status in (ACTIVE, PAUSED))
        self._reset_btn.setEnabled(status == KILLED)

    def _show_detail(self) -> None:
        self._update_buttons()
        for table in (self._positions, self._orders, self._order_trail, self._activity):
            table.setRowCount(0)
        self._plot.clear()
        self._legend.clear()
        self._order_trails = {}

        s = self._selected()
        if s is None:
            self._detail_title.setText("Select a bot.")
            return
        if s.account is None:
            self._detail_title.setText(
                f"{s.name}: not started; it opens an account on its first daily run"
            )
            return
        self._detail_title.setText(
            f"{s.name}: {s.account.name}, {s.strategy}, "
            f"started {_day(s.first_session)}"
        )

        ledger, prices = self._open()
        try:
            curve = ledger.equity_curve(s.account.id)
            bench = report.benchmark_curve(
                report._cached_adjusted(prices, report.BENCHMARK),
                curve.index, s.starting_cash,
            )
            positions = report.positions(ledger, s.account, prices)
            orders = report.orders(ledger, s.account)
            self._order_trails = {o.id: ledger.order_history(o.id) for o in orders}
            activity = report.activity(ledger, s.account)
        finally:
            ledger.close()
            prices.close()

        self._draw_equity(s, curve, bench)
        self._fill_positions(positions)
        self._fill_orders(orders)
        self._fill_activity(activity)

    def _draw_equity(self, s, curve: pd.Series, bench: pd.Series) -> None:
        if curve.empty:
            return
        self._plot.plot(
            _epoch_seconds(curve.index), curve.to_numpy(dtype=float),
            pen=pg.mkPen("#1a73e8", width=2), name=s.name,
        )
        if not bench.empty:
            self._plot.plot(
                _epoch_seconds(bench.index), bench.to_numpy(dtype=float),
                pen=pg.mkPen("#7a7a7a", width=1, style=Qt.PenStyle.DashLine),
                name=f"{report.BENCHMARK} buy-and-hold",
            )

    def _fill_positions(self, rows: list[report.PositionRow]) -> None:
        self._positions.setRowCount(len(rows))
        for r, p in enumerate(rows):
            close = DASH if p.mark is None else f"${p.mark:,.2f} ({_day(p.mark_session)})"
            cells = [
                QTableWidgetItem(p.ticker),
                _num(f"{p.qty:g}"),
                _num(_usd(p.avg_cost)),
                _num(close),
                _num(_usd(p.market_value)),
                _signed(_num(_usd(p.unrealised)), p.unrealised),
                _signed(_num(_pct(p.unrealised_pct)), p.unrealised_pct),
            ]
            for c, item in enumerate(cells):
                self._positions.setItem(r, c, item)

    def _fill_orders(self, orders: list) -> None:
        self._orders.blockSignals(True)
        self._orders.setRowCount(len(orders))
        for r, o in enumerate(orders):
            status = o.status.value
            cells = [
                _num(str(o.id)),
                QTableWidgetItem(_day(o.created_session)),
                QTableWidgetItem(o.ticker),
                QTableWidgetItem(o.side.value),
                QTableWidgetItem(o.type.value),
                _num(f"{o.qty:g}"),
                QTableWidgetItem(status),
                QTableWidgetItem(o.reject_reason or o.reason),
            ]
            if o.reject_reason:
                cells[6].setForeground(RED)
            for c, item in enumerate(cells):
                self._orders.setItem(r, c, item)
        self._orders.blockSignals(False)

    def _on_order_selected(self) -> None:
        self._order_trail.setRowCount(0)
        rows = self._orders.selectionModel().selectedRows()
        if not rows:
            return
        order_id = int(self._orders.item(rows[0].row(), 0).text())
        trail = self._order_trails.get(order_id, [])
        self._order_trail.setRowCount(len(trail))
        for r, e in enumerate(trail):
            for c, key in enumerate(("session", "status", "reject_reason", "at")):
                self._order_trail.setItem(r, c, QTableWidgetItem(e[key] or ""))

    def _fill_activity(self, rows: list[report.ActivityRow]) -> None:
        self._activity.setRowCount(len(rows))
        for r, a in enumerate(rows):
            cells = [
                QTableWidgetItem(a.session),
                QTableWidgetItem(a.kind),
                QTableWidgetItem(a.description),
                _signed(_num(DASH if a.amount is None else f"{a.amount:+,.2f}"), a.amount),
            ]
            for c, item in enumerate(cells):
                self._activity.setItem(r, c, item)

    # ---------------------------------------------------------------- controls
    # Dialogs are separate methods so the offscreen tests can answer them.

    def _ask_reason(self, title: str, prompt: str) -> str | None:
        text, ok = QInputDialog.getText(self, title, prompt)
        return text if ok else None

    def _confirm(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(
            self, title, text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _warn(self, text: str) -> None:
        QMessageBox.warning(self, "Not changed", text)

    def _apply(self, action) -> None:
        """Run one admin action on its own ledger connection, then re-read
        everything: what is on screen must be what the ledger now says."""
        ledger = Ledger(paths.ledger_path())
        try:
            action(ledger)
        except admin.AdminError as e:
            self._warn(str(e))
        finally:
            ledger.close()
        self.refresh()

    def _on_pause(self) -> None:
        s = self._selected()
        if s is None:
            return
        reason = self._ask_reason(f"Pause {s.name}", "Reason (optional):")
        if reason is None:
            return
        spec = self._spec(s.name)
        self._apply(lambda led: admin.pause(led, spec, SOURCE_UI, reason or None))

    def _on_resume(self) -> None:
        s = self._selected()
        if s is None:
            return
        spec = self._spec(s.name)
        self._apply(lambda led: admin.resume(led, spec, SOURCE_UI))

    def _on_kill(self) -> None:
        s = self._selected()
        if s is None:
            return
        spec = self._spec(s.name)
        reason = self._ask_reason(f"Kill {s.name}", "Reason (required, kept in the ledger):")
        if reason is None:
            return
        if not reason.strip():
            self._warn("A kill needs a reason.")
            return
        sells = (
            "and SELLS every position on the next session (liquidate_on_kill is on)"
            if spec.risk.liquidate_on_kill
            else "and keeps its positions (liquidate_on_kill is off)"
        )
        if not self._confirm(
            f"Kill {s.name}?",
            f"This is permanent. {s.name} cancels its {s.open_orders} resting "
            f"order(s) {sells}. It never trades again; Reset starts a fresh account.",
        ):
            return
        self._apply(lambda led: admin.kill(led, spec, SOURCE_UI, reason))

    def _on_reset(self) -> None:
        s = self._selected()
        if s is None:
            return
        spec = self._spec(s.name)
        if not self._confirm(
            f"Reset {s.name}?",
            f"Open a fresh account for {s.name} with ${spec.starting_cash:,.0f} "
            f"(from bots.yaml). {s.account.name} and its history are kept. The new "
            f"account starts trading on the next daily run.",
        ):
            return
        self._apply(lambda led: admin.reset(led, spec))
