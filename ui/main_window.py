"""Main window: the Watchlist table with sorting/filtering, the Refresh button
with its background worker thread, and the detail chart wiring.

Threading rule: yfinance and sqlite3 are touched only inside
WatchlistWorker.run, which executes on its own QThread with its own
DataFetcher (sqlite3 connections are thread-affine). The UI updates
exclusively via queued signal/slot delivery.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QSortFilterProxyModel,
    Qt,
    QThread,
    pyqtSignal,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import paths
from data.fetcher import DataFetcher, FetchError
from data.universe import SURVIVORSHIP_NOTE, describe, get_universe
from ui.common import GREEN, RED
from ui.detail_panel import DetailPanel
from ui.bots_panel import BotsPanel
from ui.investments_panel import InvestmentsPanel

log = logging.getLogger(__name__)

COLUMNS = ("Ticker", "Last close", "Day change %", "Volume")
C_TICKER, C_CLOSE, C_CHANGE, C_VOLUME = range(len(COLUMNS))

DISCLAIMER = (
    "End-of-day Yahoo Finance data, ~15 min delayed and refreshed at most once "
    "a day. " + SURVIVORSHIP_NOTE
)

# toolbar label -> the indexes get_universe() should union
UNIVERSE_CHOICES = (
    ("S&P 500 + NASDAQ-100", ("sp500", "nasdaq100")),
    ("S&P 500", ("sp500",)),
    ("NASDAQ-100", ("nasdaq100",)),
)


@dataclass(slots=True)
class WatchlistRow:
    """One ticker's latest bar. `prices` is the full cached history, kept for
    the detail chart; `error` is set when the ticker could not be fetched."""
    ticker: str
    last_close: float | None = None
    day_change: float | None = None       # fraction, e.g. +0.0123
    volume: int | None = None
    prices: pd.DataFrame | None = None
    error: str | None = None


def build_row(ticker: str, prices: pd.DataFrame) -> WatchlistRow:
    """Latest close, day-over-day change and volume from a price frame. A
    one-bar history has no previous close, so day_change stays None."""
    close = prices["close"]
    last = float(close.iloc[-1])
    change = None
    if len(close) >= 2:
        prev = float(close.iloc[-2])
        if prev > 0:
            change = last / prev - 1.0
    volume = prices["volume"].iloc[-1]
    return WatchlistRow(
        ticker=ticker,
        last_close=last,
        day_change=change,
        volume=None if pd.isna(volume) else int(volume),
        prices=prices,
    )


class WatchlistTableModel(QAbstractTableModel):
    def __init__(self) -> None:
        super().__init__()
        self._rows: list[WatchlistRow] = []

    # Qt plumbing -----------------------------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        row = self._rows[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display(row, col)
        if role == Qt.ItemDataRole.UserRole:
            return self._sort_key(row, col)
        if role == Qt.ItemDataRole.ForegroundRole and col == C_CHANGE:
            if row.day_change is not None:
                return GREEN if row.day_change >= 0 else RED
        return None

    # our API ---------------------------------------------------------------
    def row_at(self, row: int) -> WatchlistRow:
        return self._rows[row]

    def clear(self) -> None:
        self.beginResetModel()
        self._rows = []
        self.endResetModel()

    def append_row(self, row: WatchlistRow) -> None:
        n = len(self._rows)
        self.beginInsertRows(QModelIndex(), n, n)
        self._rows.append(row)
        self.endInsertRows()

    @staticmethod
    def _display(r: WatchlistRow, col: int):
        if col == C_TICKER:
            return r.ticker
        if r.error is not None:
            return "skipped" if col == C_CLOSE else "—"
        if col == C_CLOSE:
            return f"${r.last_close:,.2f}" if r.last_close is not None else "—"
        if col == C_CHANGE:
            return f"{r.day_change:+.2%}" if r.day_change is not None else "—"
        if col == C_VOLUME:
            return f"{r.volume:,}" if r.volume is not None else "—"
        return None

    @staticmethod
    def _sort_key(r: WatchlistRow, col: int):
        if col == C_TICKER:
            return r.ticker
        value = (r.last_close, r.day_change, r.volume)[col - 1]
        return value if value is not None else float("-inf")


class WatchlistFilterProxy(QSortFilterProxyModel):
    """Sorts on raw values (UserRole) and filters on ticker substring."""

    def __init__(self) -> None:
        super().__init__()
        self.setSortRole(Qt.ItemDataRole.UserRole)
        self._ticker_filter = ""

    def set_ticker_filter(self, text: str) -> None:
        self._ticker_filter = text.strip().upper()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        row: WatchlistRow = self.sourceModel().row_at(source_row)
        return not self._ticker_filter or self._ticker_filter in row.ticker


def load_watchlist(
    tickers: list[str],
    fetcher: DataFetcher,
    progress: Callable[[int, int, str], None] | None = None,
    row_ready: Callable[[WatchlistRow], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[WatchlistRow]:
    """Fetch every ticker's latest bar. One failure never stops the sweep: the
    ticker gets an error row and the loop continues."""
    total = len(tickers)

    # One batched, chunked, rate-limited pass for the whole universe instead of
    # a request per ticker. At 518 symbols the per-ticker loop this replaced
    # took minutes; the batch takes seconds and leaves every row a cache read.
    if progress is not None:
        progress(0, total, f"{total} symbols")
    errors = fetcher.refresh(tickers)

    rows: list[WatchlistRow] = []
    for i, ticker in enumerate(tickers, 1):
        if should_stop is not None and should_stop():
            break
        if progress is not None:
            progress(i, total, ticker)
        try:
            prices = fetcher.get_stock(ticker).prices
            row = build_row(ticker, prices)
        except FetchError as e:
            log.warning("%s: skipped from the watchlist: %s", ticker, e)
            row = WatchlistRow(ticker=ticker, error=errors.get(ticker, str(e)))
        except Exception as e:  # noqa: BLE001 — one ticker never kills the sweep
            log.exception("%s: watchlist row failed", ticker)
            row = WatchlistRow(ticker=ticker, error=str(e))
        rows.append(row)
        if row_ready is not None:
            row_ready(row)
    return rows


class WatchlistWorker(QObject):
    progress = pyqtSignal(int, int, str)
    row_ready = pyqtSignal(object)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, tickers: list[str]) -> None:
        super().__init__()
        self._tickers = tickers
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            fetcher = DataFetcher(paths.db_path(), should_stop=self._stop.is_set)
            try:
                rows = load_watchlist(
                    self._tickers,
                    fetcher,
                    progress=self.progress.emit,
                    row_ready=self.row_ready.emit,
                    should_stop=self._stop.is_set,
                )
            finally:
                fetcher.close()
            self.finished.emit(rows)
        except Exception as e:  # pragma: no cover - surfaced to the UI
            log.exception("watchlist worker crashed")
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NASDAQ Advisor — watchlist")
        self.resize(1320, 800)

        self._model = WatchlistTableModel()
        self._proxy = WatchlistFilterProxy()
        self._proxy.setSourceModel(self._model)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.selectionModel().currentRowChanged.connect(self._on_row_changed)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._start_refresh)

        self._universe_box = QComboBox()
        for label, indexes in UNIVERSE_CHOICES:
            self._universe_box.addItem(label, indexes)
        self._universe_box.setToolTip(
            "Which index members to list. Data is cached per ticker, so "
            "switching re-uses whatever has already been fetched."
        )
        self._universe_box.currentIndexChanged.connect(self._on_universe_changed)

        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter ticker…")
        self._filter_edit.textChanged.connect(self._proxy.set_ticker_filter)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self._refresh_btn)
        toolbar.addWidget(self._universe_box)
        toolbar.addWidget(self._filter_edit, 1)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._status = QLabel(f"Ready. Click Refresh to load {describe()}.")
        self._status.setWordWrap(True)

        left_layout = QVBoxLayout()
        left_layout.addLayout(toolbar)
        left_layout.addWidget(self._table, 1)
        left_layout.addWidget(self._progress)
        left_layout.addWidget(self._status)
        left = QWidget()
        left.setLayout(left_layout)

        self._detail = DetailPanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self._investments = InvestmentsPanel()
        self._bots = BotsPanel()
        tabs = QTabWidget()
        tabs.addTab(splitter, "Watchlist")
        tabs.addTab(self._investments, "My Investments")
        tabs.addTab(self._bots, "Bots")
        self.setCentralWidget(tabs)

        self._thread: QThread | None = None
        self._worker: WatchlistWorker | None = None
        self._has_loaded = False

    # refresh lifecycle ------------------------------------------------------
    def selected_indexes(self) -> tuple[str, ...]:
        return self._universe_box.currentData()

    def _on_universe_changed(self, _index: int) -> None:
        # only reload once the user has actually populated the table; otherwise
        # changing the selector before the first Refresh would fire a fetch
        if self._thread is None and self._has_loaded:
            self._start_refresh()

    def _start_refresh(self) -> None:
        if self._thread is not None:
            return
        universe = get_universe(self.selected_indexes())
        self._model.clear()
        self._detail.clear()
        self._refresh_btn.setEnabled(False)
        self._progress.setRange(0, len(universe))
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._status.setText("Loading…")

        self._worker = WatchlistWorker(universe)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.row_ready.connect(self._on_row_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None
        self._refresh_btn.setEnabled(True)
        self._progress.setVisible(False)

    def _on_progress(self, i: int, n: int, ticker: str) -> None:
        self._progress.setValue(i)
        self._status.setText(f"Fetching {ticker} ({i}/{n})…")

    def _on_row_ready(self, row: WatchlistRow) -> None:
        self._model.append_row(row)

    def _on_finished(self, rows: list) -> None:
        self._has_loaded = True
        skipped = [r.ticker for r in rows if r.error]
        msg = f"{len(rows) - len(skipped)} of {describe(self.selected_indexes())} loaded"
        if skipped:
            shown = ", ".join(skipped[:8])
            if len(skipped) > 8:
                shown += f", +{len(skipped) - 8} more"
            msg += f", {len(skipped)} skipped: {shown}"
        self._status.setText(f"{msg}. {DISCLAIMER}")
        self._teardown_thread()

    def _on_failed(self, err: str) -> None:
        self._status.setText(f"Refresh failed: {err}")
        self._teardown_thread()

    # selection --------------------------------------------------------------
    def _on_row_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        if not current.isValid():
            return
        source = self._proxy.mapToSource(current)
        row = self._model.row_at(source.row())
        self._detail.show_prices(row.ticker, row.prices)

    def closeEvent(self, event) -> None:
        self._investments.shutdown()
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            if not self._thread.wait(10_000):
                # never let a running QThread be destroyed — Qt aborts the
                # process; the interruptible backoff bounds this wait to
                # roughly one in-flight HTTP call
                self._thread.wait()
        super().closeEvent(event)
