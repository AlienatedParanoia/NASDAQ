"""Main window: the scan table with sorting/filtering, the Scan button with
its background worker thread, and the detail panel wiring.

Threading rule: yfinance and sqlite3 are touched only inside ScanWorker.run,
which executes on its own QThread with its own DataFetcher (sqlite3
connections are thread-affine). The UI updates exclusively via queued
signal/slot delivery.
"""
from __future__ import annotations

import logging
import threading

from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QSortFilterProxyModel,
    Qt,
    QThread,
    pyqtSignal,
)
from PyQt6.QtGui import QColor
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
from classifier.classify import Bucket
from data.fetcher import DataFetcher
from data.universe import get_universe
from signals.signal_generator import ScanRow, Signal, scan_universe
from ui.detail_panel import DetailPanel
from ui.investments_panel import InvestmentsPanel
from ui.portfolio_panel import PortfolioPanel

log = logging.getLogger(__name__)

COLUMNS = ("Ticker", "Bucket", "Signal", "Ret/yr", "Sharpe", "Max DD")

WINDOW_CHOICES = (("10 years", 10), ("5 years", 5), ("2 years", 2))

SIGNAL_COLORS = {
    Signal.BUY: QColor("#1e8e3e"),
    Signal.HOLD: QColor("#7a7a7a"),
    Signal.AVOID: QColor("#d93025"),
}

DISCLAIMER = (
    "Signals are informational only — backtests assume no costs/slippage and "
    "data is ~15 min delayed."
)


class ScanTableModel(QAbstractTableModel):
    def __init__(self) -> None:
        super().__init__()
        self._rows: list[ScanRow] = []

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
        if role == Qt.ItemDataRole.ForegroundRole and col == 2 and row.signal is not None:
            return SIGNAL_COLORS.get(row.signal)
        return None

    # our API ---------------------------------------------------------------
    def row_at(self, row: int) -> ScanRow:
        return self._rows[row]

    def clear(self) -> None:
        self.beginResetModel()
        self._rows = []
        self.endResetModel()

    def append_row(self, row: ScanRow) -> None:
        n = len(self._rows)
        self.beginInsertRows(QModelIndex(), n, n)
        self._rows.append(row)
        self.endInsertRows()

    @staticmethod
    def _display(r: ScanRow, col: int):
        if col == 0:
            return r.ticker
        if col == 1:
            return str(r.bucket)
        if col == 2:
            return str(r.signal) if r.signal else ("skipped" if r.error else "—")
        if col == 3:
            return f"{r.cagr:+.1%}" if r.cagr is not None else "—"
        if col == 4:
            return f"{r.sharpe:.2f}" if r.sharpe is not None else "—"
        if col == 5:
            return f"{r.max_dd:.1%}" if r.max_dd is not None else "—"
        return None

    @staticmethod
    def _sort_key(r: ScanRow, col: int):
        if col == 0:
            return r.ticker
        if col == 1:
            return str(r.bucket)
        if col == 2:
            return str(r.signal) if r.signal else ""
        value = (r.cagr, r.sharpe, r.max_dd)[col - 3]
        return value if value is not None else float("-inf")


class ScanFilterProxy(QSortFilterProxyModel):
    """Sorts on raw values (UserRole) and filters on ticker substring +
    bucket equality."""

    def __init__(self) -> None:
        super().__init__()
        self.setSortRole(Qt.ItemDataRole.UserRole)
        self._ticker_filter = ""
        self._bucket_filter: str | None = None

    def set_ticker_filter(self, text: str) -> None:
        self._ticker_filter = text.strip().upper()
        self.invalidateFilter()

    def set_bucket_filter(self, bucket: str | None) -> None:
        self._bucket_filter = bucket
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        row: ScanRow = self.sourceModel().row_at(source_row)
        if self._ticker_filter and self._ticker_filter not in row.ticker:
            return False
        if self._bucket_filter and str(row.bucket) != self._bucket_filter:
            return False
        return True


class ScanWorker(QObject):
    progress = pyqtSignal(int, int, str)
    row_ready = pyqtSignal(object)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, tickers: list[str], window_years: int) -> None:
        super().__init__()
        self._tickers = tickers
        self._window_years = window_years
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            fetcher = DataFetcher(paths.db_path(), should_stop=self._stop.is_set)
            try:
                rows = scan_universe(
                    self._tickers,
                    fetcher,
                    progress=self.progress.emit,
                    row_ready=self.row_ready.emit,
                    should_stop=self._stop.is_set,
                    window_years=self._window_years,
                )
            finally:
                fetcher.close()
            self.finished.emit(rows)
        except Exception as e:  # pragma: no cover - surfaced to the UI
            log.exception("scan worker crashed")
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NASDAQ Advisor — NASDAQ-100 scanner")
        self.resize(1320, 800)

        self._model = ScanTableModel()
        self._proxy = ScanFilterProxy()
        self._proxy.setSourceModel(self._model)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.selectionModel().currentRowChanged.connect(self._on_row_changed)

        self._scan_btn = QPushButton("Scan")
        self._scan_btn.clicked.connect(self._start_scan)

        self._window_box = QComboBox()
        for label, years in WINDOW_CHOICES:
            self._window_box.addItem(label, years)
        self._window_box.setToolTip(
            "Backtest window. Data is cached at 10 years; switching re-analyzes "
            "instantly without refetching."
        )
        self._window_box.currentIndexChanged.connect(self._on_window_changed)

        self._bucket_box = QComboBox()
        self._bucket_box.addItem("All buckets")
        for b in Bucket:
            self._bucket_box.addItem(str(b))
        self._bucket_box.currentIndexChanged.connect(self._on_bucket_filter)

        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter ticker…")
        self._filter_edit.textChanged.connect(self._proxy.set_ticker_filter)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self._scan_btn)
        toolbar.addWidget(self._window_box)
        toolbar.addWidget(self._bucket_box)
        toolbar.addWidget(self._filter_edit, 1)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._status = QLabel("Ready. Click Scan to fetch and analyze the NASDAQ-100.")
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

        self._portfolio = PortfolioPanel(lambda: self._window_box.currentData())
        self._investments = InvestmentsPanel()
        tabs = QTabWidget()
        tabs.addTab(splitter, "Scanner")
        tabs.addTab(self._portfolio, "Portfolio")
        tabs.addTab(self._investments, "My Investments")
        self.setCentralWidget(tabs)

        self._thread: QThread | None = None
        self._worker: ScanWorker | None = None
        self._has_scanned = False

    # scan lifecycle ---------------------------------------------------------
    def _start_scan(self) -> None:
        if self._thread is not None:
            return
        universe = get_universe()
        self._model.clear()
        self._detail.clear()
        self._scan_btn.setEnabled(False)
        self._progress.setRange(0, len(universe))
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._status.setText("Starting scan…")

        self._worker = ScanWorker(universe, self._window_box.currentData())
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
        self._scan_btn.setEnabled(True)
        self._progress.setVisible(False)

    def _on_progress(self, i: int, n: int, ticker: str) -> None:
        self._progress.setValue(i)
        self._status.setText(f"Fetching {ticker} ({i}/{n})…")

    def _on_row_ready(self, row: ScanRow) -> None:
        self._model.append_row(row)

    def _on_finished(self, rows: list) -> None:
        self._has_scanned = True
        skipped = [r.ticker for r in rows if r.error]
        msg = f"{len(rows) - len(skipped)} stocks scanned"
        if skipped:
            msg += f", {len(skipped)} skipped: {', '.join(skipped)}"
        self._status.setText(f"{msg}. {DISCLAIMER}")
        self._teardown_thread()

    def _on_failed(self, err: str) -> None:
        self._status.setText(f"Scan failed: {err}")
        self._teardown_thread()

    # filters + selection ----------------------------------------------------
    def _on_window_changed(self, _index: int) -> None:
        # data is cached, so re-analyzing under the new window is seconds
        if self._thread is None and self._has_scanned:
            self._start_scan()
        self._portfolio.window_changed()

    def _on_bucket_filter(self, index: int) -> None:
        self._proxy.set_bucket_filter(None if index == 0 else self._bucket_box.currentText())

    def _on_row_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        if not current.isValid():
            return
        source = self._proxy.mapToSource(current)
        self._detail.show_row(self._model.row_at(source.row()))

    def closeEvent(self, event) -> None:
        self._portfolio.shutdown()
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
