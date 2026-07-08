"""Daily Report tab: an on-demand digest of the day's top risk-adjusted picks.

Clicking Generate runs the same full scan the Scanner tab does (on its own
QThread with its own DataFetcher, same cancellable lifecycle as ScanWorker),
then ranks the Buy-rated rows via signals.daily_report and shows the top 5:
a dated text report, a clickable picks table, and the shared detail chart.

The scan is read-only over the daily cache, so repeat clicks are cheap.
"""
from __future__ import annotations

import logging
import threading
from datetime import date

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

import paths
from data.fetcher import DataFetcher
from data.universe import get_universe
from signals.daily_report import (
    DEFAULT_TOP_N,
    Pick,
    buy_counts,
    daily_top_picks,
    render_report,
)
from signals.signal_generator import ScanRow, scan_universe
from ui.detail_panel import DetailPanel

log = logging.getLogger(__name__)

COLS = ("#", "Ticker", "Bucket", "Signal", "Ret/yr", "Sharpe", "Max DD", "Score")

BUY_COLOR = QColor("#1e8e3e")


class DailyReportWorker(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(list)  # list[ScanRow]
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
                    should_stop=self._stop.is_set,
                    window_years=self._window_years,
                )
            finally:
                fetcher.close()
            self.finished.emit(rows)
        except Exception as e:  # pragma: no cover - surfaced to the UI
            log.exception("daily report worker crashed")
            self.failed.emit(str(e))


class DailyReportPanel(QWidget):
    def __init__(self, window_years_provider) -> None:
        """window_years_provider: zero-arg callable returning the current
        backtest window (shared with the scanner's toolbar combo)."""
        super().__init__()
        self._window_years = window_years_provider
        self._picks: list[Pick] = []
        self._thread: QThread | None = None
        self._worker: DailyReportWorker | None = None

        self._generate_btn = QPushButton("Generate report")
        self._generate_btn.clicked.connect(self._on_generate)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._status = QLabel("Click Generate to rank today's top opportunities.")
        self._status.setWordWrap(True)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self._generate_btn)
        toolbar.addWidget(self._progress, 1)

        self._report_view = QTextBrowser()

        self._table = QTableWidget(0, len(COLS))
        self._table.setHorizontalHeaderLabels(COLS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.itemSelectionChanged.connect(self._on_selection)

        left_layout = QVBoxLayout()
        left_layout.addLayout(toolbar)
        left_layout.addWidget(self._report_view, 3)
        left_layout.addWidget(self._table, 2)
        left_layout.addWidget(self._status)
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

    # ------------------------------------------------------------ lifecycle

    def window_changed(self) -> None:
        """Backtest window switched: any shown report is for the old window."""
        if self._picks and self._thread is None:
            self._status.setText(
                "Backtest window changed — click Generate to refresh the report."
            )

    def _on_generate(self) -> None:
        if self._thread is not None:
            return
        universe = get_universe()
        self._generate_btn.setEnabled(False)
        self._progress.setRange(0, len(universe))
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._status.setText("Scanning…")
        self._worker = DailyReportWorker(universe, self._window_years())
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _teardown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None
        self._generate_btn.setEnabled(True)
        self._progress.setVisible(False)

    def shutdown(self) -> None:
        """Called from MainWindow.closeEvent."""
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            if not self._thread.wait(10_000):
                self._thread.wait()

    # ------------------------------------------------------------ results

    def _on_progress(self, i: int, n: int, ticker: str) -> None:
        self._progress.setValue(i)
        self._status.setText(f"Scanning {ticker} ({i}/{n})…")

    def _on_finished(self, rows: list) -> None:
        n_buys, n_excluded = buy_counts(rows)
        self._picks = daily_top_picks(rows, DEFAULT_TOP_N)
        report = render_report(
            self._picks,
            as_of=date.today().isoformat(),
            window_years=self._window_years(),
            n_scanned=len(rows),
            n_buys=n_buys,
            n_excluded=n_excluded,
        )
        self._report_view.setPlainText(report)
        self._render_table()
        self._detail.clear()
        status = (
            f"{len(self._picks)} pick(s) from {n_buys - n_excluded} eligible Buys "
            f"of {len(rows)} scanned."
        )
        if n_excluded:
            status += f" {n_excluded} excluded (implausible Sharpe)."
        self._status.setText(status)
        self._teardown()

    def _on_failed(self, err: str) -> None:
        self._status.setText(f"Report failed: {err}")
        self._teardown()

    def _render_table(self) -> None:
        self._table.setRowCount(len(self._picks))
        for i, p in enumerate(self._picks):
            r: ScanRow = p.row
            self._set(i, 0, str(p.rank))
            self._set(i, 1, r.ticker)
            self._set(i, 2, str(r.bucket))
            sig = QTableWidgetItem(str(r.signal) if r.signal else "—")
            sig.setForeground(BUY_COLOR)
            self._table.setItem(i, 3, sig)
            self._set(i, 4, f"{r.cagr:+.1%}" if r.cagr is not None else "—")
            self._set(i, 5, f"{r.sharpe:.2f}" if r.sharpe is not None else "—")
            self._set(i, 6, f"{r.max_dd:.1%}" if r.max_dd is not None else "—")
            self._set(i, 7, f"{p.score:.2f}")

    def _set(self, row: int, col: int, text: str) -> None:
        self._table.setItem(row, col, QTableWidgetItem(text))

    def _on_selection(self) -> None:
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            return
        i = indexes[0].row()
        if 0 <= i < len(self._picks):
            self._detail.show_row(self._picks[i].row)
