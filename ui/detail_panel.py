"""Detail panel: pyqtgraph price chart (close + bucket overlays + entry/exit
markers from the backtest) above the reasoning text."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QLabel, QTextBrowser, QVBoxLayout, QWidget

from signals.signal_generator import ScanRow

pg.setConfigOptions(antialias=True, background="w", foreground="k")

OVERLAY_COLORS = ("#1f77b4", "#ff7f0e", "#9467bd", "#8c564b")
PLACEHOLDER = "Select a row to see the chart and reasoning."


def _epoch_seconds(values) -> np.ndarray:
    """tz-naive pandas datetimes -> POSIX seconds for pyqtgraph's DateAxisItem."""
    return values.astype("int64").to_numpy() / 1e9


class DetailPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._title = QLabel(PLACEHOLDER)
        self._title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._plot = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._legend = self._plot.addLegend(offset=(10, 10))
        self._text = QTextBrowser()

        layout = QVBoxLayout(self)
        layout.addWidget(self._title)
        layout.addWidget(self._plot, 3)
        layout.addWidget(self._text, 1)

    def clear(self) -> None:
        self._plot.clear()
        self._legend.clear()
        self._text.clear()
        self._title.setText(PLACEHOLDER)

    def show_row(self, row: ScanRow) -> None:
        self._plot.clear()
        self._legend.clear()

        title = f"{row.ticker} — {row.bucket}"
        if row.signal is not None:
            title += f" — {row.signal}"
        self._title.setText(title)
        self._text.setPlainText(row.reasoning)

        if row.prices is None or row.prices.empty:
            return

        x = _epoch_seconds(row.prices.index)
        self._plot.plot(
            x,
            row.prices["close"].to_numpy(),
            pen=pg.mkPen("#000000", width=2),
            name="Close",
            connect="finite",
        )
        for (name, series), color in zip(row.overlay.items(), OVERLAY_COLORS):
            aligned = series.reindex(row.prices.index)
            self._plot.plot(
                x,
                aligned.to_numpy(dtype=float),
                pen=pg.mkPen(color, width=1),
                name=name,
                connect="finite",
            )

        entries_x, entries_y, exits_x, exits_y = [], [], [], []
        for trade in row.trades:
            entries_x.append(trade.entry_date.value / 1e9)
            entries_y.append(trade.entry_price)
            if trade.exit_date is not None:
                exits_x.append(trade.exit_date.value / 1e9)
                exits_y.append(trade.exit_price)
        if entries_x:
            item = pg.ScatterPlotItem(
                x=entries_x, y=entries_y, symbol="t1", size=13,
                brush=pg.mkBrush("#1e8e3e"), pen=pg.mkPen("k"),
            )
            self._plot.addItem(item)
            self._legend.addItem(item, "Entry")
        if exits_x:
            item = pg.ScatterPlotItem(
                x=exits_x, y=exits_y, symbol="t", size=13,
                brush=pg.mkBrush("#d93025"), pen=pg.mkPen("k"),
            )
            self._plot.addItem(item)
            self._legend.addItem(item, "Exit")
