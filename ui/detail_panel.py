"""Detail panel: a pyqtgraph close-price chart for one ticker.

Deliberately minimal after the signal system was removed — it takes a ticker
and its raw price frame and draws the close line. Phase 6 grows this into the
brokerage-style chart (candlesticks, ledger fill markers, indicator overlays).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

pg.setConfigOptions(antialias=True, background="w", foreground="k")

PLACEHOLDER = "Select a row to see its chart."


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

        layout = QVBoxLayout(self)
        layout.addWidget(self._title)
        layout.addWidget(self._plot, 1)

    def clear(self) -> None:
        self._plot.clear()
        self._legend.clear()
        self._title.setText(PLACEHOLDER)

    def show_prices(self, ticker: str, prices: pd.DataFrame | None) -> None:
        """Named show_prices, not show: QWidget.show already means something."""
        self._plot.clear()
        self._legend.clear()
        self._title.setText(ticker)

        if prices is None or prices.empty:
            return

        self._plot.plot(
            _epoch_seconds(prices.index),
            prices["close"].to_numpy(dtype=float),
            pen=pg.mkPen("#000000", width=2),
            name="Close",
            connect="finite",
        )
