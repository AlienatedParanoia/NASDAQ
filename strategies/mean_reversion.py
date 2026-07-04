"""Mean-reversion strategy: oversold RSI plus a close below the lower
Bollinger band enters; reversion to the middle band, RSI recovery, or a
time stop exits. Evaluated as a stateful bar-by-bar loop so entries and
exits are deterministic."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies import indicators as ind

RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
RSI_ENTRY = 30.0
RSI_EXIT = 55.0
MAX_HOLD_BARS = 15
WARMUP = 20


class MeanReversionStrategy:
    name = "RSI(14) + Bollinger bounce"

    def _series(self, prices: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        close = prices["close"]
        return ind.rsi_wilder(close, RSI_PERIOD), ind.bollinger(close, BB_PERIOD, BB_STD)

    def _raw(self, prices: pd.DataFrame) -> pd.Series:
        """1 while the strategy wants to be long, decided at each bar's close."""
        close = prices["close"].to_numpy()
        rsi, bb = self._series(prices)
        rsi_v = rsi.to_numpy()
        lower = bb["lower"].to_numpy()
        mid = bb["mid"].to_numpy()

        raw = np.zeros(len(close))
        in_trade = False
        held = 0
        for i in range(len(close)):
            if np.isnan(rsi_v[i]) or np.isnan(lower[i]):
                in_trade = False
                continue
            if not in_trade:
                if rsi_v[i] < RSI_ENTRY and close[i] < lower[i]:
                    in_trade = True
                    held = 0
            else:
                held += 1
                if close[i] >= mid[i] or rsi_v[i] >= RSI_EXIT or held >= MAX_HOLD_BARS:
                    in_trade = False
            raw[i] = 1.0 if in_trade else 0.0
        return pd.Series(raw, index=prices.index)

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        """0/1 positions, already shifted one bar; NaN during warmup."""
        pos = self._raw(prices).shift(1)
        pos.iloc[: min(WARMUP, len(pos))] = np.nan
        return pos

    def latest_state(self, prices: pd.DataFrame) -> dict:
        rsi, bb = self._series(prices)
        close = float(prices["close"].iloc[-1])
        rsi_now = float(rsi.iloc[-1])
        lower = float(bb["lower"].iloc[-1])
        mid = float(bb["mid"].iloc[-1])
        raw = self._raw(prices)
        return {
            "rsi": rsi_now,
            "close": close,
            "lower": lower,
            "mid": mid,
            "setup": rsi_now < RSI_ENTRY and close < lower,
            "in_trade": bool(raw.iloc[-1] == 1.0),
        }

    def overlays(self, prices: pd.DataFrame) -> dict[str, pd.Series]:
        _, bb = self._series(prices)
        return {"BB upper": bb["upper"], "BB mid": bb["mid"], "BB lower": bb["lower"]}
