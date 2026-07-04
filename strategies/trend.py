"""Trend-following strategy: long while EMA(20) is above EMA(50)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies import indicators as ind

EMA_FAST = 20
EMA_SLOW = 50
WARMUP = 50


class TrendStrategy:
    name = "EMA 20/50 crossover"

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        """0/1 positions, already shifted one bar; NaN during warmup."""
        close = prices["close"]
        raw = (ind.ema(close, EMA_FAST) > ind.ema(close, EMA_SLOW)).astype(float)
        pos = raw.shift(1)
        pos.iloc[: min(WARMUP, len(pos))] = np.nan
        return pos

    def latest_state(self, prices: pd.DataFrame) -> dict:
        close = prices["close"]
        fast = float(ind.ema(close, EMA_FAST).iloc[-1])
        slow = float(ind.ema(close, EMA_SLOW).iloc[-1])
        return {"ema_fast": fast, "ema_slow": slow, "in_uptrend": fast > slow}

    def overlays(self, prices: pd.DataFrame) -> dict[str, pd.Series]:
        close = prices["close"]
        return {
            f"EMA{EMA_FAST}": ind.ema(close, EMA_FAST),
            f"EMA{EMA_SLOW}": ind.ema(close, EMA_SLOW),
        }
