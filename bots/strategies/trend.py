"""Trend following: long while EMA(20) is above EMA(50).

A port of strategies/trend.py to the Strategy interface. The logic is identical;
what changes is the shape of the output — the original emitted a 0/1 position
series over the whole history, which only makes sense when you already know the
whole history. Here the same rule is evaluated one session at a time against
data that stops at that session, which is the only form that can run forward.

Indicators are computed on **adjusted** closes: a 10:1 split would otherwise
read as a 90% crash and flip the trend for weeks.
"""
from __future__ import annotations

import pandas as pd

from bots.data_view import DataView, PortfolioView
from bots.strategy import OrderRequest, Strategy
from engine.orders import Side
from strategies import indicators as ind


class TrendStrategy(Strategy):
    """Buy when the fast EMA crosses above the slow one, sell when it crosses
    back. Long-only, one position per ticker."""

    name = "EMA crossover"

    def __init__(self, fast: int = 20, slow: int = 50, universe_cap: int = 0) -> None:
        if fast >= slow:
            raise ValueError(f"fast EMA ({fast}) must be shorter than slow ({slow})")
        super().__init__(fast=fast, slow=slow)
        self.fast = int(fast)
        self.slow = int(slow)
        self.name = f"EMA {fast}/{slow} crossover"

    def warmup_bars(self) -> int:
        # an EMA is defined from bar 1 but is not meaningful until it has seen
        # roughly its own span; require the slow span before trading
        return self.slow

    def _in_uptrend(self, closes: pd.Series) -> bool | None:
        """None when there is not enough history to say."""
        if len(closes) < self.slow:
            return None
        fast = ind.ema(closes, self.fast)
        slow = ind.ema(closes, self.slow)
        if fast.empty or slow.empty:
            return None
        latest_fast, latest_slow = float(fast.iloc[-1]), float(slow.iloc[-1])
        if pd.isna(latest_fast) or pd.isna(latest_slow):
            return None
        return latest_fast > latest_slow

    def on_bar(
        self, session: pd.Timestamp, data: DataView, portfolio: PortfolioView
    ) -> list[OrderRequest]:
        requests: list[OrderRequest] = []

        for ticker in data.tickers:
            if portfolio.has_open_order(ticker):
                continue  # already trying to trade this one
            closes = data.adjusted(ticker)
            uptrend = self._in_uptrend(closes)
            if uptrend is None:
                continue

            holding = portfolio.holds(ticker)
            if uptrend and not holding:
                fast = float(ind.ema(closes, self.fast).iloc[-1])
                slow = float(ind.ema(closes, self.slow).iloc[-1])
                requests.append(
                    OrderRequest(
                        ticker=ticker,
                        side=Side.BUY,
                        reason=(
                            f"EMA{self.fast} {fast:.2f} > EMA{self.slow} {slow:.2f} "
                            f"— uptrend"
                        ),
                    )
                )
            elif not uptrend and holding:
                fast = float(ind.ema(closes, self.fast).iloc[-1])
                slow = float(ind.ema(closes, self.slow).iloc[-1])
                requests.append(
                    OrderRequest(
                        ticker=ticker,
                        side=Side.SELL,
                        qty=portfolio.shares(ticker),
                        reason=(
                            f"EMA{self.fast} {fast:.2f} < EMA{self.slow} {slow:.2f} "
                            f"— trend over"
                        ),
                    )
                )

        return requests
