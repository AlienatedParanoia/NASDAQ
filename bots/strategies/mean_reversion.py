"""Mean reversion: buy oversold, sell on recovery.

A port of strategies/mean_reversion.py. Entry needs RSI below its threshold AND
a close under the lower Bollinger band; exit on a return to the middle band, an
RSI recovery, or a time stop.

The original ran a stateful bar-by-bar loop over the whole history to know how
long a trade had been open. Here that state lives where it belongs — in the
portfolio — so the time stop is measured from the actual fill, not from a
re-simulation. The bot therefore exits on the real holding period rather than a
reconstructed one.
"""
from __future__ import annotations

import pandas as pd

from bots.data_view import DataView, PortfolioView
from bots.strategy import OrderRequest, Strategy
from engine.orders import Side
from strategies import indicators as ind


class MeanReversionStrategy(Strategy):
    name = "RSI + Bollinger bounce"

    def __init__(
        self,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_entry: float = 30.0,
        rsi_exit: float = 55.0,
        max_hold_bars: int = 15,
    ) -> None:
        super().__init__(
            rsi_period=rsi_period, bb_period=bb_period, bb_std=bb_std,
            rsi_entry=rsi_entry, rsi_exit=rsi_exit, max_hold_bars=max_hold_bars,
        )
        self.rsi_period = int(rsi_period)
        self.bb_period = int(bb_period)
        self.bb_std = float(bb_std)
        self.rsi_entry = float(rsi_entry)
        self.rsi_exit = float(rsi_exit)
        self.max_hold_bars = int(max_hold_bars)
        self.name = f"RSI({rsi_period}) + Bollinger({bb_period}) bounce"

    def warmup_bars(self) -> int:
        return max(self.rsi_period, self.bb_period) + 1

    def _state(self, closes: pd.Series) -> dict | None:
        if len(closes) < self.warmup_bars():
            return None
        rsi = ind.rsi_wilder(closes, self.rsi_period)
        bands = ind.bollinger(closes, self.bb_period, self.bb_std)
        if rsi.empty or bands.empty:
            return None
        latest = {
            "rsi": float(rsi.iloc[-1]),
            "close": float(closes.iloc[-1]),
            "lower": float(bands["lower"].iloc[-1]),
            "mid": float(bands["mid"].iloc[-1]),
        }
        if any(pd.isna(v) for v in latest.values()):
            return None
        return latest

    def on_bar(
        self, session: pd.Timestamp, data: DataView, portfolio: PortfolioView
    ) -> list[OrderRequest]:
        requests: list[OrderRequest] = []

        for ticker in data.tickers:
            if portfolio.has_open_order(ticker):
                continue
            closes = data.adjusted(ticker)
            state = self._state(closes)
            if state is None:
                continue

            holding = portfolio.holds(ticker)

            if not holding:
                oversold = state["rsi"] < self.rsi_entry
                below_band = state["close"] < state["lower"]
                if oversold and below_band:
                    requests.append(
                        OrderRequest(
                            ticker=ticker,
                            side=Side.BUY,
                            reason=(
                                f"RSI {state['rsi']:.1f} < {self.rsi_entry:.0f} and "
                                f"close {state['close']:.2f} < lower band "
                                f"{state['lower']:.2f} — oversold"
                            ),
                        )
                    )
                continue

            exit_reason = None
            if state["close"] >= state["mid"]:
                exit_reason = (
                    f"close {state['close']:.2f} back at the mid band "
                    f"{state['mid']:.2f}"
                )
            elif state["rsi"] >= self.rsi_exit:
                exit_reason = f"RSI {state['rsi']:.1f} recovered past {self.rsi_exit:.0f}"
            else:
                held_bars = portfolio.bars_held(ticker, data.history(ticker))
                if held_bars is not None and held_bars >= self.max_hold_bars:
                    exit_reason = (
                        f"time stop: held {held_bars} bars, limit {self.max_hold_bars}"
                    )

            if exit_reason:
                requests.append(
                    OrderRequest(
                        ticker=ticker, side=Side.SELL, qty=portfolio.shares(ticker),
                        reason=exit_reason,
                    )
                )

        return requests
