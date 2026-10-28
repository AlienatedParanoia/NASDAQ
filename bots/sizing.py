"""Position sizing: turning "I want to own AAPL" into "buy 37 shares".

An invariant every policy here obeys: **a sizing policy must never author an
order the broker will reject.** It rounds down to whole shares and caps itself
at available buying power, so a zero-size result means "cannot afford any",
which the caller drops. If sizing could overreach, the blotter would fill with
rejections that look like decisions but are really arithmetic mistakes.

Sizes are computed from **as-traded** prices, because that is what the shares
will actually cost. Volatility is measured on **adjusted** closes, because a
split would otherwise read as a 90% one-day move and make a calm stock look
wild.
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from bots.data_view import DataView, PortfolioView
from strategies import indicators as ind

log = logging.getLogger(__name__)

#: Leave a sliver of cash unspent so an order sized now still fits after the
#: open moves a little. Mirrors the broker's own buying-power buffer.
CASH_SAFETY = 0.99


def _whole_shares(dollars: float, price: float) -> int:
    """Round DOWN, always. Rounding up invents money the account does not have."""
    if price <= 0 or dollars <= 0:
        return 0
    return int(math.floor(dollars / price))


class SizingPolicy(ABC):
    """How many shares to buy, given a budget and a price."""

    name: str = "sizing"

    @abstractmethod
    def target_dollars(
        self, ticker: str, data: DataView, portfolio: PortfolioView, max_positions: int
    ) -> float:
        """The dollar value this policy wants in this position."""

    def shares_for(
        self,
        ticker: str,
        data: DataView,
        portfolio: PortfolioView,
        max_positions: int,
    ) -> int:
        """Whole shares, affordable at today's close. Zero means "cannot"."""
        price = data.close(ticker)
        if price is None or price <= 0:
            log.debug("%s: no price, cannot size", ticker)
            return 0
        budget = self.target_dollars(ticker, data, portfolio, max_positions)
        affordable = portfolio.buying_power * CASH_SAFETY
        return _whole_shares(min(budget, affordable), price)

    def describe(self) -> str:
        return self.name


@dataclass(slots=True)
class FixedDollar(SizingPolicy):
    """Always aim for the same dollar amount per position."""

    amount: float
    name: str = "fixed dollar"

    def target_dollars(self, ticker, data, portfolio, max_positions) -> float:
        return self.amount

    def describe(self) -> str:
        return f"fixed ${self.amount:,.0f} per position"


@dataclass(slots=True)
class EqualWeight(SizingPolicy):
    """Split equity evenly across the maximum number of positions allowed.

    Deliberately divided by `max_positions` rather than by however many signals
    fired today. Sizing then does not swing with signal count: the first buy is
    the same size as the tenth, and a lone signal cannot claim the whole
    account only for the per-position risk cap to claw it back. The cost is
    idle cash until the bot is fully invested, which is honest.
    """

    name: str = "equal weight"

    def target_dollars(self, ticker, data, portfolio, max_positions) -> float:
        slots = max(int(max_positions), 1)
        return portfolio.equity / slots

    def describe(self) -> str:
        return "equal weight (equity / max positions)"


@dataclass(slots=True)
class VolatilityTarget(SizingPolicy):
    """Size so each position contributes roughly the same risk.

    A position whose annualised volatility is twice the target gets half the
    equal-weight allocation. Capped so a very quiet stock cannot be levered up
    into an oversized bet, and falls back to equal weight when there is not
    enough history to measure volatility.
    """

    target_annual_vol: float = 0.20
    window: int = 60
    max_multiple: float = 1.5
    name: str = "volatility target"

    def target_dollars(self, ticker, data, portfolio, max_positions) -> float:
        slots = max(int(max_positions), 1)
        base = portfolio.equity / slots

        closes = data.adjusted(ticker)
        if len(closes) < self.window + 1:
            return base
        vol_series = ind.annualized_vol(closes, window=self.window).dropna()
        if vol_series.empty:
            return base
        vol = float(vol_series.iloc[-1])
        if not math.isfinite(vol) or vol <= 0:
            return base

        multiple = min(self.target_annual_vol / vol, self.max_multiple)
        return base * multiple

    def describe(self) -> str:
        return (
            f"volatility target {self.target_annual_vol:.0%}/yr "
            f"({self.window}-bar window, max {self.max_multiple:g}x)"
        )


#: Names usable in bots.yaml -> constructor
SIZING_POLICIES: dict[str, type[SizingPolicy]] = {
    "fixed_dollar": FixedDollar,
    "equal_weight": EqualWeight,
    "volatility_target": VolatilityTarget,
}


def build_sizing(spec: dict | str | None) -> SizingPolicy:
    """Construct a sizing policy from a config block.

    Accepts either a bare name ("equal_weight") or a mapping with a `type` key
    plus that policy's parameters.
    """
    if spec is None:
        return EqualWeight()
    if isinstance(spec, str):
        spec = {"type": spec}
    params = dict(spec)
    kind = params.pop("type", None)
    if kind not in SIZING_POLICIES:
        raise ValueError(
            f"unknown sizing policy {kind!r}; expected one of {sorted(SIZING_POLICIES)}"
        )
    try:
        return SIZING_POLICIES[kind](**params)
    except TypeError as e:
        raise ValueError(f"bad parameters for sizing policy {kind!r}: {e}") from e
