"""Risk limits: the last word before an order reaches the broker.

Every OrderRequest passes through here, and every block is recorded with a
reason. That matters more than it sounds: a bot that places no orders is
otherwise indistinguishable from a bot that is broken, and "the strategy wanted
to buy NVDA but the position cap refused it" is a far more useful answer than
silence.

Risk is checked *before* the broker, not instead of it. The broker still
enforces cash and share availability — these limits are about the bot's own
policy (concentration, churn, drawdown), not about solvency.

The drawdown kill switch deserves a note. When equity falls far enough below
its own peak, the bot is killed: open orders are cancelled and no new ones are
placed. By default the positions are **kept**. Liquidating at a drawdown trough
is the classic way to turn a paper loss into a permanent one, and since this is
fake money, watching what the held positions do next is the more informative
outcome. A bot that wants the other behaviour sets `liquidate_on_kill: true`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from bots.data_view import DataView, PortfolioView
from bots.strategy import OrderRequest
from engine.orders import Side

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Per-bot policy. All limits are optional; None means "no limit"."""

    max_position_pct: float | None = 0.20      # of equity, per position
    max_positions: int = 10
    max_orders_per_day: int | None = 5
    max_drawdown_pct: float | None = 0.25      # kill below this off the peak
    liquidate_on_kill: bool = False
    min_order_value: float = 100.0             # skip trades too small to matter

    def describe(self) -> str:
        bits = [f"max {self.max_positions} positions"]
        if self.max_position_pct is not None:
            bits.append(f"{self.max_position_pct:.0%} per position")
        if self.max_orders_per_day is not None:
            bits.append(f"{self.max_orders_per_day} orders/day")
        if self.max_drawdown_pct is not None:
            bits.append(
                f"kill at -{self.max_drawdown_pct:.0%}"
                + (" (liquidate)" if self.liquidate_on_kill else " (hold)")
            )
        return ", ".join(bits)


@dataclass(frozen=True, slots=True)
class Blocked:
    """A request risk refused, and why — surfaced in the UI."""

    request: OrderRequest
    reason: str


@dataclass(slots=True)
class RiskDecision:
    """The outcome of risk-checking one session's requests."""

    allowed: list[OrderRequest] = field(default_factory=list)
    blocked: list[Blocked] = field(default_factory=list)
    kill: bool = False
    kill_reason: str = ""

    def log(self, bot_name: str) -> None:
        for block in self.blocked:
            log.info("%s: blocked %s — %s", bot_name, block.request.describe(), block.reason)
        if self.kill:
            log.warning("%s: KILLED — %s", bot_name, self.kill_reason)


class RiskManager:
    """Applies one bot's limits to one session's requests."""

    def __init__(self, limits: RiskLimits):
        self.limits = limits

    # ------------------------------------------------------------- drawdown

    def drawdown_breach(self, equity_curve: pd.Series, equity_now: float) -> str | None:
        """The kill reason, or None. Measured against the account's own peak,
        including today's equity so a crash on the current session counts."""
        if self.limits.max_drawdown_pct is None:
            return None
        peak = max(float(equity_curve.max()) if len(equity_curve) else 0.0, equity_now)
        if peak <= 0:
            return None
        drawdown = equity_now / peak - 1.0
        if drawdown <= -self.limits.max_drawdown_pct:
            return (
                f"equity ${equity_now:,.2f} is {drawdown:.1%} below its peak of "
                f"${peak:,.2f}, past the {self.limits.max_drawdown_pct:.0%} limit"
            )
        return None

    # --------------------------------------------------------------- filter

    def check(
        self,
        requests: list[OrderRequest],
        data: DataView,
        portfolio: PortfolioView,
        equity_curve: pd.Series | None = None,
        orders_today: int = 0,
    ) -> RiskDecision:
        """Filter a session's requests. Sells are judged more leniently than
        buys: reducing risk should never be blocked by a risk limit."""
        decision = RiskDecision()

        kill_reason = self.drawdown_breach(
            equity_curve if equity_curve is not None else pd.Series(dtype=float),
            portfolio.equity,
        )
        if kill_reason:
            decision.kill = True
            decision.kill_reason = kill_reason
            for request in requests:
                decision.blocked.append(Blocked(request, "bot killed by drawdown limit"))
            return decision

        # a position opened this session counts against the cap for the rest of it
        projected_positions = portfolio.position_count
        allowed_today = self.limits.max_orders_per_day
        placed = orders_today

        for request in requests:
            if allowed_today is not None and placed >= allowed_today:
                decision.blocked.append(
                    Blocked(request, f"daily order cap of {allowed_today} reached")
                )
                continue

            reason = self._check_one(request, data, portfolio, projected_positions)
            if reason is not None:
                decision.blocked.append(Blocked(request, reason))
                continue

            if request.side is Side.BUY and not portfolio.holds(request.ticker):
                projected_positions += 1
            decision.allowed.append(request)
            placed += 1

        return decision

    def _check_one(
        self,
        request: OrderRequest,
        data: DataView,
        portfolio: PortfolioView,
        projected_positions: int,
    ) -> str | None:
        price = data.close(request.ticker)
        if price is None or price <= 0:
            return f"no price for {request.ticker}"

        if request.side is Side.SELL:
            # selling reduces exposure; the broker still checks the shares exist
            if not portfolio.holds(request.ticker):
                return f"no {request.ticker} position to sell"
            return None

        if request.qty is None or request.qty <= 0:
            return "sizing produced no affordable quantity"

        value = request.qty * price
        if value < self.limits.min_order_value:
            return (
                f"order value ${value:,.2f} is below the ${self.limits.min_order_value:,.0f} "
                f"minimum — too small to be worth the commission"
            )

        if not portfolio.holds(request.ticker) and projected_positions >= self.limits.max_positions:
            return (
                f"already at the {self.limits.max_positions}-position limit; "
                f"{request.ticker} would make {projected_positions + 1}"
            )

        if self.limits.max_position_pct is not None and portfolio.equity > 0:
            existing = portfolio.shares(request.ticker) * price
            pct = (existing + value) / portfolio.equity
            if pct > self.limits.max_position_pct:
                return (
                    f"{request.ticker} would be {pct:.1%} of equity, over the "
                    f"{self.limits.max_position_pct:.0%} per-position limit"
                )

        return None


def build_risk(spec: dict | None) -> RiskLimits:
    """Construct limits from a config block, naming any bad field."""
    if not spec:
        return RiskLimits()
    known = {f for f in RiskLimits.__slots__}
    unknown = set(spec) - known
    if unknown:
        raise ValueError(
            f"unknown risk setting(s) {sorted(unknown)}; expected from {sorted(known)}"
        )
    try:
        return RiskLimits(**spec)
    except TypeError as e:
        raise ValueError(f"bad risk configuration: {e}") from e
