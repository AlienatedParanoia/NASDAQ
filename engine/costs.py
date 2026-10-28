"""Trading costs: commission and slippage.

================================ READ THIS =================================
EVERY NUMBER BELOW IS A PLACEHOLDER. They are plausible retail-broker figures,
invented for this simulator, and they are NOT your broker's fee schedule.

Costs are the difference between a strategy that looks profitable and one that
is. A 5 bps slippage assumption on a bot that trades daily is roughly 2.5% a
year of drag; set it to zero and mediocre strategies start looking excellent.
So every reported return in this app is only as trustworthy as the numbers in
CostModel.

Replace them with the real schedule before believing any result.
============================================================================

One place to change, two entry points: `commission()` and `slip()`. Nothing
else in the engine is allowed to invent a cost.
"""
from __future__ import annotations

from dataclasses import dataclass

BPS = 1e-4

# PLACEHOLDER defaults — see the warning above.
DEFAULT_PER_ORDER = 0.00     # PLACEHOLDER: flat charge per executed order
DEFAULT_PER_SHARE = 0.005    # PLACEHOLDER: per-share charge
DEFAULT_MINIMUM = 1.00       # PLACEHOLDER: minimum commission per fill
DEFAULT_SLIPPAGE_BPS = 5.0   # PLACEHOLDER: market/stop fills move this far against us


@dataclass(frozen=True, slots=True)
class CostModel:
    """What a fill costs on top of its price.

    commission = max(minimum, per_order + per_share * qty)

    slippage applies to MARKET and STOP fills only. A limit order fills at its
    price or better by definition — that is what the order type buys you — so
    limit fills are exact. (See sim_broker's fill table.)
    """

    per_order: float = DEFAULT_PER_ORDER        # PLACEHOLDER
    per_share: float = DEFAULT_PER_SHARE        # PLACEHOLDER
    minimum: float = DEFAULT_MINIMUM            # PLACEHOLDER
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS  # PLACEHOLDER

    def __post_init__(self) -> None:
        for name in ("per_order", "per_share", "minimum", "slippage_bps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")

    def commission(self, qty: float) -> float:
        """Commission for a fill of `qty` shares. Never negative; zero only if
        the whole model is zero (a deliberately cost-free test setup)."""
        if qty <= 0:
            return 0.0
        return max(self.minimum, self.per_order + self.per_share * qty)

    def slip(self, price: float, is_buy: bool) -> float:
        """Price after slippage: buys pay up, sells receive less."""
        move = price * self.slippage_bps * BPS
        return price + move if is_buy else price - move

    def slippage_cost(self, raw_price: float, filled_price: float, qty: float) -> float:
        """Dollar cost of the slippage on a fill, recorded so the blotter can
        show what the model took out."""
        return abs(filled_price - raw_price) * qty


#: The model used unless a caller supplies its own.
DEFAULT_COSTS = CostModel()

#: For tests that need to isolate fill logic from cost arithmetic.
ZERO_COSTS = CostModel(per_order=0.0, per_share=0.0, minimum=0.0, slippage_bps=0.0)
