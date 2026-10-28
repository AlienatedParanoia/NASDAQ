"""The Strategy interface.

A strategy answers exactly one question: *given everything knowable as of this
session, what would I like to do?* It returns `OrderRequest`s — wishes, not
orders. Something else decides how big (bots/sizing.py), something else decides
whether it is allowed (bots/risk.py), and only then does the broker see an
actual Order.

Keeping those three apart is what makes each testable on its own, and it means
a blocked trade can explain itself: "the strategy wanted this, risk refused it
because X" is a far better answer than a bot that silently did nothing.

A strategy must not:
  * touch the broker, the ledger, or another account;
  * look past `session` (DataView will not let it);
  * keep mutable state that assumes it is called once per session in order —
    replay and forward trading both call on_bar once per session, but a strategy
    that depends on that is fragile. Derive what you need from the data view.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from bots.data_view import DataView, PortfolioView
from engine.orders import OrderType, Side, TimeInForce


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """What a strategy wants. Not yet an order.

    `qty=None` means "I have no view on size — let the sizing policy decide",
    which is the normal case. A strategy that genuinely needs an exact share
    count (closing a position, say) states it.
    """

    ticker: str
    side: Side
    qty: float | None = None
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    tif: TimeInForce = TimeInForce.DAY
    reason: str = ""

    def with_qty(self, qty: float) -> "OrderRequest":
        return OrderRequest(
            ticker=self.ticker, side=self.side, qty=qty, type=self.type,
            limit_price=self.limit_price, stop_price=self.stop_price,
            tif=self.tif, reason=self.reason,
        )

    def describe(self) -> str:
        size = f"{self.qty:g}" if self.qty is not None else "?"
        return f"{self.side.value} {size} {self.ticker} ({self.type.value})"


class Strategy(ABC):
    """Base class for everything that decides what to trade."""

    #: Human-readable name, shown in the UI and written into order reasons.
    name: str = "unnamed strategy"

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)

    @abstractmethod
    def on_bar(
        self,
        session: pd.Timestamp,
        data: DataView,
        portfolio: PortfolioView,
    ) -> list[OrderRequest]:
        """Decide, using only what `data` exposes (nothing after `session`).

        Anything returned executes on the NEXT session at the earliest — the
        broker enforces that, so a strategy should think of its output as
        "orders for tomorrow", not "trades today".
        """

    def warmup_bars(self) -> int:
        """How much history the indicators need before the output means
        anything. Replay skips sessions below this, so a bot does not trade on
        a half-formed moving average."""
        return 0

    def describe(self) -> str:
        if not self.params:
            return self.name
        shown = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name} ({shown})"

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"<{type(self).__name__}: {self.describe()}>"
