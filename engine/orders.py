"""Orders and fills — the vocabulary the whole simulator speaks.

A bot never constructs an Order directly; it returns an `OrderRequest` (phase 4)
which risk checks may reject before the broker ever sees it. The broker assigns
the id, stamps the session, and owns the status from then on.

Statuses are a one-way street: `pending` is the only non-terminal state, and
every transition out of it is recorded in the ledger as an appended event rather
than by overwriting the row. An order's history is therefore reconstructable,
which matters when a bot's blotter has to explain why a trade did not happen.

Enums are str-valued because they are persisted; an int would make the ledger
unreadable in a SQL browser, and a reordered enum would silently rewrite
history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

#: Fractional shares are a config flag, default OFF (spec). With it off, a
#: quantity that is not a whole number is a programming error, not a rounding
#: question — the broker rejects it rather than silently truncating.
FRACTIONAL_SHARES = False


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def is_buy(self) -> bool:
        return self is Side.BUY


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class TimeInForce(StrEnum):
    DAY = "day"   # expires if it does not fill on its first eligible session
    GTC = "gtc"   # rests until filled or cancelled


class OrderStatus(StrEnum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    @property
    def is_open(self) -> bool:
        return self is OrderStatus.PENDING


#: Every status except PENDING is final.
TERMINAL_STATUSES = frozenset(s for s in OrderStatus if not s.is_open)


class OrderError(ValueError):
    """An order that is malformed, as opposed to one the broker rejects for
    business reasons (no cash, no shares) — those get a reject_reason."""


@dataclass(slots=True)
class Order:
    """A request to trade, and its current state.

    `created_session` is the session on which the order was *decided*. It can
    never fill on that session: the decision used bar t's close, so execution
    happens on t+1 or later. The broker enforces this rather than trusting
    callers (see sim_broker's eligibility check) — that is the no-look-ahead
    rule made structural.
    """

    account_id: int
    created_session: pd.Timestamp
    ticker: str
    side: Side
    qty: float
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    tif: TimeInForce = TimeInForce.DAY
    reason: str = ""                      # the bot's stated rationale
    id: int | None = None                 # assigned by the ledger on submit
    status: OrderStatus = OrderStatus.PENDING
    reject_reason: str | None = None

    def __post_init__(self) -> None:
        self.side = Side(self.side)
        self.type = OrderType(self.type)
        self.tif = TimeInForce(self.tif)
        self.status = OrderStatus(self.status)
        self.created_session = pd.Timestamp(self.created_session).normalize()
        if not self.ticker:
            raise OrderError("order has no ticker")
        self.validate_qty()
        self.validate_prices()

    def validate_qty(self) -> None:
        if self.qty <= 0:
            raise OrderError(f"{self.ticker}: quantity must be positive, got {self.qty}")
        if not FRACTIONAL_SHARES and float(self.qty) != int(self.qty):
            raise OrderError(
                f"{self.ticker}: fractional quantity {self.qty} but fractional "
                f"shares are disabled (engine.orders.FRACTIONAL_SHARES)"
            )

    def validate_prices(self) -> None:
        if self.type is OrderType.LIMIT:
            if self.limit_price is None:
                raise OrderError(f"{self.ticker}: limit order needs a limit_price")
            if self.limit_price <= 0:
                raise OrderError(f"{self.ticker}: limit_price must be positive")
        if self.type is OrderType.STOP:
            if self.stop_price is None:
                raise OrderError(f"{self.ticker}: stop order needs a stop_price")
            if self.stop_price <= 0:
                raise OrderError(f"{self.ticker}: stop_price must be positive")
        if self.type is OrderType.MARKET and (
            self.limit_price is not None or self.stop_price is not None
        ):
            raise OrderError(f"{self.ticker}: market order must not carry a trigger price")

    @property
    def is_open(self) -> bool:
        return self.status.is_open

    def eligible_on(self, session: pd.Timestamp) -> bool:
        """No look-ahead: an order decided on bar t executes on t+1 at the
        earliest."""
        return pd.Timestamp(session).normalize() > self.created_session

    def apply_split(self, ratio: float) -> None:
        """A split while this order rests: the shares it asks for multiply and
        the price it triggers at divides, so the order keeps meaning the same
        thing in money terms."""
        if ratio <= 0:
            raise OrderError(f"invalid split ratio {ratio}")
        self.qty = self.qty * ratio
        if not FRACTIONAL_SHARES:
            # a 3:2 split on 5 shares is 7.5; a share count must stay whole, and
            # rounding down never creates shares the account cannot cover
            self.qty = float(int(self.qty))
        if self.limit_price is not None:
            self.limit_price = self.limit_price / ratio
        if self.stop_price is not None:
            self.stop_price = self.stop_price / ratio

    def describe(self) -> str:
        bits = [f"{self.side.value} {self.qty:g} {self.ticker}", self.type.value]
        if self.limit_price is not None:
            bits.append(f"@{self.limit_price:.2f}")
        if self.stop_price is not None:
            bits.append(f"stop {self.stop_price:.2f}")
        bits.append(self.tif.value)
        return " ".join(bits)


@dataclass(frozen=True, slots=True)
class Fill:
    """An execution. Immutable: a fill is a historical fact."""

    order_id: int
    account_id: int
    session: pd.Timestamp
    ticker: str
    side: Side
    qty: float
    price: float            # the price actually paid/received, after slippage
    commission: float
    slippage_cost: float    # dollars the slippage model took out of this fill
    id: int | None = None

    @property
    def gross(self) -> float:
        """Share value at the fill price, before commission."""
        return self.qty * self.price

    @property
    def cash_delta(self) -> float:
        """Signed effect on cash: a buy spends gross + commission, a sell
        receives gross - commission."""
        if self.side is Side.BUY:
            return -(self.gross + self.commission)
        return self.gross - self.commission


@dataclass(frozen=True, slots=True)
class Position:
    """A derived holding — never stored, always rebuilt from fills and
    corporate actions (see engine/ledger.rebuild_positions)."""

    ticker: str
    qty: float
    avg_cost: float         # per share, split-adjusted

    @property
    def cost_basis(self) -> float:
        return self.qty * self.avg_cost

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealised(self, price: float) -> float:
        return self.market_value(price) - self.cost_basis


@dataclass(slots=True)
class AccountSummary:
    account_id: int
    name: str
    status: str
    cash: float
    equity: float
    buying_power: float
    positions: dict[str, Position] = field(default_factory=dict)
    open_orders: int = 0
    last_processed_session: pd.Timestamp | None = None
