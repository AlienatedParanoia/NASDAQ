"""The simulated broker.

Fake money, real prices. This is the only Broker implementation in the project
and the only one planned — there is no real-money trading code here.

A session is processed in exactly three steps, in this order, and the order is
not incidental:

  1. **Corporate actions with ex-date t.** Splits re-term both holdings and any
     resting orders; dividends credit cash. This happens before matching so an
     order queued the day before a split executes at post-split terms rather
     than asking for ten times the shares it meant.
  2. **Match pending orders against bar t.** Sells before buys (a sale funds a
     later purchase, as it would in a real session), each side oldest-first, so
     a replay is deterministic.
  3. **Mark to market at close t and snapshot.**

The no-look-ahead rule is structural, not a convention: an order carries the
session it was decided on, and `Order.eligible_on` refuses to match it on that
same session. A strategy that sees bar t's close can only ever trade on t+1.

Because phase 1 stores **as-traded** prices, a split genuinely drops the price
by its ratio on the ex-date. That means the same split handling is correct for
both historical replay and forward trading — there is no "backtest mode" here
to get subtly wrong.

Documented simplifications (see README):
  * Dividends credit cash on the **ex-date**, not the pay date. Real settlement
    is weeks later, so this account can re-invest a dividend sooner than reality
    would allow.
  * Fills are **all-or-nothing** at a single price: no partial fills, no queue
    position, no volume cap. A bot can "buy" more than the day's volume.
  * Cash settles instantly (no T+1), there is no margin, no shorting, no
    options.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

import pandas as pd

from engine.broker import Broker
from engine.costs import DEFAULT_COSTS, CostModel
from engine.ledger import (
    DIVIDEND,
    EPS,
    SPLIT,
    Ledger,
)
from engine.orders import (
    AccountSummary,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    TimeInForce,
)

log = logging.getLogger(__name__)


class CorporateActionLike(Protocol):
    """What the broker needs from a corporate action.

    Structural, not inherited: data.fetcher.CorporateAction already satisfies
    it, so the engine never imports the data layer and stays testable with
    plain stubs.
    """

    ticker: str
    ex_date: pd.Timestamp
    type: str
    value: float

#: Buying power is checked against the last close, which the open can gap above.
#: 1% covers ordinary overnight drift; a bigger gap still gets caught at fill
#: time, which the spec requires anyway. See the phase 3 decisions.
BUYING_POWER_BUFFER = 0.01


@dataclass(frozen=True, slots=True)
class Bar:
    """One session's OHLCV for one ticker — as-traded, never adjusted."""

    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @classmethod
    def from_row(cls, row) -> "Bar":
        return cls(
            open=float(row["open"]), high=float(row["high"]), low=float(row["low"]),
            close=float(row["close"]), volume=float(row.get("volume", 0.0) or 0.0),
        )


@dataclass(frozen=True, slots=True)
class SessionResult:
    """What one processed session did — returned so callers can log/report
    without re-querying the ledger."""

    session: pd.Timestamp
    fills: tuple[Fill, ...] = ()
    expired: tuple[int, ...] = ()
    rejected: tuple[int, ...] = ()
    dividends: float = 0.0
    splits: tuple[str, ...] = ()
    cash: float = 0.0
    equity: float = 0.0


class SimBroker(Broker):
    """A fake-money brokerage account backed by the append-only ledger."""

    def __init__(
        self,
        ledger: Ledger,
        account_id: int,
        costs: CostModel = DEFAULT_COSTS,
        buying_power_buffer: float = BUYING_POWER_BUFFER,
    ):
        self.ledger = ledger
        self.account_id = account_id
        self.costs = costs
        self.buying_power_buffer = buying_power_buffer
        self._last_close: dict[str, float] = {}

    # ------------------------------------------------------------------ state

    def cash(self) -> float:
        return self.ledger.rebuild_cash(self.account_id)

    def positions(self) -> dict[str, Position]:
        return self.ledger.rebuild_positions(self.account_id)

    def open_orders(self) -> list[Order]:
        return self.ledger.open_orders(self.account_id)

    def equity(self, mark_prices: dict[str, float]) -> float:
        total = self.cash()
        for ticker, position in self.positions().items():
            price = mark_prices.get(ticker, self._last_close.get(ticker))
            if price is None:
                # no mark available: value at cost rather than dropping the
                # position from equity, which would look like a loss
                price = position.avg_cost
            total += position.qty * price
        return total

    def buying_power(self) -> float:
        """Cash minus what resting buy orders have already claimed."""
        return max(self.cash() - self._reserved_cash(), 0.0)

    def account_summary(
        self, mark_prices: dict[str, float] | None = None
    ) -> AccountSummary:
        account = self.ledger.get_account(self.account_id)
        marks = mark_prices or dict(self._last_close)
        return AccountSummary(
            account_id=self.account_id,
            name=account.name,
            status=account.status,
            cash=self.cash(),
            equity=self.equity(marks),
            buying_power=self.buying_power(),
            positions=self.positions(),
            open_orders=len(self.open_orders()),
            last_processed_session=self.ledger.last_processed_session(self.account_id),
        )

    def set_mark(self, ticker: str, price: float) -> None:
        """Remember a close so equity can be struck without re-supplying every
        price (the UI and the buying-power check both need this)."""
        self._last_close[ticker] = float(price)

    # --------------------------------------------------------------- reserves

    def _reserved_cash(self, exclude_order_id: int | None = None) -> float:
        """Cash already claimed by resting buy orders.

        Without this, two buys that each fit in cash individually could both be
        accepted and the second would fail at fill time — a surprise the bot
        could not have predicted.
        """
        total = 0.0
        for order in self.open_orders():
            if order.side is not Side.BUY or order.id == exclude_order_id:
                continue
            price = self._estimate_price(order)
            total += order.qty * price * (1.0 + self.buying_power_buffer)
        return total

    def _reserved_shares(self, ticker: str, exclude_order_id: int | None = None) -> float:
        """Shares already claimed by resting sell orders — the mirror image of
        reserved cash, so two sells cannot between them oversell a position."""
        return sum(
            o.qty
            for o in self.open_orders()
            if o.side is Side.SELL and o.ticker == ticker and o.id != exclude_order_id
        )

    def _estimate_price(self, order: Order) -> float:
        """What this order is likely to cost per share, for the pre-trade
        check only. Never used for a fill."""
        if order.type is OrderType.LIMIT and order.limit_price:
            return order.limit_price
        if order.type is OrderType.STOP and order.stop_price:
            return order.stop_price
        return self._last_close.get(order.ticker, 0.0)

    # ----------------------------------------------------------------- submit

    def submit(self, order: Order) -> Order:
        """Accept or reject. A rejection is a returned order with a reason, not
        an exception — at a real broker a rejection is a normal outcome."""
        order.account_id = self.account_id
        rejection = self._pre_trade_check(order)
        if rejection is not None:
            order.status = OrderStatus.REJECTED
            order.reject_reason = rejection
            self.ledger.record_order(order)
            log.info("order rejected: %s — %s", order.describe(), rejection)
            return order

        order.status = OrderStatus.PENDING
        self.ledger.record_order(order)
        return order

    def _pre_trade_check(self, order: Order) -> str | None:
        """The reason to refuse this order, or None. Every branch returns a
        sentence a human can act on."""
        account = self.ledger.get_account(self.account_id)
        if account.status == "killed":
            return f"account {account.name} is killed and cannot place orders"

        if order.side is Side.SELL:
            held = self.positions().get(order.ticker)
            have = held.qty if held else 0.0
            reserved = self._reserved_shares(order.ticker)
            available = have - reserved
            if order.qty > available + EPS:
                if reserved > 0:
                    return (
                        f"sell {order.qty:g} {order.ticker} exceeds available shares: "
                        f"hold {have:g}, {reserved:g} already reserved by resting "
                        f"sell orders, {available:g} free (no shorting)"
                    )
                return (
                    f"sell {order.qty:g} {order.ticker} exceeds the {have:g} shares "
                    f"held (no shorting)"
                )
            return None

        price = self._estimate_price(order)
        if price <= 0:
            return (
                f"no reference price for {order.ticker}; cannot size a buy "
                f"(has the ticker ever been marked?)"
            )
        estimated = order.qty * price * (1.0 + self.buying_power_buffer)
        available = self.cash() - self._reserved_cash()
        if estimated > available + EPS:
            return (
                f"buy {order.qty:g} {order.ticker} needs about ${estimated:,.2f} "
                f"(incl. {self.buying_power_buffer:.0%} buffer) but only "
                f"${available:,.2f} is available"
            )
        return None

    def cancel(self, order_id: int) -> Order:
        order = self.ledger.get_order(order_id)
        if not order.is_open:
            raise ValueError(
                f"order {order_id} is already {order.status.value}; cannot cancel"
            )
        self.ledger.record_order_event(order_id, OrderStatus.CANCELLED)
        order.status = OrderStatus.CANCELLED
        return order

    def cancel_all(self, session=None) -> list[Order]:
        return [self.cancel(o.id) for o in self.open_orders()]

    # -------------------------------------------------------- session processing

    def process_session(
        self,
        session,
        bars: dict[str, Bar],
        actions: Sequence[CorporateActionLike] = (),
    ) -> SessionResult:
        """Advance the account through one session. Steps in the spec's order."""
        session = pd.Timestamp(session).normalize()

        splits_applied, dividend_cash = self._apply_corporate_actions(session, actions)
        fills, expired, rejected = self._match_orders(session, bars)

        for ticker, bar in bars.items():
            self.set_mark(ticker, bar.close)

        cash = self.cash()
        marks = {t: b.close for t, b in bars.items()}
        equity = self.equity(marks)
        positions = self.positions()
        self.ledger.write_snapshot(self.account_id, session, cash, equity, positions)
        self.ledger.set_last_processed_session(self.account_id, session)

        return SessionResult(
            session=session,
            fills=tuple(fills),
            expired=tuple(expired),
            rejected=tuple(rejected),
            dividends=dividend_cash,
            splits=tuple(splits_applied),
            cash=cash,
            equity=equity,
        )

    # step 1 -----------------------------------------------------------------

    def _apply_corporate_actions(self, session, actions) -> tuple[list[str], float]:
        """Splits re-term holdings AND resting orders; dividends credit cash.

        Equity is unchanged across a split by construction: shares multiply by
        r while the as-traded price divides by r on the same session.
        """
        already = self.ledger.applied_action_ids(self.account_id, session)
        positions = self.positions()
        splits: list[str] = []
        dividend_total = 0.0

        for action in actions:
            ex_date = pd.Timestamp(action.ex_date).normalize()
            if ex_date != session:
                continue
            ticker = action.ticker
            if (ticker, action.type) in already:
                continue  # idempotent: this session already applied it
            held = positions.get(ticker)

            if action.type == SPLIT:
                if held is None or held.qty <= EPS:
                    continue
                before = held.qty
                after = before * action.value
                self.ledger.record_corporate_action(
                    self.account_id, session, ticker, SPLIT, action.value,
                    shares_before=before, shares_after=after,
                )
                # resting orders must be re-termed too, or an order placed
                # pre-split asks for r times the shares it meant
                for order in self.open_orders():
                    if order.ticker != ticker:
                        continue
                    order.apply_split(action.value)
                    self.ledger.record_order_qty(
                        order.id, order.qty, order.limit_price, order.stop_price
                    )
                splits.append(ticker)
                positions = self.positions()

            elif action.type == DIVIDEND:
                # paid on the shares held at the previous close, i.e. before
                # anything this session trades
                if held is None or held.qty <= EPS:
                    continue
                amount = held.qty * action.value
                self.ledger.record_corporate_action(
                    self.account_id, session, ticker, DIVIDEND, action.value,
                    shares_before=held.qty, shares_after=held.qty,
                )
                self.ledger.record_cash_event(
                    self.account_id, DIVIDEND, amount, session=session,
                    note=f"{ticker} dividend {action.value:.4f}/share on {held.qty:g} shares",
                )
                dividend_total += amount

        return splits, dividend_total

    # step 2 -----------------------------------------------------------------

    def _match_orders(self, session, bars) -> tuple[list[Fill], list[int], list[int]]:
        """Sells first, then buys, each oldest-first.

        Sells before buys because that is what a real session does: proceeds
        from a morning sale are available for an afternoon purchase. Oldest
        first within each side keeps a replay deterministic.
        """
        eligible = [o for o in self.open_orders() if o.eligible_on(session)]
        eligible.sort(key=lambda o: (0 if o.side is Side.SELL else 1, o.id))

        fills: list[Fill] = []
        expired: list[int] = []
        rejected: list[int] = []

        for order in eligible:
            bar = bars.get(order.ticker)
            if bar is None:
                # halted or missing data: nothing to fill against. The order
                # stays pending unless it was a DAY order, whose day is gone.
                if order.tif is TimeInForce.DAY:
                    self._expire(order, session, "no bar for this session")
                    expired.append(order.id)
                continue

            matched = self._fill_price(order, bar)
            if matched is None:
                if order.tif is TimeInForce.DAY:
                    self._expire(order, session, "not marketable on this session")
                    expired.append(order.id)
                continue

            price, reference = matched
            fill = self._execute(order, session, price, reference)
            if fill is None:
                rejected.append(order.id)
            else:
                fills.append(fill)

        return fills, expired, rejected

    def _fill_price(self, order: Order, bar: Bar) -> tuple[float, float] | None:
        """The spec's fill table.

        Returns `(fill_price, reference_price)` — the price actually paid and
        the price before slippage — so the fill can record what the slippage
        model took out. Returns None when the order is not marketable against
        this bar.

        Slippage applies to MARKET and STOP fills. It does NOT apply to limit
        fills: a limit order fills at its price or better by definition, which
        is exactly what the order type buys you.
        """
        buy = order.side is Side.BUY

        if order.type is OrderType.MARKET:
            return self.costs.slip(bar.open, buy), bar.open

        if order.type is OrderType.LIMIT:
            limit = order.limit_price
            if buy:
                if bar.open <= limit:
                    return bar.open, bar.open   # gapped through: price improvement
                if bar.low <= limit:
                    return limit, limit         # touched intraday
                return None
            if bar.open >= limit:
                return bar.open, bar.open
            if bar.high >= limit:
                return limit, limit
            return None

        # a triggered stop becomes a market order, so slippage applies
        stop = order.stop_price
        if buy:
            if bar.open >= stop:
                return self.costs.slip(bar.open, True), bar.open
            if bar.high >= stop:
                return self.costs.slip(stop, True), stop
            return None
        if bar.open <= stop:
            return self.costs.slip(bar.open, False), bar.open
        if bar.low <= stop:
            return self.costs.slip(stop, False), stop
        return None

    def _execute(
        self, order: Order, session, price: float, reference: float
    ) -> Fill | None:
        """Write the fill, or reject it if the account cannot actually afford
        it at the real price. Cash must never go negative."""
        qty = order.qty
        commission = self.costs.commission(qty)

        if order.side is Side.BUY:
            cost = qty * price + commission
            available = self.cash()
            if cost > available + EPS:
                reason = (
                    f"buy {qty:g} {order.ticker} at ${price:,.2f} costs "
                    f"${cost:,.2f} incl. ${commission:,.2f} commission, but only "
                    f"${available:,.2f} cash is left"
                )
                self._reject(order, session, reason)
                return None
        else:
            held = self.positions().get(order.ticker)
            have = held.qty if held else 0.0
            if qty > have + EPS:
                # can happen if a split shrank the position or an earlier sell
                # this session already took the shares
                self._reject(
                    order, session,
                    f"sell {qty:g} {order.ticker} but only {have:g} shares are held",
                )
                return None

        fill = Fill(
            order_id=order.id,
            account_id=self.account_id,
            session=session,
            ticker=order.ticker,
            side=order.side,
            qty=qty,
            price=price,
            commission=commission,
            # zero for limit fills by construction: reference == price there
            slippage_cost=self.costs.slippage_cost(reference, price, qty),
        )
        stored = self.ledger.record_fill(fill)
        self.ledger.record_order_event(order.id, OrderStatus.FILLED, session=session)
        return stored

    def _expire(self, order: Order, session, why: str) -> None:
        self.ledger.record_order_event(
            order.id, OrderStatus.EXPIRED, session=session,
            reject_reason=f"day order expired: {why}",
        )

    def _reject(self, order: Order, session, reason: str) -> None:
        self.ledger.record_order_event(
            order.id, OrderStatus.REJECTED, session=session, reject_reason=reason
        )
        log.info("fill rejected: %s — %s", order.describe(), reason)
