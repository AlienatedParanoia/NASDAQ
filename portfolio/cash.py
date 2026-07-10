"""Cash ledger and Real Investment P&L.

The My Investments tab tracks positions, not money. It knows what went into
each lot, but not where that money came from: a lot funded by reinvesting a
previous sale looks exactly like a lot funded by a fresh bank transfer. Summing
`money_in` therefore answers "how much have I ever committed to positions",
which is not the question the user is asking.

This module adds the missing half. Every dollar entering or leaving the account
from outside is recorded as a CashFlow; everything else — buys, sales — merely
moves money between the cash balance and the positions. The number the user
actually wants is then:

    Real Net Profit = Current Account Value - Net External Capital

where account value is cash plus the market value of the open positions, and
net external capital is deposits minus withdrawals.

No cash balance is stored. It is replayed from the ledger on demand, because a
stored balance is a second source of truth that can silently drift out of
agreement with the lots. Deriving it means the identity above holds by
construction rather than by discipline.

That identity, expanded, is worth stating once:

    cash            = D - W - sum_all(money_in) + sum_sold(money_out)
    account_value   = cash + sum_open(worth)
    account - (D-W) = sum_all(worth - money_in)

The right-hand side is exactly the existing per-lot P&L, summed over every lot
including the closed ones. Real Net Profit and tracker.py's `overall.pnl` are
therefore the same number, always — asserted in tests/phase11_real_pnl_check.py.
What this module fixes is not the profit but the *denominator*: `overall.total_in`
grows with every reinvestment, so the return percentage it reports shrinks
against a base the user never deposited.

Nothing here imports Qt or touches price data. `compute_real_pnl` takes the
open positions' market values as an argument, which is what lets the whole
thing be tested offline in milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

if TYPE_CHECKING:  # tracker imports CashFlow for persistence; keep it one-way
    from portfolio.tracker import Lot

DEPOSIT = "deposit"
WITHDRAWAL = "withdrawal"

# money compares are float compares; a balance this close to zero is zero
EPS = 1e-6


@dataclass(slots=True)
class CashFlow:
    """External money movement — the only source of truth for what the user
    actually put in from, or took out to, their bank."""
    amount: float          # always positive; `direction` carries the sign
    direction: str         # DEPOSIT | WITHDRAWAL
    date: str              # ISO YYYY-MM-DD
    seq: int               # monotonic insertion counter; breaks same-date ties

    @property
    def delta(self) -> float:
        return self.amount if self.direction == DEPOSIT else -self.amount


@dataclass(slots=True, eq=False)
class Event:
    """One dated movement of the cash balance, from either a CashFlow or a
    leg of a Lot. `label` is user-facing: it names the offending transaction
    when a replay overdraws."""
    date: str
    seq: int
    delta: float
    label: str


@dataclass(slots=True, eq=False)
class RealPnL:
    total_deposited: float
    total_withdrawn: float
    net_external: float
    cash: float
    positions_value: float           # open lots only, at market
    account_value: float             # cash + positions_value
    real_net_profit: float           # account_value - net_external
    real_return_pct: float | None    # None when net_external <= 0
    unresolved_sales: int            # sold lots still awaiting their proceeds


def _money(v: float) -> str:
    return f"${v:,.2f}"


def is_unresolved_sale(lot: Lot) -> bool:
    """A lot sold through the UI carries a sell date before the worker has had
    a chance to price it. Its proceeds are unknown until then, so it credits
    nothing — the cash balance is understated, never overstated, which keeps
    the overdraft checks conservative."""
    return lot.sell_date is not None and lot.money_out is None


def ledger_events(
    lots: Sequence[Lot], flows: Sequence[CashFlow]
) -> list[Event]:
    """Every cash movement, in the order it happened. A buy debits on its buy
    date; a sale credits on its sell date; the two legs of one lot are
    independent events, which is what makes a backdated buy checkable."""
    events: list[Event] = []
    for flow in flows:
        verb = "Deposit" if flow.direction == DEPOSIT else "Withdrawal"
        events.append(Event(
            date=flow.date,
            seq=flow.seq,
            delta=flow.delta,
            label=f"{verb} of {_money(flow.amount)}",
        ))
    for lot in lots:
        events.append(Event(
            date=lot.buy_date,
            seq=lot.buy_seq,
            delta=-lot.money_in,
            label=f"Buy {lot.ticker} for {_money(lot.money_in)}",
        ))
        if lot.is_sold and not is_unresolved_sale(lot):
            events.append(Event(
                # a v1 record may carry proceeds with no sell date; credit it
                # on its buy date, which is the only date it has
                date=lot.sell_date or lot.buy_date,
                seq=lot.sell_seq if lot.sell_seq is not None else lot.buy_seq,
                delta=float(lot.money_out),
                label=f"Sale of {lot.ticker} for {_money(lot.money_out)}",
            ))
    events.sort(key=lambda e: (e.date, e.seq))
    return events


def replay(events: Iterable[Event]) -> list[tuple[Event, float]]:
    """Running cash balance after each event."""
    balance = 0.0
    out: list[tuple[Event, float]] = []
    for event in events:
        balance += event.delta
        out.append((event, balance))
    return out


def current_cash(lots: Sequence[Lot], flows: Sequence[CashFlow]) -> float:
    """The balance today. Equivalently D - W - sum(money_in) + sum(money_out),
    but replaying keeps one definition of the ledger."""
    trail = replay(ledger_events(lots, flows))
    return trail[-1][1] if trail else 0.0


def validate(lots: Sequence[Lot], flows: Sequence[CashFlow]) -> str | None:
    """Replay the *prospective* ledger and name the first transaction that
    drives cash negative, or return None if it never does.

    Callers build the state they are about to commit — with the new event
    inserted, or the removed one dropped — and refuse the action on a message.
    Because a backdated buy shifts every later balance down, the offending
    transaction is often not the one the user just entered; saying which one
    broke is the difference between a usable error and a baffling one.
    """
    for event, balance in replay(ledger_events(lots, flows)):
        if balance < -EPS:
            return (
                f"{event.label} on {event.date} would overdraw the account "
                f"by {_money(-balance)}. Deposit more cash first, or change "
                f"an earlier transaction."
            )
    return None


def compute_real_pnl(
    lots: Sequence[Lot],
    flows: Sequence[CashFlow],
    open_worth: Mapping[int, float],
) -> RealPnL:
    """The pure function. `open_worth` maps an index into `lots` to that
    position's current market value; only open lots need an entry, and a
    missing one (an errored ticker, say) contributes zero.

    Kept free of DataFetcher on purpose — the caller has already priced the
    positions, and the five validation scenarios need no prices at all.
    """
    deposited = sum(f.amount for f in flows if f.direction == DEPOSIT)
    withdrawn = sum(f.amount for f in flows if f.direction == WITHDRAWAL)
    net_external = deposited - withdrawn

    cash = current_cash(lots, flows)
    positions_value = sum(
        float(open_worth.get(i, 0.0))
        for i, lot in enumerate(lots)
        if not lot.is_sold
    )
    account_value = cash + positions_value
    real_net_profit = account_value - net_external

    # a return on zero or negative capital is meaningless, not infinite
    real_return_pct = (
        real_net_profit / net_external * 100.0 if net_external > EPS else None
    )

    return RealPnL(
        total_deposited=float(deposited),
        total_withdrawn=float(withdrawn),
        net_external=float(net_external),
        cash=float(cash),
        positions_value=float(positions_value),
        account_value=float(account_value),
        real_net_profit=float(real_net_profit),
        real_return_pct=real_return_pct,
        unresolved_sales=sum(1 for lot in lots if is_unresolved_sale(lot)),
    )


def next_seq(lots: Sequence[Lot], flows: Sequence[CashFlow]) -> int:
    """One counter shared by both stores, so a buy and a deposit entered on
    the same day always replay in the order they were entered."""
    seqs = [f.seq for f in flows]
    for lot in lots:
        seqs.append(lot.buy_seq)
        if lot.sell_seq is not None:
            seqs.append(lot.sell_seq)
    return max(seqs, default=-1) + 1
