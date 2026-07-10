"""Cash ledger and Real Investment P&L.

The My Investments tab tracks positions, not money. It knows what went into
each lot, but not where that money came from: a lot funded by reinvesting a
previous sale looks exactly like a lot funded by a fresh bank transfer. Summing
`money_in` therefore answers "how much have I ever committed to positions",
which is not the question the user is asking.

This module adds the missing half. Every dollar entering or leaving the account
from outside counts as external capital; everything else — buys, sales — merely
moves money between the cash balance and the positions. The number the user
actually wants is then:

    Real Net Profit = Current Account Value - Net External Capital

where account value is cash plus the market value of the open positions, and
net external capital is deposits minus withdrawals.

External capital comes from two places. The user may record a deposit
explicitly, which is the only way to say "I transferred $10,000 in and left
$8,000 of it uninvested". Otherwise a purchase that exceeds the cash on hand
is assumed to have been funded from outside for exactly the shortfall, and
`ledger_events` records that inferred deposit on the buy date. Cash on hand is
always spent first, so a purchase funded by an earlier sale creates no deposit
and reinvested profit is never mistaken for money from the bank.

Return is reported two ways, against the same profit. `overall_return_pct`
divides by every dollar put into a position; `own_money_return_pct` divides by
only the part the user did not flag as bought-with-profits. The profit figure
is identical, so during a reinvestment loss the two percentages separate: the
overall return sags because the reinvested-then-lost money is still in its
base, while the own-money return holds up because a profit-funded position was
never the user's capital to lose. This is a presentation lens, not a second
truth — the profit, the cash balance, and the ledger are unaffected by the
flag.

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

# event kinds, for the replay
BUY = "buy"
SELL = "sell"
AUTOFUND = "autofund"

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
    leg of a Lot.

    Two user-facing strings, because they are read in different places.
    `title` is short ("Buy AAPL") for the transactions table, which carries the
    amount in its own column. `label` embeds the amount ("Buy AAPL for
    $2,000.00") so an overdraft message can name the offending transaction on
    its own, without a table around it.
    """
    date: str
    seq: int
    delta: float
    kind: str                # DEPOSIT | WITHDRAWAL | BUY | SELL | AUTOFUND
    title: str
    label: str
    subject: str | None = None   # the ticker, for lot legs


@dataclass(slots=True, eq=False)
class RealPnL:
    total_deposited: float
    total_withdrawn: float
    net_external: float
    cash: float
    positions_value: float           # open lots only, at market
    account_value: float             # cash + positions_value
    real_net_profit: float           # account_value - net_external
    real_return_pct: float | None    # profit / net_external; None when <= 0
    unresolved_sales: int            # sold lots still awaiting their proceeds
    auto_funded: float               # of total_deposited, how much was inferred
    # two views of the same profit against different capital bases. The profit
    # is identical; only the denominator changes, so during a reinvestment loss
    # they diverge. See portfolio/cash.py module docstring.
    total_invested: float            # every dollar put into a position
    own_invested: float              # of that, the part not flagged from-profit
    overall_return_pct: float | None      # profit / total_invested
    own_money_return_pct: float | None    # profit / own_invested
    has_profit_positions: bool       # any lot flagged bought-with-profits


@dataclass(slots=True, eq=False)
class LotFunding:
    """How one purchase was paid for, split with no fungibility convention.

    `from_own` is money that came from outside — idle cash you deposited, plus
    any shortfall auto-funded on the buy. `from_proceeds` is money recycled
    from an earlier sale. They sum to the lot's cost. The split is exact: a buy
    is assumed to spend sale proceeds before idle deposited cash (proceeds
    first), which is the only assumption needed and the one that reads
    correctly when idle deposited cash is present.
    """
    index: int              # position in the lots sequence it was built from
    ticker: str
    cost: float             # == money_in == from_own + from_proceeds
    from_own: float
    from_proceeds: float

    @property
    def source(self) -> str:
        if self.from_proceeds <= EPS:
            return "Own"
        if self.from_own <= EPS:
            return "Proceeds"
        return "Mixed"


def _money(v: float) -> str:
    return f"${v:,.2f}"


def is_unresolved_sale(lot: Lot) -> bool:
    """A lot sold through the UI carries a sell date before the worker has had
    a chance to price it. Its proceeds are unknown until then, so it credits
    nothing — the cash balance is understated, never overstated, which keeps
    the overdraft checks conservative."""
    return lot.sell_date is not None and lot.money_out is None


def _recorded_events(
    lots: Sequence[Lot], flows: Sequence[CashFlow]
) -> list[Event]:
    """The events the user actually entered, in the order they happened. A buy
    debits on its buy date; a sale credits on its sell date; the two legs of
    one lot are independent events, which is what dates a backdated buy."""
    events: list[Event] = []
    for flow in flows:
        verb = "Deposit" if flow.direction == DEPOSIT else "Withdrawal"
        events.append(Event(
            date=flow.date,
            seq=flow.seq,
            delta=flow.delta,
            kind=flow.direction,
            title=verb,
            label=f"{verb} of {_money(flow.amount)}",
        ))
    for lot in lots:
        events.append(Event(
            date=lot.buy_date,
            seq=lot.buy_seq,
            delta=-lot.money_in,
            kind=BUY,
            title=f"Buy {lot.ticker}",
            label=f"Buy {lot.ticker} for {_money(lot.money_in)}",
            subject=lot.ticker,
        ))
        if lot.is_sold and not is_unresolved_sale(lot):
            events.append(Event(
                # a v1 record may carry proceeds with no sell date; credit it
                # on its buy date, which is the only date it has
                date=lot.sell_date or lot.buy_date,
                seq=lot.sell_seq if lot.sell_seq is not None else lot.buy_seq,
                delta=float(lot.money_out),
                kind=SELL,
                title=f"Sell {lot.ticker}",
                label=f"Sale of {lot.ticker} for {_money(lot.money_out)}",
                subject=lot.ticker,
            ))
    events.sort(key=lambda e: (e.date, e.seq))
    return events


def ledger_events(
    lots: Sequence[Lot], flows: Sequence[CashFlow], *, autofund: bool = True
) -> list[Event]:
    """Every cash movement, with the deposits the user never had to type.

    A buy spends the cash on hand. If that is not enough, the shortfall could
    only have come from outside the account, so it is recorded as a deposit
    dated on the buy — this is `autofund`. The user is never asked to deposit
    before buying, and never has to think about the order they entered things.

    The crucial property is that cash is spent *before* new capital is
    invented. A buy funded by an earlier sale finds the proceeds waiting and
    creates no deposit, so reinvested profit is never miscounted as money from
    the bank. That is precisely the guarantee the old "deposit first" rule
    bought, obtained without the friction.

    Only buys autofund. Withdrawing cash you do not have is not a shortfall to
    be covered, it is a mistake — `validate` refuses it.

    Nothing here is persisted: the inferred deposits are recomputed on every
    replay, so editing or removing a lot can never strand one behind. This is
    the same reason the cash balance is derived rather than stored.
    """
    if not autofund:
        return _recorded_events(lots, flows)

    events: list[Event] = []
    balance = 0.0
    for event in _recorded_events(lots, flows):
        if event.kind == BUY:
            shortfall = -(balance + event.delta)
            if shortfall > EPS:
                events.append(Event(
                    date=event.date,
                    seq=event.seq,   # sorted already; sits just before the buy
                    delta=shortfall,
                    kind=AUTOFUND,
                    title=f"Deposit (funds {event.subject})",
                    label=f"Deposit of {_money(shortfall)} to fund {event.subject}",
                    subject=event.subject,
                ))
                balance += shortfall
        events.append(event)
        balance += event.delta
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

    Since buys autofund, the only event that can now overdraw is a withdrawal
    — you cannot take out cash that is not there, and unlike a purchase there
    is nothing to infer. Callers build the state they are about to commit and
    refuse the action on a message.
    """
    for event, balance in replay(ledger_events(lots, flows)):
        if balance < -EPS:
            return (
                f"{event.label} on {event.date} would overdraw the account "
                f"by {_money(-balance)}. There is not that much cash in the "
                f"account on that date."
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
    # money the user typed, plus money the app inferred had to come from
    # outside to fund a purchase — both are external capital, and the return
    # must be measured against the sum
    events = ledger_events(lots, flows)
    auto_funded = sum(e.delta for e in events if e.kind == AUTOFUND)
    deposited = sum(f.amount for f in flows if f.direction == DEPOSIT) + auto_funded
    withdrawn = sum(f.amount for f in flows if f.direction == WITHDRAWAL)
    net_external = deposited - withdrawn

    trail = replay(events)
    cash = trail[-1][1] if trail else 0.0
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

    # the two headline percentages. Same profit, two bases: everything put into
    # positions, versus only the part the user did not flag as profit-funded.
    # Reinvestment losses shrink the profit and inflate the overall base, so the
    # overall return sags while the own-money return holds up.
    total_invested = float(sum(lot.money_in for lot in lots))
    own_invested = float(
        sum(lot.money_in for lot in lots if not getattr(lot, "from_profit", False))
    )
    overall_return_pct = (
        real_net_profit / total_invested * 100.0 if total_invested > EPS else None
    )
    own_money_return_pct = (
        real_net_profit / own_invested * 100.0 if own_invested > EPS else None
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
        auto_funded=float(auto_funded),
        total_invested=total_invested,
        own_invested=own_invested,
        overall_return_pct=overall_return_pct,
        own_money_return_pct=own_money_return_pct,
        has_profit_positions=any(
            getattr(lot, "from_profit", False) for lot in lots
        ),
    )


def funding_breakdown(
    lots: Sequence[Lot], flows: Sequence[CashFlow]
) -> dict[int, LotFunding]:
    """Split each purchase into money from outside vs money recycled from an
    earlier sale, keyed by the lot's index.

    A buy spends sale proceeds first, then idle deposited cash, then whatever
    is still short is auto-funded from outside. So `from_proceeds` is money
    that had already been through a position, and `from_own` is fresh external
    capital — the distinction the "Funded by" column reports, and the one the
    profit report totals as reinvested proceeds. No principal-vs-profit
    convention is involved: proceeds are proceeds, whatever their make-up.
    """
    events = []
    for f in flows:
        events.append((f.date, f.seq, f.direction, f.amount, -1))
    for i, lot in enumerate(lots):
        events.append((lot.buy_date, lot.buy_seq, BUY, lot.money_in, i))
        if lot.is_sold and not is_unresolved_sale(lot):
            events.append((
                lot.sell_date or lot.buy_date,
                lot.sell_seq if lot.sell_seq is not None else lot.buy_seq,
                SELL, float(lot.money_out), i,
            ))
    events.sort(key=lambda e: (e[0], e[1]))

    idle = 0.0       # deposited cash not yet spent
    proceeds = 0.0   # cash returned by sales, waiting to be redeployed
    out: dict[int, LotFunding] = {}
    for _date, _seq, kind, amount, idx in events:
        if kind == DEPOSIT:
            idle += amount
        elif kind == WITHDRAWAL:
            # taking your money back out comes off idle deposits first
            take = min(amount, idle)
            idle -= take
            proceeds -= amount - take
        elif kind == SELL:
            proceeds += amount
        elif kind == BUY:
            from_proceeds = min(amount, proceeds)
            proceeds -= from_proceeds
            rest = amount - from_proceeds
            from_idle = min(rest, idle)
            idle -= from_idle
            from_autofund = rest - from_idle   # the shortfall, sourced outside
            out[idx] = LotFunding(
                index=idx,
                ticker=lots[idx].ticker,
                cost=float(amount),
                from_own=float(from_idle + from_autofund),
                from_proceeds=float(from_proceeds),
            )
    return out


def next_seq(lots: Sequence[Lot], flows: Sequence[CashFlow]) -> int:
    """One counter shared by both stores, so a buy and a deposit entered on
    the same day always replay in the order they were entered."""
    seqs = [f.seq for f in flows]
    for lot in lots:
        seqs.append(lot.buy_seq)
        if lot.sell_seq is not None:
            seqs.append(lot.sell_seq)
    return max(seqs, default=-1) + 1
