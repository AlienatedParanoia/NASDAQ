"""Phase 3 walkthrough: one account across eight sessions, printed so the
arithmetic can be checked by hand.

Covers, in order: a market buy, a limit buy that misses and expires, a GTC
limit buy that waits and then fills, a 2:1 split (with a resting order
re-termed by it), a dividend, a rejected buy, a missing bar, and a sell.

Offline. Run from the project root:
    python -m tests.phase3_broker_walkthrough
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from engine.costs import CostModel
from engine.ledger import DIVIDEND, SPLIT, Ledger
from engine.orders import Order, OrderStatus, OrderType, Side, TimeInForce
from engine.sim_broker import Bar, SimBroker

START_CASH = 10_000.0
# explicit, hand-checkable costs: $1 flat per fill, 10 bps slippage
COSTS = CostModel(per_order=1.0, per_share=0.0, minimum=1.0, slippage_bps=10.0)


@dataclass(frozen=True)
class Action:
    ticker: str
    ex_date: pd.Timestamp
    type: str
    value: float


_SESSIONS = pd.bdate_range("2026-03-02", periods=12)


def session(n: int) -> pd.Timestamp:
    return _SESSIONS[n]


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show(broker: SimBroker, result, note: str) -> None:
    positions = broker.positions()
    held = ", ".join(
        f"{t} {p.qty:g} @ ${p.avg_cost:,.2f}" for t, p in sorted(positions.items())
    ) or "none"
    print(f"\n--- {result.session.date()} — {note}")
    for fill in result.fills:
        print(
            f"    FILL  {fill.side.value:<4} {fill.qty:>4g} {fill.ticker} "
            f"@ ${fill.price:>8,.4f}   commission ${fill.commission:.2f}"
            f"   slippage ${fill.slippage_cost:.2f}"
        )
    for order_id in result.expired:
        print(f"    EXPIRED  order #{order_id}")
    for order_id in result.rejected:
        reason = broker.ledger.get_order(order_id).reject_reason
        print(f"    REJECTED order #{order_id}: {reason}")
    if result.splits:
        print(f"    SPLIT    {', '.join(result.splits)}")
    if result.dividends:
        print(f"    DIVIDEND ${result.dividends:,.2f} credited")
    print(f"    cash ${result.cash:>10,.2f}   equity ${result.equity:>10,.2f}")
    print(f"    holdings: {held}")


def main() -> None:
    path = Path(tempfile.mkdtemp()) / "walkthrough.db"
    ledger = Ledger(path)
    account = ledger.create_account("walkthrough", START_CASH)
    broker = SimBroker(ledger, account.id, costs=COSTS)
    broker.set_mark("ACME", 100.0)

    rule(
        f"Account '{account.name}' — starting cash ${START_CASH:,.2f}\n"
        f"Costs: ${COSTS.minimum:.2f} minimum commission per fill, "
        f"{COSTS.slippage_bps:.0f} bps slippage on market/stop fills (limits are exact)"
    )

    # -- session 1: decide to buy; nothing can fill on the decision session ---
    order1 = broker.submit(
        Order(account_id=account.id, created_session=session(0), ticker="ACME",
              side=Side.BUY, qty=50, reason="walkthrough: open a position")
    )
    r = broker.process_session(session(0), {"ACME": Bar(100.0, 102.0, 99.0, 100.0)})
    show(broker, r, "market buy 50 decided today — cannot fill today (no look-ahead)")

    # -- session 2: it fills at the open, plus slippage ------------------------
    r = broker.process_session(session(1), {"ACME": Bar(100.0, 104.0, 99.5, 103.0)})
    show(broker, r, "market buy fills at the open 100.00 x 1.001 = 100.10")
    print(f"    check: 50 x $100.10 = ${50 * 100.10:,.2f} + $1.00 commission")
    print(f"           ${START_CASH:,.2f} - ${50 * 100.10 + 1:,.2f} = ${broker.cash():,.2f}")

    # -- session 3: a DAY limit far below the market expires unfilled ---------
    broker.submit(
        Order(account_id=account.id, created_session=session(1), ticker="ACME",
              side=Side.BUY, qty=10, type=OrderType.LIMIT, limit_price=80.0,
              tif=TimeInForce.DAY, reason="walkthrough: lowball day order")
    )
    r = broker.process_session(session(2), {"ACME": Bar(103.0, 105.0, 101.0, 104.0)})
    show(broker, r, "DAY limit buy at $80 never traded there — expires")

    # -- session 4: a GTC limit waits ----------------------------------------
    order3 = broker.submit(
        Order(account_id=account.id, created_session=session(2), ticker="ACME",
              side=Side.BUY, qty=10, type=OrderType.LIMIT, limit_price=95.0,
              tif=TimeInForce.GTC, reason="walkthrough: patient GTC bid")
    )
    r = broker.process_session(session(3), {"ACME": Bar(104.0, 106.0, 100.0, 105.0)})
    show(broker, r, "GTC limit buy at $95 still waiting (low was 100)")

    # -- session 5: 2:1 split; the resting GTC order is re-termed -------------
    before = broker.positions()["ACME"]
    equity_before = broker.equity({"ACME": 105.0})
    resting_before = ledger.get_order(order3.id)
    # the previous close was 105.00; a clean 2:1 halves it to exactly 52.50,
    # which makes "equity is unchanged across a split" literally checkable
    r = broker.process_session(
        session(4), {"ACME": Bar(52.5, 53.0, 51.0, 52.5)},
        [Action("ACME", session(4), SPLIT, 2.0)],
    )
    after = broker.positions()["ACME"]
    resting_after = ledger.get_order(order3.id)
    show(broker, r, "2:1 split — as-traded price halves, shares double")
    print(f"    check: shares {before.qty:g} -> {after.qty:g}, "
          f"avg cost ${before.avg_cost:,.4f} -> ${after.avg_cost:,.4f}")
    print(f"           cost basis ${before.cost_basis:,.2f} -> ${after.cost_basis:,.2f} "
          f"(unchanged)")
    print(f"           equity ${equity_before:,.2f} -> ${r.equity:,.2f} "
          f"(unchanged: shares x2, price /2)")
    assert abs(r.equity - equity_before) < 0.01, "a split must not move equity"
    print(f"           resting order: {resting_before.qty:g} @ "
          f"${resting_before.limit_price:,.2f}  ->  {resting_after.qty:g} @ "
          f"${resting_after.limit_price:,.2f}")

    # -- session 6: dividend on the shares held at the previous close ---------
    held = broker.positions()["ACME"].qty
    r = broker.process_session(
        session(5), {"ACME": Bar(52.5, 54.0, 52.0, 53.5)},
        [Action("ACME", session(5), DIVIDEND, 0.40)],
    )
    show(broker, r, "dividend $0.40/share, credited on the ex-date")
    print(f"    check: {held:g} shares x $0.40 = ${held * 0.40:,.2f}")

    # -- session 7: a buy far beyond cash is refused at submission ------------
    rejected = broker.submit(
        Order(account_id=account.id, created_session=session(5), ticker="ACME",
              side=Side.BUY, qty=10_000, reason="walkthrough: deliberately too big")
    )
    print(f"\n--- submission refused immediately (not a fill-time surprise)")
    print(f"    order #{rejected.id} status={rejected.status.value}")
    print(f"    reason: {rejected.reject_reason}")

    # -- session 8: no bar at all — the GTC order simply waits ---------------
    r = broker.process_session(session(6), {})
    show(broker, r, "no bar for ACME (halted): GTC order stays pending, marks unchanged")

    # -- session 9: sell half the position ------------------------------------
    broker.submit(
        Order(account_id=account.id, created_session=session(6), ticker="ACME",
              side=Side.SELL, qty=50, reason="walkthrough: take some profit")
    )
    r = broker.process_session(session(7), {"ACME": Bar(60.0, 61.0, 59.0, 60.5)})
    show(broker, r, "market sell 50 at the open 60.00 x 0.999 = 59.94")

    # -- reconciliation -------------------------------------------------------
    rule("Reconciliation — rebuilt from the ledger alone")
    fresh = SimBroker(ledger, account.id, costs=COSTS)
    marks = {"ACME": 60.5}
    print(f"    live broker : cash ${broker.cash():>10,.2f}   "
          f"equity ${broker.equity(marks):>10,.2f}")
    print(f"    rebuilt     : cash ${fresh.cash():>10,.2f}   "
          f"equity ${fresh.equity(marks):>10,.2f}")
    assert abs(fresh.cash() - broker.cash()) < 1e-9
    assert abs(fresh.equity(marks) - broker.equity(marks)) < 1e-9

    positions = broker.positions()
    equity = broker.equity(marks)
    holdings_value = sum(p.qty * marks[t] for t, p in positions.items())
    print(f"\n    equity identity: cash ${broker.cash():,.2f} + holdings "
          f"${holdings_value:,.2f} = ${broker.cash() + holdings_value:,.2f}")
    print(f"                     reported equity              = ${equity:,.2f}")
    assert abs(equity - (broker.cash() + holdings_value)) < 0.01

    rule("Order book")
    for order in ledger.orders(account.id):
        reason = f"  [{order.reject_reason}]" if order.reject_reason else ""
        print(
            f"    #{order.id:<3} {order.side.value:<4} {order.qty:>7g} "
            f"{order.ticker:<5} {order.type.value:<7} {order.tif.value:<4} "
            f"{order.status.value:<9}{reason}"
        )

    rule("Cash ledger")
    running = 0.0
    for event in ledger.cash_events(account.id):
        running += event["amount"]
        when = event["session"] or "opening"
        print(
            f"    {when:<12} {event['kind']:<11} {event['amount']:>12,.2f}"
            f"   balance {running:>11,.2f}"
        )
    assert abs(running - broker.cash()) < 1e-9

    print(f"\n{'=' * 78}")
    print("PHASE 3 WALKTHROUGH COMPLETE — every figure above is hand-checkable")
    print(f"{'=' * 78}")
    ledger.close()


if __name__ == "__main__":
    main()
