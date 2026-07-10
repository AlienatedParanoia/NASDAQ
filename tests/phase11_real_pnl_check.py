"""Phase 11 checks: the cash ledger and Real Investment P&L (portfolio/cash.py).

Fully offline and deterministic — no DataFetcher, no network, no cache.db. The
open positions' market values are injected, which is the whole point of keeping
compute_real_pnl a pure function.

Run from the project root: python -m tests.phase11_real_pnl_check
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path

from portfolio.cash import (
    DEPOSIT,
    WITHDRAWAL,
    CashFlow,
    RealPnL,
    compute_real_pnl,
    current_cash,
    funding_breakdown,
    ledger_events,
    replay,
    validate,
)
from portfolio.tracker import (
    INVESTMENTS_VERSION,
    Lot,
    load_investments_doc,
    save_investments,
)

TOL = 1e-9


def _flow(amount: float, direction: str, date: str, seq: int) -> CashFlow:
    return CashFlow(amount=amount, direction=direction, date=date, seq=seq)


def _check(name: str, lots, flows, open_worth, *, cash, account, profit) -> RealPnL:
    """Assert the headline numbers, then the invariants that must hold in
    every scenario regardless of what the headline numbers are."""
    r = compute_real_pnl(lots, flows, open_worth)

    assert abs(r.cash - cash) < TOL, f"{name}: cash {r.cash} != {cash}"
    assert abs(r.account_value - account) < TOL, (
        f"{name}: account {r.account_value} != {account}"
    )
    assert abs(r.real_net_profit - profit) < TOL, (
        f"{name}: profit {r.real_net_profit} != {profit}"
    )

    # account value is cash plus the open positions, by definition
    open_sum = sum(
        open_worth.get(i, 0.0) for i, lot in enumerate(lots) if not lot.is_sold
    )
    assert abs(r.account_value - (r.cash + open_sum)) < TOL

    # the formula under test
    assert abs(r.real_net_profit - (r.account_value - r.net_external)) < TOL

    # the identity from cash.py's docstring: Real Net Profit is exactly the
    # existing per-lot P&L summed over EVERY lot, closed ones included. This
    # cross-checks the new number against tracker.py's `overall.pnl`.
    legacy = sum(
        (lot.money_out if lot.is_sold else open_worth.get(i, 0.0)) - lot.money_in
        for i, lot in enumerate(lots)
    )
    assert abs(r.real_net_profit - legacy) < TOL, (
        f"{name}: real {r.real_net_profit} != sum-of-position-P&L {legacy}"
    )

    # cash never dips below zero anywhere in the replay
    for event, balance in replay(ledger_events(lots, flows)):
        assert balance >= -TOL, f"{name}: {event.label} overdrew to {balance}"

    print(
        f"{name}: cash ${r.cash:,.0f}  account ${r.account_value:,.0f}  "
        f"net external ${r.net_external:,.0f}  real P&L {r.real_net_profit:+,.0f}"
        + (
            f"  ({r.real_return_pct:+.1f}%)"
            if r.real_return_pct is not None else "  (n/a)"
        )
    )
    return r


def scenario_1_simple_gain() -> None:
    """Deposit $2,000. Buy A for $2,000. A grows to $2,500 (unrealized)."""
    flows = [_flow(2000, DEPOSIT, "2026-01-02", 0)]
    lots = [Lot("A", 2000.0, "2026-01-05", buy_seq=1)]
    r = _check(
        "scenario 1 (simple gain)", lots, flows, {0: 2500.0},
        cash=0.0, account=2500.0, profit=500.0,
    )
    assert abs(r.real_return_pct - 25.0) < TOL


def scenario_2_sell_hold_cash() -> None:
    """...A grows to $5,000. Sell all. Cash $5,000, profit +$3,000."""
    flows = [_flow(2000, DEPOSIT, "2026-01-02", 0)]
    lots = [
        Lot("A", 2000.0, "2026-01-05", money_out=5000.0, sell_date="2026-03-01",
            buy_seq=1, sell_seq=2),
    ]
    _check(
        "scenario 2 (sell, hold cash)", lots, flows, {},
        cash=5000.0, account=5000.0, profit=3000.0,
    )


def scenario_3_reinvest_all() -> None:
    """...Sell all. Buy B for $3,000. B drops to $500."""
    flows = [_flow(2000, DEPOSIT, "2026-01-02", 0)]
    lots = [
        Lot("A", 2000.0, "2026-01-05", money_out=5000.0, sell_date="2026-03-01",
            buy_seq=1, sell_seq=2),
        Lot("B", 3000.0, "2026-03-02", buy_seq=3),
    ]
    r = _check(
        "scenario 3 (reinvest)", lots, flows, {1: 500.0},
        cash=2000.0, account=2500.0, profit=500.0,
    )
    assert abs(r.real_return_pct - 25.0) < TOL


def scenario_4_chain_ending_in_loss() -> None:
    """The one that matters. A -> $5,000, B -> $4,000, C bought at $5,000
    drops to $1,500. Cash $1,000, account $2,500, real profit +$500.

    The brief called this the scenario that separates the new formula from the
    old one. It does not — see the identity asserted in _check — but it is the
    scenario where the *denominator* diverges hardest: sum(money_in) is
    $10,000 against $2,000 actually deposited, so the existing view reports a
    +5% return on a position the user is genuinely up 25% on.
    """
    flows = [_flow(2000, DEPOSIT, "2026-01-02", 0)]
    lots = [
        Lot("A", 2000.0, "2026-01-05", money_out=5000.0, sell_date="2026-03-01",
            buy_seq=1, sell_seq=2),
        Lot("B", 3000.0, "2026-03-02", money_out=4000.0, sell_date="2026-05-01",
            buy_seq=3, sell_seq=4),
        Lot("C", 5000.0, "2026-05-02", buy_seq=5),
    ]
    r = _check(
        "scenario 4 (chain -> loss)", lots, flows, {2: 1500.0},
        cash=1000.0, account=2500.0, profit=500.0,
    )
    assert abs(r.real_return_pct - 25.0) < TOL

    # the divergence this feature exists to fix
    legacy_total_in = sum(lot.money_in for lot in lots)
    assert legacy_total_in == 10_000.0
    assert r.net_external == 2000.0
    legacy_return = r.real_net_profit / legacy_total_in * 100.0
    assert abs(legacy_return - 5.0) < TOL
    print(
        f"  -> existing view would report Put in ${legacy_total_in:,.0f} "
        f"and {legacy_return:+.1f}% against a real {r.real_return_pct:+.1f}%"
    )


def scenario_5_partial_withdrawal() -> None:
    """...Sell all. Withdraw $1,000. Net external $1,000, profit +$3,000."""
    flows = [
        _flow(2000, DEPOSIT, "2026-01-02", 0),
        _flow(1000, WITHDRAWAL, "2026-03-05", 3),
    ]
    lots = [
        Lot("A", 2000.0, "2026-01-05", money_out=5000.0, sell_date="2026-03-01",
            buy_seq=1, sell_seq=2),
    ]
    r = _check(
        "scenario 5 (withdrawal)", lots, flows, {},
        cash=4000.0, account=4000.0, profit=3000.0,
    )
    assert r.net_external == 1000.0
    assert abs(r.real_return_pct - 300.0) < TOL


def buy_beyond_cash_auto_funds_the_shortfall() -> None:
    """A purchase larger than the cash on hand is not refused. The excess could
    only have come from outside, so it is recorded as a deposit on the buy."""
    flows = [_flow(2000, DEPOSIT, "2026-01-02", 0)]
    lots = [Lot("A", 2500.0, "2026-01-05", buy_seq=1)]
    assert validate(lots, flows) is None

    r = compute_real_pnl(lots, flows, {0: 2500.0})
    assert abs(r.auto_funded - 500.0) < TOL, r.auto_funded
    assert abs(r.total_deposited - 2500.0) < TOL   # $2,000 typed + $500 inferred
    assert abs(r.cash) < TOL
    assert abs(r.real_net_profit) < TOL            # bought at 2500, worth 2500
    print(f"buy beyond cash auto-funded ${r.auto_funded:,.0f}")


def cash_on_hand_is_spent_before_new_capital_is_invented() -> None:
    """The guarantee the old deposit-first rule bought, kept. A buy funded by
    an earlier sale must create no deposit, or reinvested profit would be
    miscounted as money from the bank — the whole point of the feature."""
    flows = [_flow(2000, DEPOSIT, "2026-01-02", 0)]
    lots = [
        Lot("A", 2000.0, "2026-01-05", money_out=5000.0, sell_date="2026-03-01",
            buy_seq=1, sell_seq=2),
        Lot("B", 3000.0, "2026-03-02", buy_seq=3),  # funded entirely by the sale
    ]
    r = compute_real_pnl(lots, flows, {1: 3000.0})
    assert r.auto_funded == 0.0, "reinvestment must not look like a deposit"
    assert abs(r.net_external - 2000.0) < TOL
    print("reinvested proceeds fund the next buy; no deposit inferred")


def withdrawal_beyond_cash_is_rejected() -> None:
    flows = [
        _flow(2000, DEPOSIT, "2026-01-02", 0),
        _flow(2500, WITHDRAWAL, "2026-01-06", 2),
    ]
    lots = [Lot("A", 2000.0, "2026-01-05", buy_seq=1)]
    err = validate(lots, flows)
    assert err is not None and "Withdrawal" in err, err
    print(f"withdrawal rejected: {err}")


def backdated_buy_funds_itself_on_its_own_date() -> None:
    """The case that drove this design: an investment bought in 2024, entered
    today. The deposit the user later typed cannot fund it — it had not
    happened yet — so the buy is funded on its own date and simply works."""
    flows = [_flow(10_000, DEPOSIT, "2026-07-10", 5)]   # dated today
    lots = [Lot("OLD", 2000.0, "2024-05-01", buy_seq=1)]
    assert validate(lots, flows) is None

    rows = [(e.date, e.kind, e.delta) for e in ledger_events(lots, flows)]
    assert rows == [
        ("2024-05-01", "autofund", 2000.0),   # inferred, dated on the buy
        ("2024-05-01", "buy", -2000.0),
        ("2026-07-10", "deposit", 10_000.0),
    ], rows

    r = compute_real_pnl(lots, flows, {0: 3000.0})
    assert abs(r.cash - 10_000.0) < TOL           # the typed deposit, uninvested
    assert abs(r.net_external - 12_000.0) < TOL   # $10,000 typed + $2,000 inferred
    assert abs(r.account_value - 13_000.0) < TOL  # cash + the position
    assert abs(r.real_net_profit - 1000.0) < TOL  # bought 2000, now worth 3000
    print("backdated buy funds itself on its own date")


def explicit_deposit_beats_inference() -> None:
    """An explicit deposit is how the user says "I transferred $10,000 in and
    left $8,000 uninvested". Without it, only the invested $2,000 would count
    as capital and the return would be flattered."""
    lots = [Lot("A", 2000.0, "2026-01-05", buy_seq=1)]

    inferred = compute_real_pnl(lots, [], {0: 2500.0})
    assert abs(inferred.net_external - 2000.0) < TOL
    assert abs(inferred.real_return_pct - 25.0) < TOL

    flows = [_flow(10_000, DEPOSIT, "2026-01-02", 0)]
    stated = compute_real_pnl(lots, flows, {0: 2500.0})
    assert stated.auto_funded == 0.0
    assert abs(stated.net_external - 10_000.0) < TOL
    assert abs(stated.cash - 8000.0) < TOL
    assert abs(stated.real_return_pct - 5.0) < TOL   # same $500 on a bigger base
    print("explicit deposit measured against the full transfer, not the buy")


def undo_sell_reassigns_the_funding_to_external_capital() -> None:
    """Reopening a sold lot withdraws its credit. A later buy it had funded is
    no longer covered by cash, so that money must have come from outside: it
    becomes an inferred deposit, raising external capital and lowering profit.
    Nothing is refused, but the reported return legitimately changes."""
    flows = [_flow(2000, DEPOSIT, "2026-01-02", 0)]
    sold = Lot("A", 2000.0, "2026-01-05", money_out=5000.0,
               sell_date="2026-03-01", buy_seq=1, sell_seq=2)
    later = Lot("B", 3000.0, "2026-03-02", buy_seq=3)
    before = compute_real_pnl([sold, later], flows, {1: 3000.0})
    assert before.auto_funded == 0.0 and before.net_external == 2000.0

    reopened = Lot("A", 2000.0, "2026-01-05", buy_seq=1)  # sale undone
    assert validate([reopened, later], flows) is None
    after = compute_real_pnl([reopened, later], flows, {0: 5000.0, 1: 3000.0})
    assert abs(after.auto_funded - 3000.0) < TOL
    assert abs(after.net_external - 5000.0) < TOL
    assert abs(after.real_net_profit - 3000.0) < TOL   # A is open and up $3,000
    print(f"undo-sell reassigned ${after.auto_funded:,.0f} to external capital")


def removing_an_open_lot_can_never_overdraw() -> None:
    """Dropping an open lot only credits its money_in back, raising every later
    balance. The validate() call on the Remove path is a defensive assertion,
    not a real gate — this pins that reasoning down."""
    flows = [_flow(2000, DEPOSIT, "2026-01-02", 0)]
    lots = [
        Lot("A", 1200.0, "2026-01-05", buy_seq=1),
        Lot("B", 800.0, "2026-01-06", buy_seq=2),
    ]
    assert validate(lots, flows) is None
    for drop in range(len(lots)):
        remaining = [l for i, l in enumerate(lots) if i != drop]
        assert validate(remaining, flows) is None
    print("removing an open lot never overdraws")


def unresolved_sale_understates_cash() -> None:
    """A lot sold in the UI carries a sell date before the worker prices it.
    Until money_out lands, the sale credits nothing — cash is understated, so
    the overdraft checks stay conservative rather than letting a buy through."""
    flows = [_flow(2000, DEPOSIT, "2026-01-02", 0)]
    pending = Lot("A", 2000.0, "2026-01-05", sell_date="2026-03-01",
                  buy_seq=1, sell_seq=2)
    r = compute_real_pnl([pending], flows, {})
    assert r.unresolved_sales == 1
    assert abs(r.cash - 0.0) < TOL, r.cash  # the $5,000 credit has not landed

    pending.money_out = 5000.0
    r = compute_real_pnl([pending], flows, {})
    assert r.unresolved_sales == 0
    assert abs(r.cash - 5000.0) < TOL
    print("unresolved sale credits nothing until priced")


def no_external_capital_has_no_return_pct() -> None:
    """Everything withdrawn: a percentage return on zero capital is meaningless,
    not infinite."""
    flows = [
        _flow(2000, DEPOSIT, "2026-01-02", 0),
        _flow(2000, WITHDRAWAL, "2026-01-03", 1),
    ]
    r = compute_real_pnl([], flows, {})
    assert r.net_external == 0.0
    assert r.real_return_pct is None
    assert r.real_net_profit == 0.0
    print("zero net external capital -> no return %")


def same_date_events_replay_in_entry_order() -> None:
    """Deposit and buy on the same day: seq decides, so the deposit funds the
    buy rather than the buy inferring capital the user has already typed in."""
    flows = [_flow(2000, DEPOSIT, "2026-01-05", 0)]
    lots = [Lot("A", 2000.0, "2026-01-05", buy_seq=1)]
    r = compute_real_pnl(lots, flows, {0: 2000.0})
    assert r.auto_funded == 0.0 and abs(r.cash) < TOL

    # entered the other way round the buy comes first, so it funds itself and
    # the deposit lands afterwards as idle cash. Nothing is lost or refused —
    # but the capital is double what the user meant, so entry order matters.
    flows = [_flow(2000, DEPOSIT, "2026-01-05", 5)]
    lots = [Lot("A", 2000.0, "2026-01-05", buy_seq=1)]
    r = compute_real_pnl(lots, flows, {0: 2000.0})
    assert abs(r.auto_funded - 2000.0) < TOL
    assert abs(r.cash - 2000.0) < TOL
    assert abs(r.net_external - 4000.0) < TOL
    print("same-date events replay in entry order")


def scenarios_reproduce_with_no_deposits_at_all() -> None:
    """Auto-funding must not change a single headline number. Scenario 4 with
    every deposit removed lands on the same cash, account value and profit —
    proof that inference reconstructs exactly the capital the user would have
    typed, and that reinvestment still creates none."""
    lots = [
        Lot("A", 2000.0, "2026-01-05", money_out=5000.0, sell_date="2026-03-01",
            buy_seq=1, sell_seq=2),
        Lot("B", 3000.0, "2026-03-02", money_out=4000.0, sell_date="2026-05-01",
            buy_seq=3, sell_seq=4),
        Lot("C", 5000.0, "2026-05-02", buy_seq=5),
    ]
    r = _check(
        "scenario 4, no deposits entered", lots, [], {2: 1500.0},
        cash=1000.0, account=2500.0, profit=500.0,
    )
    assert abs(r.auto_funded - 2000.0) < TOL, r.auto_funded
    assert abs(r.net_external - 2000.0) < TOL
    assert abs(r.real_return_pct - 25.0) < TOL
    print("  -> only the first $2,000 was inferred; both reinvestments were free")


def two_return_bases_diverge_on_reinvestment_loss() -> None:
    """Same profit, two denominators. A profit-flagged position that loses drags
    the overall return down but not the return on the user's own money, because
    that money was never their capital to lose."""
    # deposit $2,000; A bought with own money, sold for $5,000; B bought with
    # $3,000 of profit and flagged from_profit; B falls to $500
    flows = [_flow(2000, DEPOSIT, "2026-01-02", 0)]
    lots = [
        Lot("A", 2000.0, "2026-01-05", money_out=5000.0, sell_date="2026-03-01",
            buy_seq=1, sell_seq=2),
        Lot("B", 3000.0, "2026-03-02", buy_seq=3, from_profit=True),
    ]
    r = compute_real_pnl(lots, flows, {1: 500.0})

    assert abs(r.real_net_profit - 500.0) < TOL       # same profit both ways
    assert abs(r.total_invested - 5000.0) < TOL
    assert abs(r.own_invested - 2000.0) < TOL         # B excluded by its flag
    assert abs(r.overall_return_pct - 10.0) < TOL     # 500 / 5000
    assert abs(r.own_money_return_pct - 25.0) < TOL   # 500 / 2000
    assert r.has_profit_positions
    print(
        f"two bases OK: overall {r.overall_return_pct:.0f}% "
        f"vs own-money {r.own_money_return_pct:.0f}% on the same +$500"
    )


def own_money_return_absent_without_profit_flags() -> None:
    """With nothing flagged, the two bases coincide and the own-money view has
    nothing extra to say — the UI hides it."""
    flows = [_flow(2000, DEPOSIT, "2026-01-02", 0)]
    lots = [Lot("A", 2000.0, "2026-01-05", buy_seq=1)]
    r = compute_real_pnl(lots, flows, {0: 2500.0})
    assert not r.has_profit_positions
    assert abs(r.total_invested - r.own_invested) < TOL
    assert abs(r.overall_return_pct - r.own_money_return_pct) < TOL
    print("no profit flags -> both bases identical")


def funding_splits_own_from_recycled_proceeds() -> None:
    """Each purchase splits into fresh external money and recycled proceeds,
    with no fungibility convention. Proceeds are spent before idle cash."""
    # Scenario 4, no deposits: only A is external; B and C ride on proceeds
    lots = [
        Lot("A", 2000.0, "2026-01-05", money_out=5000.0, sell_date="2026-03-01",
            buy_seq=1, sell_seq=2),
        Lot("B", 3000.0, "2026-03-02", money_out=4000.0, sell_date="2026-05-01",
            buy_seq=3, sell_seq=4),
        Lot("C", 5000.0, "2026-05-02", buy_seq=5),
    ]
    fb = funding_breakdown(lots, [])
    assert (fb[0].from_own, fb[0].from_proceeds) == (2000.0, 0.0)
    assert (fb[1].from_own, fb[1].from_proceeds) == (0.0, 3000.0)
    assert (fb[2].from_own, fb[2].from_proceeds) == (0.0, 5000.0)
    assert fb[0].source == "Own" and fb[1].source == "Proceeds"
    reinvested = sum(f.from_proceeds for f in fb.values())
    assert reinvested == 8000.0, reinvested
    # each lot's split sums to its cost
    for f in fb.values():
        assert abs(f.from_own + f.from_proceeds - f.cost) < TOL
    print(f"funding split OK: ${reinvested:,.0f} reinvested, $2,000 external")


def idle_deposit_is_own_money_not_proceeds() -> None:
    """Spending cash you deposited but never invested must read as your money,
    even though it sits in the same pool sales would land in."""
    flows = [_flow(10_000, DEPOSIT, "2026-01-02", 0)]
    lots = [Lot("A", 2000.0, "2026-01-05", buy_seq=1)]
    fb = funding_breakdown(lots, flows)
    assert (fb[0].from_own, fb[0].from_proceeds) == (2000.0, 0.0)
    assert fb[0].source == "Own"

    # a mixed buy: $500 idle left, rest from a $3,000 sale
    lots = [
        Lot("A", 9500.0, "2026-01-05", money_out=3000.0, sell_date="2026-02-01",
            buy_seq=1, sell_seq=2),
        Lot("B", 3200.0, "2026-02-02", buy_seq=3),
    ]
    fb = funding_breakdown(lots, flows)
    # after A: idle 500 (10000 - 9500), then A sells -> proceeds 3000.
    # B $3,200 spends 3000 proceeds first, then 200 idle.
    assert (fb[1].from_proceeds, fb[1].from_own) == (3000.0, 200.0)
    assert fb[1].source == "Mixed"
    print("idle deposit reads as own money; proceeds spent first")


def ledger_rows_read_as_a_statement() -> None:
    """The transactions table in the UI renders replay() directly, so the event
    order, titles and running balance are load-bearing, not cosmetic."""
    flows = [
        _flow(6000, DEPOSIT, "2026-01-02", 0),
        _flow(1000, WITHDRAWAL, "2026-03-03", 4),
    ]
    lots = [
        Lot("AAPL", 2000.0, "2026-01-05", money_out=5000.0, sell_date="2026-03-01",
            buy_seq=1, sell_seq=2),
        Lot("NVDA", 5000.0, "2026-03-02", buy_seq=3),
    ]
    rows = [
        (e.date, e.title, e.delta, bal)
        for e, bal in replay(ledger_events(lots, flows))
    ]
    assert rows == [
        ("2026-01-02", "Deposit",    6000.0, 6000.0),
        ("2026-01-05", "Buy AAPL",  -2000.0, 4000.0),
        ("2026-03-01", "Sell AAPL",  5000.0, 9000.0),
        ("2026-03-02", "Buy NVDA",  -5000.0, 4000.0),
        ("2026-03-03", "Withdrawal", -1000.0, 3000.0),
    ], rows
    # the statement's closing balance is the cash balance, by construction
    assert abs(rows[-1][3] - current_cash(lots, flows)) < TOL
    print("ledger statement reads in order and closes on the cash balance")


def document_roundtrip_and_v2_upgrade() -> None:
    """Cash flows survive a save/load, and a v2 file — which has no cash_flows
    key and no seq fields — loads with the seqs defaulted from list position."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "investments.json"
        lots = [
            Lot("A", 2000.0, "2026-01-05", money_out=5000.0,
                sell_date="2026-03-01", buy_seq=1, sell_seq=2),
            Lot("B", 3000.0, "2026-03-02", buy_seq=3),
        ]
        flows = [_flow(2000, DEPOSIT, "2026-01-02", 0)]
        save_investments(lots, p, cash_flows=flows)
        got_lots, got_flows = load_investments_doc(p)
        assert got_lots == lots
        assert got_flows == flows
        assert json.loads(p.read_text())["version"] == INVESTMENTS_VERSION

        # a v2 document: no cash_flows, no seq
        p.write_text(json.dumps({
            "version": 2,
            "lots": [
                {"ticker": "AAPL", "money_in": 1000.0, "buy_date": "2024-05-01",
                 "money_out": None, "sell_date": None, "strategy": None},
                {"ticker": "SHOP", "money_in": 750.0, "buy_date": "2023-11-20",
                 "money_out": 900.0, "sell_date": "2025-02-14", "strategy": None},
            ],
        }), encoding="utf-8")
        got_lots, got_flows = load_investments_doc(p)
        assert got_flows == []
        assert [l.buy_seq for l in got_lots] == [0, 1]
        assert got_lots[1].sell_seq is None
        # a v2 file records no cash flows at all, but auto-funding infers the
        # capital each buy needed, so a legacy portfolio reports real numbers
        # instead of a meaningless zero base
        r = compute_real_pnl(got_lots, got_flows, {0: 1200.0})
        assert abs(r.auto_funded - 1750.0) < TOL     # AAPL $1,000 + SHOP $750
        assert abs(r.net_external - 1750.0) < TOL
        assert abs(r.cash - 900.0) < TOL             # SHOP's proceeds
        assert abs(r.account_value - 2100.0) < TOL   # cash + AAPL at $1,200
        assert abs(r.real_net_profit - 350.0) < TOL  # +$200 AAPL, +$150 SHOP
    print("document round-trip OK; v2 files load and infer their capital")


SCENARIOS = (
    scenario_1_simple_gain,
    scenario_2_sell_hold_cash,
    scenario_3_reinvest_all,
    scenario_4_chain_ending_in_loss,
    scenario_5_partial_withdrawal,
    scenarios_reproduce_with_no_deposits_at_all,
)

NEGATIVE_PATHS = (
    buy_beyond_cash_auto_funds_the_shortfall,
    cash_on_hand_is_spent_before_new_capital_is_invented,
    withdrawal_beyond_cash_is_rejected,
    backdated_buy_funds_itself_on_its_own_date,
    explicit_deposit_beats_inference,
    undo_sell_reassigns_the_funding_to_external_capital,
    removing_an_open_lot_can_never_overdraw,
    unresolved_sale_understates_cash,
    no_external_capital_has_no_return_pct,
    same_date_events_replay_in_entry_order,
    two_return_bases_diverge_on_reinvestment_loss,
    own_money_return_absent_without_profit_flags,
    funding_splits_own_from_recycled_proceeds,
    idle_deposit_is_own_money_not_proceeds,
    ledger_rows_read_as_a_statement,
    document_roundtrip_and_v2_upgrade,
)


def run_scenarios() -> None:
    """The five validation scenarios alone. Imported by main.py's --selftest,
    so a frozen build fails if the Real Investment maths ever breaks.

    Output is swallowed: a --windowed PyInstaller exe has no stdout, and a bare
    print() there raises on None. The assertions are the contract; the printing
    is for a human running the module directly.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        for scenario in SCENARIOS:
            scenario()


def main() -> None:
    print("--- the five scenarios ---")
    for scenario in SCENARIOS:
        scenario()
    print("\n--- negative paths ---")
    for check in NEGATIVE_PATHS:
        check()
    print("\nphase 11 OK")


if __name__ == "__main__":
    main()
