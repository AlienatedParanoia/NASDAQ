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


def buy_beyond_cash_is_rejected() -> None:
    flows = [_flow(2000, DEPOSIT, "2026-01-02", 0)]
    lots = [Lot("A", 2500.0, "2026-01-05", buy_seq=1)]
    err = validate(lots, flows)
    assert err is not None and "Buy A" in err and "$500.00" in err, err
    print(f"overdraft rejected: {err}")


def withdrawal_beyond_cash_is_rejected() -> None:
    flows = [
        _flow(2000, DEPOSIT, "2026-01-02", 0),
        _flow(2500, WITHDRAWAL, "2026-01-06", 2),
    ]
    lots = [Lot("A", 2000.0, "2026-01-05", buy_seq=1)]
    err = validate(lots, flows)
    assert err is not None and "Withdrawal" in err, err
    print(f"withdrawal rejected: {err}")


def backdated_buy_names_the_offending_event() -> None:
    """A buy inserted before its funding deposit must be refused, and the
    message must name the buy — not some later, innocent transaction."""
    flows = [_flow(2000, DEPOSIT, "2026-03-01", 0)]
    lots = [Lot("EARLY", 500.0, "2026-02-01", buy_seq=1)]
    err = validate(lots, flows)
    assert err is not None and "Buy EARLY" in err and "2026-02-01" in err, err

    # the same buy after the deposit is fine
    lots = [Lot("EARLY", 500.0, "2026-03-02", buy_seq=1)]
    assert validate(lots, flows) is None
    print("backdated buy rejected and named correctly")


def undo_sell_that_funds_a_later_buy_is_rejected() -> None:
    """Reopening a sold lot withdraws its credit, lowering every later balance.
    This is the one action whose replay check is not vacuous."""
    flows = [_flow(2000, DEPOSIT, "2026-01-02", 0)]
    sold = Lot("A", 2000.0, "2026-01-05", money_out=5000.0,
               sell_date="2026-03-01", buy_seq=1, sell_seq=2)
    later = Lot("B", 3000.0, "2026-03-02", buy_seq=3)
    assert validate([sold, later], flows) is None

    reopened = Lot("A", 2000.0, "2026-01-05", buy_seq=1)  # sale undone
    err = validate([reopened, later], flows)
    assert err is not None and "Buy B" in err, err
    print(f"undo-sell rejected: {err}")


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
    buy rather than the buy overdrawing an account it is about to fill."""
    flows = [_flow(2000, DEPOSIT, "2026-01-05", 0)]
    lots = [Lot("A", 2000.0, "2026-01-05", buy_seq=1)]
    assert validate(lots, flows) is None
    assert abs(current_cash(lots, flows)) < TOL

    # entered the other way round, the buy comes first and overdraws
    flows = [_flow(2000, DEPOSIT, "2026-01-05", 5)]
    lots = [Lot("A", 2000.0, "2026-01-05", buy_seq=1)]
    assert validate(lots, flows) is not None
    print("same-date events replay in entry order")


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
        # no deposits recorded, so a legacy portfolio reads as fully "profit"
        # until the user logs what they actually put in — it does not crash
        r = compute_real_pnl(got_lots, got_flows, {0: 1200.0})
        assert r.net_external == 0.0 and r.real_return_pct is None
    print("document round-trip OK; v2 files load with defaulted seqs")


SCENARIOS = (
    scenario_1_simple_gain,
    scenario_2_sell_hold_cash,
    scenario_3_reinvest_all,
    scenario_4_chain_ending_in_loss,
    scenario_5_partial_withdrawal,
)

NEGATIVE_PATHS = (
    buy_beyond_cash_is_rejected,
    withdrawal_beyond_cash_is_rejected,
    backdated_buy_names_the_offending_event,
    undo_sell_that_funds_a_later_buy_is_rejected,
    removing_an_open_lot_can_never_overdraw,
    unresolved_sale_understates_cash,
    no_external_capital_has_no_return_pct,
    same_date_events_replay_in_entry_order,
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
