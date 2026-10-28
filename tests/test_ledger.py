"""The ledger itself: append-only semantics and derived state. Offline."""
from __future__ import annotations

import pandas as pd
import pytest

from engine.ledger import (
    BACKTEST,
    DEPOSIT,
    DIVIDEND,
    FORWARD,
    SPLIT,
    Ledger,
    LedgerError,
)
from engine.orders import Fill, Order, OrderStatus, Side

D1 = pd.Timestamp("2026-01-05")
D2 = pd.Timestamp("2026-01-06")


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(tmp_path / "ledger.db")
    yield led
    led.close()


@pytest.fixture
def account(ledger):
    return ledger.create_account("bot", 10_000.0)


def buy(ledger, account, qty=10, price=100.0, session=D1, ticker="AAPL",
        commission=0.0) -> Fill:
    order = ledger.record_order(
        Order(account_id=account.id, created_session=session, ticker=ticker,
              side=Side.BUY, qty=qty)
    )
    return ledger.record_fill(
        Fill(order_id=order.id, account_id=account.id, session=session,
             ticker=ticker, side=Side.BUY, qty=qty, price=price,
             commission=commission, slippage_cost=0.0)
    )


def sell(ledger, account, qty=10, price=100.0, session=D2, ticker="AAPL",
         commission=0.0) -> Fill:
    order = ledger.record_order(
        Order(account_id=account.id, created_session=session, ticker=ticker,
              side=Side.SELL, qty=qty)
    )
    return ledger.record_fill(
        Fill(order_id=order.id, account_id=account.id, session=session,
             ticker=ticker, side=Side.SELL, qty=qty, price=price,
             commission=commission, slippage_cost=0.0)
    )


class TestAccounts:
    def test_starting_cash_is_recorded_as_a_cash_event(self, ledger, account):
        """So rebuild_cash needs no special case for the opening balance."""
        events = ledger.cash_events(account.id)
        assert len(events) == 1
        assert events[0]["kind"] == DEPOSIT
        assert events[0]["amount"] == pytest.approx(10_000.0)
        assert ledger.rebuild_cash(account.id) == pytest.approx(10_000.0)

    def test_account_names_are_unique(self, ledger, account):
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            ledger.create_account("bot", 1_000.0)

    def test_backtest_and_forward_accounts_are_distinguishable(self, ledger):
        """A replay must never be readable as live paper performance."""
        ledger.create_account("fwd", 1_000.0, kind=FORWARD)
        ledger.create_account("bt", 1_000.0, kind=BACKTEST)
        assert [a.name for a in ledger.accounts(kind=FORWARD)] == ["fwd"]
        assert [a.name for a in ledger.accounts(kind=BACKTEST)] == ["bt"]

    def test_an_unknown_kind_is_refused(self, ledger):
        with pytest.raises(LedgerError):
            ledger.create_account("weird", 1_000.0, kind="live")

    def test_negative_starting_cash_is_refused(self, ledger):
        with pytest.raises(LedgerError):
            ledger.create_account("broke", -5.0)

    def test_status_can_be_paused_and_killed(self, ledger, account):
        ledger.set_account_status(account.id, "paused")
        assert ledger.account_status(account.id) == "paused"
        ledger.set_account_status(account.id, "killed")
        assert ledger.account_status(account.id) == "killed"

    def test_an_unknown_status_is_refused(self, ledger, account):
        with pytest.raises(LedgerError):
            ledger.set_account_status(account.id, "sleepy")


class TestAppendOnlyOrders:
    def test_status_is_the_newest_event_not_a_column(self, ledger, account):
        order = ledger.record_order(
            Order(account_id=account.id, created_session=D1, ticker="AAPL",
                  side=Side.BUY, qty=5)
        )
        assert ledger.get_order(order.id).status is OrderStatus.PENDING
        ledger.record_order_event(order.id, OrderStatus.FILLED, session=D2)
        assert ledger.get_order(order.id).status is OrderStatus.FILLED

    def test_the_whole_history_survives(self, ledger, account):
        """"Why didn't my bot buy?" must be answerable months later."""
        order = ledger.record_order(
            Order(account_id=account.id, created_session=D1, ticker="AAPL",
                  side=Side.BUY, qty=5)
        )
        ledger.record_order_event(
            order.id, OrderStatus.REJECTED, session=D2, reject_reason="not enough cash"
        )
        history = ledger.order_history(order.id)
        assert [h["status"] for h in history] == ["pending", "rejected"]
        assert history[-1]["reject_reason"] == "not enough cash"

    def test_open_orders_excludes_terminal_ones(self, ledger, account):
        a = ledger.record_order(Order(account_id=account.id, created_session=D1,
                                      ticker="AAPL", side=Side.BUY, qty=1))
        b = ledger.record_order(Order(account_id=account.id, created_session=D1,
                                      ticker="MSFT", side=Side.BUY, qty=1))
        ledger.record_order_event(a.id, OrderStatus.CANCELLED)
        assert [o.id for o in ledger.open_orders(account.id)] == [b.id]


class TestDerivedPositions:
    def test_a_buy_creates_a_position_at_its_price(self, ledger, account):
        buy(ledger, account, 10, 100.0)
        position = ledger.rebuild_positions(account.id)["AAPL"]
        assert position.qty == 10
        assert position.avg_cost == pytest.approx(100.0)

    def test_commission_is_capitalised_into_the_basis(self, ledger, account):
        buy(ledger, account, 10, 100.0, commission=5.0)
        position = ledger.rebuild_positions(account.id)["AAPL"]
        assert position.avg_cost == pytest.approx(100.5)

    def test_averaging_up_blends_the_cost(self, ledger, account):
        buy(ledger, account, 10, 100.0, session=D1)
        buy(ledger, account, 10, 200.0, session=D2)
        position = ledger.rebuild_positions(account.id)["AAPL"]
        assert position.qty == 20
        assert position.avg_cost == pytest.approx(150.0)

    def test_a_sale_leaves_the_average_cost_of_the_remainder_untouched(
        self, ledger, account
    ):
        buy(ledger, account, 10, 100.0, session=D1)
        sell(ledger, account, 4, 300.0, session=D2)
        position = ledger.rebuild_positions(account.id)["AAPL"]
        assert position.qty == 6
        assert position.avg_cost == pytest.approx(100.0), (
            "realised profit belongs in cash, not in the basis of what is left"
        )

    def test_a_fully_closed_position_disappears(self, ledger, account):
        buy(ledger, account, 10, 100.0, session=D1)
        sell(ledger, account, 10, 120.0, session=D2)
        assert "AAPL" not in ledger.rebuild_positions(account.id)

    def test_a_split_multiplies_shares_and_preserves_the_basis(self, ledger, account):
        buy(ledger, account, 10, 100.0, session=D1)
        before = ledger.rebuild_positions(account.id)["AAPL"]
        ledger.record_corporate_action(account.id, D2, "AAPL", SPLIT, 2.0,
                                       shares_before=10, shares_after=20)
        after = ledger.rebuild_positions(account.id)["AAPL"]
        assert after.qty == pytest.approx(20)
        assert after.avg_cost == pytest.approx(50.0)
        assert after.cost_basis == pytest.approx(before.cost_basis)

    def test_positions_are_per_ticker(self, ledger, account):
        buy(ledger, account, 10, 100.0, ticker="AAPL")
        buy(ledger, account, 5, 50.0, ticker="MSFT")
        positions = ledger.rebuild_positions(account.id)
        assert positions["AAPL"].qty == 10 and positions["MSFT"].qty == 5


class TestDerivedCash:
    def test_cash_is_the_sum_of_its_events(self, ledger, account):
        buy(ledger, account, 10, 100.0, commission=1.0)
        assert ledger.rebuild_cash(account.id) == pytest.approx(10_000 - 1000 - 1)

    def test_a_sale_returns_cash(self, ledger, account):
        buy(ledger, account, 10, 100.0)
        sell(ledger, account, 10, 150.0)
        assert ledger.rebuild_cash(account.id) == pytest.approx(10_000 - 1000 + 1500)

    def test_a_dividend_credits_cash(self, ledger, account):
        ledger.record_cash_event(account.id, DIVIDEND, 25.0, session=D2)
        assert ledger.rebuild_cash(account.id) == pytest.approx(10_025.0)

    def test_realised_pnl_ignores_the_opening_deposit(self, ledger, account):
        buy(ledger, account, 10, 100.0)
        sell(ledger, account, 10, 130.0)
        assert ledger.realised_pnl(account.id) == pytest.approx(300.0)


class TestSnapshots:
    def test_snapshots_round_trip(self, ledger, account):
        buy(ledger, account, 10, 100.0)
        positions = ledger.rebuild_positions(account.id)
        ledger.write_snapshot(account.id, D2, 9_000.0, 10_200.0, positions)
        snaps = ledger.snapshots(account.id)
        assert len(snaps) == 1
        assert snaps[0].cash == pytest.approx(9_000.0)
        assert snaps[0].equity == pytest.approx(10_200.0)
        assert snaps[0].positions == {"AAPL": 10}

    def test_rewriting_a_session_replaces_rather_than_duplicates(self, ledger, account):
        """Re-running a session must be idempotent (phase 5 depends on it)."""
        ledger.write_snapshot(account.id, D2, 1.0, 2.0, {})
        ledger.write_snapshot(account.id, D2, 3.0, 4.0, {})
        snaps = ledger.snapshots(account.id)
        assert len(snaps) == 1 and snaps[0].equity == pytest.approx(4.0)

    def test_equity_curve_is_a_time_series(self, ledger, account):
        ledger.write_snapshot(account.id, D1, 1.0, 100.0, {})
        ledger.write_snapshot(account.id, D2, 1.0, 110.0, {})
        curve = ledger.equity_curve(account.id)
        assert list(curve) == pytest.approx([100.0, 110.0])
        assert isinstance(curve.index, pd.DatetimeIndex)

    def test_an_empty_account_has_an_empty_curve(self, ledger, account):
        assert ledger.equity_curve(account.id).empty


class TestSessionCursor:
    def test_it_starts_unset_and_then_advances(self, ledger, account):
        assert ledger.last_processed_session(account.id) is None
        ledger.set_last_processed_session(account.id, D1)
        assert ledger.last_processed_session(account.id) == D1
        ledger.set_last_processed_session(account.id, D2)
        assert ledger.last_processed_session(account.id) == D2


class TestHeldTickers:
    def test_it_reports_what_is_actually_held(self, ledger, account):
        buy(ledger, account, 10, 100.0, ticker="AAPL")
        buy(ledger, account, 5, 50.0, ticker="MSFT")
        assert ledger.held_tickers() == {"AAPL", "MSFT"}

    def test_a_closed_position_is_not_held(self, ledger, account):
        buy(ledger, account, 10, 100.0, ticker="AAPL", session=D1)
        sell(ledger, account, 10, 110.0, ticker="AAPL", session=D2)
        assert ledger.held_tickers() == set()

    def test_backtest_accounts_do_not_drive_the_forward_fetch_list(self, ledger):
        bt = ledger.create_account("bt", 1_000.0, kind=BACKTEST)
        buy(ledger, bt, 1, 10.0, ticker="ZZZZ")
        assert "ZZZZ" not in ledger.held_tickers()
        assert "ZZZZ" in ledger.held_tickers(kinds=(BACKTEST,))


class TestIdempotency:
    def test_applied_actions_are_reported_per_session(self, ledger, account):
        assert ledger.applied_action_ids(account.id, D2) == set()
        ledger.record_corporate_action(account.id, D2, "AAPL", SPLIT, 2.0)
        assert ledger.applied_action_ids(account.id, D2) == {("AAPL", SPLIT)}
        assert ledger.applied_action_ids(account.id, D1) == set()


class TestPersistence:
    def test_state_survives_reopening_the_file(self, tmp_path):
        path = tmp_path / "ledger.db"
        with Ledger(path) as led:
            acct = led.create_account("bot", 10_000.0)
            buy(led, acct, 10, 100.0)
            cash, positions = led.rebuild_cash(acct.id), led.rebuild_positions(acct.id)

        with Ledger(path) as reopened:
            assert reopened.rebuild_cash(acct.id) == pytest.approx(cash)
            rebuilt = reopened.rebuild_positions(acct.id)
            assert rebuilt["AAPL"].qty == positions["AAPL"].qty
            assert rebuilt["AAPL"].avg_cost == pytest.approx(positions["AAPL"].avg_cost)
