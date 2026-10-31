"""SimBroker: fill rules, cash and share invariants, corporate actions.

Offline, synthetic bars. Every done-condition the phase 3 spec lists has a
named test here. Costs are zeroed in most tests so a fill-price assertion is
about the fill rule and not about commission arithmetic; the cost model gets
its own section.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from engine.costs import ZERO_COSTS, CostModel
from engine.ledger import DIVIDEND, SPLIT, Ledger
from engine.orders import (
    Order,
    OrderError,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from engine.sim_broker import Bar, SimBroker

D1 = pd.Timestamp("2026-01-05")
D2 = pd.Timestamp("2026-01-06")
D3 = pd.Timestamp("2026-01-07")
D4 = pd.Timestamp("2026-01-08")


@dataclass(frozen=True)
class Action:
    """Stands in for data.fetcher.CorporateAction (structural typing)."""

    ticker: str
    ex_date: pd.Timestamp
    type: str
    value: float


def bar(open_, high, low, close, volume=1_000_000.0) -> Bar:
    return Bar(open=open_, high=high, low=low, close=close, volume=volume)


def flat(price: float) -> Bar:
    return bar(price, price, price, price)


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(tmp_path / "ledger.db")
    yield led
    led.close()


@pytest.fixture
def broker(ledger):
    account = ledger.create_account("test-bot", 10_000.0)
    b = SimBroker(ledger, account.id, costs=ZERO_COSTS)
    b.set_mark("AAPL", 100.0)
    return b


def submit(broker, side=Side.BUY, qty=10, created=D1, **kw) -> Order:
    return broker.submit(
        Order(
            account_id=broker.account_id, created_session=created, ticker="AAPL",
            side=side, qty=qty, **kw,
        )
    )


def hold(broker, qty=10, price=100.0, session=D2):
    """Give the account a position by filling a market buy."""
    submit(broker, Side.BUY, qty, created=D1)
    broker.process_session(session, {"AAPL": flat(price)})


# ------------------------------------------------------------------ look-ahead


class TestNoLookAhead:
    def test_an_order_cannot_fill_on_the_session_it_was_decided(self, broker):
        order = submit(broker, created=D2)
        result = broker.process_session(D2, {"AAPL": flat(100.0)})
        assert result.fills == ()
        assert broker.ledger.get_order(order.id).status is OrderStatus.PENDING

    def test_it_fills_on_the_next_session(self, broker):
        order = submit(broker, created=D1, tif=TimeInForce.GTC)
        broker.process_session(D1, {"AAPL": flat(100.0)})
        result = broker.process_session(D2, {"AAPL": flat(100.0)})
        assert len(result.fills) == 1
        assert broker.ledger.get_order(order.id).status is OrderStatus.FILLED


# ------------------------------------------------------------------ fill rules


class TestMarketOrders:
    def test_market_buy_fills_at_the_open_with_slippage(self, ledger):
        account = ledger.create_account("slip", 10_000.0)
        b = SimBroker(ledger, account.id, costs=CostModel(0, 0, 0, slippage_bps=5.0))
        b.set_mark("AAPL", 100.0)
        submit(b, Side.BUY, 10)
        result = b.process_session(D2, {"AAPL": bar(100.0, 110.0, 95.0, 105.0)})
        # 5 bps against us: 100 * 1.0005
        assert result.fills[0].price == pytest.approx(100.05)

    def test_market_sell_fills_at_the_open_less_slippage(self, ledger):
        account = ledger.create_account("slip2", 10_000.0)
        b = SimBroker(ledger, account.id, costs=CostModel(0, 0, 0, slippage_bps=5.0))
        b.set_mark("AAPL", 100.0)
        submit(b, Side.BUY, 10)
        b.process_session(D2, {"AAPL": flat(100.0)})
        submit(b, Side.SELL, 10, created=D2)
        result = b.process_session(D3, {"AAPL": bar(200.0, 210.0, 190.0, 205.0)})
        assert result.fills[0].price == pytest.approx(200.0 * 0.9995)

    def test_commission_is_charged_and_recorded(self, ledger):
        account = ledger.create_account("comm", 10_000.0)
        b = SimBroker(ledger, account.id, costs=CostModel(0.0, 0.005, 1.00, 0.0))
        b.set_mark("AAPL", 100.0)
        submit(b, Side.BUY, 10)
        result = b.process_session(D2, {"AAPL": flat(100.0)})
        fill = result.fills[0]
        assert fill.commission == pytest.approx(1.00)  # minimum beats 10*0.005
        assert b.cash() == pytest.approx(10_000.0 - 1000.0 - 1.00)


class TestLimitBuy:
    def test_fills_at_the_open_when_the_open_gaps_below_the_limit(self, broker):
        submit(broker, Side.BUY, 10, type=OrderType.LIMIT, limit_price=100.0)
        result = broker.process_session(D2, {"AAPL": bar(95.0, 99.0, 94.0, 98.0)})
        assert result.fills[0].price == pytest.approx(95.0), "price improvement kept"

    def test_fills_at_the_limit_when_only_the_low_touches(self, broker):
        submit(broker, Side.BUY, 10, type=OrderType.LIMIT, limit_price=100.0)
        result = broker.process_session(D2, {"AAPL": bar(105.0, 106.0, 99.0, 104.0)})
        assert result.fills[0].price == pytest.approx(100.0)

    def test_does_not_fill_when_the_low_stays_above_the_limit(self, broker):
        order = submit(
            broker, Side.BUY, 10, type=OrderType.LIMIT, limit_price=100.0,
            tif=TimeInForce.GTC,
        )
        result = broker.process_session(D2, {"AAPL": bar(105.0, 106.0, 101.0, 104.0)})
        assert result.fills == ()
        assert broker.ledger.get_order(order.id).status is OrderStatus.PENDING

    def test_limit_fills_take_no_slippage(self, ledger):
        account = ledger.create_account("nolimslip", 10_000.0)
        b = SimBroker(ledger, account.id, costs=CostModel(0, 0, 0, slippage_bps=50.0))
        b.set_mark("AAPL", 100.0)
        submit(b, Side.BUY, 10, type=OrderType.LIMIT, limit_price=100.0)
        result = b.process_session(D2, {"AAPL": bar(105.0, 106.0, 99.0, 104.0)})
        assert result.fills[0].price == pytest.approx(100.0), "a limit fills at its price"
        assert result.fills[0].slippage_cost == pytest.approx(0.0)


class TestLimitSell:
    def test_fills_at_the_open_when_the_open_gaps_above_the_limit(self, broker):
        hold(broker, 10, 100.0)
        submit(broker, Side.SELL, 10, created=D2, type=OrderType.LIMIT, limit_price=110.0)
        result = broker.process_session(D3, {"AAPL": bar(120.0, 125.0, 118.0, 122.0)})
        assert result.fills[0].price == pytest.approx(120.0)

    def test_fills_at_the_limit_when_only_the_high_touches(self, broker):
        hold(broker, 10, 100.0)
        submit(broker, Side.SELL, 10, created=D2, type=OrderType.LIMIT, limit_price=110.0)
        result = broker.process_session(D3, {"AAPL": bar(105.0, 112.0, 104.0, 106.0)})
        assert result.fills[0].price == pytest.approx(110.0)

    def test_does_not_fill_when_the_high_stays_below(self, broker):
        hold(broker, 10, 100.0)
        o = submit(
            broker, Side.SELL, 10, created=D2, type=OrderType.LIMIT,
            limit_price=110.0, tif=TimeInForce.GTC,
        )
        result = broker.process_session(D3, {"AAPL": bar(105.0, 108.0, 104.0, 106.0)})
        assert result.fills == ()
        assert broker.ledger.get_order(o.id).status is OrderStatus.PENDING


class TestStopOrders:
    def test_stop_sell_fills_at_the_open_when_the_open_is_already_below(self, broker):
        hold(broker, 10, 100.0)
        submit(broker, Side.SELL, 10, created=D2, type=OrderType.STOP, stop_price=90.0)
        result = broker.process_session(D3, {"AAPL": bar(85.0, 88.0, 84.0, 86.0)})
        assert result.fills[0].price == pytest.approx(85.0)

    def test_stop_sell_fills_at_the_stop_when_only_the_low_reaches_it(self, broker):
        hold(broker, 10, 100.0)
        submit(broker, Side.SELL, 10, created=D2, type=OrderType.STOP, stop_price=90.0)
        result = broker.process_session(D3, {"AAPL": bar(95.0, 96.0, 89.0, 94.0)})
        assert result.fills[0].price == pytest.approx(90.0)

    def test_stop_sell_does_not_fill_above_the_stop(self, broker):
        hold(broker, 10, 100.0)
        o = submit(
            broker, Side.SELL, 10, created=D2, type=OrderType.STOP,
            stop_price=90.0, tif=TimeInForce.GTC,
        )
        result = broker.process_session(D3, {"AAPL": bar(95.0, 96.0, 91.0, 94.0)})
        assert result.fills == ()
        assert broker.ledger.get_order(o.id).status is OrderStatus.PENDING

    def test_stop_buy_fills_at_the_open_when_the_open_is_already_above(self, broker):
        submit(broker, Side.BUY, 10, type=OrderType.STOP, stop_price=110.0)
        result = broker.process_session(D2, {"AAPL": bar(115.0, 118.0, 114.0, 116.0)})
        assert result.fills[0].price == pytest.approx(115.0)

    def test_stop_buy_fills_at_the_stop_when_only_the_high_reaches_it(self, broker):
        submit(broker, Side.BUY, 10, type=OrderType.STOP, stop_price=110.0)
        result = broker.process_session(D2, {"AAPL": bar(105.0, 112.0, 104.0, 106.0)})
        assert result.fills[0].price == pytest.approx(110.0)

    def test_a_market_fill_records_what_slippage_cost(self, ledger):
        """The blotter must be able to show the drag, not just the price."""
        account = ledger.create_account("slipcost", 10_000.0)
        b = SimBroker(ledger, account.id, costs=CostModel(0, 0, 0, slippage_bps=10.0))
        b.set_mark("AAPL", 100.0)
        submit(b, Side.BUY, 10)
        fill = b.process_session(D2, {"AAPL": flat(100.0)}).fills[0]
        # paid 100.10 instead of 100.00, on 10 shares
        assert fill.price == pytest.approx(100.10)
        assert fill.slippage_cost == pytest.approx(1.0)

    def test_stop_fills_take_slippage(self, ledger):
        account = ledger.create_account("stopslip", 10_000.0)
        b = SimBroker(ledger, account.id, costs=CostModel(0, 0, 0, slippage_bps=10.0))
        b.set_mark("AAPL", 100.0)
        submit(b, Side.BUY, 10, type=OrderType.STOP, stop_price=110.0)
        result = b.process_session(D2, {"AAPL": bar(105.0, 112.0, 104.0, 106.0)})
        # a triggered stop becomes a market order, so it pays slippage
        assert result.fills[0].price == pytest.approx(110.0 * 1.001)


# -------------------------------------------------------------- time in force


class TestTimeInForce:
    def test_a_day_order_expires_after_one_eligible_session(self, broker):
        order = submit(broker, Side.BUY, 10, type=OrderType.LIMIT, limit_price=50.0)
        broker.process_session(D2, {"AAPL": bar(100.0, 101.0, 99.0, 100.0)})
        assert broker.ledger.get_order(order.id).status is OrderStatus.EXPIRED

    def test_a_gtc_order_persists_across_sessions(self, broker):
        order = submit(
            broker, Side.BUY, 10, type=OrderType.LIMIT, limit_price=50.0,
            tif=TimeInForce.GTC,
        )
        for session in (D2, D3, D4):
            broker.process_session(session, {"AAPL": bar(100.0, 101.0, 99.0, 100.0)})
        assert broker.ledger.get_order(order.id).status is OrderStatus.PENDING

    def test_a_gtc_order_eventually_fills_when_price_comes_to_it(self, broker):
        order = submit(
            broker, Side.BUY, 10, type=OrderType.LIMIT, limit_price=90.0,
            tif=TimeInForce.GTC,
        )
        broker.process_session(D2, {"AAPL": bar(100.0, 101.0, 99.0, 100.0)})
        result = broker.process_session(D3, {"AAPL": bar(95.0, 96.0, 88.0, 94.0)})
        assert len(result.fills) == 1
        assert broker.ledger.get_order(order.id).status is OrderStatus.FILLED


class TestMissingBars:
    def test_a_gtc_order_stays_pending_when_there_is_no_bar(self, broker):
        order = submit(broker, Side.BUY, 10, tif=TimeInForce.GTC)
        result = broker.process_session(D2, {})       # halted / no data
        assert result.fills == ()
        assert broker.ledger.get_order(order.id).status is OrderStatus.PENDING

    def test_it_fills_once_the_bar_reappears(self, broker):
        order = submit(broker, Side.BUY, 10, tif=TimeInForce.GTC)
        broker.process_session(D2, {})
        broker.process_session(D3, {"AAPL": flat(100.0)})
        assert broker.ledger.get_order(order.id).status is OrderStatus.FILLED

    def test_a_day_order_still_expires_without_a_bar(self, broker):
        order = submit(broker, Side.BUY, 10, tif=TimeInForce.DAY)
        broker.process_session(D2, {})
        assert broker.ledger.get_order(order.id).status is OrderStatus.EXPIRED


# ------------------------------------------------------------ risk invariants


class TestBuyingPower:
    def test_a_buy_beyond_cash_is_rejected_with_a_reason(self, broker):
        order = submit(broker, Side.BUY, 1000)        # 1000 * 100 >> 10,000
        assert order.status is OrderStatus.REJECTED
        assert "available" in order.reject_reason
        assert "1000" in order.reject_reason

    def test_resting_buys_reserve_cash(self, broker):
        first = submit(broker, Side.BUY, 90, tif=TimeInForce.GTC)   # ~9,090 reserved
        assert first.status is OrderStatus.PENDING
        second = submit(broker, Side.BUY, 50, tif=TimeInForce.GTC)
        assert second.status is OrderStatus.REJECTED, "cash was already claimed"

    def test_cash_never_goes_negative_on_a_gap_up(self, ledger):
        """Passes submission against the last close, then the open gaps far
        above it — the fill must be rejected, not overdraw the account."""
        account = ledger.create_account("gap", 1_000.0)
        b = SimBroker(ledger, account.id, costs=ZERO_COSTS)
        b.set_mark("AAPL", 100.0)
        order = submit(b, Side.BUY, 9)                 # ~909 at the last close
        assert order.status is OrderStatus.PENDING
        result = b.process_session(D2, {"AAPL": bar(500.0, 510.0, 495.0, 505.0)})
        assert result.fills == ()
        assert b.cash() == pytest.approx(1_000.0)
        assert b.cash() >= 0
        assert "cash is left" in ledger.get_order(order.id).reject_reason

    def test_buying_power_reflects_reserved_cash(self, broker):
        assert broker.buying_power() == pytest.approx(10_000.0)
        submit(broker, Side.BUY, 50, tif=TimeInForce.GTC)
        assert broker.buying_power() < 10_000.0


class TestNoShorting:
    def test_selling_more_than_held_is_rejected(self, broker):
        hold(broker, 10, 100.0)
        order = submit(broker, Side.SELL, 25, created=D2)
        assert order.status is OrderStatus.REJECTED
        assert "no shorting" in order.reject_reason

    def test_selling_with_no_position_is_rejected(self, broker):
        order = submit(broker, Side.SELL, 1)
        assert order.status is OrderStatus.REJECTED

    def test_resting_sells_reserve_shares(self, broker):
        """Two sells that each cover the whole position cannot both rest."""
        hold(broker, 10, 100.0)
        first = submit(broker, Side.SELL, 10, created=D2, tif=TimeInForce.GTC)
        assert first.status is OrderStatus.PENDING
        second = submit(broker, Side.SELL, 10, created=D2, tif=TimeInForce.GTC)
        assert second.status is OrderStatus.REJECTED
        assert "reserved" in second.reject_reason

    def test_share_count_never_goes_negative(self, broker):
        hold(broker, 10, 100.0)
        submit(broker, Side.SELL, 10, created=D2, tif=TimeInForce.GTC)
        broker.process_session(D3, {"AAPL": flat(100.0)})
        assert broker.positions() == {}
        assert all(p.qty >= 0 for p in broker.positions().values())


class TestFillOrdering:
    """Within a session: sells settle before buys, each side oldest-first.

    Note the two different checks a buy faces. At SUBMISSION it is measured
    against cash on hand — a resting sell that has not happened yet does not
    count, which is the conservative and predictable choice. At FILL time the
    sell has already settled, so its proceeds are spendable.
    """

    def test_sells_fill_before_buys_within_a_session(self, ledger):
        account = ledger.create_account("order", 10_000.0)
        b = SimBroker(ledger, account.id, costs=ZERO_COSTS)
        b.set_mark("AAPL", 100.0)
        b.set_mark("MSFT", 100.0)
        b.submit(Order(account_id=b.account_id, created_session=D1, ticker="AAPL",
                       side=Side.BUY, qty=9))
        b.process_session(D2, {"AAPL": flat(100.0), "MSFT": flat(100.0)})

        # queue the BUY first, so submission order alone would put it first
        b.submit(Order(account_id=b.account_id, created_session=D2, ticker="MSFT",
                       side=Side.BUY, qty=8, tif=TimeInForce.GTC))
        b.submit(Order(account_id=b.account_id, created_session=D2, ticker="AAPL",
                       side=Side.SELL, qty=9, tif=TimeInForce.GTC))
        result = b.process_session(D3, {"AAPL": flat(100.0), "MSFT": flat(100.0)})

        assert [f.side for f in result.fills] == [Side.SELL, Side.BUY]
        assert b.cash() >= 0

    def test_a_sale_funds_a_buy_that_cash_alone_could_not_cover(self, ledger):
        """The buy is sized against MSFT's last close and passes submission.
        Then MSFT gaps up 50% overnight, so at fill time it costs more than the
        cash on hand — and only clears because the AAPL sale settled first."""
        account = ledger.create_account("fund", 2_000.0)
        b = SimBroker(ledger, account.id, costs=ZERO_COSTS)
        b.set_mark("AAPL", 100.0)
        b.set_mark("MSFT", 100.0)

        b.submit(Order(account_id=b.account_id, created_session=D1, ticker="AAPL",
                       side=Side.BUY, qty=9))
        b.process_session(D2, {"AAPL": flat(100.0), "MSFT": flat(100.0)})
        assert b.cash() == pytest.approx(1_100.0)

        b.submit(Order(account_id=b.account_id, created_session=D2, ticker="AAPL",
                       side=Side.SELL, qty=9, tif=TimeInForce.GTC))
        buy = b.submit(Order(account_id=b.account_id, created_session=D2,
                             ticker="MSFT", side=Side.BUY, qty=10,
                             tif=TimeInForce.GTC))
        assert buy.status is OrderStatus.PENDING, buy.reject_reason

        # MSFT opens at 150: the buy now needs $1,500 against $1,100 of cash
        result = b.process_session(
            D3, {"AAPL": flat(100.0), "MSFT": flat(150.0)}
        )
        assert [f.side for f in result.fills] == [Side.SELL, Side.BUY]
        assert b.positions()["MSFT"].qty == 10
        assert b.cash() == pytest.approx(1_100.0 + 900.0 - 1_500.0)
        assert b.cash() >= 0

    def test_a_buy_is_rejected_at_submission_when_cash_is_not_there_yet(self, ledger):
        """A resting sell does not pre-fund a new buy: buying power is cash on
        hand, so a bot cannot spend proceeds it has not received."""
        account = ledger.create_account("nopre", 1_000.0)
        b = SimBroker(ledger, account.id, costs=ZERO_COSTS)
        b.set_mark("AAPL", 100.0)
        b.set_mark("MSFT", 100.0)
        b.submit(Order(account_id=b.account_id, created_session=D1, ticker="AAPL",
                       side=Side.BUY, qty=9))
        b.process_session(D2, {"AAPL": flat(100.0), "MSFT": flat(100.0)})

        b.submit(Order(account_id=b.account_id, created_session=D2, ticker="AAPL",
                       side=Side.SELL, qty=9, tif=TimeInForce.GTC))
        too_big = b.submit(Order(account_id=b.account_id, created_session=D2,
                                 ticker="MSFT", side=Side.BUY, qty=50,
                                 tif=TimeInForce.GTC))
        assert too_big.status is OrderStatus.REJECTED
        assert "available" in too_big.reject_reason


# ------------------------------------------------------------ corporate actions


class TestSplits:
    def test_a_two_for_one_doubles_shares_and_halves_average_cost(self, broker):
        hold(broker, 10, 100.0)
        before = broker.positions()["AAPL"]
        assert (before.qty, before.avg_cost) == (10, pytest.approx(100.0))

        broker.process_session(
            D3, {"AAPL": flat(50.0)},
            [Action("AAPL", D3, SPLIT, 2.0)],
        )
        after = broker.positions()["AAPL"]
        assert after.qty == pytest.approx(20)
        assert after.avg_cost == pytest.approx(50.0)
        assert after.cost_basis == pytest.approx(before.cost_basis)

    def test_equity_is_unchanged_across_a_split(self, broker):
        hold(broker, 10, 100.0)
        equity_before = broker.equity({"AAPL": 100.0})
        # as-traded prices really do halve on the ex-date
        result = broker.process_session(
            D3, {"AAPL": flat(50.0)}, [Action("AAPL", D3, SPLIT, 2.0)]
        )
        assert result.equity == pytest.approx(equity_before)

    def test_a_resting_order_is_re_termed_by_the_split(self, broker):
        hold(broker, 10, 100.0)
        order = submit(
            broker, Side.SELL, 10, created=D2, type=OrderType.LIMIT,
            limit_price=200.0, tif=TimeInForce.GTC,
        )
        broker.process_session(
            D3, {"AAPL": flat(50.0)}, [Action("AAPL", D3, SPLIT, 2.0)]
        )
        reterm = broker.ledger.get_order(order.id)
        assert reterm.qty == pytest.approx(20), "share count doubles"
        assert reterm.limit_price == pytest.approx(100.0), "limit halves"

    def test_a_split_is_applied_only_once_per_session(self, broker):
        hold(broker, 10, 100.0)
        action = [Action("AAPL", D3, SPLIT, 2.0)]
        broker.process_session(D3, {"AAPL": flat(50.0)}, action)
        broker.process_session(D3, {"AAPL": flat(50.0)}, action)   # re-run
        assert broker.positions()["AAPL"].qty == pytest.approx(20)

    def test_a_split_on_a_ticker_we_do_not_hold_is_ignored(self, broker):
        broker.process_session(
            D2, {"AAPL": flat(100.0)}, [Action("AAPL", D2, SPLIT, 2.0)]
        )
        assert broker.positions() == {}


class TestDividends:
    def test_a_dividend_credits_cash_on_the_ex_date(self, broker):
        hold(broker, 10, 100.0)
        cash_before = broker.cash()
        result = broker.process_session(
            D3, {"AAPL": flat(100.0)}, [Action("AAPL", D3, DIVIDEND, 0.50)]
        )
        assert result.dividends == pytest.approx(5.0)      # 10 shares * 0.50
        assert broker.cash() == pytest.approx(cash_before + 5.0)

    def test_it_pays_on_the_shares_held_at_the_previous_close(self, broker):
        """A buy filling on the ex-date must NOT collect that dividend."""
        hold(broker, 10, 100.0)
        submit(broker, Side.BUY, 5, created=D2, tif=TimeInForce.GTC)
        result = broker.process_session(
            D3, {"AAPL": flat(100.0)}, [Action("AAPL", D3, DIVIDEND, 1.0)]
        )
        assert len(result.fills) == 1, "the extra 5 shares did fill"
        assert result.dividends == pytest.approx(10.0), "but paid on the prior 10 only"

    def test_no_dividend_without_a_position(self, broker):
        result = broker.process_session(
            D2, {"AAPL": flat(100.0)}, [Action("AAPL", D2, DIVIDEND, 1.0)]
        )
        assert result.dividends == pytest.approx(0.0)

    def test_a_dividend_is_credited_only_once_per_session(self, broker):
        hold(broker, 10, 100.0)
        action = [Action("AAPL", D3, DIVIDEND, 0.50)]
        broker.process_session(D3, {"AAPL": flat(100.0)}, action)
        cash = broker.cash()
        broker.process_session(D3, {"AAPL": flat(100.0)}, action)   # re-run
        assert broker.cash() == pytest.approx(cash)

    def test_an_action_on_another_session_is_not_applied(self, broker):
        hold(broker, 10, 100.0)
        result = broker.process_session(
            D3, {"AAPL": flat(100.0)}, [Action("AAPL", D4, DIVIDEND, 1.0)]
        )
        assert result.dividends == pytest.approx(0.0)


# ------------------------------------------------------------------- rebuild


class TestRebuild:
    def test_state_rebuilt_from_the_ledger_matches_the_live_broker(self, ledger, broker):
        """Positions and cash are derived, so a fresh broker on the same ledger
        must agree exactly."""
        hold(broker, 10, 100.0)
        submit(broker, Side.BUY, 5, created=D2, tif=TimeInForce.GTC)
        broker.process_session(
            D3, {"AAPL": flat(120.0)}, [Action("AAPL", D3, DIVIDEND, 0.25)]
        )
        broker.process_session(
            D4, {"AAPL": flat(60.0)}, [Action("AAPL", D4, SPLIT, 2.0)]
        )

        fresh = SimBroker(ledger, broker.account_id, costs=ZERO_COSTS)
        assert fresh.cash() == pytest.approx(broker.cash())
        assert fresh.positions().keys() == broker.positions().keys()
        for ticker, position in broker.positions().items():
            other = fresh.positions()[ticker]
            assert other.qty == pytest.approx(position.qty)
            assert other.avg_cost == pytest.approx(position.avg_cost)
        assert fresh.equity({"AAPL": 60.0}) == pytest.approx(broker.equity({"AAPL": 60.0}))

    def test_snapshots_record_each_processed_session(self, broker):
        hold(broker, 10, 100.0)
        broker.process_session(D3, {"AAPL": flat(110.0)})
        snaps = broker.ledger.snapshots(broker.account_id)
        assert [s.session for s in snaps] == [D2, D3]
        assert snaps[-1].equity == pytest.approx(snaps[-1].cash + 10 * 110.0)

    def test_last_processed_session_advances(self, broker):
        assert broker.ledger.last_processed_session(broker.account_id) is None
        broker.process_session(D2, {"AAPL": flat(100.0)})
        assert broker.ledger.last_processed_session(broker.account_id) == D2


class TestEquityIdentity:
    def test_equity_equals_cash_plus_marked_positions(self, broker):
        hold(broker, 10, 100.0)
        result = broker.process_session(D3, {"AAPL": flat(130.0)})
        assert result.equity == pytest.approx(result.cash + 10 * 130.0)

    def test_commission_comes_out_of_equity(self, ledger):
        account = ledger.create_account("fees", 10_000.0)
        b = SimBroker(ledger, account.id, costs=CostModel(0.0, 0.0, 5.00, 0.0))
        b.set_mark("AAPL", 100.0)
        submit(b, Side.BUY, 10)
        result = b.process_session(D2, {"AAPL": flat(100.0)})
        # bought at the mark, so the only change to equity is the commission
        assert result.equity == pytest.approx(10_000.0 - 5.00)


# -------------------------------------------------------------- order hygiene


class TestOrderValidation:
    def test_fractional_quantities_are_refused_while_the_flag_is_off(self):
        with pytest.raises(OrderError, match="fractional"):
            Order(account_id=1, created_session=D1, ticker="AAPL",
                  side=Side.BUY, qty=1.5)

    def test_a_limit_order_needs_a_limit_price(self):
        with pytest.raises(OrderError, match="limit_price"):
            Order(account_id=1, created_session=D1, ticker="AAPL",
                  side=Side.BUY, qty=1, type=OrderType.LIMIT)

    def test_a_market_order_must_not_carry_a_trigger(self):
        with pytest.raises(OrderError, match="trigger"):
            Order(account_id=1, created_session=D1, ticker="AAPL", side=Side.BUY,
                  qty=1, type=OrderType.MARKET, limit_price=10.0)

    def test_quantity_must_be_positive(self):
        with pytest.raises(OrderError, match="positive"):
            Order(account_id=1, created_session=D1, ticker="AAPL",
                  side=Side.BUY, qty=0)


class TestCancel:
    def test_cancelling_a_resting_order_works(self, broker):
        order = submit(broker, Side.BUY, 10, tif=TimeInForce.GTC)
        broker.cancel(order.id)
        assert broker.ledger.get_order(order.id).status is OrderStatus.CANCELLED
        assert broker.open_orders() == []

    def test_a_cancelled_order_never_fills(self, broker):
        order = submit(broker, Side.BUY, 10, tif=TimeInForce.GTC)
        broker.cancel(order.id)
        result = broker.process_session(D2, {"AAPL": flat(100.0)})
        assert result.fills == ()

    def test_cancelling_twice_is_an_error_not_a_silent_no_op(self, broker):
        order = submit(broker, Side.BUY, 10, tif=TimeInForce.GTC)
        broker.cancel(order.id)
        with pytest.raises(ValueError, match="already"):
            broker.cancel(order.id)

    def test_a_killed_account_cannot_place_orders(self, broker, ledger):
        ledger.set_account_status(broker.account_id, "killed", source="manual")
        order = submit(broker, Side.BUY, 1)
        assert order.status is OrderStatus.REJECTED
        assert "killed" in order.reject_reason


class TestRejectionsAreExplained:
    def test_every_rejection_carries_a_human_readable_reason(self, broker):
        rejections = [
            submit(broker, Side.BUY, 100_000),
            submit(broker, Side.SELL, 5),
        ]
        for order in rejections:
            assert order.status is OrderStatus.REJECTED
            assert order.reject_reason and len(order.reject_reason) > 20
            assert order.ticker in order.reject_reason

    def test_the_rejection_survives_in_the_order_history(self, broker):
        order = submit(broker, Side.BUY, 100_000)
        history = broker.ledger.order_history(order.id)
        assert history[-1]["status"] == "rejected"
        assert history[-1]["reject_reason"]
