"""Sizing policies and risk limits. Offline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bots.data_view import DataView, PortfolioView
from bots.risk import RiskLimits, RiskManager, build_risk
from bots.sizing import (
    EqualWeight,
    FixedDollar,
    VolatilityTarget,
    build_sizing,
)
from bots.strategy import OrderRequest
from engine.orders import Side

DATES = pd.bdate_range("2026-01-05", periods=200)


def frame(closes) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    idx = DATES[: len(closes)]
    return pd.DataFrame(
        {"open": closes, "high": closes * 1.01, "low": closes * 0.99,
         "close": closes, "volume": np.full(len(closes), 1e6)},
        index=idx,
    )


#: every ticker the tests below invent, so "no price" never masks a real result
TICKERS = tuple(["AAA", "BBB", "A", "B", "C", "X", "Y"] + [f"T{i}" for i in range(25)])


def view(price: float = 100.0, n: int = 200, vol: float = 0.0) -> DataView:
    rng = np.random.default_rng(7)
    if vol > 0:
        steps = rng.normal(0.0, vol / np.sqrt(252), n)
        closes = price * np.exp(np.cumsum(steps))
    else:
        closes = np.full(n, price)
    f = frame(closes)
    return DataView(
        session=f.index[-1],
        frames={t: f for t in TICKERS},
        adjusted_closes={t: f["close"] for t in TICKERS},
    )


def portfolio(equity=100_000.0, buying_power=100_000.0, **kw) -> PortfolioView:
    return PortfolioView(
        session=DATES[-1], cash=kw.pop("cash", buying_power), equity=equity,
        buying_power=buying_power, **kw,
    )


class TestFixedDollar:
    def test_it_buys_the_configured_amount(self):
        policy = FixedDollar(10_000.0)
        # the budget fits inside buying power, so the full $10,000 is deployed
        assert policy.shares_for("AAA", view(100.0), portfolio(), 10) == 100

    def test_it_rounds_down_never_up(self):
        policy = FixedDollar(1_050.0)
        # $1,050 / $100 = 10.5 shares -> 10, never 11
        assert policy.shares_for("AAA", view(100.0), portfolio(), 10) == 10

    def test_it_is_capped_by_buying_power(self):
        policy = FixedDollar(50_000.0)
        shares = policy.shares_for("AAA", view(100.0), portfolio(buying_power=1_000.0), 10)
        assert shares == 9, "cannot size beyond the cash actually available"

    def test_no_buying_power_means_no_shares(self):
        policy = FixedDollar(10_000.0)
        assert policy.shares_for("AAA", view(100.0), portfolio(buying_power=0.0), 10) == 0


class TestEqualWeight:
    def test_it_divides_equity_by_max_positions(self):
        policy = EqualWeight()
        # $100,000 equity / 10 slots = $10,000 -> 100 shares at $100
        assert policy.shares_for("AAA", view(100.0), portfolio(), 10) == 100

    def test_the_size_does_not_depend_on_how_many_signals_fired(self):
        """The first buy must be the same size as the tenth."""
        policy = EqualWeight()
        first = policy.shares_for("AAA", view(100.0), portfolio(), 10)
        later = policy.shares_for(
            "AAA", view(100.0),
            portfolio(positions={"X": 1, "Y": 1, "Z": 1}), 10,
        )
        assert first == later

    def test_more_slots_means_smaller_positions(self):
        policy = EqualWeight()
        few = policy.shares_for("AAA", view(100.0), portfolio(), 4)
        many = policy.shares_for("AAA", view(100.0), portfolio(), 20)
        assert few > many

    def test_a_single_slot_deploys_the_account(self):
        policy = EqualWeight()
        assert policy.shares_for("AAA", view(100.0), portfolio(), 1) == 990


class TestVolatilityTarget:
    def test_a_quiet_stock_gets_a_bigger_position_than_a_wild_one(self):
        policy = VolatilityTarget(target_annual_vol=0.20, window=60)
        quiet = policy.shares_for("AAA", view(100.0, vol=0.10), portfolio(), 10)
        wild = policy.shares_for("AAA", view(100.0, vol=0.80), portfolio(), 10)
        assert quiet > wild

    def test_it_is_capped_so_a_flat_stock_is_not_levered_up(self):
        policy = VolatilityTarget(target_annual_vol=0.20, window=60, max_multiple=1.5)
        quiet = view(100.0, vol=0.01)
        shares = policy.shares_for("AAA", quiet, portfolio(), 10)
        equal = EqualWeight().shares_for("AAA", quiet, portfolio(), 10)
        assert shares <= equal * 1.5 + 1, "the multiple must be capped"

    def test_it_falls_back_to_equal_weight_without_enough_history(self):
        policy = VolatilityTarget(window=60)
        short = view(100.0, n=20)
        assert policy.shares_for("AAA", short, portfolio(), 10) == pytest.approx(
            EqualWeight().shares_for("AAA", short, portfolio(), 10)
        )


class TestSizingNeverOverreaches:
    @pytest.mark.parametrize(
        "policy", [FixedDollar(1e9), EqualWeight(), VolatilityTarget()]
    )
    def test_no_policy_sizes_beyond_buying_power(self, policy):
        """A sizing policy must not author an order the broker will reject."""
        pv = portfolio(equity=1_000_000.0, buying_power=5_000.0)
        data = view(100.0, vol=0.3)
        shares = policy.shares_for("AAA", data, pv, 10)
        assert shares * data.close("AAA") <= pv.buying_power


class TestBuildSizing:
    def test_a_bare_name_works(self):
        assert isinstance(build_sizing("equal_weight"), EqualWeight)

    def test_a_mapping_with_params_works(self):
        policy = build_sizing({"type": "fixed_dollar", "amount": 5000})
        assert isinstance(policy, FixedDollar) and policy.amount == 5000

    def test_an_unknown_policy_is_named_in_the_error(self):
        with pytest.raises(ValueError, match="nonsense"):
            build_sizing("nonsense")

    def test_bad_params_are_reported(self):
        with pytest.raises(ValueError, match="bad parameters"):
            build_sizing({"type": "fixed_dollar", "wrong": 1})


# --------------------------------------------------------------------- risk


def request(ticker="AAA", side=Side.BUY, qty=10) -> OrderRequest:
    return OrderRequest(ticker=ticker, side=side, qty=qty, reason="test")


class TestPositionLimits:
    def test_a_new_position_beyond_the_cap_is_blocked(self):
        manager = RiskManager(RiskLimits(max_positions=2, max_position_pct=None))
        pv = portfolio(positions={"X": 1.0, "Y": 1.0})
        decision = manager.check([request("AAA")], view(100.0), pv)
        assert not decision.allowed
        assert "position limit" in decision.blocked[0].reason

    def test_adding_to_an_existing_position_is_not_blocked_by_the_count(self):
        manager = RiskManager(RiskLimits(max_positions=2, max_position_pct=None))
        pv = portfolio(positions={"AAA": 1.0, "Y": 1.0})
        decision = manager.check([request("AAA")], view(100.0), pv)
        assert decision.allowed

    def test_the_cap_counts_positions_opened_earlier_in_the_same_session(self):
        manager = RiskManager(
            RiskLimits(max_positions=1, max_position_pct=None, max_orders_per_day=None)
        )
        decision = manager.check(
            [request("AAA"), request("BBB")], view(100.0), portfolio()
        )
        assert len(decision.allowed) == 1 and len(decision.blocked) == 1

    def test_a_position_over_the_percentage_cap_is_blocked(self):
        manager = RiskManager(RiskLimits(max_position_pct=0.05, max_positions=10))
        # 10 shares * 100 = 1,000 on 10,000 equity = 10% > 5%
        decision = manager.check(
            [request(qty=10)], view(100.0), portfolio(equity=10_000.0)
        )
        assert not decision.allowed
        assert "per-position limit" in decision.blocked[0].reason

    def test_the_cap_includes_what_is_already_held(self):
        manager = RiskManager(RiskLimits(max_position_pct=0.15, max_positions=10))
        pv = portfolio(equity=10_000.0, positions={"AAA": 10.0})
        decision = manager.check([request(qty=10)], view(100.0), pv)
        assert not decision.allowed, "1,000 held + 1,000 new = 20% of equity"


class TestOrderCap:
    def test_the_daily_cap_blocks_the_surplus(self):
        manager = RiskManager(
            RiskLimits(max_orders_per_day=2, max_positions=10, max_position_pct=None)
        )
        decision = manager.check(
            [request("A"), request("B"), request("C")], view(100.0), portfolio()
        )
        assert len(decision.allowed) == 2
        assert "daily order cap" in decision.blocked[0].reason

    def test_orders_already_placed_today_count(self):
        manager = RiskManager(
            RiskLimits(max_orders_per_day=2, max_positions=10, max_position_pct=None)
        )
        decision = manager.check(
            [request("A")], view(100.0), portfolio(), orders_today=2
        )
        assert not decision.allowed

    def test_no_cap_means_no_limit(self):
        manager = RiskManager(
            RiskLimits(max_orders_per_day=None, max_positions=50, max_position_pct=None)
        )
        decision = manager.check(
            [request(f"T{i}") for i in range(20)], view(100.0), portfolio()
        )
        assert len(decision.allowed) == 20


class TestSellsAreNotBlockedByRiskLimits:
    def test_a_sell_passes_even_at_the_position_cap(self):
        manager = RiskManager(RiskLimits(max_positions=1, max_position_pct=0.01))
        pv = portfolio(positions={"AAA": 10.0, "X": 1.0})
        decision = manager.check(
            [request("AAA", Side.SELL)], view(100.0), pv
        )
        assert decision.allowed, "reducing exposure must never be risk-blocked"

    def test_selling_something_not_held_is_blocked(self):
        manager = RiskManager(RiskLimits())
        decision = manager.check(
            [request("AAA", Side.SELL)], view(100.0), portfolio()
        )
        assert not decision.allowed
        assert "no AAA position" in decision.blocked[0].reason


class TestMinimumOrderValue:
    def test_a_trade_too_small_to_be_worth_commission_is_skipped(self):
        manager = RiskManager(RiskLimits(min_order_value=500.0, max_position_pct=None))
        decision = manager.check([request(qty=1)], view(100.0), portfolio())
        assert not decision.allowed
        assert "below the $500 minimum" in decision.blocked[0].reason


class TestDrawdownKillSwitch:
    def test_it_fires_past_the_limit(self):
        manager = RiskManager(RiskLimits(max_drawdown_pct=0.20))
        curve = pd.Series([100_000.0, 110_000.0, 95_000.0])
        decision = manager.check(
            [request()], view(100.0), portfolio(equity=80_000.0), equity_curve=curve
        )
        assert decision.kill
        assert "below its peak" in decision.kill_reason

    def test_it_does_not_fire_inside_the_limit(self):
        manager = RiskManager(RiskLimits(max_drawdown_pct=0.20, max_position_pct=None))
        curve = pd.Series([100_000.0, 110_000.0])
        decision = manager.check(
            [request()], view(100.0), portfolio(equity=100_000.0), equity_curve=curve
        )
        assert not decision.kill

    def test_every_request_is_blocked_once_killed(self):
        manager = RiskManager(RiskLimits(max_drawdown_pct=0.10))
        curve = pd.Series([100_000.0])
        decision = manager.check(
            [request("A"), request("B")], view(100.0),
            portfolio(equity=50_000.0), equity_curve=curve,
        )
        assert decision.kill and len(decision.blocked) == 2 and not decision.allowed

    def test_todays_equity_counts_toward_the_peak(self):
        """A bot that rises then crashes within one session is still measured
        against the high."""
        manager = RiskManager(RiskLimits(max_drawdown_pct=0.20))
        decision = manager.check(
            [], view(100.0), portfolio(equity=100.0),
            equity_curve=pd.Series([1000.0]),
        )
        assert decision.kill

    def test_no_limit_means_no_kill(self):
        manager = RiskManager(RiskLimits(max_drawdown_pct=None, max_position_pct=None))
        decision = manager.check(
            [request()], view(100.0), portfolio(equity=1.0),
            equity_curve=pd.Series([1_000_000.0]),
        )
        assert not decision.kill


class TestEveryBlockExplainsItself:
    def test_blocked_requests_always_carry_a_reason(self):
        manager = RiskManager(RiskLimits(max_positions=0, max_orders_per_day=1))
        decision = manager.check(
            [request("A"), request("B")], view(100.0), portfolio()
        )
        assert decision.blocked
        for block in decision.blocked:
            assert block.reason and len(block.reason) > 10


class TestBuildRisk:
    def test_defaults_when_unspecified(self):
        assert build_risk(None) == RiskLimits()

    def test_values_are_applied(self):
        limits = build_risk({"max_positions": 3, "liquidate_on_kill": True})
        assert limits.max_positions == 3 and limits.liquidate_on_kill

    def test_an_unknown_setting_is_named(self):
        with pytest.raises(ValueError, match="max_postions"):
            build_risk({"max_postions": 3})
