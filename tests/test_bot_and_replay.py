"""Bots end to end: the session loop, isolation, kill switch, and replay.

Offline — synthetic price paths, synthetic ledgers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bots.bot import Bot
from bots.data_view import DataView
from bots.registry import ConfigError, parse_spec
from bots.risk import RiskLimits
from bots.sizing import EqualWeight, FixedDollar
from bots.strategy import OrderRequest, Strategy
from engine.costs import ZERO_COSTS
from engine.ledger import BACKTEST, KILLED, Ledger
from engine.orders import Side

DATES = pd.bdate_range("2026-01-05", periods=80)


def ramp(n: int = 80, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    closes = start + np.arange(n, dtype=float) * step
    return pd.DataFrame(
        {"open": closes, "high": closes * 1.01, "low": closes * 0.99,
         "close": closes, "volume": np.full(n, 1e6)},
        index=DATES[:n],
    )


def make_view(frames: dict[str, pd.DataFrame], session=None) -> DataView:
    session = session if session is not None else DATES[0]
    return DataView(
        session=session, frames=frames,
        adjusted_closes={t: f["close"] for t, f in frames.items()},
    )


class BuyOnce(Strategy):
    """Buys one named ticker the first time it is asked, then nothing."""

    name = "buy once"

    def __init__(self, ticker="AAA"):
        super().__init__(ticker=ticker)
        self.ticker = ticker

    def on_bar(self, session, data, portfolio):
        if portfolio.holds(self.ticker) or portfolio.has_open_order(self.ticker):
            return []
        return [OrderRequest(ticker=self.ticker, side=Side.BUY, reason="always")]


class SellEverything(Strategy):
    name = "sell everything"

    def on_bar(self, session, data, portfolio):
        return [
            OrderRequest(ticker=t, side=Side.SELL, qty=q, reason="exit")
            for t, q in portfolio.positions.items() if q > 0
        ]


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(tmp_path / "ledger.db")
    yield led
    led.close()


def make_bot(ledger, strategy=None, cash=100_000.0, name="bot", **kw) -> Bot:
    return Bot.create(
        name=name, strategy=strategy or BuyOnce(), ledger=ledger,
        starting_cash=cash, costs=ZERO_COSTS,
        sizing=kw.pop("sizing", FixedDollar(10_000.0)),
        risk=kw.pop("risk", RiskLimits(max_position_pct=None, min_order_value=0.0)),
        **kw,
    )


class TestSessionLoop:
    def test_an_order_is_queued_on_one_session_and_fills_on_the_next(self, ledger):
        frames = {"AAA": ramp()}
        bot = make_bot(ledger)

        first = bot.run_session(DATES[0], make_view(frames, DATES[0]))
        assert first.submitted == 1
        assert first.broker.fills == (), "cannot fill on the decision session"

        second = bot.run_session(DATES[1], make_view(frames, DATES[1]))
        assert len(second.broker.fills) == 1
        assert bot.broker.positions()["AAA"].qty == 100   # $10,000 / $100

    def test_the_strategy_sees_the_account_after_todays_fills(self, ledger):
        """Step order matters: the broker processes before the strategy looks."""
        frames = {"AAA": ramp()}
        bot = make_bot(ledger)
        bot.run_session(DATES[0], make_view(frames, DATES[0]))
        bot.run_session(DATES[1], make_view(frames, DATES[1]))
        # BuyOnce refuses to buy again because it can already see the holding
        third = bot.run_session(DATES[2], make_view(frames, DATES[2]))
        assert third.submitted == 0

    def test_equity_is_snapshotted_every_session(self, ledger):
        frames = {"AAA": ramp()}
        bot = make_bot(ledger)
        for d in DATES[:5]:
            bot.run_session(d, make_view(frames, d))
        curve = ledger.equity_curve(bot.account_id)
        assert len(curve) == 5
        assert curve.index[0] == DATES[0]

    def test_a_sell_closes_the_position(self, ledger):
        frames = {"AAA": ramp()}
        bot = make_bot(ledger)
        bot.run_session(DATES[0], make_view(frames, DATES[0]))
        bot.run_session(DATES[1], make_view(frames, DATES[1]))
        assert bot.broker.positions()

        bot.strategy = SellEverything()
        bot.run_session(DATES[2], make_view(frames, DATES[2]))
        bot.run_session(DATES[3], make_view(frames, DATES[3]))
        assert bot.broker.positions() == {}


class TestIsolation:
    def test_two_bots_on_the_same_data_never_affect_each_other(self, ledger):
        """The spec's done-condition."""
        frames = {"AAA": ramp()}
        a = make_bot(ledger, BuyOnce(), name="bot-a", cash=100_000.0)
        b = make_bot(ledger, BuyOnce(), name="bot-b", cash=50_000.0,
                     sizing=FixedDollar(5_000.0))

        for d in DATES[:3]:
            view = make_view(frames, d)
            a.run_session(d, view)
            b.run_session(d, view)

        # sized off session 0's close ($100), filled at session 1's open ($101)
        assert a.broker.positions()["AAA"].qty == 100
        assert b.broker.positions()["AAA"].qty == 50
        assert a.broker.cash() == pytest.approx(100_000.0 - 100 * 101.0)
        assert b.broker.cash() == pytest.approx(50_000.0 - 50 * 101.0)

    def test_killing_one_bot_leaves_the_other_trading(self, ledger):
        frames = {"AAA": ramp()}
        a = make_bot(ledger, BuyOnce(), name="bot-a")
        b = make_bot(ledger, BuyOnce(), name="bot-b")
        a.kill("test")

        for d in DATES[:3]:
            view = make_view(frames, d)
            a.run_session(d, view)
            b.run_session(d, view)

        assert a.broker.positions() == {}
        assert b.broker.positions()["AAA"].qty == 100

    def test_one_bots_view_cannot_corrupt_anothers(self, ledger):
        frames = {"AAA": ramp()}
        view = make_view(frames, DATES[5])
        history = view.history("AAA")
        history.loc[:, "close"] = -1.0
        assert view.close("AAA") > 0, "the shared cache survived a mutation"


class TestKillSwitch:
    def test_a_drawdown_breach_kills_the_bot(self, ledger):
        crash = ramp(n=10, start=100.0, step=-9.0)   # 100 down to 19
        frames = {"AAA": crash}
        bot = make_bot(
            ledger, BuyOnce(), cash=10_000.0,
            sizing=FixedDollar(10_000.0),
            risk=RiskLimits(max_drawdown_pct=0.20, max_position_pct=None,
                            min_order_value=0.0),
        )
        killed = False
        for d in crash.index:
            result = bot.run_session(d, make_view(frames, d))
            killed = killed or result.killed
        assert killed
        assert bot.status == KILLED

    def test_a_killed_bot_holds_its_positions_by_default(self, ledger):
        crash = ramp(n=10, start=100.0, step=-9.0)
        frames = {"AAA": crash}
        bot = make_bot(
            ledger, BuyOnce(), cash=10_000.0, sizing=FixedDollar(5_000.0),
            risk=RiskLimits(max_drawdown_pct=0.20, max_position_pct=None,
                            min_order_value=0.0, liquidate_on_kill=False),
        )
        for d in crash.index:
            bot.run_session(d, make_view(frames, d))
        assert bot.status == KILLED
        assert bot.broker.positions(), "positions are held, not dumped at the trough"

    def test_liquidate_on_kill_sells_out(self, ledger):
        """The liquidation is an ORDER, so it fills on the session after the
        kill — a bot cannot sell on the same bar it decided to."""
        crash = ramp(n=14, start=100.0, step=-6.0)
        frames = {"AAA": crash}
        bot = make_bot(
            ledger, BuyOnce(), cash=10_000.0, sizing=FixedDollar(5_000.0),
            risk=RiskLimits(max_drawdown_pct=0.20, max_position_pct=None,
                            min_order_value=0.0, liquidate_on_kill=True),
        )
        killed_on = None
        for d in crash.index:
            result = bot.run_session(d, make_view(frames, d))
            if result.killed and killed_on is None:
                killed_on = d
        assert killed_on is not None, "the drawdown limit never fired"
        assert killed_on < crash.index[-1], "need a session after the kill to fill on"
        assert bot.status == KILLED
        assert bot.broker.positions() == {}, "liquidation sold everything"

    def test_a_killed_bot_places_no_further_orders(self, ledger):
        frames = {"AAA": ramp()}
        bot = make_bot(ledger)
        bot.kill("manual")
        result = bot.run_session(DATES[0], make_view(frames, DATES[0]))
        assert result.submitted == 0 and result.skipped == "bot is killed"

    def test_a_paused_bot_still_marks_to_market(self, ledger):
        frames = {"AAA": ramp()}
        bot = make_bot(ledger)
        bot.pause()
        result = bot.run_session(DATES[0], make_view(frames, DATES[0]))
        assert result.submitted == 0 and result.skipped == "bot is paused"
        assert result.equity > 0, "a paused account is still valued"

    def test_resume_restores_trading(self, ledger):
        frames = {"AAA": ramp()}
        bot = make_bot(ledger)
        bot.pause()
        bot.run_session(DATES[0], make_view(frames, DATES[0]))
        bot.resume()
        assert bot.run_session(DATES[1], make_view(frames, DATES[1])).submitted == 1


class TestBlockedRequestsAreExplained:
    def test_a_blocked_request_reaches_the_result(self, ledger):
        frames = {"AAA": ramp()}
        bot = make_bot(
            ledger, BuyOnce(),
            risk=RiskLimits(max_positions=0, min_order_value=0.0,
                            max_position_pct=None),
        )
        result = bot.run_session(DATES[0], make_view(frames, DATES[0]))
        assert result.submitted == 0
        assert result.blocked and "position limit" in result.blocked[0][1]


class TestEntrySessions:
    def test_it_reports_when_a_position_was_opened(self, ledger):
        frames = {"AAA": ramp()}
        bot = make_bot(ledger)
        bot.run_session(DATES[0], make_view(frames, DATES[0]))
        bot.run_session(DATES[1], make_view(frames, DATES[1]))
        view = bot.portfolio_view(DATES[2], {"AAA": 102.0})
        assert view.entry_sessions["AAA"] == DATES[1]

    def test_re_entering_reports_the_latest_entry(self, ledger):
        frames = {"AAA": ramp()}
        bot = make_bot(ledger)
        bot.run_session(DATES[0], make_view(frames, DATES[0]))
        bot.run_session(DATES[1], make_view(frames, DATES[1]))

        bot.strategy = SellEverything()
        bot.run_session(DATES[2], make_view(frames, DATES[2]))
        bot.run_session(DATES[3], make_view(frames, DATES[3]))
        assert bot.broker.positions() == {}

        bot.strategy = BuyOnce()
        bot.run_session(DATES[4], make_view(frames, DATES[4]))
        bot.run_session(DATES[5], make_view(frames, DATES[5]))
        view = bot.portfolio_view(DATES[6], {"AAA": 106.0})
        assert view.entry_sessions["AAA"] == DATES[5], "the time stop restarts"


class TestRegistry:
    def test_a_valid_block_parses(self):
        spec = parse_spec("x", {
            "strategy": "trend",
            "params": {"fast": 5, "slow": 20},
            "starting_cash": 50_000,
            "sizing": "equal_weight",
            "risk": {"max_positions": 3},
        })
        assert spec.strategy_name == "trend"
        assert spec.risk.max_positions == 3
        assert spec.build_strategy().fast == 5

    def test_an_unknown_strategy_names_the_bot_and_the_options(self):
        with pytest.raises(ConfigError, match="unknown strategy"):
            parse_spec("mybot", {"strategy": "momentum"})

    def test_a_missing_strategy_is_refused(self):
        with pytest.raises(ConfigError, match="no strategy"):
            parse_spec("mybot", {"starting_cash": 1000})

    def test_bad_strategy_params_are_reported(self):
        spec = parse_spec("x", {"strategy": "trend", "params": {"nonsense": 1}})
        with pytest.raises(ConfigError, match="bad params"):
            spec.build_strategy()

    def test_an_invalid_strategy_configuration_is_refused(self):
        spec = parse_spec("x", {"strategy": "trend", "params": {"fast": 50, "slow": 20}})
        with pytest.raises(ValueError, match="shorter"):
            spec.build_strategy()

    def test_tickers_are_normalised(self):
        spec = parse_spec("x", {"strategy": "trend", "universe": ["brk.b", "aapl"]})
        assert spec.universe == ("BRK-B", "AAPL")

    def test_a_bad_risk_setting_names_the_bot(self):
        with pytest.raises(ConfigError, match="mybot"):
            parse_spec("mybot", {"strategy": "trend", "risk": {"max_postions": 1}})

    def test_negative_starting_cash_is_refused(self):
        with pytest.raises(ConfigError, match="starting_cash"):
            parse_spec("x", {"strategy": "trend", "starting_cash": -1})

    def test_the_shipped_config_loads(self):
        from bots.registry import load_specs

        specs = load_specs()
        assert len(specs) >= 2
        for spec in specs:
            spec.build_strategy()          # every bot must actually construct


class TestReplay:
    def test_a_replay_writes_to_the_backtest_ledger_only(self, tmp_path, ledger):
        """Backtest results must never be readable as live paper performance."""
        from bots.replay import replay_bot

        frames = {"AAA": ramp()}

        class FakeFetcher:
            def _load_prices(self, ticker):
                return frames.get(ticker, pd.DataFrame())

            def actions(self, ticker):
                return []

        spec = parse_spec("replay-bot", {
            "strategy": "trend",
            "params": {"fast": 3, "slow": 8},
            "universe": ["AAA"],
            "starting_cash": 100000,
            "sizing": {"type": "fixed_dollar", "amount": 10000},
            "risk": {"max_position_pct": None, "min_order_value": 0},
        })
        bt_path = tmp_path / "backtest.db"
        result = replay_bot(
            spec, FakeFetcher(), DATES[0], DATES[-1],
            ledger_path=bt_path, costs=ZERO_COSTS, universe=["AAA"],
        )

        assert result.sessions > 0
        with Ledger(bt_path) as bt:
            accounts = bt.accounts()
            assert len(accounts) == 1
            assert accounts[0].kind == BACKTEST
            # and a backtest holding never reaches the forward fetch list
            assert bt.held_tickers() == set()

        # the forward ledger from the fixture is untouched
        assert ledger.accounts() == []

    def test_replaying_twice_does_not_reuse_the_account(self, tmp_path):
        from bots.replay import replay_bot

        frames = {"AAA": ramp()}

        class FakeFetcher:
            def _load_prices(self, ticker):
                return frames.get(ticker, pd.DataFrame())

            def actions(self, ticker):
                return []

        spec = parse_spec("twice", {
            "strategy": "trend", "params": {"fast": 3, "slow": 8},
            "universe": ["AAA"], "starting_cash": 10000,
        })
        bt = tmp_path / "bt.db"
        first = replay_bot(spec, FakeFetcher(), DATES[0], DATES[-1],
                           ledger_path=bt, costs=ZERO_COSTS, universe=["AAA"])
        second = replay_bot(spec, FakeFetcher(), DATES[0], DATES[-1],
                            ledger_path=bt, costs=ZERO_COSTS, universe=["AAA"])
        assert first.bot_name != second.bot_name, "a rerun starts clean"
        assert first.final_equity == pytest.approx(second.final_equity)

    def test_an_end_before_the_start_is_refused(self, tmp_path):
        from bots.replay import replay_bot

        spec = parse_spec("x", {"strategy": "trend", "universe": ["AAA"]})
        with pytest.raises(ValueError, match="before start"):
            replay_bot(spec, None, DATES[5], DATES[0], ledger_path=tmp_path / "x.db")
