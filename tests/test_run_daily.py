"""The headless daily runner (phase 5). Offline — synthetic bars, synthetic
fetcher, a real but tiny XNYS calendar window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bots.bot import Bot
from bots.risk import RiskLimits
from bots.run_daily import process_all, run_bot, sessions_to_run
from bots.sizing import FixedDollar
from bots.strategy import OrderRequest, Strategy
from data import calendar
from engine.costs import ZERO_COSTS
from engine.ledger import Ledger
from engine.orders import Side

# ten consecutive real XNYS sessions, no holiday inside the window
SESSIONS = list(calendar.trading_days("2026-01-05", "2026-01-16"))
assert len(SESSIONS) == 10


def ramp(n: int, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    closes = start + np.arange(n, dtype=float) * step
    return pd.DataFrame(
        {"open": closes, "high": closes * 1.01, "low": closes * 0.99,
         "close": closes, "volume": np.full(n, 1e6)},
        index=SESSIONS[:n],
    )


class FakeFetcher:
    """Duck-types the two DataFetcher methods load_market_data needs."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def _load_prices(self, ticker: str) -> pd.DataFrame:
        return self.frames.get(ticker, pd.DataFrame())

    def actions(self, ticker: str) -> list:
        return []


class BuyOnce(Strategy):
    name = "buy once"

    def __init__(self, ticker="AAA"):
        super().__init__(ticker=ticker)
        self.ticker = ticker

    def on_bar(self, session, data, portfolio):
        if portfolio.holds(self.ticker) or portfolio.has_open_order(self.ticker):
            return []
        return [OrderRequest(ticker=self.ticker, side=Side.BUY, reason="always")]


class RaisesOn(Strategy):
    """Blows up the moment it is asked to decide on a named session —
    simulates a bot bug mid-session."""

    name = "boom"

    def __init__(self, fail_on):
        super().__init__()
        self.fail_on = pd.Timestamp(fail_on).normalize()

    def on_bar(self, session, data, portfolio):
        if session == self.fail_on:
            raise RuntimeError("bot blew up mid-session")
        return []


class FakeSpec:
    """Duck-types just what run_bot needs from a BotSpec, so a test can hand
    it an arbitrary Strategy instance without going through bots.yaml."""

    def __init__(self, name, strategy_factory, universe=("AAA",),
                 starting_cash=100_000.0, enabled=True):
        self.name = name
        self.strategy_factory = strategy_factory
        self.universe = universe
        self.starting_cash = starting_cash
        self.enabled = enabled

    def resolve_universe(self):
        return self.universe

    def build(self, ledger, costs=ZERO_COSTS, kind="forward", universe=None):
        return Bot.create(
            name=self.name, strategy=self.strategy_factory(), ledger=ledger,
            starting_cash=self.starting_cash,
            universe=universe if universe is not None else self.universe,
            sizing=FixedDollar(10_000.0),
            risk=RiskLimits(max_position_pct=None, min_order_value=0.0),
            costs=costs, kind=kind, enabled=self.enabled,
        )


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(tmp_path / "ledger.db")
    yield led
    led.close()


class TestSessionsToRun:
    def test_no_cursor_means_just_the_latest_session(self, ledger):
        bot = Bot.create("x", BuyOnce(), ledger, 10_000.0)
        assert sessions_to_run(ledger, bot.account_id, SESSIONS[4]) == [SESSIONS[4]]

    def test_an_existing_cursor_walks_forward_to_latest(self, ledger):
        bot = Bot.create("x", BuyOnce(), ledger, 10_000.0)
        ledger.set_last_processed_session(bot.account_id, SESSIONS[2])
        assert sessions_to_run(ledger, bot.account_id, SESSIONS[5]) == SESSIONS[3:6]

    def test_already_caught_up_is_empty(self, ledger):
        bot = Bot.create("x", BuyOnce(), ledger, 10_000.0)
        ledger.set_last_processed_session(bot.account_id, SESSIONS[5])
        assert sessions_to_run(ledger, bot.account_id, SESSIONS[5]) == []


class TestCatchUpAndIdempotency:
    def test_a_missed_day_is_caught_up_without_double_trading(self, ledger):
        frames = {"AAA": ramp(10)}
        fetcher = FakeFetcher(frames)
        spec = FakeSpec("bot", BuyOnce)

        # first run: no cursor yet -> processes exactly SESSIONS[2] alone
        outcome = run_bot(spec, ledger, fetcher, SESSIONS[2], costs=ZERO_COSTS)
        assert not outcome.failed
        assert [r.session for r in outcome.results] == [SESSIONS[2]]
        account_id = ledger.account_by_name("bot").id
        assert ledger.last_processed_session(account_id) == SESSIONS[2]

        # SESSIONS[3] is "missed" (the script simply never ran that day) —
        # the next run, at SESSIONS[4], must catch both up in one call
        outcome2 = run_bot(spec, ledger, fetcher, SESSIONS[4], costs=ZERO_COSTS)
        assert not outcome2.failed
        assert [r.session for r in outcome2.results] == [SESSIONS[3], SESSIONS[4]]
        assert ledger.last_processed_session(account_id) == SESSIONS[4]
        assert len(ledger.fills(account_id)) == 1, "BuyOnce trades exactly once total"

    def test_running_twice_at_the_same_latest_is_a_no_op(self, ledger):
        frames = {"AAA": ramp(10)}
        fetcher = FakeFetcher(frames)
        spec = FakeSpec("bot", BuyOnce)

        run_bot(spec, ledger, fetcher, SESSIONS[4], costs=ZERO_COSTS)
        account_id = ledger.account_by_name("bot").id
        cash_before = ledger.rebuild_cash(account_id)
        positions_before = ledger.rebuild_positions(account_id)
        fills_before = ledger.fills(account_id)
        cursor_before = ledger.last_processed_session(account_id)

        outcome = run_bot(spec, ledger, fetcher, SESSIONS[4], costs=ZERO_COSTS)
        assert outcome.results == [] and not outcome.failed

        assert ledger.rebuild_cash(account_id) == cash_before
        assert ledger.rebuild_positions(account_id) == positions_before
        assert ledger.fills(account_id) == fills_before
        assert ledger.last_processed_session(account_id) == cursor_before


class TestCrashSafety:
    def test_a_crash_mid_session_leaves_the_ledger_untouched_for_that_session(self, ledger):
        frames = {"AAA": ramp(10)}
        fetcher = FakeFetcher(frames)
        spec = FakeSpec("boom-bot", lambda: RaisesOn(SESSIONS[2]))

        outcome = run_bot(spec, ledger, fetcher, SESSIONS[2], costs=ZERO_COSTS)
        assert outcome.failed
        account_id = ledger.account_by_name("boom-bot").id
        # the broker's own process_session (mark-to-market + cursor advance)
        # ran first, inside the SAME transaction as the raising strategy —
        # the crash must roll that back too
        assert ledger.last_processed_session(account_id) is None
        assert ledger.rebuild_cash(account_id) == pytest.approx(100_000.0)
        assert ledger.snapshots(account_id) == []

    def test_a_crash_leaves_the_cursor_put_so_the_next_run_retries_it(self, ledger):
        frames = {"AAA": ramp(10)}
        fetcher = FakeFetcher(frames)
        fail_on = SESSIONS[2]
        spec = FakeSpec("boom-bot", lambda: RaisesOn(fail_on))

        first = run_bot(spec, ledger, fetcher, fail_on, costs=ZERO_COSTS)
        assert first.failed

        # a fixed strategy, same account: the next run must retry SESSIONS[2],
        # not skip past it
        spec2 = FakeSpec("boom-bot", BuyOnce)
        second = run_bot(spec2, ledger, fetcher, fail_on, costs=ZERO_COSTS)
        assert not second.failed
        assert [r.session for r in second.results] == [fail_on]


class TestIsolationAcrossBots:
    def test_one_failing_bot_does_not_stop_the_others(self, ledger):
        frames = {"AAA": ramp(10)}
        fetcher = FakeFetcher(frames)
        good_a = FakeSpec("good-a", BuyOnce)
        bad = FakeSpec("bad", lambda: RaisesOn(SESSIONS[2]))
        good_b = FakeSpec("good-b", BuyOnce)

        outcomes = process_all(
            [good_a, bad, good_b], ledger, fetcher, SESSIONS[2], costs=ZERO_COSTS
        )
        by_name = {o.spec_name: o for o in outcomes}
        assert not by_name["good-a"].failed and by_name["good-a"].results
        assert by_name["bad"].failed
        assert not by_name["good-b"].failed and by_name["good-b"].results


class TestPauseAndKill:
    def test_a_killed_bot_is_skipped_but_still_marked_to_market(self, ledger):
        frames = {"AAA": ramp(10)}
        fetcher = FakeFetcher(frames)
        spec = FakeSpec("k", BuyOnce)

        bot = Bot.create("k", BuyOnce(), ledger, 100_000.0, universe=("AAA",),
                         sizing=FixedDollar(10_000.0),
                         risk=RiskLimits(max_position_pct=None, min_order_value=0.0),
                         costs=ZERO_COSTS)
        bot.kill("test")

        outcome = run_bot(spec, ledger, fetcher, SESSIONS[2], costs=ZERO_COSTS)
        assert not outcome.failed
        assert outcome.results[0].skipped == "bot is killed"
        assert ledger.snapshots(bot.account_id), "still marked to market"
        assert ledger.rebuild_positions(bot.account_id) == {}

    def test_a_paused_bot_is_skipped_but_still_marked_to_market(self, ledger):
        frames = {"AAA": ramp(10)}
        fetcher = FakeFetcher(frames)
        spec = FakeSpec("p", BuyOnce)

        bot = Bot.create("p", BuyOnce(), ledger, 100_000.0, universe=("AAA",),
                         sizing=FixedDollar(10_000.0),
                         risk=RiskLimits(max_position_pct=None, min_order_value=0.0),
                         costs=ZERO_COSTS)
        bot.pause()

        outcome = run_bot(spec, ledger, fetcher, SESSIONS[2], costs=ZERO_COSTS)
        assert not outcome.failed
        assert outcome.results[0].skipped == "bot is paused"
        assert ledger.snapshots(bot.account_id)


def many(n: int, missing_on: pd.Timestamp | None = None, n_missing: int = 0):
    """n tickers T00..T{n-1}; the first n_missing have no bar on missing_on."""
    frames = {}
    for i in range(n):
        frame = ramp(10, start=100.0 + i)
        if i < n_missing:
            frame = frame.drop(missing_on)
        frames[f"T{i:02d}"] = frame
    return frames


class TestMissingBar:
    def test_a_lone_ticker_missing_its_bar_stalls_instead_of_advancing(self, ledger):
        """AAA has no row for SESSIONS[2]: 0 of 1 tickers is far below the
        coverage threshold, so the session is left for the next run rather
        than marked done on yesterday's price."""
        gapped = ramp(10).drop(SESSIONS[2])
        fetcher = FakeFetcher({"AAA": gapped})
        spec = FakeSpec("gapped", BuyOnce)

        outcome = run_bot(spec, ledger, fetcher, SESSIONS[2], costs=ZERO_COSTS)
        assert not outcome.failed
        assert outcome.stalled and outcome.results == []
        account_id = ledger.account_by_name("gapped").id
        assert ledger.last_processed_session(account_id) is None
        assert ledger.snapshots(account_id) == []

    def test_a_stale_refresh_stalls_then_resumes_once_the_bars_arrive(self, ledger):
        universe = tuple(f"T{i:02d}" for i in range(20))
        spec = FakeSpec("stale", BuyOnce, universe=universe)
        run_bot(spec, ledger, FakeFetcher(many(20)), SESSIONS[1], costs=ZERO_COSTS)
        account_id = ledger.account_by_name("stale").id

        # half the universe has not got SESSIONS[2] yet
        stale = FakeFetcher(many(20, missing_on=SESSIONS[2], n_missing=10))
        outcome = run_bot(spec, ledger, stale, SESSIONS[3], costs=ZERO_COSTS)
        assert not outcome.failed
        assert "10/20" in outcome.stalled
        assert outcome.results == [], "SESSIONS[3] must not run past the gap"
        assert ledger.last_processed_session(account_id) == SESSIONS[1]

        outcome = run_bot(spec, ledger, FakeFetcher(many(20)), SESSIONS[3],
                          costs=ZERO_COSTS)
        assert not outcome.stalled
        assert [r.session for r in outcome.results] == [SESSIONS[2], SESSIONS[3]]

    def test_one_halted_ticker_in_a_large_universe_is_tolerated_and_never_matched(
        self, ledger
    ):
        """19 of 20 is exactly the 95% threshold: the session runs, and the
        ticker without a bar never fills against nothing."""
        universe = tuple(f"T{i:02d}" for i in range(20))
        spec = FakeSpec("halt", BuyOnce, universe=universe)
        # BuyOnce trades T00, which is the ticker with the gap
        spec.strategy_factory = lambda: BuyOnce("T00")
        frames = many(20, missing_on=SESSIONS[3], n_missing=1)

        run_bot(spec, ledger, FakeFetcher(frames), SESSIONS[2], costs=ZERO_COSTS)
        outcome = run_bot(spec, ledger, FakeFetcher(frames), SESSIONS[3],
                          costs=ZERO_COSTS)
        assert not outcome.stalled and not outcome.failed
        assert [r.session for r in outcome.results] == [SESSIONS[3]]
        assert outcome.results[0].broker.fills == ()


class TestSetupFailure:
    def test_a_bot_failing_in_setup_does_not_stop_the_others(self, ledger):
        class BrokenUniverse(FakeSpec):
            def resolve_universe(self):
                raise RuntimeError("universe file unreadable")

        fetcher = FakeFetcher({"AAA": ramp(10)})
        outcomes = process_all(
            [FakeSpec("a", BuyOnce), BrokenUniverse("broken", BuyOnce),
             FakeSpec("b", BuyOnce)],
            ledger, fetcher, SESSIONS[2], costs=ZERO_COSTS,
        )
        by_name = {o.spec_name: o for o in outcomes}
        assert by_name["broken"].failed
        assert by_name["broken"].error.startswith("setup:")
        assert by_name["a"].results and by_name["b"].results


class TestMainCli:
    """`main` end to end with the network, the real data dir and bots.yaml
    swapped out."""

    @pytest.fixture
    def cli(self, tmp_path, monkeypatch):
        import bots.run_daily as rd

        frames = {"AAA": ramp(10)}
        specs = [FakeSpec("on", BuyOnce), FakeSpec("off", BuyOnce, enabled=False)]

        class Fetcher(FakeFetcher):
            def __init__(self, _path):
                super().__init__(frames)

            def close(self):
                pass

        monkeypatch.setattr(rd.paths, "setup_logging", lambda: None)
        monkeypatch.setattr(rd.paths, "ledger_path", lambda: tmp_path / "l.db")
        monkeypatch.setattr(rd.paths, "db_path", lambda: tmp_path / "c.db")
        monkeypatch.setattr(rd, "configure_yfinance", lambda _d: None)
        monkeypatch.setattr(rd, "load_specs", lambda: specs)
        monkeypatch.setattr(rd, "DataFetcher", Fetcher)
        monkeypatch.setattr(
            rd.calendar, "latest_complete_session", lambda now=None: SESSIONS[4]
        )
        return rd, frames

    def test_a_normal_run_exits_zero(self, cli):
        rd, _ = cli
        assert rd.main(["--no-fetch"]) == rd.EXIT_OK

    def test_until_after_the_latest_complete_session_is_refused(self, cli):
        rd, _ = cli
        assert rd.main(["--no-fetch", "--until", str(SESSIONS[5].date())]) == 2

    def test_until_on_a_weekend_is_refused(self, cli):
        rd, _ = cli
        assert rd.main(["--no-fetch", "--until", "2026-01-10"]) == 2

    def test_until_that_is_not_a_date_is_refused(self, cli):
        rd, _ = cli
        assert rd.main(["--no-fetch", "--until", "next tuesday"]) == 2

    def test_naming_a_disabled_bot_is_an_error(self, cli):
        rd, _ = cli
        assert rd.main(["--no-fetch", "--bot", "off"]) == 2

    def test_a_bot_waiting_for_data_exits_three(self, cli):
        rd, frames = cli
        frames["AAA"] = frames["AAA"].drop(SESSIONS[4])
        assert rd.main(["--no-fetch"]) == rd.EXIT_WAITING


class TestNoQtImport:
    def test_run_daily_imports_no_qt(self):
        import subprocess
        import sys

        code = (
            "import bots.run_daily\n"
            "import sys\n"
            "assert not any(m.startswith('PyQt6') for m in sys.modules), "
            "'Qt leaked into bots.run_daily import graph'\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stdout + result.stderr
