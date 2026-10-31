"""Phase 6: append-only account status, bot -> account mapping and reset,
the admin controls, the Bots-tab report, and an offscreen smoke of the panel.
Offline: synthetic bars and a fake price cache.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from bots import admin, report
from bots.bot import Bot
from bots.registry import parse_spec
from bots.run_daily import run_bot
from data import calendar
from engine.costs import ZERO_COSTS
from engine.ledger import (
    ACTIVE,
    KILLED,
    PAUSED,
    SOURCE_CLI,
    SOURCE_MIGRATION,
    SOURCE_UI,
    Ledger,
    LedgerError,
)
from engine.orders import Order, OrderStatus, Side
from engine.sim_broker import Bar

SESSIONS = list(calendar.trading_days("2026-01-05", "2026-01-16"))
assert len(SESSIONS) == 10


def ramp(n: int = 10, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    closes = start + np.arange(n, dtype=float) * step
    return pd.DataFrame(
        {"open": closes, "high": closes * 1.01, "low": closes * 0.99,
         "close": closes, "volume": np.full(n, 1e6)},
        index=SESSIONS[:n],
    )


class FakePrices:
    """Duck-types the DataFetcher reads that load_market_data and the report
    make: `_load_prices`, `adjusted_closes`, `actions`."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def _load_prices(self, ticker):
        return self.frames.get(ticker, pd.DataFrame())

    def adjusted_closes(self, ticker):
        return self.frames[ticker]["close"]

    def actions(self, ticker):
        return []

    def close(self):
        pass


def spec(name="bot", liquidate=False, cash=100_000.0):
    return parse_spec(name, {
        "strategy": "trend",
        "universe": ["AAA"],
        "starting_cash": cash,
        "risk": {"max_position_pct": None, "min_order_value": 0.0,
                 "liquidate_on_kill": liquidate},
    })


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(tmp_path / "ledger.db")
    yield led
    led.close()


def open_bot(ledger, s) -> Bot:
    return s.build(ledger, costs=ZERO_COSTS, universe=("AAA",))


def buy(bot: Bot, session, qty=10) -> Order:
    bot.broker.set_mark("AAA", 100.0)
    return bot.broker.submit(Order(
        account_id=bot.account_id, created_session=session, ticker="AAA",
        side=Side.BUY, qty=qty, reason="test",
    ))


def bar(price=100.0) -> Bar:
    return Bar(price, price * 1.01, price * 0.99, price)


# ------------------------------------------------------------------ status


class TestStatusIsAppendOnly:
    def test_each_change_appends_a_row_and_the_latest_wins(self, ledger):
        account = ledger.create_account("x", 1_000.0)
        ledger.set_account_status(account.id, PAUSED, source=SOURCE_UI, reason="hols")
        ledger.set_account_status(account.id, ACTIVE, source=SOURCE_CLI)
        history = ledger.status_history(account.id)
        assert [h["status"] for h in history] == [ACTIVE, PAUSED, ACTIVE]
        assert [h["source"] for h in history][1:] == [SOURCE_UI, SOURCE_CLI]
        assert history[1]["reason"] == "hols"
        assert ledger.account_status(account.id) == ACTIVE

    def test_the_v1_column_is_never_updated(self, ledger):
        account = ledger.create_account("x", 1_000.0)
        ledger.set_account_status(account.id, KILLED, source=SOURCE_UI, reason="r")
        raw = ledger.connection.execute(
            "SELECT status FROM accounts WHERE id=?", (account.id,)
        ).fetchone()[0]
        assert raw == ACTIVE
        assert ledger.account_status(account.id) == KILLED

    @pytest.mark.parametrize("after", [ACTIVE, PAUSED, KILLED])
    def test_killed_is_final(self, ledger, after):
        account = ledger.create_account("x", 1_000.0)
        ledger.set_account_status(account.id, KILLED, source=SOURCE_UI, reason="r")
        with pytest.raises(LedgerError, match="killed"):
            ledger.set_account_status(account.id, after, source=SOURCE_UI)
        assert len(ledger.status_history(account.id)) == 2

    def test_a_drawdown_kill_records_the_risk_source(self, ledger):
        bot = open_bot(ledger, spec())
        bot._kill(SESSIONS[3], "drawdown 30% > 25%", {}, "risk")
        last = ledger.status_history(bot.account_id)[-1]
        assert (last["status"], last["source"], last["session"]) == (
            KILLED, "risk", "2026-01-08",
        )
        assert "drawdown" in last["reason"]


class TestMigrationFromV1:
    def _make_v1(self, path, status="paused"):
        led = Ledger(path)
        account = led.create_account("old-bot", 5_000.0)
        led.close()
        conn = sqlite3.connect(str(path))
        conn.execute("DELETE FROM account_status_events")
        conn.execute("DELETE FROM bot_accounts")
        conn.execute("UPDATE accounts SET status=?", (status,))
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()
        return account.id

    def test_status_and_mapping_are_seeded_once(self, tmp_path):
        path = tmp_path / "v1.db"
        account_id = self._make_v1(path)
        with Ledger(path) as led:
            assert led.account_status(account_id) == PAUSED
            assert led.status_history(account_id)[0]["source"] == SOURCE_MIGRATION
            assert led.bot_account("old-bot").id == account_id
        with Ledger(path) as led:  # reopening a v2 file seeds nothing more
            assert len(led.status_history(account_id)) == 1
            assert len(led.bot_account_history("old-bot")) == 1


# ------------------------------------------------------------------ admin


class TestControls:
    def test_a_bot_with_no_account_is_refused_and_none_is_opened(self, ledger):
        with pytest.raises(admin.AdminError, match="no account yet"):
            admin.pause(ledger, spec(), SOURCE_UI)
        assert ledger.accounts() == []

    def test_pause_resume_round_trip_and_wrong_state_refusals(self, ledger):
        s = spec()
        open_bot(ledger, s)
        with pytest.raises(admin.AdminError, match="only a paused"):
            admin.resume(ledger, s, SOURCE_UI)
        admin.pause(ledger, s, SOURCE_UI, "earnings")
        with pytest.raises(admin.AdminError, match="only an active"):
            admin.pause(ledger, s, SOURCE_UI)
        admin.resume(ledger, s, SOURCE_UI)
        assert ledger.bot_account("bot").status == ACTIVE

    def test_kill_needs_a_reason(self, ledger):
        s = spec()
        open_bot(ledger, s)
        with pytest.raises(admin.AdminError, match="reason"):
            admin.kill(ledger, s, SOURCE_UI, "  ")
        assert ledger.bot_account("bot").status == ACTIVE

    def test_kill_cancels_resting_orders_so_nothing_fills_next_session(self, ledger):
        s = spec()
        bot = open_bot(ledger, s)
        bot.broker.process_session(SESSIONS[0], {"AAA": bar()})
        order = buy(bot, SESSIONS[0])
        assert order.status is OrderStatus.PENDING

        admin.kill(ledger, s, SOURCE_UI, "retired")
        assert ledger.get_order(order.id).status is OrderStatus.CANCELLED

        outcome = run_bot(s, ledger, FakePrices({"AAA": ramp()}), SESSIONS[1],
                          costs=ZERO_COSTS)
        assert not outcome.failed
        assert ledger.fills(bot.account_id) == []
        assert ledger.bot_account("bot").status == KILLED

    def test_a_kill_that_fails_midway_changes_nothing(self, ledger, monkeypatch):
        s = spec()
        bot = open_bot(ledger, s)
        bot.broker.process_session(SESSIONS[0], {"AAA": bar()})
        order = buy(bot, SESSIONS[0])

        def boom(*a, **k):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(Ledger, "set_account_status", boom)
        with pytest.raises(admin.AdminError, match="busy"):
            admin.kill(ledger, s, SOURCE_UI, "retired")
        monkeypatch.undo()
        assert ledger.get_order(order.id).status is OrderStatus.PENDING
        assert ledger.bot_account("bot").status == ACTIVE

    def test_kill_follows_liquidate_on_kill(self, ledger):
        s = spec(liquidate=True)
        bot = open_bot(ledger, s)
        bot.broker.process_session(SESSIONS[0], {"AAA": bar()})
        buy(bot, SESSIONS[0], qty=10)
        bot.broker.process_session(SESSIONS[1], {"AAA": bar()})
        assert ledger.rebuild_positions(bot.account_id)["AAA"].qty == 10

        admin.kill(ledger, s, SOURCE_UI, "retired")
        sells = [o for o in ledger.open_orders(bot.account_id) if o.side is Side.SELL]
        assert len(sells) == 1 and sells[0].qty == 10
        # created on the last processed session, so it fills on the next one
        assert sells[0].created_session == SESSIONS[1]

        bot.broker.process_session(SESSIONS[2], {"AAA": bar(101.0)})
        assert ledger.rebuild_positions(bot.account_id) == {}

    def test_kill_without_liquidate_holds_positions(self, ledger):
        s = spec(liquidate=False)
        bot = open_bot(ledger, s)
        bot.broker.process_session(SESSIONS[0], {"AAA": bar()})
        buy(bot, SESSIONS[0])
        bot.broker.process_session(SESSIONS[1], {"AAA": bar()})
        admin.kill(ledger, s, SOURCE_UI, "retired")
        assert ledger.open_orders(bot.account_id) == []
        assert ledger.rebuild_positions(bot.account_id)["AAA"].qty == 10

    def test_the_runner_honours_a_pause_written_between_runs(self, ledger):
        s = spec()
        prices = FakePrices({"AAA": ramp()})
        run_bot(s, ledger, prices, SESSIONS[2], costs=ZERO_COSTS)
        admin.pause(ledger, s, SOURCE_UI)
        outcome = run_bot(s, ledger, prices, SESSIONS[5], costs=ZERO_COSTS)
        account_id = ledger.bot_account("bot").id
        assert ledger.last_processed_session(account_id) == SESSIONS[5]
        assert ledger.orders(account_id) == []
        assert all(r.skipped == "bot is paused" for r in outcome.results)


class TestReset:
    def test_reset_is_refused_unless_killed(self, ledger):
        s = spec()
        open_bot(ledger, s)
        with pytest.raises(admin.AdminError, match="kill it before"):
            admin.reset(ledger, s)

    def test_reset_opens_a_fresh_account_and_keeps_the_old_one(self, ledger):
        s = spec(cash=50_000.0)
        old = open_bot(ledger, s)
        old.broker.process_session(SESSIONS[0], {"AAA": bar()})
        buy(old, SESSIONS[0])
        old.broker.process_session(SESSIONS[1], {"AAA": bar()})
        fills_before = ledger.fills(old.account_id)
        admin.kill(ledger, s, SOURCE_UI, "retired")

        fresh = admin.reset(ledger, s)
        assert fresh.name == "bot#2"
        assert fresh.status == ACTIVE
        assert ledger.rebuild_cash(fresh.id) == 50_000.0
        assert ledger.bot_account("bot").id == fresh.id
        assert [a.name for a in ledger.bot_account_history("bot")] == ["bot", "bot#2"]
        assert ledger.fills(old.account_id) == fills_before
        assert ledger.account_status(old.account_id) == KILLED
        # the runner and every Bot now bind to the new account
        assert open_bot(ledger, s).account_id == fresh.id

    def test_a_reset_account_starts_at_the_next_run_not_a_backfill(self, ledger):
        s = spec()
        prices = FakePrices({"AAA": ramp()})
        run_bot(s, ledger, prices, SESSIONS[2], costs=ZERO_COSTS)
        admin.kill(ledger, s, SOURCE_UI, "retired")
        fresh = admin.reset(ledger, s)
        outcome = run_bot(s, ledger, prices, SESSIONS[6], costs=ZERO_COSTS)
        assert not outcome.failed
        assert ledger.last_processed_session(fresh.id) == SESSIONS[6]
        assert [snap.session for snap in ledger.snapshots(fresh.id)] == [SESSIONS[6]]

    def test_a_second_reset_numbers_on(self, ledger):
        s = spec()
        open_bot(ledger, s)
        admin.kill(ledger, s, SOURCE_UI, "one")
        admin.reset(ledger, s)
        admin.kill(ledger, s, SOURCE_UI, "two")
        assert admin.reset(ledger, s).name == "bot#3"


# ------------------------------------------------------------------ report


class TestReportMaths:
    def test_max_drawdown(self):
        curve = pd.Series([100.0, 120.0, 90.0, 110.0, 130.0, 117.0])
        assert report.max_drawdown(curve) == pytest.approx(0.25)
        assert report.max_drawdown(pd.Series(dtype=float)) is None
        assert str(report.max_drawdown(pd.Series([5.0, 5.0]))) == "0.0"

    def test_benchmark_return_needs_both_closes(self):
        adj = pd.Series([400.0, 410.0, 420.0], index=SESSIONS[:3])
        assert report.benchmark_return(adj, SESSIONS[0], SESSIONS[2]) == pytest.approx(0.05)
        assert report.benchmark_return(adj, SESSIONS[0], SESSIONS[5]) is None
        assert report.benchmark_return(pd.Series(dtype=float), SESSIONS[0], SESSIONS[2]) is None

    def test_benchmark_curve_is_rebased_to_starting_cash(self):
        adj = pd.Series([400.0, 410.0, 420.0], index=SESSIONS[:3])
        curve = report.benchmark_curve(adj, pd.DatetimeIndex(SESSIONS[1:3]), 1_000.0)
        assert list(curve.round(4)) == [1_000.0, pytest.approx(1_024.3902, abs=1e-4)]


class TestSummary:
    def test_a_bot_that_never_ran(self, ledger):
        summary = report.summarise(ledger, spec(), SESSIONS[5])
        assert summary.status == "not started"
        assert summary.equity is None and not summary.behind

    def test_return_benchmark_and_behind(self, ledger):
        s = spec(cash=100_000.0)
        prices = FakePrices({"AAA": ramp(), "QQQ": ramp(start=400.0, step=4.0)})
        bot = open_bot(ledger, s)
        bot.broker.process_session(SESSIONS[0], {"AAA": bar()})
        buy(bot, SESSIONS[0], qty=100)
        bot.broker.process_session(SESSIONS[1], {"AAA": bar(100.0)})
        bot.broker.process_session(SESSIONS[2], {"AAA": bar(110.0)})

        summary = report.summarise(ledger, s, SESSIONS[4], prices)
        # bought 100 @ 100 on S1; marked 110 on S2 -> +1,000 on 100,000
        assert summary.equity == pytest.approx(101_000.0)
        assert summary.total_return == pytest.approx(0.01)
        # QQQ 400 (S0) -> 408 (S2): the first snapshot is S0
        assert summary.benchmark_return == pytest.approx(0.02)
        assert summary.excess_return == pytest.approx(-0.01)
        assert summary.positions == 1
        assert summary.behind  # last session S2 < latest S4

    def test_positions_are_marked_at_the_last_processed_session(self, ledger):
        s = spec()
        prices = FakePrices({"AAA": ramp()})  # close on S1 = 101
        bot = open_bot(ledger, s)
        bot.broker.process_session(SESSIONS[0], {"AAA": bar()})
        buy(bot, SESSIONS[0], qty=10)
        bot.broker.process_session(SESSIONS[1], {"AAA": bar(100.0)})
        account = ledger.bot_account("bot")
        [row] = report.positions(ledger, account, prices)
        assert row.mark == 101.0 and row.mark_session == SESSIONS[1]
        assert row.unrealised == pytest.approx(10.0)

    def test_activity_is_newest_first_and_skips_fill_mirrors(self, ledger):
        s = spec()
        bot = open_bot(ledger, s)
        bot.broker.process_session(SESSIONS[0], {"AAA": bar()})
        buy(bot, SESSIONS[0], qty=10)
        bot.broker.process_session(SESSIONS[1], {"AAA": bar()})
        admin.pause(ledger, s, SOURCE_UI, "hols")
        rows = report.activity(ledger, ledger.bot_account("bot"))
        kinds = [r.kind for r in rows]
        assert "buy" not in kinds and "commission" not in kinds
        assert kinds.count("fill") == 1
        assert "deposit" in kinds
        sessions = [r.session for r in rows if r.session]
        assert sessions == sorted(sessions, reverse=True)


# ------------------------------------------------------------------ UI smoke


@pytest.fixture
def qt_app():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class TestPanelSmoke:
    def _panel(self, tmp_path, monkeypatch, specs):
        import paths
        import ui.bots_panel as bp

        monkeypatch.setattr(paths, "ledger_path", lambda: tmp_path / "ledger.db")
        monkeypatch.setattr(paths, "db_path", lambda: tmp_path / "cache.db")
        monkeypatch.setattr(bp, "load_specs", lambda: specs)
        panel = bp.BotsPanel()
        panel.refresh()
        return panel

    def test_builds_on_an_empty_ledger(self, qt_app, tmp_path, monkeypatch):
        panel = self._panel(tmp_path, monkeypatch, [spec("a"), spec("b")])
        assert panel._overview.rowCount() == 2
        assert panel._overview.item(0, 1).text() == "not started"
        assert not panel._pause_btn.isEnabled() and not panel._kill_btn.isEnabled()

    def test_pause_kill_and_reset_through_the_buttons(self, qt_app, tmp_path, monkeypatch):
        s = spec("a")
        with Ledger(tmp_path / "ledger.db") as led:
            bot = open_bot(led, s)
            bot.broker.process_session(SESSIONS[0], {"AAA": bar()})
            buy(bot, SESSIONS[0])
            bot.broker.process_session(SESSIONS[1], {"AAA": bar()})

        panel = self._panel(tmp_path, monkeypatch, [s])
        warnings = []
        monkeypatch.setattr(panel, "_ask_reason", lambda *a: "from the test")
        monkeypatch.setattr(panel, "_confirm", lambda *a: True)
        monkeypatch.setattr(panel, "_warn", warnings.append)

        assert panel._positions.rowCount() == 1
        assert panel._orders.rowCount() == 1
        assert panel._activity.rowCount() > 0

        panel._on_pause()
        assert panel._overview.item(0, 1).text() == PAUSED
        assert panel._resume_btn.isEnabled() and not panel._pause_btn.isEnabled()

        panel._on_kill()
        assert panel._overview.item(0, 1).text() == KILLED
        assert panel._reset_btn.isEnabled() and not panel._kill_btn.isEnabled()

        panel._on_reset()
        assert panel._overview.item(0, 1).text() == ACTIVE
        assert panel._positions.rowCount() == 0
        assert warnings == []
        panel.deleteLater()
