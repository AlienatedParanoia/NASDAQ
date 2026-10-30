"""The headless daily forward-trading runner (phase 5).

`python -m bots.run_daily`, scheduled once a day (see scripts/install_task.ps1)
well after the close so `data.calendar.latest_complete_session()` never has to
guess. Nobody is watching when this runs, so the properties that matter are
guarantees, not features:

  * **Never acts on a forming bar.** The upper bound on every session walked is
    `latest_complete_session()`, which respects early closes.
  * **Re-running is a no-op.** Each account's own `last_processed_session`
    cursor decides what is left to do; nothing here tracks "today" separately.
  * **Catch-up cannot double-trade.** A missed day is simply more sessions
    between the cursor and `latest` — same cursor, same guarantee.
  * **A crash leaves a consistent ledger.** Each bot's each session runs inside
    one `ledger.transaction()` (see engine/ledger.py): a failure rolls back
    that session in full, including the broker's own mark-to-market and cursor
    advance, so the retry next run sees exactly the state it would have seen
    had this run never touched that session.
  * **Never trades a session whose prices have not arrived.** If fewer than
    `MIN_SESSION_COVERAGE` of a bot's tickers have a bar for the session (a
    stale cache, a failed refresh), the bot stops *before* that session and
    leaves its cursor alone, so the next run retries it once the data is in.
    Advancing anyway would mark the day done forever on yesterday's prices.
  * **One bad bot is contained.** A failure — in setup or mid-session — is
    caught per bot, logged with its traceback, and does not stop the rest —
    but the process still exits non-zero so the failure is not silent.

Exit codes: 0 fine / nothing to do, 1 a bot failed, 2 a configuration or
argument error before any bot ran, 3 a bot is waiting for price data (no
failure; it retries next run).

No Qt import anywhere in this module's import graph (checked by
tests/test_run_daily.py): a stray `from ui...` would make the scheduled task
need a desktop session.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field

import pandas as pd

import paths
from bots.bot import BotSessionResult
from bots.data_view import DataView
from bots.registry import BotSpec, ConfigError, load_specs
from bots.replay import load_market_data
from data import calendar
from data.fetcher import DataFetcher, configure_yfinance, tickers_to_fetch
from engine.costs import DEFAULT_COSTS, CostModel
from engine.ledger import Ledger

log = logging.getLogger(__name__)

# Share of a bot's tickers that must have a bar on a session before it is
# processed. Below 1.0 so a single halted stock in a large universe does not
# stall every bot; well above what a stale or partially failed refresh leaves.
MIN_SESSION_COVERAGE = 0.95

EXIT_OK, EXIT_FAILED, EXIT_SETUP, EXIT_WAITING = 0, 1, 2, 3


class _DryRunAbort(Exception):
    """Raised once, after every bot has been processed, to roll back the
    single outer transaction a --dry-run wraps everything in. Never escapes
    `main`."""


@dataclass(slots=True)
class BotRunOutcome:
    """What happened to one bot over however many sessions it owed."""

    spec_name: str
    results: list[BotSessionResult] = field(default_factory=list)
    failed: bool = False
    error: str = ""
    skipped: str = ""  # e.g. "no cached price data" — a setup problem, not a crash
    stalled: str = ""  # waiting for a session's prices; cursor left in place


def sessions_to_run(
    ledger: Ledger, account_id: int, latest: pd.Timestamp
) -> list[pd.Timestamp]:
    """Every session this account owes, through `latest`.

    No cursor yet means the account has never processed a single session —
    its first live session is `latest` itself, not a years-long backfill. An
    account opened today should look like it started trading today, not like
    it has been live since inception.
    """
    cursor = ledger.last_processed_session(account_id)
    if cursor is None:
        return [latest]
    if cursor >= latest:
        return []
    start = calendar.next_session(cursor)
    return list(calendar.trading_days(start, latest))


def _max_bars(view: DataView, frames: dict) -> int:
    return max((view.bars_available(t) for t in frames), default=0)


def session_coverage(view: DataView, frames: dict) -> tuple[int, int]:
    """`(have, expected)` for the view's session: `expected` counts tickers
    that had started trading by then (first bar on or before the session),
    `have` those of them with a bar on the session itself. A ticker that
    lists later never counts against an earlier session."""
    session = view.session
    expected = have = 0
    for ticker, frame in frames.items():
        if frame is None or frame.empty or frame.index.min() > session:
            continue
        expected += 1
        if view.bar(ticker) is not None:
            have += 1
    return have, expected


def _actions_on(actions_by_ticker: dict, session: pd.Timestamp) -> list:
    out = []
    for actions in actions_by_ticker.values():
        out.extend(
            a for a in actions if pd.Timestamp(a.ex_date).normalize() == session
        )
    return out


def run_bot(
    spec: BotSpec,
    ledger: Ledger,
    fetcher: DataFetcher,
    latest: pd.Timestamp,
    costs: CostModel = DEFAULT_COSTS,
) -> BotRunOutcome:
    """Advance one bot through every session it owes, one transaction per
    session. Never raises — a bot's own failure is caught and returned."""
    outcome = BotRunOutcome(spec_name=spec.name)
    try:
        universe = spec.resolve_universe()
        bot = spec.build(ledger, costs=costs, universe=universe)

        sessions = sessions_to_run(ledger, bot.account_id, latest)
        if not sessions:
            return outcome

        frames, adjusted, actions_by_ticker = load_market_data(
            fetcher, list(universe), end=latest
        )
        if not frames:
            outcome.skipped = (
                f"no cached price data for any of {len(universe)} ticker(s)"
            )
            log.warning("%s: %s", spec.name, outcome.skipped)
            return outcome

        base_view = DataView(
            session=sessions[0], frames=frames, adjusted_closes=adjusted
        )
        warmup = bot.strategy.warmup_bars()
    except Exception as e:  # noqa: BLE001 — isolate this bot, keep the run going
        outcome.failed = True
        outcome.error = f"setup: {e}"
        log.exception("%s: setup failed; nothing written for this bot", spec.name)
        return outcome

    for session in sessions:
        view = base_view.for_session(session)
        have, expected = session_coverage(view, frames)
        if expected == 0 or have < MIN_SESSION_COVERAGE * expected:
            outcome.stalled = (
                f"{session.date()}: only {have}/{expected} ticker(s) have a bar "
                f"(need {MIN_SESSION_COVERAGE:.0%}); cursor unchanged — will "
                f"retry next run"
            )
            log.warning("%s: waiting for price data — %s", spec.name, outcome.stalled)
            break
        actions = _actions_on(actions_by_ticker, session)
        try:
            with ledger.transaction():
                if _max_bars(view, frames) < warmup:
                    # still inside the indicator warmup: advance the broker (so
                    # marks and the equity curve exist) without consulting a
                    # strategy whose indicators are not fully formed yet
                    bot.broker.process_session(
                        session, bot._bars_for(session, view), actions
                    )
                else:
                    outcome.results.append(bot.run_session(session, view, actions))
        except Exception as e:  # noqa: BLE001 — isolate this bot, keep the run going
            outcome.failed = True
            outcome.error = f"{session.date()}: {e}"
            log.exception(
                "%s: session %s failed; rolled back, cursor unchanged — will "
                "retry next run",
                spec.name, session.date(),
            )
            break

    return outcome


def process_all(
    specs: list[BotSpec],
    ledger: Ledger,
    fetcher: DataFetcher,
    latest: pd.Timestamp,
    costs: CostModel = DEFAULT_COSTS,
) -> list[BotRunOutcome]:
    return [run_bot(spec, ledger, fetcher, latest, costs=costs) for spec in specs]


def _log_outcome(outcome: BotRunOutcome) -> None:
    if outcome.skipped:
        log.info("%s: skipped (%s)", outcome.spec_name, outcome.skipped)
        return
    if not outcome.results and not outcome.failed and not outcome.stalled:
        log.info("%s: already up to date, nothing to do", outcome.spec_name)
        return
    for result in outcome.results:
        log.info(
            "%s: %s — %d requested, %d submitted, equity %.2f%s",
            outcome.spec_name, result.session.date(), result.requested,
            result.submitted, result.equity,
            f", KILLED ({result.kill_reason})" if result.killed else "",
        )
    if outcome.stalled:
        log.warning("%s: WAITING FOR DATA — %s", outcome.spec_name, outcome.stalled)
    if outcome.failed:
        log.error("%s: FAILED — %s", outcome.spec_name, outcome.error)


def _resolve_latest(until: str | None) -> pd.Timestamp:
    """The last session to process. `--until` may only move it *back*: a date
    after the latest complete session would trade on a bar still forming, and
    a non-session date would process a day the market never had. Neither is
    snapped to a nearby session — that would be guessing."""
    latest = calendar.latest_complete_session()
    if until is None:
        return latest
    try:
        day = pd.Timestamp(until).normalize()
    except (ValueError, TypeError) as e:
        raise ConfigError(f"--until {until!r} is not a date: {e}") from e
    if day > latest:
        raise ConfigError(
            f"--until {day.date()} is after the latest complete session "
            f"({latest.date()}); its bar may still be forming"
        )
    if not calendar.is_trading_day(day):
        raise ConfigError(f"--until {day.date()} is not an XNYS trading day")
    return day


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advance every enabled bot through the sessions it owes."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="compute everything but write nothing to the ledger",
    )
    parser.add_argument(
        "--bot", metavar="NAME", help="run only the named bot (see bots.yaml)",
    )
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="use the price cache as-is; do not refresh from Yahoo",
    )
    parser.add_argument(
        "--until", metavar="DATE",
        help="stop at DATE instead of the latest complete session; must be a "
        "trading day no later than that session (for testing catch-up)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    paths.setup_logging()
    args = _parse_args(argv)

    try:
        configure_yfinance(paths.data_dir())
        specs = load_specs()
        if args.bot is not None:
            specs = [s for s in specs if s.name == args.bot]
            if not specs:
                log.error("no such bot %r in bots.yaml", args.bot)
                return EXIT_SETUP
            if not specs[0].enabled:
                log.error("bot %r is disabled in bots.yaml", args.bot)
                return EXIT_SETUP
        specs = [s for s in specs if s.enabled]
        if not specs:
            log.info("no enabled bots; nothing to do")
            return EXIT_OK

        latest = _resolve_latest(args.until)

        fetcher = DataFetcher(paths.db_path())
        if not args.no_fetch:
            errors = fetcher.refresh(tickers_to_fetch())
            if errors:
                log.warning("refresh had %d error(s): %s", len(errors), errors)
    except ConfigError as e:
        log.error("configuration error: %s", e)
        return EXIT_SETUP
    except Exception:
        log.exception("setup failed before any bot ran")
        return EXIT_SETUP

    ledger = Ledger(paths.ledger_path())
    try:
        if args.dry_run:
            outcomes: list[BotRunOutcome] = []
            try:
                with ledger.transaction():
                    outcomes = process_all(specs, ledger, fetcher, latest)
                    raise _DryRunAbort()
            except _DryRunAbort:
                pass
            log.info("--dry-run: nothing written to the ledger")
        else:
            outcomes = process_all(specs, ledger, fetcher, latest)

        for outcome in outcomes:
            _log_outcome(outcome)

        if any(o.failed for o in outcomes):
            return EXIT_FAILED
        return EXIT_WAITING if any(o.stalled for o in outcomes) else EXIT_OK
    except Exception:
        # run_bot never raises, so this is a bug in the runner itself — still
        # log it to advisor.log, which is all a scheduled task leaves behind
        log.exception("run failed outside any single bot")
        return EXIT_FAILED
    finally:
        ledger.close()
        fetcher.close()


if __name__ == "__main__":
    sys.exit(main())
