"""Pause, resume, kill and reset a bot: the only writes the Bots tab makes.

Both the Bots tab (ui/bots_panel.py) and the command line go through here, so
the rules live in one place rather than in a widget:

  * **pause** an active bot: it stops making decisions. Orders it already
    placed still fill, because pause is the soft stop.
  * **resume** a paused bot.
  * **kill**: final. Cancels every resting order in the same transaction,
    because the broker fills yesterday's orders *before* a bot checks its
    status, so a status-only kill would let them through next morning.
    Liquidates if the bot's `liquidate_on_kill` says so, exactly like the
    drawdown kill.
  * **reset** a killed bot: open a fresh account (`<bot>#2`, `#3`...) at the
    starting cash in bots.yaml. The old account and its history are untouched.

Each one is a single ledger transaction. The ledger is WAL with a 5s busy
timeout, so a click during the 06:30 run waits rather than failing; if the lock
still is not free it is reported, never retried blindly.

    python -m bots.admin status
    python -m bots.admin pause trend-follower --reason "earnings week"
    python -m bots.admin kill mean-reverter --reason "strategy retired"
    python -m bots.admin reset mean-reverter
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import sqlite3
import sys

import paths
from bots.bot import Bot
from bots.registry import BotSpec, ConfigError, load_specs
from engine.ledger import (
    ACTIVE,
    FORWARD,
    KILLED,
    PAUSED,
    SOURCE_CLI,
    Account,
    Ledger,
    LedgerError,
)

log = logging.getLogger(__name__)


class AdminError(Exception):
    """A control that cannot be applied, with a sentence saying why."""


def _current(ledger: Ledger, spec: BotSpec) -> Account:
    account = ledger.bot_account(spec.name)
    if account is None:
        raise AdminError(
            f"{spec.name} has no account yet; it opens one on its first daily run"
        )
    return account


def _bind(ledger: Ledger, spec: BotSpec, account: Account) -> Bot:
    """A Bot on the bot's current account. Not `spec.build`: that would open
    an account as a side effect, and a control must never create one."""
    return Bot(
        name=spec.name, strategy=spec.build_strategy(), ledger=ledger,
        account_id=account.id, sizing=spec.sizing, risk=spec.risk,
        enabled=spec.enabled,
    )


@contextlib.contextmanager
def _writing():
    """Turn the two ways a write can be refused into an AdminError: the lock
    is held past the busy timeout, or the status changed underneath us (the
    daily run's drawdown kill landed between our check and our write)."""
    try:
        yield
    except sqlite3.OperationalError as e:
        raise AdminError(
            f"the ledger is busy (the daily run may be in progress); nothing was "
            f"changed, try again in a minute ({e})"
        ) from e
    except LedgerError as e:
        raise AdminError(str(e)) from e


def pause(ledger: Ledger, spec: BotSpec, source: str, reason: str | None = None) -> None:
    account = _current(ledger, spec)
    if account.status != ACTIVE:
        raise AdminError(f"{spec.name} is {account.status}; only an active bot can be paused")
    with _writing():
        _bind(ledger, spec, account).pause(reason=reason, source=source)
    log.info("%s paused (%s): %s", spec.name, source, reason or "no reason given")


def resume(ledger: Ledger, spec: BotSpec, source: str, reason: str | None = None) -> None:
    account = _current(ledger, spec)
    if account.status != PAUSED:
        raise AdminError(f"{spec.name} is {account.status}; only a paused bot can be resumed")
    with _writing():
        _bind(ledger, spec, account).resume(reason=reason, source=source)
    log.info("%s resumed (%s): %s", spec.name, source, reason or "no reason given")


def kill(ledger: Ledger, spec: BotSpec, source: str, reason: str) -> None:
    if not reason or not reason.strip():
        raise AdminError("a kill needs a reason; it is final and stays in the ledger")
    account = _current(ledger, spec)
    if account.status == KILLED:
        raise AdminError(f"{spec.name} is already killed")
    with _writing():
        _bind(ledger, spec, account).kill(reason=reason.strip(), source=source)


def reset(ledger: Ledger, spec: BotSpec) -> Account:
    """Open the bot's next account. Only after a kill: an active account with
    open positions is never abandoned silently."""
    account = _current(ledger, spec)
    if account.status != KILLED:
        raise AdminError(
            f"{spec.name} is {account.status}; kill it before resetting (a reset "
            f"never abandons a live account)"
        )
    n = len(ledger.bot_account_history(spec.name)) + 1
    while ledger.account_by_name(f"{spec.name}#{n}") is not None:
        n += 1
    with _writing():
        fresh = ledger.create_account(
            f"{spec.name}#{n}", spec.starting_cash, kind=FORWARD, bot_name=spec.name,
        )
    log.info(
        "%s reset: new account %s with $%s; %s kept as history",
        spec.name, fresh.name, f"{spec.starting_cash:,.2f}", account.name,
    )
    return fresh


# ----------------------------------------------------------------------- CLI


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control a bot from the command line.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="list every bot and its status")
    for name in ("pause", "resume", "kill", "reset"):
        p = sub.add_parser(name, help=f"{name} a bot")
        p.add_argument("bot")
        if name != "reset":
            p.add_argument("--reason", required=(name == "kill"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    paths.setup_logging()
    args = _parse_args(argv)
    try:
        specs = {s.name: s for s in load_specs()}
    except ConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 2

    with Ledger(paths.ledger_path()) as ledger:
        if args.command == "status":
            for spec in specs.values():
                account = ledger.bot_account(spec.name)
                state = account.status if account else "no account yet"
                where = f" ({account.name})" if account else ""
                print(f"{spec.name}: {state}{where}")
            return 0

        spec = specs.get(args.bot)
        if spec is None:
            print(f"no such bot {args.bot!r} in bots.yaml", file=sys.stderr)
            return 2
        try:
            if args.command == "pause":
                pause(ledger, spec, SOURCE_CLI, args.reason)
            elif args.command == "resume":
                resume(ledger, spec, SOURCE_CLI, args.reason)
            elif args.command == "kill":
                kill(ledger, spec, SOURCE_CLI, args.reason)
            else:
                fresh = reset(ledger, spec)
                print(f"{spec.name} now trades through {fresh.name}")
        except AdminError as e:
            print(str(e), file=sys.stderr)
            return 1
    print(f"{args.bot}: {args.command} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
