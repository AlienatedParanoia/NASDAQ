"""The trade ledger: append-only SQLite, and the single source of truth.

Two rules shape this module, and both exist because a silent disagreement
between "what we think we hold" and "what we actually traded" would poison
every performance number downstream without ever raising an error.

**1. Positions are derived, never stored.** There is no positions table. A
holding is computed from its fills plus the corporate actions that touched it
(`rebuild_positions`). That makes a position incapable of disagreeing with the
trades that produced it: there is only one record, not two to drift apart.
Cash is the same — summed from cash_events, never kept as a running total.

**2. Status changes are appended, not overwritten.** An order's current status
is the newest row in `order_events`, not a mutable column on `orders`. So "why
didn't my bot buy?" is answerable months later: the rejection is still there,
with its reason. Accounts work the same way (`account_status_events`), so a
pause or kill keeps who did it, when and why.

The ledger lives in its own file, never inside cache.db — that one is wiped on
CACHE_VERSION bumps, and a record of what the bots did must not be collateral
damage of a schema change.
"""
from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from engine.orders import (
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    TimeInForce,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2

# account kinds — a replay must never be mistaken for live paper performance
FORWARD = "forward"
BACKTEST = "backtest"

# cash_events.kind
DEPOSIT = "deposit"
WITHDRAWAL = "withdrawal"
DIVIDEND = "dividend"
COMMISSION = "commission"
BUY = "buy"
SELL = "sell"

# account status
ACTIVE = "active"
PAUSED = "paused"
KILLED = "killed"
STATUSES = (ACTIVE, PAUSED, KILLED)

# account_status_events.source: who changed the status
SOURCE_RISK = "risk"            # the risk manager's drawdown kill
SOURCE_UI = "ui"                # the Bots tab
SOURCE_CLI = "cli"              # python -m bots.admin
SOURCE_MANUAL = "manual"        # Bot.pause()/resume()/kill() called from code
SOURCE_MIGRATION = "migration"  # seeded from the v1 status column

# corporate_action_events.type
SPLIT = "split"

#: Money comparisons tolerance — float arithmetic on dollars, same convention
#: as portfolio/cash.py.
EPS = 1e-6

_SCHEMA = """
-- `status` here and on account_state is the v1 mutable column: written once at
-- creation, never read or updated since v2. SQLite cannot cleanly drop a
-- column, so it stays; account_status_events is the truth.
CREATE TABLE IF NOT EXISTS accounts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL UNIQUE,
    kind          TEXT NOT NULL CHECK (kind IN ('forward','backtest')),
    starting_cash REAL NOT NULL,
    status        TEXT NOT NULL DEFAULT 'active',
    created_at    TEXT NOT NULL
);

-- append-only: an account's status is the LATEST row here
CREATE TABLE IF NOT EXISTS account_status_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL REFERENCES accounts(id),
    status     TEXT NOT NULL,
    source     TEXT NOT NULL,
    reason     TEXT,
    session    TEXT,
    at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_status_events_account ON account_status_events(account_id);

-- which account a bot trades through; its newest row is the current one. A
-- reset appends a fresh account instead of wiping the old one's history.
CREATE TABLE IF NOT EXISTS bot_accounts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_name   TEXT NOT NULL,
    account_id INTEGER NOT NULL UNIQUE REFERENCES accounts(id),
    started_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bot_accounts_bot ON bot_accounts(bot_name);

CREATE TABLE IF NOT EXISTS orders (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id       INTEGER NOT NULL REFERENCES accounts(id),
    created_session  TEXT NOT NULL,
    ticker           TEXT NOT NULL,
    side             TEXT NOT NULL CHECK (side IN ('buy','sell')),
    type             TEXT NOT NULL CHECK (type IN ('market','limit','stop')),
    qty              REAL NOT NULL,
    limit_price      REAL,
    stop_price       REAL,
    tif              TEXT NOT NULL CHECK (tif IN ('day','gtc')),
    reason           TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_account ON orders(account_id);

-- append-only: an order's status is the LATEST row here, never a column above
CREATE TABLE IF NOT EXISTS order_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      INTEGER NOT NULL REFERENCES orders(id),
    session       TEXT,
    status        TEXT NOT NULL,
    reject_reason TEXT,
    at            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_order_events_order ON order_events(order_id);

CREATE TABLE IF NOT EXISTS fills (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      INTEGER NOT NULL REFERENCES orders(id),
    account_id    INTEGER NOT NULL REFERENCES accounts(id),
    session       TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    side          TEXT NOT NULL CHECK (side IN ('buy','sell')),
    qty           REAL NOT NULL,
    price         REAL NOT NULL,
    commission    REAL NOT NULL DEFAULT 0,
    slippage_cost REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_fills_account ON fills(account_id, session);

CREATE TABLE IF NOT EXISTS cash_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL REFERENCES accounts(id),
    session    TEXT,
    kind       TEXT NOT NULL,
    amount     REAL NOT NULL,      -- signed: credits positive, debits negative
    ref        INTEGER,            -- fill id / order id when applicable
    note       TEXT
);
CREATE INDEX IF NOT EXISTS idx_cash_account ON cash_events(account_id);

CREATE TABLE IF NOT EXISTS corporate_action_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id    INTEGER NOT NULL REFERENCES accounts(id),
    session       TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    type          TEXT NOT NULL,
    value         REAL NOT NULL,
    shares_before REAL,
    shares_after  REAL
);
CREATE INDEX IF NOT EXISTS idx_ca_account ON corporate_action_events(account_id, ticker);

CREATE TABLE IF NOT EXISTS snapshots (
    account_id     INTEGER NOT NULL REFERENCES accounts(id),
    session        TEXT NOT NULL,
    cash           REAL NOT NULL,
    equity         REAL NOT NULL,
    positions_json TEXT NOT NULL,
    PRIMARY KEY (account_id, session)
);

CREATE TABLE IF NOT EXISTS account_state (
    account_id              INTEGER PRIMARY KEY REFERENCES accounts(id),
    last_processed_session  TEXT,
    status                  TEXT NOT NULL DEFAULT 'active'
);
"""


class LedgerError(Exception):
    """The ledger was asked for something inconsistent."""


@dataclass(frozen=True, slots=True)
class Account:
    id: int
    name: str
    kind: str
    starting_cash: float
    status: str


@dataclass(frozen=True, slots=True)
class Snapshot:
    account_id: int
    session: pd.Timestamp
    cash: float
    equity: float
    positions: dict[str, float]


def _iso(ts) -> str:
    return pd.Timestamp(ts).normalize().strftime("%Y-%m-%d")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# an account's status is its newest status event, never a column
_ACCOUNT_SELECT = (
    "SELECT a.id, a.name, a.kind, a.starting_cash,"
    " COALESCE((SELECT e.status FROM account_status_events e"
    "  WHERE e.account_id = a.id ORDER BY e.id DESC LIMIT 1), 'active') AS status"
    " FROM accounts a"
)


def _account(row: sqlite3.Row) -> Account:
    return Account(row["id"], row["name"], row["kind"], row["starting_cash"],
                   row["status"])


class Ledger:
    """Append-only store of everything the bots did.

    Not thread-safe (sqlite3 connections are thread-affine) — one Ledger per
    thread, same rule as DataFetcher.
    """

    def __init__(self, db_path: Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        self._conn.executescript(_SCHEMA)
        if version < 2:
            self._migrate_to_v2()
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()
        self._tx_depth = 0

    def _migrate_to_v2(self) -> None:
        """v1 kept status in a mutable column and found a bot's account by
        name. Seed one status event per account from that column, and map
        each account to the bot of the same name. No-ops on a fresh file."""
        self._conn.execute(
            "INSERT INTO account_status_events"
            " (account_id, status, source, reason, session, at)"
            " SELECT id, status, ?, 'seeded from the v1 status column', NULL, ?"
            " FROM accounts WHERE id NOT IN"
            " (SELECT account_id FROM account_status_events)",
            (SOURCE_MIGRATION, _now()),
        )
        self._conn.execute(
            "INSERT INTO bot_accounts (bot_name, account_id, started_at)"
            " SELECT name, id, created_at FROM accounts WHERE id NOT IN"
            " (SELECT account_id FROM bot_accounts)"
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """Raw access for read-only queries outside this class. Writers
        must go through `transaction()` / the internal `_tx()`, not this —
        plain `with self._conn:` does not nest, so using it directly for
        writes would commit early inside an outer `transaction()`."""
        return self._conn

    @contextlib.contextmanager
    def _tx(self):
        """Reentrant transaction. The outermost call owns the commit/rollback;
        nested calls (every internal write method uses this) join it instead
        of each committing on its own — plain `sqlite3` `with conn:` blocks do
        not nest, so without this a crash mid-session would leave the writes
        before the crash committed and the ones after it missing."""
        if self._tx_depth > 0:
            self._tx_depth += 1
            try:
                yield self._conn
            finally:
                self._tx_depth -= 1
            return
        self._tx_depth = 1
        try:
            yield self._conn
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        finally:
            self._tx_depth = 0

    def transaction(self):
        """Public entry point for a caller that needs several ledger writes
        to land atomically (phase 5's runner wraps each bot's session in
        one), e.g.::

            with ledger.transaction():
                ledger.record_fill(fill)
                ledger.set_last_processed_session(account_id, session)
        """
        return self._tx()

    # ---------------------------------------------------------------- accounts

    def create_account(
        self, name: str, starting_cash: float, kind: str = FORWARD,
        bot_name: str | None = None,
    ) -> Account:
        """Open an account and make it `bot_name`'s current one (default: the
        bot of the same name)."""
        if kind not in (FORWARD, BACKTEST):
            raise LedgerError(f"unknown account kind {kind!r}")
        if starting_cash < 0:
            raise LedgerError("starting cash cannot be negative")
        with self._tx():
            cur = self._conn.execute(
                "INSERT INTO accounts (name, kind, starting_cash, status, created_at)"
                " VALUES (?,?,?,?,?)",
                (name, kind, float(starting_cash), ACTIVE, _now()),
            )
            account_id = int(cur.lastrowid)
            self._conn.execute(
                "INSERT INTO account_state (account_id, last_processed_session, status)"
                " VALUES (?,NULL,?)",
                (account_id, ACTIVE),
            )
            self._conn.execute(
                "INSERT INTO account_status_events"
                " (account_id, status, source, reason, session, at)"
                " VALUES (?,?,?,?,NULL,?)",
                (account_id, ACTIVE, SOURCE_MANUAL, "account opened", _now()),
            )
            self._conn.execute(
                "INSERT INTO bot_accounts (bot_name, account_id, started_at)"
                " VALUES (?,?,?)",
                (bot_name or name, account_id, _now()),
            )
            # starting cash is itself a cash event, so rebuild_cash needs no
            # special case for "where did the first dollar come from"
            self._conn.execute(
                "INSERT INTO cash_events (account_id, session, kind, amount, ref, note)"
                " VALUES (?,NULL,?,?,NULL,?)",
                (account_id, DEPOSIT, float(starting_cash), "opening balance"),
            )
        return Account(account_id, name, kind, float(starting_cash), ACTIVE)

    def get_account(self, account_id: int) -> Account:
        row = self._conn.execute(_ACCOUNT_SELECT + " WHERE a.id=?", (account_id,)).fetchone()
        if row is None:
            raise LedgerError(f"no such account: {account_id}")
        return _account(row)

    def account_by_name(self, name: str) -> Account | None:
        row = self._conn.execute(_ACCOUNT_SELECT + " WHERE a.name=?", (name,)).fetchone()
        return None if row is None else _account(row)

    def accounts(self, kind: str | None = None) -> list[Account]:
        sql = _ACCOUNT_SELECT
        params: tuple = ()
        if kind is not None:
            sql += " WHERE a.kind=?"
            params = (kind,)
        return [_account(r) for r in self._conn.execute(sql + " ORDER BY a.id", params)]

    def set_account_status(
        self, account_id: int, status: str, *, source: str,
        reason: str | None = None, session=None,
    ) -> None:
        """Append a status change. Killed is final: a killed account is never
        revived; a reset opens a new one instead (see `bot_accounts`)."""
        if status not in STATUSES:
            raise LedgerError(f"unknown account status {status!r}")
        with self._tx():
            current = self.get_account(account_id).status
            if current == KILLED:
                raise LedgerError(
                    f"account {account_id} is killed; a killed account cannot "
                    f"become {status!r}"
                )
            self._conn.execute(
                "INSERT INTO account_status_events"
                " (account_id, status, source, reason, session, at)"
                " VALUES (?,?,?,?,?,?)",
                (
                    account_id, status, source, reason,
                    _iso(session) if session is not None else None, _now(),
                ),
            )

    def account_status(self, account_id: int) -> str:
        return self.get_account(account_id).status

    def status_history(self, account_id: int) -> list[dict]:
        return [
            dict(r)
            for r in self._conn.execute(
                "SELECT status, source, reason, session, at FROM account_status_events"
                " WHERE account_id=? ORDER BY id",
                (account_id,),
            )
        ]

    # ------------------------------------------------------------ bot accounts

    def bot_account(self, bot_name: str) -> Account | None:
        """The account `bot_name` currently trades through, or None."""
        row = self._conn.execute(
            "SELECT account_id FROM bot_accounts WHERE bot_name=?"
            " ORDER BY id DESC LIMIT 1",
            (bot_name,),
        ).fetchone()
        return None if row is None else self.get_account(row["account_id"])

    def bot_account_history(self, bot_name: str) -> list[Account]:
        """Every account `bot_name` has traded through, oldest first."""
        return [
            self.get_account(r["account_id"])
            for r in self._conn.execute(
                "SELECT account_id FROM bot_accounts WHERE bot_name=? ORDER BY id",
                (bot_name,),
            )
        ]

    # ------------------------------------------------------------------ orders

    def record_order(self, order: Order) -> Order:
        """Persist a new order and its opening status event. Assigns the id."""
        with self._tx():
            cur = self._conn.execute(
                "INSERT INTO orders (account_id, created_session, ticker, side, type,"
                " qty, limit_price, stop_price, tif, reason, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    order.account_id,
                    _iso(order.created_session),
                    order.ticker,
                    order.side.value,
                    order.type.value,
                    float(order.qty),
                    order.limit_price,
                    order.stop_price,
                    order.tif.value,
                    order.reason,
                    _now(),
                ),
            )
            order.id = int(cur.lastrowid)
            self._conn.execute(
                "INSERT INTO order_events (order_id, session, status, reject_reason, at)"
                " VALUES (?,?,?,?,?)",
                (
                    order.id,
                    _iso(order.created_session),
                    order.status.value,
                    order.reject_reason,
                    _now(),
                ),
            )
        return order

    def record_order_event(
        self,
        order_id: int,
        status: OrderStatus,
        session=None,
        reject_reason: str | None = None,
    ) -> None:
        """Append a status change. Never updates the orders row — the history
        of an order is the sequence of its events."""
        with self._tx():
            self._conn.execute(
                "INSERT INTO order_events (order_id, session, status, reject_reason, at)"
                " VALUES (?,?,?,?,?)",
                (
                    order_id,
                    _iso(session) if session is not None else None,
                    OrderStatus(status).value,
                    reject_reason,
                    _now(),
                ),
            )

    def record_order_qty(self, order_id: int, qty: float,
                         limit_price: float | None,
                         stop_price: float | None) -> None:
        """A split re-terms a resting order. This is the one place an orders
        row is updated, and it is a restatement of the same economic order —
        the event log records that the split happened."""
        with self._tx():
            self._conn.execute(
                "UPDATE orders SET qty=?, limit_price=?, stop_price=? WHERE id=?",
                (float(qty), limit_price, stop_price, order_id),
            )

    def _order_from_row(self, row: sqlite3.Row) -> Order:
        order = Order(
            account_id=row["account_id"],
            created_session=pd.Timestamp(row["created_session"]),
            ticker=row["ticker"],
            side=Side(row["side"]),
            qty=row["qty"],
            type=OrderType(row["type"]),
            limit_price=row["limit_price"],
            stop_price=row["stop_price"],
            tif=TimeInForce(row["tif"]),
            reason=row["reason"] or "",
            id=row["id"],
        )
        status, reason = self._latest_status(row["id"])
        order.status = status
        order.reject_reason = reason
        return order

    def _latest_status(self, order_id: int) -> tuple[OrderStatus, str | None]:
        """An order's status IS the newest event; there is no status column."""
        row = self._conn.execute(
            "SELECT status, reject_reason FROM order_events"
            " WHERE order_id=? ORDER BY id DESC LIMIT 1",
            (order_id,),
        ).fetchone()
        if row is None:
            raise LedgerError(f"order {order_id} has no status events")
        return OrderStatus(row["status"]), row["reject_reason"]

    def get_order(self, order_id: int) -> Order:
        row = self._conn.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
        if row is None:
            raise LedgerError(f"no such order: {order_id}")
        return self._order_from_row(row)

    def orders(self, account_id: int) -> list[Order]:
        return [
            self._order_from_row(r)
            for r in self._conn.execute(
                "SELECT * FROM orders WHERE account_id=? ORDER BY id", (account_id,)
            )
        ]

    def open_orders(self, account_id: int) -> list[Order]:
        return [o for o in self.orders(account_id) if o.is_open]

    def order_history(self, order_id: int) -> list[dict]:
        return [
            dict(r)
            for r in self._conn.execute(
                "SELECT session, status, reject_reason, at FROM order_events"
                " WHERE order_id=? ORDER BY id",
                (order_id,),
            )
        ]

    # ------------------------------------------------------------------- fills

    def record_fill(self, fill: Fill) -> Fill:
        """A fill and the cash movements it causes, written together.

        The gross and the commission are separate cash events so a blotter can
        show what costs took out without re-deriving them.
        """
        with self._tx():
            cur = self._conn.execute(
                "INSERT INTO fills (order_id, account_id, session, ticker, side, qty,"
                " price, commission, slippage_cost) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    fill.order_id, fill.account_id, _iso(fill.session), fill.ticker,
                    fill.side.value, float(fill.qty), float(fill.price),
                    float(fill.commission), float(fill.slippage_cost),
                ),
            )
            fill_id = int(cur.lastrowid)
            gross = fill.qty * fill.price
            self._conn.execute(
                "INSERT INTO cash_events (account_id, session, kind, amount, ref, note)"
                " VALUES (?,?,?,?,?,?)",
                (
                    fill.account_id, _iso(fill.session),
                    BUY if fill.side is Side.BUY else SELL,
                    -gross if fill.side is Side.BUY else gross,
                    fill_id,
                    f"{fill.side.value} {fill.qty:g} {fill.ticker} @ {fill.price:.4f}",
                ),
            )
            if fill.commission:
                self._conn.execute(
                    "INSERT INTO cash_events (account_id, session, kind, amount, ref, note)"
                    " VALUES (?,?,?,?,?,?)",
                    (
                        fill.account_id, _iso(fill.session), COMMISSION,
                        -float(fill.commission), fill_id,
                        f"commission on fill {fill_id}",
                    ),
                )
        return Fill(
            order_id=fill.order_id, account_id=fill.account_id, session=fill.session,
            ticker=fill.ticker, side=fill.side, qty=fill.qty, price=fill.price,
            commission=fill.commission, slippage_cost=fill.slippage_cost, id=fill_id,
        )

    def fills(self, account_id: int, ticker: str | None = None) -> list[Fill]:
        sql = "SELECT * FROM fills WHERE account_id=?"
        params: list = [account_id]
        if ticker is not None:
            sql += " AND ticker=?"
            params.append(ticker)
        return [
            Fill(
                order_id=r["order_id"], account_id=r["account_id"],
                session=pd.Timestamp(r["session"]), ticker=r["ticker"],
                side=Side(r["side"]), qty=r["qty"], price=r["price"],
                commission=r["commission"], slippage_cost=r["slippage_cost"], id=r["id"],
            )
            for r in self._conn.execute(sql + " ORDER BY id", params)
        ]

    # -------------------------------------------------------------- cash events

    def record_cash_event(
        self, account_id: int, kind: str, amount: float, session=None,
        ref: int | None = None, note: str = "",
    ) -> None:
        with self._tx():
            self._conn.execute(
                "INSERT INTO cash_events (account_id, session, kind, amount, ref, note)"
                " VALUES (?,?,?,?,?,?)",
                (
                    account_id, _iso(session) if session is not None else None,
                    kind, float(amount), ref, note,
                ),
            )

    def cash_events(self, account_id: int) -> list[dict]:
        return [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM cash_events WHERE account_id=? ORDER BY id", (account_id,)
            )
        ]

    # --------------------------------------------------- corporate action events

    def record_corporate_action(
        self, account_id: int, session, ticker: str, type_: str, value: float,
        shares_before: float | None = None, shares_after: float | None = None,
    ) -> None:
        with self._tx():
            self._conn.execute(
                "INSERT INTO corporate_action_events (account_id, session, ticker,"
                " type, value, shares_before, shares_after) VALUES (?,?,?,?,?,?,?)",
                (
                    account_id, _iso(session), ticker, type_, float(value),
                    shares_before, shares_after,
                ),
            )

    def corporate_actions(self, account_id: int, ticker: str | None = None) -> list[dict]:
        sql = "SELECT * FROM corporate_action_events WHERE account_id=?"
        params: list = [account_id]
        if ticker is not None:
            sql += " AND ticker=?"
            params.append(ticker)
        return [dict(r) for r in self._conn.execute(sql + " ORDER BY id", params)]

    def applied_action_ids(self, account_id: int, session) -> set[tuple[str, str]]:
        """(ticker, type) pairs already applied on this session — the guard that
        makes re-running a session idempotent (phase 5)."""
        return {
            (r["ticker"], r["type"])
            for r in self._conn.execute(
                "SELECT ticker, type FROM corporate_action_events"
                " WHERE account_id=? AND session=?",
                (account_id, _iso(session)),
            )
        }

    # ------------------------------------------------------------- derived state

    def rebuild_cash(self, account_id: int) -> float:
        """Cash is the sum of every cash event. Never a stored running total —
        a stored total can drift from the events that made it."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(amount), 0.0) AS total FROM cash_events WHERE account_id=?",
            (account_id,),
        ).fetchone()
        return float(row["total"])

    def rebuild_positions(self, account_id: int) -> dict[str, Position]:
        """Derive holdings from fills and corporate actions, in event order.

        Average cost follows the standard convention: a buy raises the basis by
        what it cost, a sell removes shares at the current average (so realised
        P&L comes out of cash, not out of the basis of what is left). A split
        multiplies shares and divides the average, leaving the basis unchanged.
        """
        events: list[tuple[int, str, dict]] = []
        for r in self._conn.execute(
            "SELECT id, session, ticker, side, qty, price, commission FROM fills"
            " WHERE account_id=? ORDER BY id",
            (account_id,),
        ):
            events.append((r["id"], "fill", dict(r)))
        for r in self._conn.execute(
            "SELECT id, session, ticker, type, value FROM corporate_action_events"
            " WHERE account_id=? ORDER BY id",
            (account_id,),
        ):
            events.append((r["id"], "action", dict(r)))

        # fills and actions are interleaved by the session they happened on;
        # within a session the broker applies actions first (see SimBroker), and
        # recording order preserves that
        qty: dict[str, float] = {}
        basis: dict[str, float] = {}

        for _, kind, e in sorted(events, key=lambda x: (x[2].get("session", ""), x[0])):
            ticker = e["ticker"]
            held = qty.get(ticker, 0.0)
            cost = basis.get(ticker, 0.0)
            if kind == "fill":
                if e["side"] == Side.BUY.value:
                    qty[ticker] = held + e["qty"]
                    # commission capitalised into the basis: it is part of what
                    # the shares cost you
                    basis[ticker] = cost + e["qty"] * e["price"] + e["commission"]
                else:
                    avg = cost / held if held > EPS else 0.0
                    sold = min(e["qty"], held)
                    qty[ticker] = held - sold
                    basis[ticker] = max(cost - sold * avg, 0.0)
            elif e["type"] == SPLIT and held > EPS:
                qty[ticker] = held * e["value"]   # basis unchanged by a split

        out: dict[str, Position] = {}
        for ticker, held in qty.items():
            if held <= EPS:
                continue
            cost = basis.get(ticker, 0.0)
            out[ticker] = Position(
                ticker=ticker, qty=held, avg_cost=cost / held if held > EPS else 0.0
            )
        return out

    def realised_pnl(self, account_id: int) -> float:
        """Cash from sells minus cash into buys minus commissions — i.e. the
        money actually banked, ignoring what is still held."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(amount), 0.0) AS total FROM cash_events"
            " WHERE account_id=? AND kind IN (?,?,?)",
            (account_id, BUY, SELL, COMMISSION),
        ).fetchone()
        return float(row["total"])

    # --------------------------------------------------------------- snapshots

    def write_snapshot(
        self, account_id: int, session, cash: float, equity: float,
        positions: dict[str, Position],
    ) -> None:
        payload = json.dumps(
            {t: {"qty": p.qty, "avg_cost": p.avg_cost} for t, p in sorted(positions.items())}
        )
        with self._tx():
            self._conn.execute(
                "INSERT INTO snapshots (account_id, session, cash, equity, positions_json)"
                " VALUES (?,?,?,?,?)"
                " ON CONFLICT(account_id, session) DO UPDATE SET"
                " cash=excluded.cash, equity=excluded.equity,"
                " positions_json=excluded.positions_json",
                (account_id, _iso(session), float(cash), float(equity), payload),
            )

    def snapshots(self, account_id: int) -> list[Snapshot]:
        return [
            Snapshot(
                account_id=r["account_id"],
                session=pd.Timestamp(r["session"]),
                cash=r["cash"],
                equity=r["equity"],
                positions={
                    t: v["qty"] for t, v in json.loads(r["positions_json"]).items()
                },
            )
            for r in self._conn.execute(
                "SELECT * FROM snapshots WHERE account_id=? ORDER BY session", (account_id,)
            )
        ]

    def equity_curve(self, account_id: int) -> pd.Series:
        snaps = self.snapshots(account_id)
        if not snaps:
            return pd.Series(dtype=float)
        return pd.Series(
            [s.equity for s in snaps],
            index=pd.DatetimeIndex([s.session for s in snaps]),
            name="equity",
        )

    # ----------------------------------------------------------- session cursor

    def last_processed_session(self, account_id: int) -> pd.Timestamp | None:
        row = self._conn.execute(
            "SELECT last_processed_session FROM account_state WHERE account_id=?",
            (account_id,),
        ).fetchone()
        if row is None or row["last_processed_session"] is None:
            return None
        return pd.Timestamp(row["last_processed_session"])

    def set_last_processed_session(self, account_id: int, session) -> None:
        with self._tx():
            self._conn.execute(
                "INSERT INTO account_state (account_id, last_processed_session, status)"
                " VALUES (?,?,?)"
                " ON CONFLICT(account_id) DO UPDATE SET"
                " last_processed_session=excluded.last_processed_session",
                (account_id, _iso(session), ACTIVE),
            )

    # ------------------------------------------------------------------ queries

    def held_tickers(self, kinds: Sequence[str] = (FORWARD,)) -> set[str]:
        """Every ticker any account still holds.

        data/fetcher.tickers_to_fetch() calls this: a position does not stop
        existing when its company leaves the index, so its prices must keep
        updating or the bot's equity silently freezes.
        """
        held: set[str] = set()
        for account in self.accounts():
            if account.kind not in kinds:
                continue
            held.update(self.rebuild_positions(account.id))
        return held


def open_ledger(path: Path | None = None, backtest: bool = False) -> Ledger:
    import paths as _paths

    if path is None:
        path = _paths.backtest_ledger_path() if backtest else _paths.ledger_path()
    return Ledger(path)
