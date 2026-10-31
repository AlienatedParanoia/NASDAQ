# Phase 6 — Bots tab (brokerage UI)

## Context

Phase 5 made the bots trade unattended. Nothing yet lets the owner *see* them
without opening `data/ledger.db` by hand, or stop one without editing code. Phase 6
adds a **Bots** tab to the desktop app: a read-only window onto the ledger, plus
the three controls the runner already honours — pause, resume, kill — and a reset.

Scope confirmed with the owner on 2026-10-31:

- overview table + per-bot detail, read-only except for status controls
- pause / resume / kill written to the ledger; the runner picks them up next session
- kill stays permanent; **Reset** opens a fresh account and keeps the old history
- benchmark: **QQQ buy-and-hold** over the same dates
- **no "Run now"** — the runner stays headless (scheduled task / CLI only)

Out of scope: ML evaluation, the My Investments backlog, editing `bots.yaml` in the UI.

## Two things the ledger has to change first

### 1. Account status is overwritten, which breaks a hard rule

`Ledger.set_account_status` (`engine/ledger.py:327`) runs `UPDATE accounts SET
status=?` and the same on `account_state`. CLAUDE.md: *"Status changes are
appended, never overwritten."* Today a kill leaves no record of when, why, or who
did it — which is exactly what a UI needs to show.

**Fix:** a new append-only table, read the same way `order_events` already is:

```sql
CREATE TABLE IF NOT EXISTS account_status_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL REFERENCES accounts(id),
    status     TEXT NOT NULL,          -- active | paused | killed
    source     TEXT NOT NULL,          -- 'risk' | 'ui' | 'cli' | 'migration'
    reason     TEXT,
    session    TEXT,                   -- trading session it applied to, if any
    at         TEXT NOT NULL
);
```

- An account's status **is** its newest event. `Account.status` is derived, not
  read from a column.
- `set_account_status(account_id, status, source, reason=None, session=None)`
  appends. Moving out of `killed` raises `LedgerError` (kill is final).
- `status_history(account_id)` returns the trail for the UI.
- **Migration** (`SCHEMA_VERSION` bump): every existing account gets one seed event
  from its current column value with `source='migration'`. The old `status`
  columns stay in the schema (SQLite can't cleanly drop them) but nothing reads or
  writes them afterwards; a comment says so.
- Callers in `bots/bot.py` (`_kill`, `pause`, `resume`, `kill`) pass a source and
  reason — the drawdown kill records the drawdown figure.

### 2. "Reset" needs a bot → account mapping

`accounts.name` is `UNIQUE` and `Bot.open` finds its account by the bot's name, so
a bot can only ever have one account. Reset adds:

```sql
CREATE TABLE IF NOT EXISTS bot_accounts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_name   TEXT NOT NULL,
    account_id INTEGER NOT NULL UNIQUE REFERENCES accounts(id),
    started_at TEXT NOT NULL
);
```

- A bot's current account is its newest `bot_accounts` row. Migration seeds one
  row per existing forward account (`bot_name = accounts.name`).
- Reset creates account `"<bot>#2"`, `"#3"`… with the starting cash from
  `bots.yaml` and appends a row. Reset is only offered for a **killed** bot — an
  active account with open positions is never abandoned silently.
- `Bot.open` resolves through this table. The runner's `last_processed_session`
  lives on `account_state`, so a reset account starts from the next session —
  it does not replay history.

## Controls go through one module, not the widget

New `bots/admin.py` — `pause(bot)`, `resume(bot)`, `kill(bot, reason)`,
`reset(bot)` — each one ledger transaction. The UI calls these; so can a small
CLI (`python -m bots.admin pause trend-follower`) for when the app isn't open.

**Kill must cancel open orders in the same transaction.** `Bot.run_session`
processes the broker's pending orders *before* checking status, so a status-only
kill from the UI would still let yesterday's orders fill next morning. `admin.kill`
appends the status event and a `cancelled` event for every pending order together.

Concurrency: the ledger is WAL with a 5s busy timeout. The UI's only writes are
these short transactions; if one collides with the 06:30 run it waits, and if it
still can't get the lock it shows an error rather than retrying blindly. The UI
holds no long-lived write transaction and re-reads the ledger on refresh.

## The tab

`ui/bots_panel.py`, added in `ui/main_window.py` after My Investments.

**Overview table** — one row per bot in `bots.yaml` (disabled bots greyed):

| Bot | Status | Equity | Cash | Return | vs QQQ | Max DD | Positions | Last session |

- Equity / cash / return from the latest snapshot; max drawdown from the equity
  curve; "Last session" from `account_state.last_processed_session`, coloured if
  it's behind `latest_complete_session()` (runner hasn't run, or exit code 3).
- Buttons under the table act on the selected bot: **Pause**, **Resume**,
  **Kill…** (asks for a reason, confirms), **Reset…** (killed bots only, confirms).

**Detail pane** for the selected bot, as sub-tabs:

- **Equity** — equity curve from `snapshots` with QQQ on the same axis, both
  rebased to the account's starting cash.
- **Positions** — from `rebuild_positions()`: qty, avg cost, last close,
  market value, unrealised P&L. Marks come from the price cache's latest stored
  close — no fetching from the UI.
- **Orders** — every order with its current status; selecting one shows its
  `order_history()` trail.
- **Activity** — fills, cash events, corporate actions, status history, newest first.

**QQQ benchmark** — buy-and-hold from the close of the account's first snapshot
session, using adjusted closes (dividends reinvested), since the bots receive
dividends as cash and a price-only QQQ would flatter them. QQQ is not an index
member, so `tickers_to_fetch()` gains a `BENCHMARKS = ("QQQ",)` addition so the
daily refresh keeps it current. If QQQ isn't cached yet the column shows "—".

All of this is read through a plain-Python `bots/report.py` (no Qt) so the
numbers are unit-tested without a GUI; the panel only formats them.

## Tests

- Status: append-only (row count grows, no UPDATE), latest wins, killed→active
  raises, migration seeds one event per account, history order.
- Reset: new account + mapping row, old account's fills/snapshots untouched,
  refused unless killed, `Bot.open` picks the new account, runner starts it at
  the next session.
- `admin.kill` cancels pending orders atomically; a kill then a run fills nothing.
- Runner honours a UI pause/kill written between two sessions of a catch-up run.
- `report.py`: return, max drawdown, QQQ-relative return on a hand-computed
  fixture; missing QQQ → `None`.
- UI smoke: panel builds on an empty ledger and on a populated one (offscreen
  Qt, as the existing UI tests do). `main.py --selftest` covers the new tab.

Verify with the CLAUDE.md commands, then one commit `phase6: ...`.

## Open questions for the owner

1. **Manual kill and `liquidate_on_kill`.** The drawdown kill sells everything when
   `liquidate_on_kill: true`; the existing manual `Bot.kill` never does. Should a
   kill from the UI follow the bot's `liquidate_on_kill` setting (recommended), or
   always hold positions?
2. **Pause and pending orders.** Paused today means "no new decisions", but orders
   already placed still fill on the next session. Keep that (recommended — pause
   is a soft stop, kill is the hard one), or have pause cancel pending orders too?
