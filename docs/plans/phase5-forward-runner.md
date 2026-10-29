# Phase 5 — Forward paper trading runner

## Context

Everything so far has been exercised by hand or by tests. Phase 5 is what makes
the simulator actually *run*: a headless daily job that advances every enabled bot
one session at a time against real market data, unattended, forever.

That changes the failure modes. Nobody is watching when this runs at 06:30, so the
properties that matter are not features but guarantees: it must never double-trade
on a re-run, never act on a half-formed bar, never leave the ledger half-written,
and never let one broken bot stop the rest.

## The spec's per-session transaction cannot work as the Ledger stands

The spec says to "wrap each session per bot in a transaction". I tested whether
that is possible and it is not — Python's `sqlite3` `with conn:` does **not**
nest:

```
with conn:                    # outer "transaction"
    insert 1
    with conn:                # an inner Ledger write
        insert 2              # <- this COMMITS everything
    raise                     # rollback saves nothing
rows surviving: [(1,), (2,)]
```

`engine/ledger.py` has **10** internal `with self._conn:` blocks, so a crash
halfway through a session would leave the earlier writes committed and the later
ones missing — a fill recorded with no cash event, say. The ledger would be
quietly wrong rather than obviously broken.

**Fix:** give `Ledger` a reentrant `transaction()` context manager with a depth
counter. The outermost call owns the commit/rollback; inner writes join it instead
of committing. Every existing `with self._conn:` becomes `with self._tx():`, which
behaves exactly as today when there is no outer transaction, so nothing else in
the codebase changes. A test crashes a bot mid-session and asserts the ledger is
unchanged.

## Decisions taken (confirmed with you)

1. **One scheduled task at 06:30 SGT daily.** Comfortably after both 04:00 (EDT)
   and 05:00 (EST), so DST never matters and there is nothing to switch twice a
   year. Running late is always safe because the runner refuses incomplete bars;
   running early would silently skip a day.
2. **A failing bot is isolated, not fatal.** Its session is rolled back, the
   traceback is logged, its cursor stays put so it retries tomorrow, and the other
   bots carry on. The run still exits non-zero so the failure is visible.

## Files to add

### `bots/run_daily.py`
`python -m bots.run_daily`. **No Qt import anywhere in its import graph** — I will
assert this in a test rather than trust it, since a stray `from ui...` would make
the scheduled job need a desktop session.

```
1. latest = calendar.latest_complete_session()      # never a forming bar
2. fetcher.refresh(tickers_to_fetch())              # universe ∪ held
3. for each bot (enabled in bots.yaml, not killed/paused in the ledger):
       for session in sessions after its cursor, up to latest:
           with ledger.transaction():               # atomic per bot per session
               bot.run_session(session, view, actions)
```

Flags: `--dry-run` (compute everything, write nothing), `--bot NAME` (one bot),
`--no-fetch` (use the cache as-is), `--until DATE` (for testing catch-up).

Exit codes: `0` nothing to do or all fine, `1` one or more bots failed, `2`
configuration or data error before any bot ran.

### `scripts/install_task.ps1`
Registers a Windows Task Scheduler job at 06:30 SGT running the venv's python with
`-m bots.run_daily`. Idempotent (re-running updates the existing task), with
`-Remove` to unregister and `-WhatIf` to preview. Prints the resolved paths it will
bake in, because a task that silently points at the wrong interpreter is tedious
to diagnose later.

### Tests
`tests/test_run_daily.py` — offline, with a fake fetcher and synthetic bars:
- a 5-session run with **one missed day** catches up without double-trading;
- running twice is a **no-op** (same fills, same cursor, same equity);
- a crash mid-session leaves the ledger **byte-identical** to before;
- a session whose bar is incomplete is **never** processed;
- a killed or paused bot is skipped but still marked to market;
- one failing bot does not stop the others, and the run exits non-zero;
- `run_daily` imports no Qt.

## Files to change

- `engine/ledger.py` — the reentrant `transaction()` described above.
- `main.py --selftest` — unchanged; the runner gets its own test coverage.
- `README.md` — a section on running daily, the scheduled task, what the exit
  codes mean, and where to look when a bot stops trading.

## The guarantees, and how each is enforced

| Guarantee | Mechanism |
|---|---|
| Never acts on a forming bar | `calendar.latest_complete_session()` — respects early closes |
| Re-running changes nothing | `last_processed_session` cursor; only sessions strictly after it are processed |
| Catch-up cannot double-trade | Same cursor; a missed day is simply more sessions to walk |
| Crash leaves a consistent ledger | One transaction per bot per session |
| One bad bot is contained | Per-bot try/except with rollback; others continue; exit code still non-zero |
| Honours pause/kill from the UI | Status read from the ledger each session (phase 6 writes it) |

## Verification

```
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python -m pytest tests -q --network
.venv\Scripts\python -m bots.run_daily --dry-run
.venv\Scripts\python -m bots.run_daily            # a real forward run
.venv\Scripts\python -m bots.run_daily            # twice: must be a no-op
.venv\Scripts\python -m tests.phase4_replay_walkthrough
.venv\Scripts\python -m tests.phase3_broker_walkthrough
...the remaining script tests...
.venv\Scripts\python main.py --selftest
build.bat
powershell -File scripts\install_task.ps1 -WhatIf
```

I will run the real runner twice and show the ledger is untouched the second
time, then report what the bots actually did on their first live session.

## Commit

One commit, `phase5: headless daily runner and scheduled task`. No push. Then I
stop and wait for Phase 6 approval.
