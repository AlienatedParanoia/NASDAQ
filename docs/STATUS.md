# Status — 2026-10-31

## Where things stand

- Phases 0–6 are committed (see `docs/ROADMAP.md`). Phase 5 (`1ab0407`) is
  pushed; its review-fix commit (`fad8921`) and Phase 6 are not, until the owner
  says so.
- Offline suite: **339 passed, 8 skipped** (`pytest tests -q`); the 8 skips are the
  `--network` checks.
- Phase 6 (plan: `docs/plans/phase6-bots-tab.md`) added the **Bots** tab, the
  `bots/admin.py` controls (also a CLI) and `bots/report.py`. Owner decisions:
  kill is final and Reset opens `<bot>#N`; QQQ buy-and-hold is the benchmark;
  no "Run now" button; a manual kill follows `liquidate_on_kill`; pause leaves
  resting orders to fill.
- Ledger schema is now **v2**: `account_status_events` (append-only; fixes the
  old in-place `UPDATE` of account status, which broke the append-only rule)
  and `bot_accounts` (bot to account, for reset). Opening a v1 ledger migrates it
  in place. The real `data/ledger.db` was migrated on 2026-10-31 by a read-only
  `python -m bots.admin status`; both bots came through active, with their
  accounts mapped.
- Found and fixed along the way: a status-only kill would not have stopped
  orders already placed, because the broker fills them before the bot checks
  its status. Kill now cancels them in the same transaction.
- QQQ is not cached until the next daily run fetches it, so the "vs QQQ" column
  shows "—" until then.
- **The scheduled task is not registered yet.** The owner runs
  `scripts\install_task.ps1` from an Administrator PowerShell.

## Next step

**Moving to a new device (2026-10-31).** The owner is continuing on another
machine through GitHub. The bot ledger travelled as `transfer/ledger-export.db`;
`docs/NEW_DEVICE.md` is the checklist for Claude Code on the new machine
(set up, verify, import the ledger, catch up, register the task). The old
machine no longer runs the bots.

After that:

Stop for the owner's approval. The phase after this ("ML evaluation that reads
its truth from the ledger") is reconstructed and must be scoped with the owner
before a plan is written. The My Investments backlog below is the other
candidate.

## Findings worth knowing

- **Replay vs backtest gap** (Phase 4): the naive comparison showed +9.7%, but +7.3%
  of that was the engine warming its EMAs inside the window. Warm-started: engine
  +20.67%, replay +23.01% — -0.53% costs, +2.87% structural (next-open fills,
  integer shares, dividends as cash). `tests/phase4_replay_walkthrough.py` prints both.
- **Costs are placeholders** (`engine/costs.py`); every reported return depends on them.

## Open backlog from the July 2026 review (My Investments tab)

A whole-project review on 2026-07-11 confirmed 65 findings. Many were in code
removed in Phase 0 (signals, old paper trading, Daily Report). These were in code
that **still exists** and had not been fixed when last checked — re-verify against
current code before fixing:

- `ui/investments_panel.py` `_resolve_proceeds` can freeze sale proceeds from a
  stale or partial (in-progress intraday) price bar, so realized P&L is permanently
  wrong. Highest priority.
- `ui/investments_panel.py` headline "Real return" uses `overall_return_pct`
  (profit / lifetime turnover) — the denominator the cash ledger was meant to replace.
- `ui/investments_panel.py`: a lot added during a transient fetch error is shown
  but never persisted; a failed-save warning is wiped by the next re-render.
- `portfolio/storage.py` `read_json_or_backup` treats an `OSError` (e.g. a OneDrive
  lock) as corruption — the app can start empty and the next save clobbers the
  real file. `backup_file` swallows `OSError`; no `fsync` before replace.
- No edit/remove for persisted cash flows; undo/reset paths have no tests.
- Minor: `indicators.py` EWM seeding differs from canonical Wilder RSI for the first
  ~80 bars (mean-reversion warm-up of 20 may be too short); `backtest_engine.py`
  fills at the signal close, which flatters mean-reversion backtests slightly.

These are separate from the bot conversion; the owner decides when to schedule them.
