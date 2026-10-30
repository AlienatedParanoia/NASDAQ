# Status — 2026-10-30

## Where things stand

- Phases 0–5 are committed (see `docs/ROADMAP.md`). Phase 5 (`1ab0407`) is
  pushed; the review-fix commit that follows it is not, until the owner says so.
- Offline suite: **311 passed, 8 skipped** (`pytest tests -q`); the 8 skips are the
  `--network` checks.
- Phase 5 review fixes (owner decisions: 95% coverage, task runs logged out):
  - `bots/run_daily.py` stops a bot *before* a session where under 95% of its
    tickers have a bar (stale cache / failed refresh) instead of marking the day
    done on old prices; exit code `3` = waiting for data.
  - A bot's setup failure (universe, build, data load) is isolated like a
    mid-session one; anything escaping still reaches `advisor.log`.
  - `--until` must be a trading day no later than the latest complete session.
  - `--bot NAME` for a disabled bot is an error (exit 2), not a silent no-op.
  - `scripts/install_task.ps1`: S4U logon (runs logged out; needs an admin
    prompt once), `-Force` re-registration so principal/description update, and
    pure ASCII — the committed version did not parse in Windows PowerShell 5.1.
- **The scheduled task is not registered yet.** The owner runs
  `scripts\install_task.ps1` from an Administrator PowerShell.

## Next step: Phase 6

Stop for the owner's approval. Phase 6's scope is reconstructed (see the caveat
in `docs/ROADMAP.md`) and must be confirmed before a plan is written.

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
