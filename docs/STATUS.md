# Status — 2026-10-29

## Where things stand

- Phases 0–4 are committed and pushed (see `docs/ROADMAP.md`).
- Offline suite: **286 passed, 8 skipped** (`pytest tests -q`); the 8 skips are the
  `--network` checks.
- Phase 4 is complete: `--selftest` includes broker and replay steps,
  `nasdaq-advisor.spec` bundles `bots.yaml`, and the README has a Bots section.

## Next step: Phase 5

Implement `docs/plans/phase5-forward-runner.md` as written. The plan is approved,
including two owner decisions: a single 06:30 SGT daily run, and a failing bot is
isolated (rolled back, logged, retried next day) rather than fatal.

Start with the `Ledger.transaction()` fix — `engine/ledger.py` still has 10 plain
`with self._conn:` blocks, which do not nest in `sqlite3`, so nothing can be made
atomic per session until that lands. Nothing from Phase 5 has been written yet.

After Phase 5: commit (`phase5: headless daily runner and scheduled task`), do not
push, and stop for the owner's approval of Phase 6.

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
