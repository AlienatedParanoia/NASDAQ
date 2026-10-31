# Continuing on another device

Everything needed to keep building is in this repository. What is **not** in it
is runtime data and personal data, on purpose: the repo is public. This page
says what to do on the new machine, what you have to carry over by hand (and
how), and what notes from earlier sessions existed only on the old machine.

## 1. Get the code running

Windows, Python 3.12, Git:

```
git clone https://github.com/AlienatedParanoia/NASDAQ.git
cd NASDAQ
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -r requirements-dev.txt
git config user.name  "<your name>"
git config user.email "<your email>"
```

Check it works (all offline except the selftest's one Yahoo fetch):

```
.venv\Scripts\python -m pytest tests -q        # expect 339 passed, 8 skipped
.venv\Scripts\python main.py --selftest        # exit code 0; details in data\advisor.log
.venv\Scripts\python main.py                   # the app
```

**Clone somewhere outside OneDrive** (e.g. `C:\dev\NASDAQ`). On the old machine
the repo sat in a OneDrive-synced Desktop, and OneDrive turned `dist\` into
read-only placeholders that PyInstaller cannot delete, so `build.bat` fails.
OneDrive can also briefly lock `data\*.db` while syncing it.

## 2. What GitHub does not carry

| File / thing | What it is | On the new machine |
|---|---|---|
| `data/cache.db` | Price cache | Nothing to do: it rebuilds on the first fetch (the app's Refresh, or `python -m bots.run_daily`). |
| `data/ledger.db` | **The bots' entire trading history** (accounts, orders, fills, snapshots, status changes) | Copy it over (section 3) to continue the same bots. Without it, each bot opens a fresh account on its first run. |
| `data/backtest_ledger.db` | Replay results | Disposable; not worth copying. |
| `data/investments.json` | **Personal real-money data** (My Investments tab) | Copy privately if you want it. Never commit it. |
| `data/advisor.log*` | Logs | Not needed. |
| The 06:30 scheduled task | Windows Task Scheduler entry, per machine | Register it again (section 4). |
| Claude Code memory | Notes from earlier sessions, stored in the old machine's user profile | The parts that still matter are in section 5. |

The frozen exe keeps its data in `%LOCALAPPDATA%\nasdaq-advisor\` instead of
`data\`; the same table applies.

## 3. Moving the bot ledger (only if you want continuity)

> The 2026-10-31 move used GitHub instead: the export is committed as
> `transfer/ledger-export.db`, and `docs/NEW_DEVICE.md` covers importing it.
> The steps below are the private route, for any later move.

**Only one machine may run the bots.** Two machines each running
`bots.run_daily` against their own copy of `ledger.db` produce two diverging
histories that cannot be merged. Pick one, and remove the scheduled task from
the other (`scripts\install_task.ps1 -Remove`, from an Administrator PowerShell).

On the **old** machine, with the app closed and the daily run not in progress,
write one consistent file. The ledger uses WAL mode, so copying `ledger.db`
alone can miss recent writes:

```
.venv\Scripts\python -c "import sqlite3; s=sqlite3.connect('data/ledger.db'); d=sqlite3.connect('ledger-export.db'); s.backup(d); d.close(); s.close()"
```

Move `ledger-export.db` by USB or private cloud storage, **not through git**. On
the new machine, save it as `data\ledger.db`. Do the same for
`data\investments.json` if you use the My Investments tab. Then check:

```
.venv\Scripts\python -m bots.admin status     # every bot listed with its status
```

The first daily run catches up any sessions missed in between, one at a time,
with no double-trading. The cursor rules are in `bots/run_daily.py`.

## 4. Turn the daily run on

From an **Administrator** PowerShell, in the repo:

```
scripts\install_task.ps1            # registers the 06:30 run (runs while logged out)
scripts\install_task.ps1 -WhatIf    # preview without changing anything
```

As of 2026-10-31 it had **not** been registered on the old machine either.

## 5. Picking up with Claude Code

Open the repo in Claude Code and ask "where are we at". It reads `CLAUDE.md`
automatically, which points at `docs/STATUS.md` (current state and next step),
`docs/ROADMAP.md` and `docs/plans/`. Everything the build depends on is in those
files. The working agreement in `CLAUDE.md` is: one phase at a time, plan,
approval, build, verify, one commit, stop; never push unless asked.

The old machine's Claude memory also held the full findings of a whole-project
review from **2026-07-11**. Much of it concerned code that Phase 0 removed. The
findings below are about files that **still exist**. They have **not been
re-verified** since July, so check each one against the current code before
fixing it. The most important are already listed in `docs/STATUS.md`.

**High: wrong real-money numbers or silent data loss**
- `ui/investments_panel.py` `_resolve_proceeds`: sale proceeds can be frozen from
  a stale or partial (still-forming intraday) price bar, so realised P&L stays
  wrong permanently. Suggested fix: a provisional flag; only freeze final
  valuations.
- `ui/investments_panel.py`: the headline "Real return" uses
  `overall_return_pct` (profit / lifetime turnover). That is the denominator the
  cash ledger was built to replace; `tests/phase11_real_pnl_check.py`
  scenario 4 treats it as the legacy-wrong number.
- `ui/investments_panel.py`: a lot added during a transient fetch error is shown
  but never saved, so it is lost when the app closes. A failed-save warning is
  wiped by the next re-render, so failed saves go unnoticed.
- `portfolio/storage.py` `read_json_or_backup` treats an `OSError` (a OneDrive
  lock, say) as corruption. The app then starts empty and the next save
  overwrites the real file.

**Medium**
- `portfolio/storage.py`: `backup_file` swallows `OSError` (a Reset can wipe
  without the backup it promises); no `fsync` before the atomic replace.
- `portfolio/tracker.py`: a headline return inflated by turnover (there are FLAG
  comments in the code); a sold lot whose ticker is delisted drops out of
  realised P&L because it still needs a price fetch; the transient-error list
  misreads an outage as a bad symbol.
- `data/fetcher.py`: a truncated Yahoo response can replace a ticker's whole
  cache; a corrupt `cache.db` breaks every tab, with no wipe-and-recreate.
- `ui/investments_panel.py`: buttons stay live while a worker is running, so a
  re-sell can overwrite a sale; no edit or remove for saved cash flows;
  undo/reset paths have no tests.

**Low**
- `strategies/indicators.py`: the EWM seeding differs from canonical Wilder RSI
  for about the first 80 bars, so a mean-reversion warm-up of 20 may be too
  short.
- `strategies/backtest_engine.py` fills at the signal bar's close, which
  slightly flatters mean-reversion backtests. The bots' replay does not have
  this problem: it fills at the next open.
- `ui/common.py` `money()` rounds to whole dollars, which can stop ledger
  figures reconciling to the cent.
- `portfolio/storage.py` reads JSON as `utf-8`, not `utf-8-sig` (a BOM breaks it).

Already resolved; don't re-raise: the July P0, a personal `investments.json.bak`
tracked in git, was purged from history before any push, and `data/*.bak` is
ignored.
