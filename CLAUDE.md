# CLAUDE.md

> **New device: move in progress.** If `transfer/ledger-export.db` exists, this
> machine has not finished taking over from the old one. Before anything else,
> follow `docs/NEW_DEVICE.md` step by step with the owner. Remove this block
> when its step 6 is done.

Guidance for Claude Code (or anyone) picking this project up on a fresh machine.

## What this is

`nasdaq-advisor`: a PyQt6 + pandas + yfinance + SQLite desktop app. It is being
converted from a signal/recommendation tool into an environment where several
trading bots trade **fake money** against real end-of-day market data through a
simulated broker. There is no real-money trading code and none is planned.

The app has three tabs: **Watchlist** (S&P 500 / NASDAQ-100), **My
Investments** (the owner's real-money lot tracker with a cash ledger — treat its
data as personal; see "Never commit" below) and **Bots** (the fake-money bots,
read-only apart from pause / resume / kill / reset via `bots/admin.py`).

## Read these first

1. `docs/STATUS.md` — where the build stands and what to do next.
2. `docs/ROADMAP.md` — the phase plan (reconstructed; see its caveat).
3. `docs/plans/` — the detailed, approved plan for the next phase.
4. `README.md` — how each finished layer works (data, broker, bots, tests, build).
5. `docs/HANDOFF.md` — moving to a new machine: what git does not carry (the bot
   ledger, personal data, the scheduled task) and older review findings.

## Setup (Windows)

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -r requirements-dev.txt   # pytest, pyinstaller
```

`data/cache.db` (prices) is not in git; it rebuilds itself on first fetch.
`data/ledger.db` (bot trades) is created on first run.

## Verify

```
.venv\Scripts\python -m pytest tests -q              # offline, ~30s
.venv\Scripts\python -m pytest tests -q --network    # adds the Yahoo checks
.venv\Scripts\python main.py --selftest              # headless end-to-end smoke
.venv\Scripts\python -m tests.phase3_broker_walkthrough
.venv\Scripts\python -m tests.phase4_replay_walkthrough
build.bat                                            # PyInstaller exe + selftest
```

Older phase scripts under `tests/` run as `python -m tests.<name>` from the repo
root; `README.md` → Tests lists them.

## Hard rules

- **No look-ahead (project rule 5).** A decision made on bar *t* may only execute
  on bar *t+1* or later. Enforced structurally: strategies see the world through
  `bots/data_view.py` (asking for a later date raises `LookAheadError`) and the
  broker refuses to fill an order on its creation session. Never add a path
  around either.
- **Prices are stored as-traded**, never adjusted. Corporate actions are separate
  rows; `adjusted_closes()` derives the adjusted series for indicators. A stored
  bar is final.
- **The ledger is the source of truth.** Positions are never stored — they are
  derived from fills + corporate actions. Status changes (orders and accounts)
  are appended, never overwritten. A kill is final; a reset opens a new account.
- **Ambiguities touching fill prices, cash or share counts are never guessed** —
  ask the owner.
- `engine/costs.py` figures are placeholders and labelled as such; keep it that way
  until they are calibrated.

## Working agreement with the owner

- Build one phase at a time. Write the plan, get it approved, implement, verify
  with the commands above, then **one commit per phase** (`phaseN: <summary>`),
  and **stop for approval before the next phase**.
- **Do not push without being asked.** The GitHub repo is **public**.
- Commit messages explain *why* and record anything found and fixed along the way
  (see `git log` for the house style).

## Never commit

(One-off exception, owner's decision 2026-10-31: `transfer/ledger-export.db`,
the fake-money bot ledger, was committed to move it to a new machine. It is
removed again in `docs/NEW_DEVICE.md` step 3. Never commit `investments.json`.)

Personal financial data and runtime state: `data/investments.json*`,
`data/paper_trading.json*`, `data/*.bak`, `data/*.db*`, logs. `.gitignore` covers
these — check `git status` before every commit anyway, because the repo is public.
