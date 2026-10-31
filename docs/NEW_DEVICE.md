# New device: instructions for Claude Code

**Claude: if `transfer/ledger-export.db` exists in this repo, the move to this
machine is not finished. Work through this file top to bottom before anything
else, ticking off each step with the owner. When step 6 is done, delete this
file's pointer from `CLAUDE.md` (the "New device" block at the top).**

Background: on 2026-10-31 the owner moved development from one Windows machine
to another, using GitHub as the only channel. The old machine is retired from
running the bots. The repo is **public**; `docs/HANDOFF.md` explains what git
does and does not carry.

## 1. Set up the environment

The repo should live **outside OneDrive** (OneDrive locked build files on the
old machine). If it was cloned into a OneDrive folder, tell the owner before
continuing.

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -r requirements-dev.txt
```

Check `git config user.name` and `user.email` are set. If not, ask the owner;
don't invent them.

## 2. Verify the code

```
.venv\Scripts\python -m pytest tests -q          # expect 339 passed, 8 skipped
.venv\Scripts\python main.py --selftest          # exit 0; details in data\advisor.log
```

If anything fails, stop and debug it before going on. Don't paper over it.

## 3. Import the bots' ledger

`transfer/ledger-export.db` is the old machine's `data/ledger.db`, as of
2026-10-31 14:23 SGT: schema v2, 2 accounts (`trend-follower`,
`mean-reverter`, both active, $100,000 each), 10 orders, 0 fills, 2 snapshots,
last processed session 2026-10-30. It is fake money only. The owner chose to
send it through the public repo.

1. If `data/ledger.db` already exists here, **stop and ask the owner**: it would
   mean bots already ran on this machine, and the two histories cannot be
   merged.
2. Otherwise copy (don't move) it into place and check it:
   ```
   copy transfer\ledger-export.db data\ledger.db
   .venv\Scripts\python -m bots.admin status
   ```
   Expect `trend-follower: active (trend-follower)` and
   `mean-reverter: active (mean-reverter)`.
3. Remove the transfer copy from the repo, so nobody re-imports a stale ledger
   over a live one later:
   ```
   git rm transfer/ledger-export.db
   ```
   Commit that together with the `CLAUDE.md` cleanup in step 6. It stays in git
   history; that's accepted (fake money, the owner's decision).

## 4. Catch the bots up and register the daily run

1. Dry run first. It writes nothing:
   ```
   .venv\Scripts\python -m bots.run_daily --dry-run
   ```
   It fetches prices (the price cache starts empty here, so the first fetch of
   about 520 tickers takes a while) and walks each bot through the sessions it
   has missed since 2026-10-30. Exit code 0 = fine, 3 = waiting for a
   session's prices (normal right after the close), 1 or 2 = a problem to
   investigate.
2. If the dry run is clean, ask the owner before running it for real
   (`python -m bots.run_daily`): that writes to the ledger.
3. The owner registers the 06:30 task themselves, from an **Administrator**
   PowerShell in the repo: `scripts\install_task.ps1` (preview with `-WhatIf`).
   It had never been registered on the old machine. Confirm with the owner
   that it's done.
4. Open the app (`python main.py`), go to the **Bots** tab, and check both bots
   show up. After a real run, "vs QQQ" should show a number instead of "—".

## 5. Packaging (optional)

`build.bat` builds the exe and runs its selftest. On the old machine it failed
only because of OneDrive; outside OneDrive it should pass. Run it if the owner
wants the exe.

## 6. Finish the move

- Remove the "New device" block at the top of `CLAUDE.md`.
- Update `docs/STATUS.md`: the move is complete, the ledger was imported, and
  the scheduled task is registered (or not, as it stands).
- One commit: `docs: finish move to new device` (house style: say why, in the
  body).
- **Don't push without the owner saying so.**

## 7. Then: the next phase

Phases 0–6 are done (see `docs/STATUS.md`, `docs/ROADMAP.md`). Nothing is
planned past Phase 6, so **ask the owner** which of these comes next. Don't
start either without them:

- **ML evaluation that reads its truth from the ledger.** This is pieced
  together from a Phase 3 commit message; the original spec was never saved. Its
  scope must be confirmed before any plan is written.
- **The My Investments backlog**: real-money correctness fixes, listed in
  `docs/STATUS.md` and `docs/HANDOFF.md` §5. The top item: sale proceeds can be
  frozen from a stale price, so realised P&L is permanently wrong. Re-verify
  each finding against current code first; they date from July 2026.

Whatever is chosen, follow the working agreement in `CLAUDE.md`: write the plan
to `docs/plans/`, get it approved, build, verify with the commands in
`CLAUDE.md`, one commit `phaseN: ...`, stop.
