# Roadmap

> **Caveat:** the original project spec was never saved to the repo. Phases 0–5
> below are accurate (from commits and the approved Phase 5 plan). Phase 6 onward
> is **reconstructed from references in the code and commit messages** and must be
> confirmed with the owner before planning.

| Phase | What | State |
|---|---|---|
| 0 | Remove the signal system (classifier, signals, Daily Report, old paper simulator) | Done — `ce797a8` |
| 1 | As-traded prices, corporate actions, XNYS calendar, batched incremental fetches | Done — `c5d3153` |
| 2 | Universe = S&P 500 ∪ NASDAQ-100 from CSVs in `data/constituents/` | Done — `69e30c4` |
| 3 | Simulated broker behind an abstract `Broker`, append-only SQLite ledger | Done — `545a035` |
| 4 | Bot framework: DataView look-ahead barrier, sizing, risk/kill switch, trend + mean-reversion strategies, `bots.yaml` registry, replay | Done — `3b219b4` |
| 5 | Headless daily runner (`python -m bots.run_daily`) + Windows scheduled task at 06:30 SGT | **Planned & approved** — `docs/plans/phase5-forward-runner.md` |
| 6 | Brokerage UI for the bots — includes writing pause/kill status to the ledger, which Phase 5 reads | Reconstructed — confirm scope |
| later | ML evaluation that reads its truth from the ledger (mentioned in the Phase 3 commit) | Reconstructed — confirm scope |

Known spec rules: **rule 5** — a decision on bar *t* executes on *t+1* or later.
Other numbered rules existed in the original spec but were not recorded.
