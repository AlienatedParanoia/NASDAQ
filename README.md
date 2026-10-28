# nasdaq-advisor

Desktop app (PyQt6 + pandas + yfinance + SQLite) with two tabs:

- **Watchlist** — the NASDAQ-100 with each stock's last close, day change and
  volume, off a daily SQLite cache, plus a price chart for the selected ticker.
- **My Investments** — real-money lot tracking with a cash ledger.

**Fake money and informational use only.** Price data is end-of-day Yahoo
Finance, ~15 minutes delayed. Long windows also carry survivorship bias: the
universe is *today's* index members — the winners — so historical returns read
flattering.

> **Conversion in progress.** The signal/recommendation system (classifier,
> Buy/Hold/Avoid signals, Daily Report, old paper-trading simulator) was removed
> in phase 0. It is being replaced by an environment where multiple trading bots
> trade fake money against real market data through a simulated broker behind an
> abstract `Broker` interface. There is no real-money trading code in this
> repository and none is planned.

## Run from source

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py
```

The first Refresh cold-fetches all 101 tickers (~16s, batched); results are
cached in SQLite (`data/cache.db`) and refreshed at most once a day, so later
refreshes are served from cache.

## Data

Prices are stored **as traded**, not adjusted. This matters more than it sounds:
yfinance's default rewrites every historical price after each dividend, and even
`auto_adjust=False` leaves prices split back-adjusted — NVDA's 2024-06-07 bar
reads $119.77 there, though it actually traded at ~$1,197. A permanent trade
ledger cannot sit on numbers that move, and filling a 2023 order at a price that
encodes a 2024 split is look-ahead bias in disguise. So the fetcher un-adjusts
Yahoo's output on the way in and records **corporate actions** (splits with their
ratio, dividends with their cash amount) as separate rows keyed on ex-date. A
stored bar, once written, is final.

Strategies that want a continuous series call `adjusted_closes()`, which rebuilds
the split- and dividend-adjusted total-return series from as-traded prices plus
actions; it agrees with yfinance's own adjusted output to within rounding.

Fetches are batched (`yf.download` over ~50 tickers at a time) and incremental —
only bars after the last cached date are requested, with a few days of overlap to
pick up Yahoo's revisions. A cold fetch of the whole NASDAQ-100 takes ~16s in two
requests; a same-day re-read is pure cache. Trading sessions, early closes and
"which bar is finished" come from `data/calendar.py`, backed by the real NYSE
(XNYS) calendar rather than a weekday approximation.

## My Investments (real positions)

The My Investments tab records what you **actually** did: any Yahoo ticker, how
much you put in, the buy date (backfillable years into the past), and — once
sold — how much you got out and when. Open positions are valued live (shares
derived from your amount at the buy-date close × latest close); sold positions
use **your entered proceeds as the truth for P&L** (they include your real
fills and fees), with market data supplying only the risk stats.

Per lot you get worth/proceeds, P&L, return, annualized return (holdings
under 30 days aren't annualized — that would be noise), risk (annualized
volatility over your actual holding window) and max drawdown. The overall strip
is computed from your portfolio's genuine day-by-day value curve — cash before
each buy, market value while held, proceeds afterwards — so total P&L equals
the sum of per-lot P&Ls exactly while risk/Sharpe/drawdown reflect the real
combined history. Risk labels: Low < 15%/yr vol, Moderate < 25%, High < 40%,
else Very high.

The account also holds uninvested cash. Buying spends it, selling refills it,
and it only grows from outside via an explicit Deposit — so a position funded
by reinvested profits is never mistaken for fresh money from your bank. Stored
in `investments.json` (atomic writes, corruption-safe backups).

## Tests

New modules are covered by pytest, fully offline against synthetic fixtures:

```
.venv\Scripts\python -m pytest tests -q              # offline
.venv\Scripts\python -m pytest tests -q --network    # adds the Yahoo checks
```

The checks that genuinely need Yahoo (verifying our split handling against the
real NVDA 10:1, our adjusted series against yfinance's own) live in
`tests/test_network_data.py` and are skipped unless `--network` is passed.

The earlier build phases each have a runnable script under `tests/` (run from the
repo root with `python -m tests.<name>`): `phase1_data_check`,
`phase2_indicator_check`, `phase2_universe_check`, `phase3_backtest_check`,
`phase5_ui_smoke`, `phase7_investments_check`, `phase11_real_pnl_check`, and
`regression_review_fixes`.

`phase11_real_pnl_check` is fully offline and covers the cash ledger and Real
Investment P&L; its five scenarios also run inside `--selftest`, so a build
cannot ship with that maths broken.

## Build the Windows exe

```
.venv\Scripts\pip install -r requirements-dev.txt
build.bat
```

`build.bat` wraps `pyinstaller nasdaq-advisor.spec --noconfirm` and then runs
the frozen `--selftest` (exit code 0 and a "selftest OK" line in
`%LOCALAPPDATA%\nasdaq-advisor\advisor.log` mean the build is good). Ship the
whole `dist\nasdaq-advisor\` folder (zip it). The frozen app stores its cache
and log in `%LOCALAPPDATA%\nasdaq-advisor\`.
