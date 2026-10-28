# nasdaq-advisor

Desktop app (PyQt6 + pandas + yfinance + SQLite) with two tabs:

- **Watchlist** — the S&P 500 and/or NASDAQ-100 (selectable in the toolbar) with
  each stock's last close, day change and volume, off a daily SQLite cache, plus
  a price chart for the selected ticker.
- **My Investments** — real-money lot tracking with a cash ledger.

**Fake money and informational use only.** Price data is end-of-day Yahoo
Finance, ~15 minutes delayed. See the survivorship-bias note under Universe:
backtests on this universe read optimistic by construction.

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

The first Refresh cold-fetches the selected universe (batched; ~16s for the
NASDAQ-100, longer for the full 518-symbol union); results are cached in SQLite
(`data/cache.db`) and refreshed at most once a day, so later refreshes are
served from cache.

## Universe

Membership lives in `data/constituents/sp500.csv` and `nasdaq100.csv` (columns
`ticker,name,sector,date_added`), not in code, so a reconstitution arrives as a
reviewable diff. The toolbar selects S&P 500, NASDAQ-100, or both; the union is
**518 unique symbols** (503 + 101 − 86 in both).

Rebuild the files from Wikipedia with:

```
.venv\Scripts\python scripts/refresh_constituents.py --dry-run   # show the diff
.venv\Scripts\python scripts/refresh_constituents.py             # write it
```

It prints added / removed / renamed members and refuses to write a file that
lost more than 10% of its members (`--force` overrides), since that is what a
source-page restructure looks like rather than a real reconstitution.

The two sources do not describe themselves identically: `sector` is GICS for the
S&P 500 and ICB for the NASDAQ-100 — comparable within an index, not across —
and `date_added` is empty for every NASDAQ-100 row because the source table has
no such column. The full NASDAQ composite is deliberately **not** included.

### Survivorship bias

These files list *today's* members. A company that was in the S&P 500 in 2018 and
then went bankrupt, got acquired, or was demoted is not here — and yfinance
mostly cannot serve its prices anyway. Every historical backtest therefore runs
on a pre-filtered list of survivors and **reads optimistically**: the losers were
removed before the test began. Forward paper trading, where the universe is not
yet known to have survived, is the honest test. This caveat is repeated in the
app's Watchlist footer.

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

## Simulated broker

Bots trade fake money through `engine/` — an abstract `Broker` interface with a
single simulated implementation. There is no real-broker adapter and none is
planned.

**Positions are never stored.** A holding is derived from its fills plus the
corporate actions that touched it, and cash is the sum of its events. There is
one record, not two to drift apart, so a position cannot silently disagree with
the trades that produced it. Order status changes are *appended* to an event
log rather than overwriting a row, so "why didn't my bot buy?" stays answerable
months later, rejection reason included.

Each session is processed in three steps, in this order: corporate actions with
that ex-date (splits re-term holdings **and** resting orders; dividends credit
cash), then order matching against the session's bar, then mark-to-market and a
snapshot. Because prices are stored as-traded, a split genuinely halves the
price on its ex-date — so the same handling is correct for historical replay and
forward trading alike. There is no "backtest mode" to get subtly wrong.

Fills are decided against the raw OHLC:

| Order | Condition | Fill price |
|---|---|---|
| market buy / sell | always | open ± slippage |
| limit buy | open ≤ limit | open (price improvement kept) |
| limit buy | low ≤ limit | limit |
| limit sell | open ≥ limit | open |
| limit sell | high ≥ limit | limit |
| stop buy | open ≥ stop / high ≥ stop | open / stop, + slippage |
| stop sell | open ≤ stop / low ≤ stop | open / stop, − slippage |

Limit fills take **no** slippage — a limit order fills at its price or better by
definition. Within a session, sells settle before buys (so a sale can fund a
purchase), each side oldest-first. An order decided on bar *t* can never fill
before *t+1*; the broker enforces that rather than trusting callers.

Cash can never go negative and shares can never go short: buys reserve cash and
resting sells reserve shares at submission, and both are re-checked at fill
time. Every rejection carries a human-readable reason.

### Costs are placeholders

`engine/costs.py` ships invented figures — $1.00 minimum commission, 0.5¢ per
share, 5 bps slippage. **They are not your broker's fee schedule, and every
reported return is only as trustworthy as they are.** At 5 bps a daily-trading
bot loses roughly 2.5%/yr to slippage alone; set it to zero and mediocre
strategies start looking excellent. Replace them before believing any result.

### Deliberate simplifications

- Dividends credit cash on the **ex-date**, not the pay date, so a dividend is
  re-investable sooner than reality allows.
- Fills are all-or-nothing at one price: no partial fills, no queue position,
  no volume cap — a bot can "buy" more than a day's volume.
- Cash settles instantly (no T+1). No margin, no shorting, no options.

`python -m tests.phase3_broker_walkthrough` prints a worked example across eight
sessions — a split, a dividend, an expiry, a rejection, a halted session — with
every figure shown so the arithmetic can be checked by hand.

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
`regression_review_fixes`, and `phase3_broker_walkthrough`.
`phase2_universe_check` is the slow one — it cold-fetches the whole union, then
proves a same-day re-read makes zero network calls.

The broker's invariants (cash never negative, no accidental shorting,
`equity == cash + Σ shares × close`, and state rebuilt from the ledger matching
the live account) are checked by property tests over random order sequences on
random-walk prices, with fixed seeds so a failure is reproducible.

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
