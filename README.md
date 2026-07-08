# nasdaq-advisor

Desktop app that scans the NASDAQ-100, classifies each stock into a strategy
bucket (Long-term hold / Trend / Mean-reversion), backtests the assigned
strategy over up to 10 years of daily data (window selectable in the toolbar:
2 / 5 / 10 years — switching re-analyzes instantly from cache), and shows a
Buy/Hold/Avoid signal with plain-English reasoning. The table's Ret/yr column
is annualized (CAGR) so stocks with different listing ages are comparable;
total return and the actual span covered appear in the reasoning text.

**Signals are informational only.** Backtests assume no transaction costs or
slippage, fills at daily closes, and use ~15-minute-delayed Yahoo Finance
data. Long windows also carry survivorship bias: the universe is *today's*
index members — the winners — so historical returns read flattering.

## Run from source

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py
```

The first Scan cold-fetches all 101 tickers (~4–6 min at rate-limit-safe
pacing); results are cached in SQLite (`data/cache.db`) and refreshed at most
once a day, so later scans take seconds.

## How signals are derived

1. **Classify** (priority order; missing data fails a rule closed):
   Long-term hold — 200-day vol < 30%, debt/equity < 1.5, revenue growth > 5%,
   beta 0.7–1.3. Trend — ADX(14) > 25 and price > 5% from the 200-day MA.
   Mean-reversion — ADX < 20 and vol > 25%. Otherwise Unclassified, no signal.
2. **Backtest** the bucket's strategy: EMA(20/50) crossover, RSI(14) + Bollinger
   bounce, or buy-and-hold scored by a 0–100 fundamentals composite — always
   benchmarked against buy-and-hold on the identical window. Stocks listed
   less than 10 years ago simply use whatever history exists.
3. **Signal** from the latest indicator state, gated by backtest quality
   (Sharpe ≥ 0.5 and positive return), with the reasoning string citing the
   actual numbers and ending with the strategy's exit cue.

## Daily Report

The Daily Report tab produces an on-demand digest of the day's best
opportunities. Click **Generate report** and it runs the same NASDAQ-100 scan
as the Scanner (over the toolbar's selected 2/5/10-year window, off the daily
cache), then ranks every stock with a current **Buy** signal and shows the
top 5.

Ranking is **risk-adjusted**: among that day's Buys, each of Sharpe, annualized
return (CAGR), and max drawdown is normalized across the field and weighted
**50% Sharpe / 30% CAGR / 20% shallower-drawdown**, so a name only reaches the
top by pairing return with risk control. Rows the pipeline flags with an
implausible Sharpe (over 3 — typically a freshly listed ticker with a couple of
trades) are **excluded** from recommendations and the exclusion is noted in the
report. The digest is dated, cites each pick's actual numbers and plain-English
reasoning, and carries the standard informational-only disclaimer; clicking a
pick opens its chart. It ranks the app's own backtest-derived signals — it is
not a prediction or advice.

## Portfolio (real positions)

The Portfolio tab records what you **actually** did: any Yahoo ticker, how much
you put in, the buy date (backfillable years into the past), and — once sold
— how much you got out and when. Open positions are valued live (shares
derived from your amount at the buy-date close × latest close); sold
positions use **your entered proceeds as the truth for P&L** (they include
your real fills and fees), with market data supplying only the risk stats.

Per lot you get worth/proceeds, P&L, return, annualized return (holdings
under 30 days aren't annualized — that would be noise), risk (annualized
volatility over your actual holding window), max drawdown, and the current
signal for open lots. The overall strip is computed from your portfolio's
genuine day-by-day value curve — cash before each buy, market value while
held, proceeds afterwards — so total P&L equals the sum of per-lot P&Ls
exactly while risk/Sharpe/drawdown reflect the real combined history. Risk
labels: Low < 15%/yr vol, Moderate < 25%, High < 40%, else Very high.
Stored in `investments.json` (atomic writes, corruption-safe backups).

## Tests

Each build phase has a runnable check under `tests/` (run from the repo root
with `python -m tests.<name>`): `phase1_data_check`, `phase2_indicator_check`,
`phase2_universe_check`, `phase3_backtest_check`, `phase4_signal_check`,
`phase5_ui_smoke`, `phase7_investments_check`, `phase8_daily_report_check`,
and `regression_review_fixes`.

## Build the Windows exe

```
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\pyinstaller main.py --name nasdaq-advisor --windowed --noconfirm ^
  --collect-all yfinance --collect-all curl_cffi --collect-all certifi ^
  --hidden-import _cffi_backend --hidden-import peewee --hidden-import multitasking ^
  --exclude-module PySide6 --exclude-module PyQt5 --exclude-module matplotlib
```

Ship the whole `dist\nasdaq-advisor\` folder (zip it). The frozen app stores
its cache and log in `%LOCALAPPDATA%\nasdaq-advisor\`. Verify a build with
`dist\nasdaq-advisor\nasdaq-advisor.exe --selftest` (exit code 0 and a
"selftest OK" line in `%LOCALAPPDATA%\nasdaq-advisor\advisor.log`).
