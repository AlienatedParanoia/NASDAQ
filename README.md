# nasdaq-advisor

Desktop app that scans the NASDAQ-100, classifies each stock into a strategy
bucket (Long-term hold / Trend / Mean-reversion), backtests the assigned
strategy over 2 years of daily data, and shows a Buy/Hold/Avoid signal with
plain-English reasoning.

**Signals are informational only.** Backtests assume no transaction costs or
slippage, fills at daily closes, and use ~15-minute-delayed Yahoo Finance data.

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
   benchmarked against buy-and-hold on the identical window.
3. **Signal** from the latest indicator state, gated by backtest quality
   (Sharpe ≥ 0.5 and positive return), with the reasoning string citing the
   actual numbers.

## Tests

Each build phase has a runnable check under `tests/` (run from the repo root
with `python -m tests.<name>`): `phase1_data_check`, `phase2_indicator_check`,
`phase2_universe_check`, `phase3_backtest_check`, `phase4_signal_check`,
`phase5_ui_smoke`, and `regression_review_fixes`.

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
