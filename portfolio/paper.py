"""Forward-simulated paper trading session.

A paper session is a fake portfolio anchored to a start date. The user adds
tickers with a fake dollar budget and (optionally) a chosen algorithm; each
holding deploys that algorithm and the session reports what it has made SINCE
the day that holding's money went in (its `added` date, never earlier than the
session start), marked to market at the latest cached close.

No new trade-execution engine is needed. generate_signal already runs the
chosen strategy and returns its daily strategy returns (row.returns) and its
discrete entry/exit fills (row.trades). A forward session is that output sliced
to the session start: compound the daily returns onto the budget for the value
curve, and keep only the fills executed on/after the start date for the
blotter. Because prices are cached daily, each new end-of-day bar extends the
slice on its own — that is the forward simulation. "Recommended (auto)" =
strategy_override None, so classify() picks the suitable algorithm.

Aggregation mirrors portfolio.tracker: the portfolio value curve V(t) is the
sum of each holding's equity curve on the union calendar (leading gaps = that
holding's uninvested cash), so overall profit equals the sum of per-holding
profits exactly while Sharpe/vol/drawdown come from the combined day-by-day
series.
"""
from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

import paths
from classifier.classify import Bucket
from data.fetcher import DataFetcher, FetchError, StockData
from portfolio.storage import backup_file, read_json_or_backup, write_json_atomic
from signals.signal_generator import ScanRow, Signal, generate_signal
from strategies.backtest_engine import Trade, perf_stats, time_base

log = logging.getLogger(__name__)

PAPER_VERSION = 1
DEFAULT_SESSION_NAME = "Paper session"

_RISK_CUTS = ((0.15, "Low"), (0.25, "Moderate"), (0.40, "High"))

# error fragments that mean "network hiccup", not "bad symbol" — used by the
# UI panel to decide whether a failed new ticker should be rejected
_TRANSIENT_MARKERS = ("cooldown", "cancelled", "rate limit", "timed out", "curl")


def looks_transient(error: str) -> bool:
    e = error.lower()
    return any(marker in e for marker in _TRANSIENT_MARKERS)


def risk_label(ann_vol: float) -> str:
    for cut, label in _RISK_CUTS:
        if ann_vol < cut:
            return label
    return "Very high"


def today_iso() -> str:
    return _dt.date.today().isoformat()


# ------------------------------------------------------------- data model

@dataclass(slots=True)
class PaperHolding:
    ticker: str
    budget: float                 # fake dollars deployed on this ticker
    strategy: str | None = None   # a Bucket value string; None = recommended
    added: str | None = None      # ISO date the money went in; None = session start


@dataclass(slots=True)
class PaperSession:
    name: str
    start_date: str               # ISO YYYY-MM-DD; the forward anchor
    holdings: list[PaperHolding] = field(default_factory=list)


def holding_start(holding: PaperHolding, session_start: str) -> str:
    """ISO date this holding's fake money was deployed.

    Sessions written before per-holding dates existed have no `added`, so their
    money is treated as deployed at the session start."""
    return holding.added or session_start


def _holding_anchor(holding: PaperHolding, session_start: pd.Timestamp) -> pd.Timestamp:
    """The bar on/after which this holding's returns start counting. Never
    earlier than the session itself, whatever the file says."""
    try:
        added = pd.Timestamp(holding.added) if holding.added else session_start
    except ValueError:
        log.warning("%s: unparseable added date %r", holding.ticker, holding.added)
        return session_start
    return max(added, session_start)


# ------------------------------------------------------------- persistence

def load_paper(path: Path | None = None) -> PaperSession:
    path = path or paths.paper_trading_path()
    raw = read_json_or_backup(path)
    if raw is None:
        return PaperSession(DEFAULT_SESSION_NAME, today_iso(), [])
    try:
        sess = raw["session"]
        holdings = [
            PaperHolding(
                str(h["ticker"]).upper(),
                float(h["budget"]),
                h.get("strategy"),
                h.get("added"),
            )
            for h in raw["holdings"]
            if float(h["budget"]) > 0
        ]
        return PaperSession(
            name=str(sess["name"]),
            start_date=str(sess["start_date"]),
            holdings=holdings,
        )
    except Exception:
        # structurally wrong but valid JSON: back it up too, or the next save
        # silently overwrites the user's session
        log.exception("paper session file has unexpected structure; starting empty")
        backup_file(path)
        return PaperSession(DEFAULT_SESSION_NAME, today_iso(), [])


def save_paper(session: PaperSession, path: Path | None = None) -> None:
    path = path or paths.paper_trading_path()
    write_json_atomic(path, {
        "version": PAPER_VERSION,
        "session": {"name": session.name, "start_date": session.start_date},
        "holdings": [
            {
                "ticker": h.ticker,
                "budget": h.budget,
                "strategy": h.strategy,
                "added": h.added,
            }
            for h in session.holdings
        ],
    })


# ---------------------------------------------------------------- analysis

@dataclass(slots=True, eq=False)
class BlotterFill:
    date: pd.Timestamp
    ticker: str
    side: str                     # "Buy" or "Sell"
    price: float


@dataclass(slots=True, eq=False)
class PaperHoldingResult:
    holding: PaperHolding
    row: ScanRow | None           # None when the ticker produced no data
    strategy_label: str           # which algorithm actually ran
    start_date: str               # ISO date this holding's money was deployed
    value: float | None           # current fake value since this holding's start
    pnl: float | None
    ret: float | None             # return since this holding's start
    position: str                 # "In", "Flat", or "—"
    n_trades: int                 # fills executed since the session start
    signal: Signal | None
    error: str | None = None


@dataclass(slots=True, eq=False)
class PaperReport:
    results: list[PaperHoldingResult]
    start_date: str
    total_budget: float           # all holdings, including errored ones
    current_value: float | None   # analyzable holdings only
    pnl: float | None
    total_return: float | None
    cagr: float | None
    ann_vol: float | None
    risk: str | None
    sharpe: float | None
    max_drawdown: float | None
    years: float | None
    value_curve: pd.Series | None  # portfolio V(t) since the session start
    blotter: list[BlotterFill]
    skipped: list[str]


def _holding_pnl(
    returns: pd.Series, budget: float, start: pd.Timestamp
) -> tuple[pd.Series, float, float, float]:
    """Pure slice-and-compound: (equity curve since start, value, pnl, ret).

    `returns` is the deployed algorithm's daily strategy return series. Only
    bars on/after the session start count; before any bar exists the holding
    is still all cash, so value == budget and pnl == 0."""
    since = returns.loc[returns.index >= start].dropna() if len(returns) else returns
    if len(since) == 0:
        return pd.Series(dtype=float), float(budget), 0.0, 0.0
    equity = budget * (1.0 + since).cumprod()
    value = float(equity.iloc[-1])
    pnl = value - budget
    return equity, value, pnl, pnl / budget


def _position_of(row: ScanRow) -> str:
    """Whether the deployed algorithm is currently holding the stock."""
    if row.bucket is Bucket.UNCLASSIFIED:
        return "In"  # plain buy-and-hold until the user picks a strategy
    if row.trades and row.trades[-1].exit_date is None:
        return "In"
    return "Flat"


def _fills_since(ticker: str, trades: Sequence[Trade], start: pd.Timestamp) -> list[BlotterFill]:
    fills: list[BlotterFill] = []
    for t in trades:
        if t.entry_date >= start:
            fills.append(BlotterFill(t.entry_date, ticker, "Buy", t.entry_price))
        if t.exit_date is not None and t.exit_date >= start:
            fills.append(BlotterFill(t.exit_date, ticker, "Sell", float(t.exit_price)))
    return fills


def _analyze_holding(
    holding: PaperHolding, data: StockData, start: pd.Timestamp, window_years: int
) -> tuple[PaperHoldingResult, pd.Series, list[BlotterFill]]:
    """Deploy the holding's algorithm and measure it since `start` — the date
    this holding's money went in, not necessarily the session start.

    Returns (result, this holding's equity curve since start indexed on the
    stock's own calendar for the portfolio V(t) aggregation, fills executed
    since start for the blotter)."""
    try:
        override = Bucket(holding.strategy) if holding.strategy else None
    except ValueError:
        log.warning("%s: unknown strategy %r, using recommended", holding.ticker, holding.strategy)
        override = None

    row = generate_signal(data, window_years, strategy_override=override)

    if row.bucket is Bucket.UNCLASSIFIED:
        # no recommended strategy exists; hold the stock plainly until the user
        # picks one (matches the retired paper-portfolio's buy-and-hold fallback)
        returns = (
            row.prices["close"].pct_change() if row.prices is not None
            else pd.Series(dtype=float)
        )
        label = "Buy & hold (unclassified)"
        fills: list[BlotterFill] = []
    else:
        returns = row.returns if row.returns is not None else pd.Series(dtype=float)
        label = str(row.bucket) + ("" if holding.strategy else " (recommended)")
        fills = _fills_since(holding.ticker, row.trades, start)

    equity, value, pnl, ret = _holding_pnl(returns, holding.budget, start)
    result = PaperHoldingResult(
        holding=holding,
        row=row,
        strategy_label=label,
        start_date=str(start.date()),
        value=value,
        pnl=pnl,
        ret=ret,
        position=_position_of(row),
        n_trades=len(fills),
        signal=row.signal,
    )
    return result, equity, fills


def _error_result(
    holding: PaperHolding, start_date: str, error: str
) -> PaperHoldingResult:
    return PaperHoldingResult(
        holding=holding, row=None, strategy_label="—", start_date=start_date,
        value=None, pnl=None, ret=None, position="—", n_trades=0, signal=None,
        error=error,
    )


def analyze_paper(
    session: PaperSession,
    fetcher: DataFetcher,
    window_years: int = 10,
    should_stop: Callable[[], bool] | None = None,
) -> PaperReport:
    start = pd.Timestamp(session.start_date)
    results: list[PaperHoldingResult] = []
    curves: list[tuple[pd.Series, float]] = []  # (equity curve, budget cash-fill)
    # budgets of holdings deployed too recently to have a single bar yet: still
    # fake cash the session is holding, so V(t) must carry them or day one shows
    # a phantom loss against total_budget_usable
    idle_cash = 0.0
    blotter: list[BlotterFill] = []

    for h in session.holdings:
        if should_stop is not None and should_stop():
            break
        anchor = _holding_anchor(h, start)
        try:
            result, equity, fills = _analyze_holding(
                h, fetcher.get_stock(h.ticker), anchor, window_years
            )
        except FetchError as e:
            log.warning("%s: excluded from paper session: %s", h.ticker, e)
            results.append(_error_result(h, str(anchor.date()), str(e)))
            continue
        except Exception as e:  # noqa: BLE001 — one holding never kills the report
            log.exception("%s: paper analysis failed", h.ticker)
            results.append(_error_result(h, str(anchor.date()), str(e)))
            continue
        results.append(result)
        blotter.extend(fills)
        if len(equity) > 0:
            curves.append((equity, h.budget))
        else:
            idle_cash += h.budget

    blotter.sort(key=lambda f: (f.date, f.ticker))
    total_budget = float(sum(h.budget for h in session.holdings))
    usable = [r for r in results if r.error is None]
    skipped = [r.holding.ticker for r in results if r.error is not None]

    if not usable:
        return PaperReport(
            results=results, start_date=session.start_date, total_budget=total_budget,
            current_value=None, pnl=None, total_return=None, cagr=None, ann_vol=None,
            risk=None, sharpe=None, max_drawdown=None, years=None, value_curve=None,
            blotter=blotter, skipped=skipped,
        )

    total_budget_usable = float(sum(r.holding.budget for r in usable))
    current_value = float(sum(r.value for r in usable))
    pnl = current_value - total_budget_usable
    total_return = pnl / total_budget_usable if total_budget_usable > 0 else None

    # portfolio V(t): union calendar, forward-filled per holding, leading gaps
    # (before that holding's first post-start bar) filled with its cash budget
    value_curve = None
    cagr = ann_vol = risk = sharpe = max_drawdown = years = None
    if curves:
        frame = pd.concat([c for c, _ in curves], axis=1)
        frame.columns = range(len(curves))
        frame = frame.ffill()
        for i, (_, budget) in enumerate(curves):
            frame[i] = frame[i].fillna(budget)
        v = frame.sum(axis=1) + idle_cash
        value_curve = v
        if len(v) >= 2 and total_budget_usable > 0:
            r_v = v.pct_change()
            r_v.iloc[0] = v.iloc[0] / total_budget_usable - 1.0  # move from budget on day one
            _, sharpe, max_drawdown, cagr, _ = perf_stats(r_v)
            years, ppy = time_base(r_v)
            std = r_v.std(ddof=1)
            ann_vol = float(std * np.sqrt(ppy)) if pd.notna(std) and std > 0 else 0.0
            risk = risk_label(ann_vol)

    return PaperReport(
        results=results,
        start_date=session.start_date,
        total_budget=total_budget,
        current_value=current_value,
        pnl=pnl,
        total_return=total_return,
        cagr=cagr,
        ann_vol=ann_vol,
        risk=risk,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        years=years,
        value_curve=value_curve,
        blotter=blotter,
        skipped=skipped,
    )
