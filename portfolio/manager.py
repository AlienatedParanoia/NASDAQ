"""Paper portfolio: persistence and joint performance math.

The portfolio is user data and lives in its own JSON file (see
paths.portfolio_path), never in cache.db — the cache is wiped whenever
CACHE_VERSION bumps.

Aggregation is buy-and-hold weighting (no daily rebalancing): portfolio
equity is the dollar-weighted sum of each holding's strategy equity curve on
the common (intersection) window. Overall profit therefore equals the sum of
per-holding profits exactly, while volatility/Sharpe/drawdown come from the
combined day-by-day series, so diversification between holdings shows up in
the risk numbers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

import paths
from classifier.classify import Bucket
from data.fetcher import DataFetcher, FetchError
from portfolio.storage import backup_file, read_json_or_backup, write_json_atomic
from signals.signal_generator import ScanRow, generate_signal
from strategies.backtest_engine import perf_stats, time_base

log = logging.getLogger(__name__)

PORTFOLIO_VERSION = 1
MIN_JOINT_BARS = 60  # below this, joint metrics are flagged unreliable

_RISK_CUTS = ((0.15, "Low"), (0.25, "Moderate"), (0.40, "High"))

# error fragments that mean "network hiccup", not "bad symbol" — used by the
# UI panels to decide whether a failed new ticker should be rejected
_TRANSIENT_MARKERS = ("cooldown", "cancelled", "rate limit", "timed out", "curl")


def looks_transient(error: str) -> bool:
    e = error.lower()
    return any(marker in e for marker in _TRANSIENT_MARKERS)


def risk_label(ann_vol: float) -> str:
    for cut, label in _RISK_CUTS:
        if ann_vol < cut:
            return label
    return "Very high"


@dataclass(slots=True)
class Holding:
    ticker: str
    amount: float
    strategy: str | None = None  # a Bucket value string; None = recommended


# ------------------------------------------------------------- persistence

def load_portfolio(path: Path | None = None) -> list[Holding]:
    path = path or paths.portfolio_path()
    raw = read_json_or_backup(path)
    if raw is None:
        return []
    try:
        return [
            Holding(str(h["ticker"]).upper(), float(h["amount"]), h.get("strategy"))
            for h in raw["holdings"]
            if float(h["amount"]) > 0
        ]
    except Exception:
        # structurally wrong but valid JSON: back it up too, or the next
        # save silently overwrites the user's data
        log.exception("portfolio file has unexpected structure; starting empty")
        backup_file(path)
        return []


def save_portfolio(holdings: Sequence[Holding], path: Path | None = None) -> None:
    path = path or paths.portfolio_path()
    write_json_atomic(path, {
        "version": PORTFOLIO_VERSION,
        "holdings": [
            {"ticker": h.ticker, "amount": h.amount, "strategy": h.strategy}
            for h in holdings
        ],
    })


# ---------------------------------------------------------------- analysis

@dataclass(slots=True, eq=False)
class HoldingResult:
    holding: Holding
    row: ScanRow | None          # None when the ticker produced no data
    strategy_label: str          # what actually ran
    weight: float                # share of total invested dollars
    profit: float | None         # over the common window, in dollars
    common_return: float | None  # this holding's return over the common window
    error: str | None = None


@dataclass(slots=True, eq=False)
class PortfolioReport:
    results: list[HoldingResult]
    total_invested: float
    total_return: float | None
    profit: float | None
    cagr: float | None
    ann_vol: float | None
    risk: str | None
    sharpe: float | None
    max_drawdown: float | None
    years: float | None
    limiting_ticker: str | None  # youngest holding truncating the window
    short_window: bool
    skipped: list[str]


@dataclass(slots=True)
class JointMetrics:
    common_returns: dict[str, float]
    total_return: float
    sharpe: float
    max_drawdown: float
    cagr: float
    ann_vol: float
    years: float
    n_bars: int
    common_start: pd.Timestamp
    excluded: list[str]  # holdings whose history ends before the joint window


def joint_metrics(
    series: dict[str, pd.Series], invested: dict[str, float]
) -> JointMetrics | None:
    """Pure joint-portfolio math over per-ticker daily strategy returns.

    Aligns on the intersection window (latest common start; union calendar
    inside it with missing days as 0), builds the dollar-weighted equity
    curve, and measures everything on its daily returns. Annualization is
    calendar-based (see time_base) so mixed calendars — crypto's 7-day week,
    foreign holidays — don't skew years/CAGR/vol. A holding whose history
    ENDS before the joint window opens is excluded (it would otherwise sit
    in the frame as zero-filled 'cash'). Returns None when fewer than 2
    common bars exist.
    """
    tickers = [t for t, s in series.items() if len(s) > 0]
    excluded: list[str] = []
    common_start = None
    while tickers:
        common_start = max(series[t].index[0] for t in tickers)
        dead = [t for t in tickers if series[t].index[-1] < common_start]
        if not dead:
            break
        excluded.extend(dead)
        tickers = [t for t in tickers if t not in dead]
    if not tickers:
        return None
    frame = pd.DataFrame({t: series[t] for t in tickers})
    frame = frame.loc[frame.index >= common_start].fillna(0.0)
    if len(frame) < 2:
        return None

    equities = (1.0 + frame).cumprod()
    weights = pd.Series({t: invested[t] for t in tickers})
    total_inv = float(weights.sum())
    if total_inv <= 0:
        return None
    port_equity = equities.mul(weights).sum(axis=1) / total_inv

    r_port = port_equity.pct_change()
    r_port.iloc[0] = port_equity.iloc[0] - 1.0  # return from the implicit 1.0 start
    total, sharpe, drawdown, cagr, _ = perf_stats(r_port)
    years, ppy = time_base(r_port)
    ann_vol = float(r_port.std(ddof=1) * np.sqrt(ppy)) if len(r_port) > 1 else 0.0

    return JointMetrics(
        common_returns={t: float(equities[t].iloc[-1] - 1.0) for t in tickers},
        total_return=total,
        sharpe=sharpe,
        max_drawdown=drawdown,
        cagr=cagr,
        ann_vol=ann_vol,
        years=years,
        n_bars=len(r_port),
        common_start=common_start,
        excluded=excluded,
    )


def _merged(holdings: Sequence[Holding]) -> list[Holding]:
    out: dict[str, Holding] = {}
    for h in holdings:
        t = h.ticker.strip().upper()
        if t in out:
            out[t].amount += h.amount
            out[t].strategy = h.strategy
        else:
            out[t] = Holding(t, float(h.amount), h.strategy)
    return [h for h in out.values() if h.amount > 0]


def _empty_report(results, total_invested, skipped) -> PortfolioReport:
    return PortfolioReport(
        results=results, total_invested=total_invested,
        total_return=None, profit=None, cagr=None, ann_vol=None, risk=None,
        sharpe=None, max_drawdown=None, years=None, limiting_ticker=None,
        short_window=False, skipped=skipped,
    )


def analyze_portfolio(
    holdings: Sequence[Holding],
    fetcher: DataFetcher,
    window_years: int = 10,
    should_stop: Callable[[], bool] | None = None,
) -> PortfolioReport:
    holdings = _merged(holdings)
    total_invested = float(sum(h.amount for h in holdings))
    results: list[HoldingResult] = []
    series: dict[str, pd.Series] = {}

    for h in holdings:
        if should_stop is not None and should_stop():
            break
        try:
            override = Bucket(h.strategy) if h.strategy else None
        except ValueError:
            log.warning("%s: unknown strategy %r, using recommended", h.ticker, h.strategy)
            override = None
        try:
            data = fetcher.get_stock(h.ticker)
            row = generate_signal(data, window_years, strategy_override=override)
            if row.bucket is Bucket.UNCLASSIFIED:
                # no recommended strategy exists; hold the stock plainly until
                # the user picks one
                r = (
                    row.prices["close"].pct_change().dropna()
                    if row.prices is not None else pd.Series(dtype=float)
                )
                label = "Buy & hold (unclassified)"
            else:
                r = row.returns if row.returns is not None else pd.Series(dtype=float)
                label = str(row.bucket) + ("" if h.strategy else " (recommended)")
            series[h.ticker] = r
            results.append(HoldingResult(h, row, label, 0.0, None, None))
        except FetchError as e:
            log.warning("%s: excluded from portfolio: %s", h.ticker, e)
            results.append(HoldingResult(h, None, "—", 0.0, None, None, error=str(e)))
        except Exception as e:  # noqa: BLE001 — one holding never kills the report
            log.exception("%s: portfolio analysis failed", h.ticker)
            results.append(HoldingResult(h, None, "—", 0.0, None, None, error=str(e)))

    if total_invested > 0:
        for res in results:
            res.weight = res.holding.amount / total_invested

    usable: list[HoldingResult] = []
    skipped: list[str] = []
    for res in results:
        if res.error is None and len(series.get(res.holding.ticker, ())) >= 2:
            usable.append(res)
        else:
            skipped.append(res.holding.ticker)

    if not usable or total_invested <= 0:
        return _empty_report(results, total_invested, skipped)

    invested = {res.holding.ticker: res.holding.amount for res in usable}
    jm = joint_metrics({t: series[t] for t in invested}, invested)
    if jm is None:
        return _empty_report(results, total_invested, skipped + sorted(invested))
    if jm.excluded:
        gone = set(jm.excluded)
        skipped.extend(sorted(gone))
        usable = [res for res in usable if res.holding.ticker not in gone]
        invested = {t: a for t, a in invested.items() if t not in gone}
        if not usable:
            return _empty_report(results, total_invested, skipped)

    for res in usable:
        res.common_return = jm.common_returns[res.holding.ticker]
        res.profit = res.holding.amount * res.common_return
    starts = {t: series[t].index[0] for t in invested}
    limiting = max(starts, key=starts.get)
    limiting_ticker = (
        limiting
        if len(invested) > 1 and starts[limiting] > min(starts.values())
        else None
    )

    return PortfolioReport(
        results=results,
        total_invested=total_invested,
        total_return=jm.total_return,
        profit=float(sum(res.profit for res in usable)),
        cagr=jm.cagr,
        ann_vol=jm.ann_vol,
        risk=risk_label(jm.ann_vol),
        sharpe=jm.sharpe,
        max_drawdown=jm.max_drawdown,
        years=jm.years,
        limiting_ticker=limiting_ticker,
        short_window=jm.n_bars < MIN_JOINT_BARS,
        skipped=skipped,
    )
