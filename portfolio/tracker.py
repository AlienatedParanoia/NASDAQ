"""Real-money investment tracking: user-entered lots (money in / money out)
valued against actual market history.

Unlike the paper portfolio (a strategy backtest), this records what the user
actually did. For sold lots the user's entered proceeds are the truth for
P&L — they include fees and real fills; market data supplies only the risk
statistics. The overall view is the genuine portfolio value curve V(t): each
lot contributes its cash before purchase, its market value while held, and
its realized proceeds after sale. V therefore starts at total money in, ends
at (open worth + realized proceeds), and overall P&L equals the sum of
per-lot P&Ls exactly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

import paths
from data.fetcher import DataFetcher, FetchError, StockData
from portfolio.manager import risk_label
from portfolio.storage import backup_file, read_json_or_backup, write_json_atomic
from signals.signal_generator import ScanRow, Signal, generate_signal
from strategies.backtest_engine import perf_stats, time_base

log = logging.getLogger(__name__)

INVESTMENTS_VERSION = 1
MIN_ANNUALIZE_DAYS = 30  # annualizing anything shorter is noise


@dataclass(slots=True)
class Lot:
    ticker: str
    money_in: float
    buy_date: str                    # ISO YYYY-MM-DD
    money_out: float | None = None   # None = still holding
    sell_date: str | None = None


def load_investments(path: Path | None = None) -> list[Lot]:
    path = path or paths.investments_path()
    raw = read_json_or_backup(path)
    if raw is None:
        return []
    try:
        return [
            Lot(
                ticker=str(entry["ticker"]).upper(),
                money_in=float(entry["money_in"]),
                buy_date=str(entry["buy_date"]),
                money_out=(
                    None if entry.get("money_out") is None
                    else float(entry["money_out"])
                ),
                sell_date=entry.get("sell_date"),
            )
            for entry in raw["lots"]
            if float(entry["money_in"]) > 0
        ]
    except Exception:
        # structurally wrong but valid JSON: back it up too, or the next
        # save silently overwrites the user's data
        log.exception("investments file has unexpected structure; starting empty")
        backup_file(path)
        return []


def save_investments(lots: Sequence[Lot], path: Path | None = None) -> None:
    path = path or paths.investments_path()
    write_json_atomic(path, {
        "version": INVESTMENTS_VERSION,
        "lots": [
            {
                "ticker": lot.ticker,
                "money_in": lot.money_in,
                "buy_date": lot.buy_date,
                "money_out": lot.money_out,
                "sell_date": lot.sell_date,
            }
            for lot in lots
        ],
    })


@dataclass(slots=True, eq=False)
class LotResult:
    lot: Lot
    shares: float | None
    worth: float | None              # current value (open) or proceeds (sold)
    pnl: float | None
    ret: float | None
    annualized: float | None         # None when held < MIN_ANNUALIZE_DAYS
    ann_vol: float | None            # over the actual holding window
    max_dd: float | None
    signal: Signal | None            # current recommendation (open lots only)
    scan_row: ScanRow | None = None  # for the detail chart (open lots only)
    buy_eff: pd.Timestamp | None = None
    note: str | None = None          # e.g. buy date clamped to history
    error: str | None = None


@dataclass(slots=True, eq=False)
class InvestmentReport:
    results: list[LotResult]
    total_in: float                  # all lots, including errored ones
    current_value: float | None     # analyzable lots only
    pnl: float | None
    total_return: float | None
    annualized: float | None
    ann_vol: float | None
    risk: str | None
    sharpe: float | None
    max_drawdown: float | None
    since: str | None                # ISO date of the earliest effective buy
    notes: list[str]
    skipped: list[str]


def _error_result(lot: Lot, error: str) -> LotResult:
    return LotResult(
        lot=lot, shares=None, worth=None, pnl=None, ret=None, annualized=None,
        ann_vol=None, max_dd=None, signal=None, error=error,
    )


def _analyze_lot(lot: Lot, data: StockData) -> tuple[LotResult, pd.Series]:
    """Returns (result, this lot's contribution to the portfolio value curve,
    indexed on the stock's own calendar)."""
    close = data.prices["close"]
    buy_ts = pd.Timestamp(lot.buy_date)
    notes: list[str] = []
    if buy_ts < close.index[0]:
        notes.append(
            f"{lot.ticker}: buy date {lot.buy_date} predates available history — "
            f"using earliest close ({close.index[0].date()})"
        )
    held = close.loc[close.index >= buy_ts]
    if held.empty:
        # bought after the latest cached bar (e.g. today, pre-close)
        held = close.iloc[-1:]
        notes.append(
            f"{lot.ticker}: buy date {lot.buy_date} is after the latest data — "
            f"valued from the latest close"
        )
    age_days = (pd.Timestamp.today().normalize() - close.index[-1]).days
    if data.stale or age_days > 5:
        notes.append(
            f"{lot.ticker}: prices last updated {close.index[-1].date()} — "
            f"worth may be out of date"
        )
    buy_eff = held.index[0]
    shares = lot.money_in / float(held.iloc[0])

    if lot.money_out is None:
        window = held
        end_ts = close.index[-1]
        worth = shares * float(close.iloc[-1])
    else:
        sell_ts = pd.Timestamp(lot.sell_date) if lot.sell_date else close.index[-1]
        window = held.loc[held.index <= sell_ts]
        if window.empty:
            window = held.iloc[:1]
        end_ts = window.index[-1]
        worth = float(lot.money_out)

    pnl = worth - lot.money_in
    ret = pnl / lot.money_in
    days_held = max((end_ts - buy_eff).days, 0)
    annualized = None
    if days_held >= MIN_ANNUALIZE_DAYS and 1.0 + ret > 0:
        annualized = float((1.0 + ret) ** (365.25 / days_held) - 1.0)

    ann_vol = max_dd = None
    r = window.pct_change().dropna()
    if len(r) > 1:
        _, _, max_dd, _, _ = perf_stats(r)
        _, ppy = time_base(r)
        ann_vol = float(r.std(ddof=1) * np.sqrt(ppy))

    signal = scan_row = None
    if lot.money_out is None:
        scan_row = generate_signal(data, 10)
        signal = scan_row.signal

    # contribution to V(t): cash before buying, market value while held,
    # realized proceeds afterwards
    value = pd.Series(np.nan, index=close.index)
    value.loc[close.index < buy_eff] = lot.money_in
    if lot.money_out is None:
        held_idx = close.index >= buy_eff
        value.loc[held_idx] = shares * close.loc[held_idx]
    else:
        held_idx = (close.index >= buy_eff) & (close.index <= end_ts)
        value.loc[held_idx] = shares * close.loc[held_idx]
        # the sale day itself ends at the entered proceeds (your real fill),
        # which also keeps V(t)'s endpoint equal to current_value when the
        # sale lands on the latest bar
        value.loc[end_ts] = float(lot.money_out)
        value.loc[close.index > end_ts] = float(lot.money_out)

    result = LotResult(
        lot=lot, shares=shares, worth=worth, pnl=pnl, ret=ret,
        annualized=annualized, ann_vol=ann_vol, max_dd=max_dd,
        signal=signal, scan_row=scan_row, buy_eff=buy_eff,
        note="; ".join(notes) if notes else None,
    )
    return result, value


def analyze_investments(
    lots: Sequence[Lot],
    fetcher: DataFetcher,
    should_stop: Callable[[], bool] | None = None,
) -> InvestmentReport:
    results: list[LotResult] = []
    curves: list[tuple[pd.Series, float]] = []  # (curve, cash fill for pre-history)
    for lot in lots:
        if should_stop is not None and should_stop():
            break
        try:
            result, curve = _analyze_lot(lot, fetcher.get_stock(lot.ticker))
        except FetchError as e:
            log.warning("%s: excluded from investments: %s", lot.ticker, e)
            result, curve = _error_result(lot, str(e)), None
        except Exception as e:  # noqa: BLE001 — one lot never kills the report
            log.exception("%s: investment analysis failed", lot.ticker)
            result, curve = _error_result(lot, str(e)), None
        results.append(result)
        if curve is not None:
            curves.append((curve, lot.money_in))

    total_in = float(sum(lot.money_in for lot in lots))
    usable = [r for r in results if r.error is None]
    skipped = [r.lot.ticker for r in results if r.error is not None]
    notes = [r.note for r in usable if r.note]

    if not usable:
        return InvestmentReport(
            results=results, total_in=total_in, current_value=None, pnl=None,
            total_return=None, annualized=None, ann_vol=None, risk=None,
            sharpe=None, max_drawdown=None, since=None, notes=notes,
            skipped=skipped,
        )

    total_in_usable = float(sum(r.lot.money_in for r in usable))
    current_value = float(sum(r.worth for r in usable))
    pnl = current_value - total_in_usable
    total_return = pnl / total_in_usable if total_in_usable > 0 else None
    earliest = min(r.buy_eff for r in usable)
    days = max((pd.Timestamp.today().normalize() - earliest).days, 0)
    annualized = None
    if total_return is not None and days >= MIN_ANNUALIZE_DAYS and 1.0 + total_return > 0:
        annualized = float((1.0 + total_return) ** (365.25 / days) - 1.0)

    # the real value curve: union calendar, forward-filled per stock, leading
    # gaps = that lot's cash, trimmed to start at the earliest effective buy
    frame = pd.concat([c for c, _ in curves], axis=1)
    frame.columns = range(len(curves))
    frame = frame.ffill()
    for i, (_, cash) in enumerate(curves):
        frame[i] = frame[i].fillna(cash)
    v = frame.sum(axis=1)
    v = v.loc[v.index >= earliest]

    ann_vol = risk = sharpe = max_drawdown = None
    if len(v) >= 2 and total_in_usable > 0:
        r_v = v.pct_change()
        r_v.iloc[0] = v.iloc[0] / total_in_usable - 1.0  # move on the buy day itself
        _, sharpe, max_drawdown, _, _ = perf_stats(r_v)
        _, ppy = time_base(r_v)
        std = r_v.std(ddof=1)
        ann_vol = float(std * np.sqrt(ppy)) if pd.notna(std) and std > 0 else 0.0
        risk = risk_label(ann_vol)

    return InvestmentReport(
        results=results,
        total_in=total_in,
        current_value=current_value,
        pnl=pnl,
        total_return=total_return,
        annualized=annualized,
        ann_vol=ann_vol,
        risk=risk,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        since=str(earliest.date()),
        notes=notes,
        skipped=skipped,
    )
