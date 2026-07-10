"""Real-money investment tracking: user-entered lots valued against actual
market history.

This is the Portfolio tab's backend: it records what the user actually did,
not a strategy backtest. A lot is closed once it has a sell date; its worth
then freezes at the close on that date, so a sold position keeps the return
it had earned up to the day it was sold and never moves again. (A lot may
carry an explicit `money_out` instead — older records, and manual overrides
that capture real fills and fees; that amount wins when present.)

The overall view is the genuine portfolio value curve V(t): each lot
contributes its cash before purchase, its market value while held, and its
realized proceeds after sale. V therefore starts at total money in, ends at
(open worth + realized proceeds), and overall P&L equals the sum of per-lot
P&Ls exactly. The same aggregation runs twice: once over every lot, once
over the open ones alone, so the money still at risk can be read on its own.
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
from data.fetcher import DataFetcher, FetchError, StockData
from portfolio.cash import CashFlow
from portfolio.storage import backup_file, read_json_or_backup, write_json_atomic
from signals.signal_generator import ScanRow, Signal, generate_signal
from strategies.backtest_engine import perf_stats, time_base

log = logging.getLogger(__name__)

INVESTMENTS_VERSION = 3
MIN_ANNUALIZE_DAYS = 30  # annualizing anything shorter is noise

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


@dataclass(slots=True)
class Lot:
    ticker: str
    money_in: float
    buy_date: str                    # ISO YYYY-MM-DD
    money_out: float | None = None   # explicit proceeds; None = value at the sell close
    sell_date: str | None = None     # set = position closed
    strategy: str | None = None      # a Bucket value string; None = recommended
    # ordering within the cash ledger (portfolio/cash.py): the buy debits the
    # balance, the sale credits it, and these break ties between events that
    # share a date so a replay is stable across loads
    buy_seq: int = 0
    sell_seq: int | None = None
    # user's declaration that this position was bought entirely with profits,
    # not money from their bank. A reporting tag only: it changes the
    # "return on your own money" view (portfolio/cash.py) and nothing else —
    # not the cash balance, not the honest profit, not the ledger.
    from_profit: bool = False

    @property
    def is_sold(self) -> bool:
        # a v1 record may carry proceeds with no sell date; it is still closed
        return self.sell_date is not None or self.money_out is not None


def load_investments_doc(
    path: Path | None = None,
) -> tuple[list[Lot], list[CashFlow]]:
    """The whole persisted document: the lots and the external cash flows that
    funded them. A v2 file has no `cash_flows` key and no `seq` fields; both
    default from list position, so it loads without migration."""
    path = path or paths.investments_path()
    raw = read_json_or_backup(path)
    if raw is None:
        return [], []
    try:
        lots = [
            Lot(
                ticker=str(entry["ticker"]).upper(),
                money_in=float(entry["money_in"]),
                buy_date=str(entry["buy_date"]),
                money_out=(
                    None if entry.get("money_out") is None
                    else float(entry["money_out"])
                ),
                sell_date=entry.get("sell_date"),
                strategy=entry.get("strategy"),
                buy_seq=int(entry.get("buy_seq", i)),
                sell_seq=(
                    None if entry.get("sell_seq") is None
                    else int(entry["sell_seq"])
                ),
                from_profit=bool(entry.get("from_profit", False)),
            )
            for i, entry in enumerate(raw["lots"])
            if float(entry["money_in"]) > 0
        ]
        flows = [
            CashFlow(
                amount=float(entry["amount"]),
                direction=str(entry["direction"]),
                date=str(entry["date"]),
                seq=int(entry.get("seq", i)),
            )
            for i, entry in enumerate(raw.get("cash_flows", []))
            if float(entry["amount"]) > 0
        ]
        return lots, flows
    except Exception:
        # structurally wrong but valid JSON: back it up too, or the next
        # save silently overwrites the user's data
        log.exception("investments file has unexpected structure; starting empty")
        backup_file(path)
        return [], []


def load_investments(path: Path | None = None) -> list[Lot]:
    return load_investments_doc(path)[0]


def save_investments(
    lots: Sequence[Lot],
    path: Path | None = None,
    cash_flows: Sequence[CashFlow] = (),
) -> None:
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
                "strategy": lot.strategy,
                "buy_seq": lot.buy_seq,
                "sell_seq": lot.sell_seq,
                "from_profit": lot.from_profit,
            }
            for lot in lots
        ],
        "cash_flows": [
            {
                "amount": flow.amount,
                "direction": flow.direction,
                "date": flow.date,
                "seq": flow.seq,
            }
            for flow in cash_flows
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
class Totals:
    """Aggregate performance over some subset of the analyzable lots."""
    count: int
    total_in: float
    current_value: float
    pnl: float
    total_return: float | None
    annualized: float | None         # None when the subset is younger than 30 days
    ann_vol: float | None
    risk: str | None
    sharpe: float | None
    max_drawdown: float | None
    since: str                       # ISO date of the earliest effective buy


@dataclass(slots=True, eq=False)
class InvestmentReport:
    results: list[LotResult]
    total_in: float                  # all lots, including errored ones
    overall: Totals | None           # sold + open; None when nothing is analyzable
    open_only: Totals | None         # open positions alone; None when none are open
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
    if (data.stale or age_days > 5) and not lot.is_sold:
        # a sold lot is frozen at its sell close, so stale prices can't move it
        notes.append(
            f"{lot.ticker}: prices last updated {close.index[-1].date()} — "
            f"worth may be out of date"
        )
    buy_eff = held.index[0]
    shares = lot.money_in / float(held.iloc[0])

    if not lot.is_sold:
        window = held
        end_ts = close.index[-1]
        worth = shares * float(close.iloc[-1])
    else:
        sell_ts = pd.Timestamp(lot.sell_date) if lot.sell_date else close.index[-1]
        window = held.loc[held.index <= sell_ts]
        if window.empty:
            window = held.iloc[:1]
        end_ts = window.index[-1]
        # the sale freezes the lot at the close on its sell date; an explicit
        # money_out (a real fill, or a v1 record) overrides that valuation
        worth = (
            float(lot.money_out) if lot.money_out is not None
            else shares * float(window.iloc[-1])
        )
        if lot.sell_date and end_ts < sell_ts:
            notes.append(
                f"{lot.ticker}: no trading on {lot.sell_date} or later — "
                f"sale valued at the close of {end_ts.date()}"
            )

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
    if not lot.is_sold:
        # the trend the user invested under decides the recommendation; None
        # lets classify() pick the suitable algorithm
        try:
            override = Bucket(lot.strategy) if lot.strategy else None
        except ValueError:
            log.warning(
                "%s: unknown strategy %r, using recommended", lot.ticker, lot.strategy
            )
            override = None
        scan_row = generate_signal(data, 10, strategy_override=override)
        signal = scan_row.signal

    # contribution to V(t): cash before buying, market value while held,
    # realized proceeds afterwards
    value = pd.Series(np.nan, index=close.index)
    value.loc[close.index < buy_eff] = lot.money_in
    if not lot.is_sold:
        held_idx = close.index >= buy_eff
        value.loc[held_idx] = shares * close.loc[held_idx]
    else:
        held_idx = (close.index >= buy_eff) & (close.index <= end_ts)
        value.loc[held_idx] = shares * close.loc[held_idx]
        # the sale day itself ends at the realized proceeds, which keeps
        # V(t)'s endpoint equal to current_value when the sale lands on the
        # latest bar (a no-op unless money_out overrides the sell close)
        value.loc[end_ts] = worth
        value.loc[close.index > end_ts] = worth

    result = LotResult(
        lot=lot, shares=shares, worth=worth, pnl=pnl, ret=ret,
        annualized=annualized, ann_vol=ann_vol, max_dd=max_dd,
        signal=signal, scan_row=scan_row, buy_eff=buy_eff,
        note="; ".join(notes) if notes else None,
    )
    return result, value


def _aggregate(pairs: Sequence[tuple[LotResult, pd.Series]]) -> Totals | None:
    """Roll a subset of analyzable lots up into one performance view. `pairs`
    is (result, that lot's contribution to V(t)); returns None if empty."""
    if not pairs:
        return None

    # FLAG (real-investment-pnl pass): `total_in` sums every lot's money_in, so
    # each reinvestment of a previous sale inflates it — the UI labels this
    # "Put in", which reads as "deposited from my bank" and is not that. The
    # derived `total_return` therefore divides a correct profit by a base the
    # user never deposited. portfolio/cash.py computes the honest figures
    # (net external capital, real return %); this view is left untouched
    # pending review. Note `pnl` itself is correct and provably equals
    # RealPnL.real_net_profit.
    total_in = float(sum(r.lot.money_in for r, _ in pairs))
    current_value = float(sum(r.worth for r, _ in pairs))
    pnl = current_value - total_in
    total_return = pnl / total_in if total_in > 0 else None
    earliest = min(r.buy_eff for r, _ in pairs)
    days = max((pd.Timestamp.today().normalize() - earliest).days, 0)
    annualized = None
    if total_return is not None and days >= MIN_ANNUALIZE_DAYS and 1.0 + total_return > 0:
        annualized = float((1.0 + total_return) ** (365.25 / days) - 1.0)

    # the real value curve: union calendar, forward-filled per stock, leading
    # gaps = that lot's cash, trimmed to start at the earliest effective buy
    frame = pd.concat([c for _, c in pairs], axis=1)
    frame.columns = range(len(pairs))
    frame = frame.ffill()
    for i, (r, _) in enumerate(pairs):
        frame[i] = frame[i].fillna(r.lot.money_in)
    v = frame.sum(axis=1)
    v = v.loc[v.index >= earliest]

    ann_vol = risk = sharpe = max_drawdown = None
    if len(v) >= 2 and total_in > 0:
        r_v = v.pct_change()
        r_v.iloc[0] = v.iloc[0] / total_in - 1.0  # move on the buy day itself
        _, sharpe, max_drawdown, _, _ = perf_stats(r_v)
        _, ppy = time_base(r_v)
        std = r_v.std(ddof=1)
        ann_vol = float(std * np.sqrt(ppy)) if pd.notna(std) and std > 0 else 0.0
        risk = risk_label(ann_vol)

    return Totals(
        count=len(pairs),
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
    )


def analyze_investments(
    lots: Sequence[Lot],
    fetcher: DataFetcher,
    should_stop: Callable[[], bool] | None = None,
) -> InvestmentReport:
    results: list[LotResult] = []
    pairs: list[tuple[LotResult, pd.Series]] = []  # analyzable lots + their V(t) leg
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
            pairs.append((result, curve))

    total_in = float(sum(lot.money_in for lot in lots))
    skipped = [r.lot.ticker for r in results if r.error is not None]
    notes = [r.note for r, _ in pairs if r.note]

    return InvestmentReport(
        results=results,
        total_in=total_in,
        overall=_aggregate(pairs),
        open_only=_aggregate([p for p in pairs if not p[0].lot.is_sold]),
        notes=notes,
        skipped=skipped,
    )
