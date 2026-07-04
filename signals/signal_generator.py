"""Ties the pipeline together per stock: classify -> run the assigned
strategy's backtest -> derive the current Buy/Hold/Avoid signal from the
latest indicator state gated by backtest quality -> compose a human-readable
reasoning string."""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

import pandas as pd

from classifier.classify import Bucket, classify, compute_metrics
from data.fetcher import DataFetcher, FetchError, StockData
from strategies import indicators as ind
from strategies.backtest_engine import BacktestResult, Trade, run_backtest
from strategies.long_term import LongTermStrategy, composite_score
from strategies.mean_reversion import MeanReversionStrategy
from strategies.trend import TrendStrategy

log = logging.getLogger(__name__)

MIN_SHARPE = 0.5            # backtest quality gate, with...
MIN_RETURN = 0.0            # ...total return that must be positive
SHARPE_BUG_TRIPWIRE = 3.0   # above this, flag a probable pipeline bug
LT_BUY = 70.0
LT_AVOID = 45.0
FALLING_KNIFE = 0.8         # long-term Buy blocked below 80% of SMA200
MAX_MISSING_COMPONENTS = 3


class Signal(enum.StrEnum):
    BUY = "Buy"
    HOLD = "Hold"
    AVOID = "Avoid"


@dataclass(slots=True)
class ScanRow:
    ticker: str
    bucket: Bucket
    signal: Signal | None            # None for Unclassified or errors
    total_return: float | None
    sharpe: float | None
    max_dd: float | None
    bh_return: float | None
    reasoning: str
    prices: pd.DataFrame | None = None
    trades: list[Trade] = field(default_factory=list)
    overlay: dict[str, pd.Series] = field(default_factory=dict)
    error: str | None = None


def _quality(bt: BacktestResult) -> bool:
    return bt.sharpe >= MIN_SHARPE and bt.total_return > MIN_RETURN


def _bt_text(bt: BacktestResult) -> str:
    s = (
        f"Backtest 2y: {bt.total_return:+.1%} vs buy-and-hold "
        f"{bt.bh_total_return:+.1%}, Sharpe {bt.sharpe:.2f}, "
        f"MaxDD {bt.max_drawdown:.1%}, {bt.n_trades} trades."
    )
    if bt.sharpe > SHARPE_BUG_TRIPWIRE:
        s += " WARNING: implausibly high Sharpe — verify pipeline."
    return s


def _row(data: StockData, bucket: Bucket, signal: Signal | None, bt: BacktestResult,
         reasoning: str, overlay: dict[str, pd.Series]) -> ScanRow:
    return ScanRow(
        ticker=data.ticker,
        bucket=bucket,
        signal=signal,
        total_return=bt.total_return,
        sharpe=bt.sharpe,
        max_dd=bt.max_drawdown,
        bh_return=bt.bh_total_return,
        reasoning=reasoning,
        prices=data.prices,
        trades=bt.trades,
        overlay=overlay,
    )


def _trend_row(data: StockData, rule_trace: str) -> ScanRow:
    strat = TrendStrategy()
    px = data.prices
    bt = run_backtest(px, strat.positions(px))
    state = strat.latest_state(px)
    quality = _quality(bt)
    if state["in_uptrend"]:
        signal = Signal.BUY if quality else Signal.HOLD
        now_txt = (
            f"EMA20 ({state['ema_fast']:.2f}) is above EMA50 ({state['ema_slow']:.2f}) — uptrend intact"
            + ("" if quality else "; the strategy's own backtest is weak here, so Hold rather than Buy")
        )
    else:
        signal = Signal.AVOID
        now_txt = (
            f"EMA20 ({state['ema_fast']:.2f}) is below EMA50 ({state['ema_slow']:.2f}) — no uptrend to follow"
        )
    reasoning = (
        f"TREND bucket ({rule_trace}). {now_txt}. {_bt_text(bt)} "
        f"Signal: {signal.value.upper()}."
    )
    return _row(data, Bucket.TREND, signal, bt, reasoning, strat.overlays(px))


def _mean_reversion_row(data: StockData, rule_trace: str) -> ScanRow:
    strat = MeanReversionStrategy()
    px = data.prices
    bt = run_backtest(px, strat.positions(px))
    state = strat.latest_state(px)
    quality = _quality(bt)
    if state["setup"]:
        signal = Signal.BUY if quality else Signal.AVOID
        now_txt = (
            f"entry setup live: RSI {state['rsi']:.1f} < 30 and close {state['close']:.2f} "
            f"below the lower band {state['lower']:.2f}"
            + ("" if quality else ", but this stock hasn't rewarded dip-buying in backtest — Avoid")
        )
    elif state["in_trade"]:
        signal = Signal.HOLD
        now_txt = (
            f"a bounce position is currently open (RSI {state['rsi']:.1f}, "
            f"waiting for the middle band {state['mid']:.2f})"
        )
    else:
        signal = Signal.HOLD if quality else Signal.AVOID
        now_txt = (
            f"no oversold setup right now (RSI {state['rsi']:.1f}, close {state['close']:.2f} "
            f"vs lower band {state['lower']:.2f})"
            + ("; wait for one" if quality else ", and the backtest is weak — Avoid")
        )
    reasoning = (
        f"MEAN-REVERSION bucket ({rule_trace}). Currently {now_txt}. {_bt_text(bt)} "
        f"Signal: {signal.value.upper()}."
    )
    return _row(data, Bucket.MEAN_REVERSION, signal, bt, reasoning, strat.overlays(px))


def _long_term_row(data: StockData, rule_trace: str) -> ScanRow:
    strat = LongTermStrategy()
    px = data.prices
    bt = run_backtest(px, strat.positions(px))
    assert data.fundamentals is not None  # the long-term gates require them
    score, components, n_missing = composite_score(data.fundamentals)

    comp_txt = ", ".join(
        f"{k} {v:.0f}" if v is not None else f"{k} missing"
        for k, v in components.items()
    )
    if n_missing >= MAX_MISSING_COMPONENTS:
        signal = Signal.AVOID
        score_txt = (
            f"Only {len(components) - n_missing}/{len(components)} fundamentals "
            f"components are available — insufficient data to endorse ({comp_txt})"
        )
    else:
        if score >= LT_BUY:
            signal = Signal.BUY
        elif score >= LT_AVOID:
            signal = Signal.HOLD
        else:
            signal = Signal.AVOID
        score_txt = f"Fundamentals score {score:.0f}/100 ({comp_txt})"
        sma200 = ind.sma(px["close"], 200).iloc[-1]
        if (
            signal is Signal.BUY
            and pd.notna(sma200)
            and float(px["close"].iloc[-1]) < FALLING_KNIFE * float(sma200)
        ):
            signal = Signal.HOLD
            pct = float(px["close"].iloc[-1]) / float(sma200) - 1.0
            score_txt += (
                f"; price is {pct:.0%} vs its 200MA — falling-knife guard downgrades Buy to Hold"
            )
    reasoning = (
        f"LONG-TERM bucket ({rule_trace}). {score_txt}. "
        f"Strategy is buy-and-hold. {_bt_text(bt)} Signal: {signal.value.upper()}."
    )
    return _row(data, Bucket.LONG_TERM, signal, bt, reasoning, strat.overlays(px))


def generate_signal(data: StockData) -> ScanRow:
    bucket, rule_trace = classify(compute_metrics(data))
    if bucket is Bucket.TREND:
        row = _trend_row(data, rule_trace)
    elif bucket is Bucket.MEAN_REVERSION:
        row = _mean_reversion_row(data, rule_trace)
    elif bucket is Bucket.LONG_TERM:
        row = _long_term_row(data, rule_trace)
    else:
        row = ScanRow(
            ticker=data.ticker,
            bucket=Bucket.UNCLASSIFIED,
            signal=None,
            total_return=None,
            sharpe=None,
            max_dd=None,
            bh_return=None,
            reasoning=f"UNCLASSIFIED: {rule_trace}. No forced signal for this stock.",
            prices=data.prices,
            overlay={"SMA200": ind.sma(data.prices["close"], 200)},
        )
    if data.stale:
        row.reasoning += " (Note: served from a stale cache after a failed refresh.)"
    return row


def scan_universe(
    tickers: Sequence[str],
    fetcher: DataFetcher,
    progress: Callable[[int, int, str], None] | None = None,
    row_ready: Callable[[ScanRow], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[ScanRow]:
    """Runs the full pipeline per ticker. Every per-ticker failure is caught,
    logged, and turned into an error row — one bad ticker never kills a scan."""
    rows: list[ScanRow] = []
    n = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        if should_stop is not None and should_stop():
            log.info("scan cancelled at %d/%d", i - 1, n)
            break
        if progress is not None:
            progress(i, n, ticker)
        try:
            row = generate_signal(fetcher.get_stock(ticker))
        except FetchError as e:
            log.warning("%s skipped: %s", ticker, e)
            row = ScanRow(
                ticker=ticker, bucket=Bucket.UNCLASSIFIED, signal=None,
                total_return=None, sharpe=None, max_dd=None, bh_return=None,
                reasoning=f"Skipped: {e}", error=str(e),
            )
        except Exception as e:  # noqa: BLE001 — the scan must survive anything
            log.exception("%s: unexpected error during scan", ticker)
            row = ScanRow(
                ticker=ticker, bucket=Bucket.UNCLASSIFIED, signal=None,
                total_return=None, sharpe=None, max_dd=None, bh_return=None,
                reasoning=f"Error while analyzing: {e}", error=str(e),
            )
        rows.append(row)
        if row_ready is not None:
            row_ready(row)
    return rows
