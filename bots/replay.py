"""Historical replay: run a bot over a past date range.

Two rules, both of which exist so a replay cannot flatter itself.

**Same code path as live.** Replay calls `Bot.run_session`, exactly what the
daily runner calls. There is no backtest-only branch anywhere — if there were,
the two would drift and the replay would stop saying anything about the live
bot.

**Separate ledger.** Results go to `paths.backtest_ledger_path()`, never the
forward one, and the accounts are created with `kind="backtest"`. A replay must
never be readable as live paper performance, and `held_tickers()` ignores
backtest accounts so a 2019 holding cannot drag a delisted symbol into today's
price fetch.

Survivorship bias applies to everything here: the universe is today's index
members, so the companies that failed are already absent. Replay numbers read
optimistically by construction — see data/universe.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import paths
from bots.bot import Bot, BotSessionResult
from bots.data_view import DataView
from bots.registry import BotSpec
from data.fetcher import DataFetcher, adjust_closes
from engine.costs import DEFAULT_COSTS, CostModel
from engine.ledger import BACKTEST, Ledger
from engine.orders import Fill

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ReplayResult:
    """What a replay produced — enough to report on without re-querying."""

    bot_name: str
    start: pd.Timestamp
    end: pd.Timestamp
    sessions: int
    starting_cash: float
    final_equity: float
    equity_curve: pd.Series
    fills: list[Fill] = field(default_factory=list)
    blocked: list[tuple[str, str, str]] = field(default_factory=list)  # session, req, why
    killed_on: pd.Timestamp | None = None
    kill_reason: str = ""

    @property
    def total_return(self) -> float:
        if self.starting_cash <= 0:
            return 0.0
        return self.final_equity / self.starting_cash - 1.0

    @property
    def years(self) -> float:
        days = max((self.end - self.start).days, 1)
        return days / 365.25

    @property
    def cagr(self) -> float:
        if self.total_return <= -1.0 or self.years <= 0:
            return 0.0
        return (1.0 + self.total_return) ** (1.0 / self.years) - 1.0

    @property
    def max_drawdown(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        peak = self.equity_curve.cummax()
        return float((self.equity_curve / peak - 1.0).min())

    @property
    def total_commission(self) -> float:
        return sum(f.commission for f in self.fills)

    @property
    def total_slippage(self) -> float:
        return sum(f.slippage_cost for f in self.fills)

    def summary(self) -> str:
        return (
            f"{self.bot_name}: {self.total_return:+.1%} over {self.years:.1f}y "
            f"({self.cagr:+.1%}/yr), {len(self.fills)} fills, "
            f"maxDD {self.max_drawdown:.1%}, "
            f"costs ${self.total_commission + self.total_slippage:,.2f}"
        )


def load_market_data(
    fetcher: DataFetcher,
    tickers: list[str],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], dict[str, list]]:
    """Read price history, adjusted closes and corporate actions once.

    The whole history is loaded, not just the replay window: indicators need
    the warmup bars that precede `start`. DataView truncates per session, so
    loading extra past data cannot leak forward.
    """
    frames: dict[str, pd.DataFrame] = {}
    adjusted: dict[str, pd.Series] = {}
    actions: dict[str, list] = {}

    for ticker in tickers:
        prices = fetcher._load_prices(ticker)
        if prices.empty:
            continue
        if end is not None:
            prices = prices.loc[prices.index <= pd.Timestamp(end)]
        if prices.empty:
            continue
        ticker_actions = fetcher.actions(ticker)
        frames[ticker] = prices
        adjusted[ticker] = adjust_closes(prices, ticker_actions)
        actions[ticker] = ticker_actions
    return frames, adjusted, actions


def replay_bot(
    spec: BotSpec,
    fetcher: DataFetcher,
    start,
    end,
    ledger_path: Path | None = None,
    costs: CostModel = DEFAULT_COSTS,
    universe: list[str] | None = None,
    account_suffix: str = "",
    progress=None,
) -> ReplayResult:
    """Run one bot over [start, end] through a backtest ledger."""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        raise ValueError(f"end {end.date()} is before start {start.date()}")

    tickers = list(universe) if universe is not None else list(spec.resolve_universe())
    frames, adjusted, actions_by_ticker = load_market_data(fetcher, tickers, start, end)
    if not frames:
        raise ValueError(f"no cached price data for any of {len(tickers)} ticker(s)")

    ledger = Ledger(ledger_path or paths.backtest_ledger_path())
    try:
        name = f"{spec.name}{account_suffix}"
        # a replay is reproducible: start it from a clean account every time
        existing = ledger.account_by_name(name)
        if existing is not None:
            name = _unique_name(ledger, name)

        bot = Bot.create(
            name=name,
            strategy=spec.build_strategy(),
            ledger=ledger,
            starting_cash=spec.starting_cash,
            universe=tuple(frames),
            sizing=spec.sizing,
            risk=spec.risk,
            costs=costs,
            kind=BACKTEST,
        )

        base_view = DataView(session=start, frames=frames, adjusted_closes=adjusted)
        sessions = _sessions_in(frames, start, end)
        warmup = bot.strategy.warmup_bars()

        blocked: list[tuple[str, str, str]] = []
        killed_on = None
        kill_reason = ""

        for i, session in enumerate(sessions, 1):
            view = base_view.for_session(session)
            actions = _actions_on(actions_by_ticker, session)

            if _max_bars(view, frames) < warmup:
                # still inside the indicator warmup: let the broker advance
                # (so marks and snapshots exist) but do not let the strategy
                # trade on a half-formed indicator
                bot.broker.process_session(session, bot._bars_for(session, view), actions)
                continue

            result = bot.run_session(session, view, actions)
            for description, reason in result.blocked:
                blocked.append((str(session.date()), description, reason))
            if result.killed and killed_on is None:
                killed_on = session
                kill_reason = result.kill_reason
            if progress is not None:
                progress(i, len(sessions), session, result)

        curve = ledger.equity_curve(bot.account_id)
        fills = ledger.fills(bot.account_id)
        final_equity = float(curve.iloc[-1]) if len(curve) else spec.starting_cash

        return ReplayResult(
            bot_name=name,
            start=sessions[0] if sessions else start,
            end=sessions[-1] if sessions else end,
            sessions=len(sessions),
            starting_cash=spec.starting_cash,
            final_equity=final_equity,
            equity_curve=curve,
            fills=fills,
            blocked=blocked,
            killed_on=killed_on,
            kill_reason=kill_reason,
        )
    finally:
        ledger.close()


def _unique_name(ledger: Ledger, base: str) -> str:
    n = 2
    while ledger.account_by_name(f"{base}#{n}") is not None:
        n += 1
    return f"{base}#{n}"


def _sessions_in(frames, start, end) -> list[pd.Timestamp]:
    """Every session any ticker traded on, within the window."""
    index = pd.DatetimeIndex([])
    for frame in frames.values():
        index = index.union(frame.index)
    return list(index[(index >= start) & (index <= end)])


def _max_bars(view: DataView, frames) -> int:
    return max((view.bars_available(t) for t in frames), default=0)


def _actions_on(actions_by_ticker, session) -> list:
    out = []
    for actions in actions_by_ticker.values():
        out.extend(a for a in actions if pd.Timestamp(a.ex_date).normalize() == session)
    return out
