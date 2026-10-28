"""The look-ahead barrier.

The project's hard rule 5 says a decision made on bar *t* may only execute on
*t+1* or later, and that it must be enforced in code rather than by convention.
That rule has two halves, and they need different defences:

  * **Execution** — phase 3's broker refuses to fill an order on the session it
    was created, so even a strategy that cheated could not act on the cheat.
  * **Observation** — enforced here. A strategy is handed a DataView pinned to
    one session, and there is simply no way to reach a later bar through it.
    Every accessor truncates at the session, and asking explicitly for a later
    date raises LookAheadError.

Truncation is the real protection; the raise exists to catch a caller who tries
to be clever and would otherwise get a confusing empty frame instead of a clear
error.

Two kinds of price live here, and the distinction matters:

  * `history()` / `close()` return **as-traded** prices — what you would really
    have paid. Sizing and affordability use these.
  * `adjusted()` returns the split- and dividend-adjusted series, which is
    continuous across corporate actions. Indicators use these, because a 10:1
    split would otherwise look like a 90% crash to a moving average.

Frames are handed out as copies. A bot that mutated a slice would corrupt the
cache that the next bot in the same run reads, and that bug would be invisible
until two bots disagreed about history.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)


class LookAheadError(RuntimeError):
    """A strategy tried to look at data it could not have had at the time."""


@dataclass(slots=True)
class DataView:
    """Everything a strategy is allowed to know, as of one session.

    `frames` holds full as-traded OHLCV per ticker and `adjusted_closes` the
    matching adjusted series; both are truncated on the way out, never on the
    way in, so one cached copy serves every session of a replay.
    """

    session: pd.Timestamp
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    adjusted_closes: dict[str, pd.Series] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.session = pd.Timestamp(self.session).normalize()

    # ------------------------------------------------------------------ guard

    def _check(self, upto: pd.Timestamp) -> pd.Timestamp:
        upto = pd.Timestamp(upto).normalize()
        if upto > self.session:
            raise LookAheadError(
                f"asked for data up to {upto.date()} while standing on "
                f"{self.session.date()} — a decision made on this bar cannot "
                f"use a later one (project rule 5)"
            )
        return upto

    # ----------------------------------------------------------------- access

    @property
    def tickers(self) -> list[str]:
        return sorted(self.frames)

    def has(self, ticker: str) -> bool:
        """True when this ticker has at least one bar at or before the session."""
        frame = self.frames.get(ticker)
        if frame is None or frame.empty:
            return False
        return bool((frame.index <= self.session).any())

    def history(self, ticker: str, upto: pd.Timestamp | None = None) -> pd.DataFrame:
        """As-traded OHLCV up to and including the session. A copy, so a
        strategy cannot corrupt the shared cache."""
        limit = self.session if upto is None else self._check(upto)
        frame = self.frames.get(ticker)
        if frame is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return frame.loc[frame.index <= limit].copy()

    def adjusted(self, ticker: str, upto: pd.Timestamp | None = None) -> pd.Series:
        """Split- and dividend-adjusted closes, for indicators."""
        limit = self.session if upto is None else self._check(upto)
        series = self.adjusted_closes.get(ticker)
        if series is None:
            # fall back to as-traded rather than failing: a ticker with no
            # corporate actions has an identical adjusted series anyway
            frame = self.frames.get(ticker)
            if frame is None or frame.empty:
                return pd.Series(dtype=float)
            series = frame["close"]
        return series.loc[series.index <= limit].copy()

    def bar(self, ticker: str) -> pd.Series | None:
        """The session's own bar, or None when the ticker did not trade."""
        frame = self.frames.get(ticker)
        if frame is None or self.session not in frame.index:
            return None
        return frame.loc[self.session].copy()

    def close(self, ticker: str) -> float | None:
        """Latest as-traded close at or before the session — the price a
        decision made now would be based on."""
        history = self.history(ticker)
        if history.empty:
            return None
        return float(history["close"].iloc[-1])

    def bars_available(self, ticker: str) -> int:
        """How much history exists so far — strategies use this to refuse to
        trade inside their indicator warmup."""
        frame = self.frames.get(ticker)
        if frame is None:
            return 0
        return int((frame.index <= self.session).sum())

    def for_session(self, session: pd.Timestamp) -> "DataView":
        """A view on the same cached data, pinned to a different session.

        This is how a replay walks forward without re-reading the database:
        one load, one view per session.
        """
        return DataView(
            session=session,
            frames=self.frames,
            adjusted_closes=self.adjusted_closes,
        )


@dataclass(frozen=True, slots=True)
class PortfolioView:
    """What a strategy may know about its own account.

    Read-only by construction: a strategy states what it wants and the broker
    decides what happens. Deliberately excludes other bots entirely — nothing
    here can reach another account.
    """

    session: pd.Timestamp
    cash: float
    equity: float
    buying_power: float
    positions: dict[str, float] = field(default_factory=dict)   # ticker -> shares
    avg_costs: dict[str, float] = field(default_factory=dict)
    # session each open position was first established, so a strategy can run a
    # time stop against the REAL holding period rather than a re-simulated one
    entry_sessions: dict[str, pd.Timestamp] = field(default_factory=dict)
    open_order_tickers: frozenset[str] = frozenset()

    def holds(self, ticker: str) -> bool:
        return self.positions.get(ticker, 0.0) > 0

    def shares(self, ticker: str) -> float:
        return self.positions.get(ticker, 0.0)

    def has_open_order(self, ticker: str) -> bool:
        """So a strategy does not queue a second order for something it is
        already trying to trade."""
        return ticker in self.open_order_tickers

    @property
    def position_count(self) -> int:
        return sum(1 for q in self.positions.values() if q > 0)

    def bars_held(self, ticker: str, history: pd.DataFrame) -> int | None:
        """Sessions elapsed since this position was opened, on the ticker's own
        calendar. None when the entry session is unknown."""
        entry = self.entry_sessions.get(ticker)
        if entry is None or history.empty:
            return None
        return int((history.index > pd.Timestamp(entry)).sum())
