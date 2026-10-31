"""What the Bots tab shows, computed without Qt so it can be unit-tested.

Everything here is read-only and comes from two places: the ledger (what the
bots did) and the price cache (what things were worth). Nothing fetches from
the network; a number that cannot be computed from what is stored is `None`
and the tab shows "-", never an estimate.

**Benchmark.** Each bot is compared against QQQ bought and held over the same
sessions, from the close of the bot's first snapshot to the close of its last.
The adjusted series is used (dividends reinvested) because the bots receive
dividends as cash: a price-only QQQ would understate what holding it earned
and flatter every bot. If either end's QQQ close is not cached, there is no
comparison: an end date nudged to a nearby session would be a guess.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bots.registry import BotSpec
from engine.ledger import (
    BUY,
    COMMISSION,
    SELL,
    Account,
    Ledger,
)

#: Buy-and-hold yardstick. Not an index member, so the daily runner fetches it
#: explicitly (see bots/run_daily.py).
BENCHMARK = "QQQ"

# cash_events kinds that only mirror a fill; the Activity list shows the fill
_FILL_CASH_KINDS = {BUY, SELL, COMMISSION}


@dataclass(frozen=True, slots=True)
class BotSummary:
    """One row of the overview table."""

    name: str
    strategy: str
    enabled: bool
    account: Account | None          # None: the bot has never run
    starting_cash: float
    equity: float | None
    cash: float | None
    total_return: float | None       # fraction, 0.05 = +5%
    benchmark_return: float | None
    max_drawdown: float | None       # fraction below the running peak, >= 0
    positions: int
    open_orders: int
    first_session: pd.Timestamp | None
    last_session: pd.Timestamp | None
    behind: bool                     # last session older than the latest complete one
    previous_accounts: int           # accounts retired by a reset

    @property
    def status(self) -> str:
        return self.account.status if self.account else "not started"

    @property
    def excess_return(self) -> float | None:
        if self.total_return is None or self.benchmark_return is None:
            return None
        return self.total_return - self.benchmark_return


@dataclass(frozen=True, slots=True)
class PositionRow:
    ticker: str
    qty: float
    avg_cost: float
    mark: float | None
    mark_session: pd.Timestamp | None

    @property
    def market_value(self) -> float | None:
        return None if self.mark is None else self.qty * self.mark

    @property
    def unrealised(self) -> float | None:
        return None if self.mark is None else self.qty * (self.mark - self.avg_cost)

    @property
    def unrealised_pct(self) -> float | None:
        if self.mark is None or self.avg_cost <= 0:
            return None
        return self.mark / self.avg_cost - 1.0


@dataclass(frozen=True, slots=True)
class ActivityRow:
    session: str          # YYYY-MM-DD, or "" when the event had no session
    kind: str
    description: str
    amount: float | None


# ------------------------------------------------------------------ helpers


def max_drawdown(curve: pd.Series) -> float | None:
    """Largest fall from a running peak, as a positive fraction."""
    if curve.empty:
        return None
    peak = curve.cummax()
    return max(0.0, float(-(curve / peak - 1.0).min()))  # no -0.0 on a flat curve


def benchmark_return(
    adjusted: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> float | None:
    """Buy-and-hold return from `start`'s close to `end`'s close, or None if
    either close is missing (see the module docstring)."""
    if adjusted is None or adjusted.empty:
        return None
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if start not in adjusted.index or end not in adjusted.index:
        return None
    first = float(adjusted.loc[start])
    if first <= 0:
        return None
    return float(adjusted.loc[end]) / first - 1.0


def benchmark_curve(
    adjusted: pd.Series, sessions: pd.DatetimeIndex, starting_value: float
) -> pd.Series:
    """QQQ rebased to `starting_value` on the first of `sessions`, for the
    equity chart. Empty if the first session's close is missing."""
    if adjusted is None or adjusted.empty or len(sessions) == 0:
        return pd.Series(dtype=float)
    first = sessions[0]
    if first not in adjusted.index:
        return pd.Series(dtype=float)
    window = adjusted.loc[(adjusted.index >= first) & (adjusted.index <= sessions[-1])]
    return window / float(adjusted.loc[first]) * starting_value


def _cached_adjusted(prices, ticker: str) -> pd.Series:
    """The adjusted series from the cache, or empty. `prices` is a DataFetcher
    (or anything with its `_load_prices` / `adjusted_closes`)."""
    if prices is None or prices._load_prices(ticker).empty:
        return pd.Series(dtype=float)
    return prices.adjusted_closes(ticker)


# ------------------------------------------------------------------ reports


def summarise(
    ledger: Ledger, spec: BotSpec, latest: pd.Timestamp, prices=None,
) -> BotSummary:
    """The overview row for one bot. `prices` (a DataFetcher) supplies the
    benchmark; without it the benchmark column is empty."""
    account = ledger.bot_account(spec.name)
    history = ledger.bot_account_history(spec.name)
    if account is None:
        return BotSummary(
            name=spec.name, strategy=spec.strategy_name, enabled=spec.enabled,
            account=None, starting_cash=spec.starting_cash, equity=None, cash=None,
            total_return=None, benchmark_return=None, max_drawdown=None,
            positions=0, open_orders=0, first_session=None, last_session=None,
            behind=False, previous_accounts=0,
        )

    curve = ledger.equity_curve(account.id)
    snaps = ledger.snapshots(account.id)
    last_session = ledger.last_processed_session(account.id)
    equity = float(curve.iloc[-1]) if len(curve) else None
    first = curve.index[0] if len(curve) else None
    end = curve.index[-1] if len(curve) else None

    bench = None
    if first is not None:
        bench = benchmark_return(_cached_adjusted(prices, BENCHMARK), first, end)

    return BotSummary(
        name=spec.name,
        strategy=spec.strategy_name,
        enabled=spec.enabled,
        account=account,
        starting_cash=account.starting_cash,
        equity=equity,
        cash=snaps[-1].cash if snaps else ledger.rebuild_cash(account.id),
        total_return=(
            equity / account.starting_cash - 1.0
            if equity is not None and account.starting_cash > 0 else None
        ),
        benchmark_return=bench,
        max_drawdown=max_drawdown(curve),
        positions=len(ledger.rebuild_positions(account.id)),
        open_orders=len(ledger.open_orders(account.id)),
        first_session=first,
        last_session=last_session,
        behind=last_session is not None and last_session < pd.Timestamp(latest).normalize(),
        previous_accounts=len(history) - 1,
    )


def positions(ledger: Ledger, account: Account, prices=None) -> list[PositionRow]:
    """Holdings marked at the last cached close on or before the account's
    last processed session, so they agree with its snapshot equity. Prices
    are as-traded and positions already carry their splits, so the two
    match without adjustment."""
    cursor = ledger.last_processed_session(account.id)
    rows = []
    for ticker, position in sorted(ledger.rebuild_positions(account.id).items()):
        mark = mark_session = None
        if prices is not None and cursor is not None:
            frame = prices._load_prices(ticker)
            frame = frame.loc[frame.index <= cursor] if not frame.empty else frame
            if not frame.empty:
                mark = float(frame["close"].iloc[-1])
                mark_session = frame.index[-1]
        rows.append(PositionRow(ticker, position.qty, position.avg_cost, mark, mark_session))
    return rows


def activity(ledger: Ledger, account: Account) -> list[ActivityRow]:
    """Fills, non-fill cash events, corporate actions and status changes,
    newest first."""
    rows: list[tuple[str, int, ActivityRow]] = []
    for f in ledger.fills(account.id):
        session = str(f.session.date())
        rows.append((session, 1, ActivityRow(
            session, "fill",
            f"{f.side.value} {f.qty:g} {f.ticker} @ {f.price:.4f}"
            + (f" (commission {f.commission:.2f})" if f.commission else ""),
            (-1 if f.side.value == "buy" else 1) * f.qty * f.price - f.commission,
        )))
    for e in ledger.cash_events(account.id):
        if e["kind"] in _FILL_CASH_KINDS:
            continue
        rows.append((e["session"] or "", 0, ActivityRow(
            e["session"] or "", e["kind"], e["note"] or "", e["amount"],
        )))
    for a in ledger.corporate_actions(account.id):
        held = ""
        if a["shares_before"] is not None and a["shares_after"] is not None:
            held = f": {a['shares_before']:g} -> {a['shares_after']:g} shares"
        rows.append((a["session"], 0, ActivityRow(
            a["session"], a["type"], f"{a['ticker']} {a['type']} {a['value']:g}{held}", None,
        )))
    for s in ledger.status_history(account.id):
        session = s["session"] or s["at"][:10]
        why = f": {s['reason']}" if s["reason"] else ""
        rows.append((session, 2, ActivityRow(
            session, "status", f"{s['status']} ({s['source']}){why}", None,
        )))
    # stable sort: newest session first; inside a session, status changes
    # above fills above everything else, each kept in recording order
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    return [r[2] for r in rows]


def orders(ledger: Ledger, account: Account) -> list:
    """Every order, newest first; `ledger.order_history(id)` has its trail."""
    return list(reversed(ledger.orders(account.id)))
