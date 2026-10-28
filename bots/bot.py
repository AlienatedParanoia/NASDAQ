"""A bot: one strategy, one account, one set of limits.

`run_session` is deliberately the **only** code path that advances a bot, and
both replay (phase 4) and the daily runner (phase 5) call it. If backtesting
used a different path from live trading, the two would drift and the backtest
would stop predicting anything about the live bot — which is the usual reason
backtests lie.

The order of work inside a session mirrors reality:

  1. The broker processes the session — corporate actions, then any orders
     queued *yesterday* fill against today's bar, then mark to market.
  2. Only then does the strategy see the session, through a DataView that
     cannot reach past it.
  3. Its requests are sized, risk-checked, and submitted — to be executed
     tomorrow at the earliest.

Step 1 before step 2 matters: the strategy must decide against the account as
it stands *after* today's fills, not before them.

Bots are completely isolated from each other. A Bot touches exactly one
account_id, and nothing here can read another bot's positions or cash.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from bots.data_view import DataView, PortfolioView
from bots.risk import RiskLimits, RiskManager
from bots.sizing import EqualWeight, SizingPolicy
from bots.strategy import OrderRequest, Strategy
from engine.costs import DEFAULT_COSTS, CostModel
from engine.ledger import ACTIVE, KILLED, PAUSED, Ledger
from engine.orders import Order, OrderStatus, Side
from engine.sim_broker import SessionResult, SimBroker

log = logging.getLogger(__name__)


@dataclass(slots=True)
class BotSessionResult:
    """What one bot did on one session — enough for a log line or a UI row."""

    session: pd.Timestamp
    broker: SessionResult
    requested: int = 0
    submitted: int = 0
    blocked: tuple[tuple[str, str], ...] = ()   # (request description, reason)
    killed: bool = False
    kill_reason: str = ""
    skipped: str = ""                            # why nothing happened, if so

    @property
    def equity(self) -> float:
        return self.broker.equity

    @property
    def cash(self) -> float:
        return self.broker.cash


@dataclass(slots=True)
class Bot:
    """A strategy bound to its own simulated account."""

    name: str
    strategy: Strategy
    ledger: Ledger
    account_id: int
    universe: tuple[str, ...] = ()
    sizing: SizingPolicy = field(default_factory=EqualWeight)
    risk: RiskLimits = field(default_factory=RiskLimits)
    costs: CostModel = DEFAULT_COSTS
    enabled: bool = True
    broker: SimBroker = field(init=False)
    _risk_manager: RiskManager = field(init=False)

    def __post_init__(self) -> None:
        self.broker = SimBroker(self.ledger, self.account_id, costs=self.costs)
        self._risk_manager = RiskManager(self.risk)

    # ------------------------------------------------------------- factories

    @classmethod
    def create(
        cls,
        name: str,
        strategy: Strategy,
        ledger: Ledger,
        starting_cash: float,
        universe: tuple[str, ...] = (),
        sizing: SizingPolicy | None = None,
        risk: RiskLimits | None = None,
        costs: CostModel = DEFAULT_COSTS,
        kind: str = "forward",
        enabled: bool = True,
    ) -> "Bot":
        """Open a fresh account for this bot, or reuse the existing one."""
        account = ledger.account_by_name(name)
        if account is None:
            account = ledger.create_account(name, starting_cash, kind=kind)
        return cls(
            name=name, strategy=strategy, ledger=ledger, account_id=account.id,
            universe=tuple(universe), sizing=sizing or EqualWeight(),
            risk=risk or RiskLimits(), costs=costs, enabled=enabled,
        )

    # ----------------------------------------------------------------- state

    @property
    def status(self) -> str:
        return self.ledger.get_account(self.account_id).status

    def portfolio_view(self, session: pd.Timestamp, marks: dict[str, float]) -> PortfolioView:
        positions = self.broker.positions()
        return PortfolioView(
            session=session,
            cash=self.broker.cash(),
            equity=self.broker.equity(marks),
            buying_power=self.broker.buying_power(),
            positions={t: p.qty for t, p in positions.items()},
            avg_costs={t: p.avg_cost for t, p in positions.items()},
            entry_sessions=self._entry_sessions(positions),
            open_order_tickers=frozenset(o.ticker for o in self.broker.open_orders()),
        )

    def _entry_sessions(self, positions) -> dict[str, pd.Timestamp]:
        """When each currently-open position was established.

        Walks the fills forward and resets on a full exit, so a ticker bought,
        sold and bought again reports the LATEST entry — which is what a time
        stop should measure from.
        """
        entries: dict[str, pd.Timestamp] = {}
        running: dict[str, float] = {}
        for fill in self.ledger.fills(self.account_id):
            held = running.get(fill.ticker, 0.0)
            if fill.side is Side.BUY:
                if held <= 0:
                    entries[fill.ticker] = fill.session
                running[fill.ticker] = held + fill.qty
            else:
                running[fill.ticker] = max(held - fill.qty, 0.0)
                if running[fill.ticker] <= 0:
                    entries.pop(fill.ticker, None)
        return {t: s for t, s in entries.items() if t in positions}

    def orders_today(self, session: pd.Timestamp) -> int:
        """Orders already created for this session — so a catch-up run cannot
        blow through the daily cap by processing several days at once."""
        return sum(
            1 for o in self.ledger.orders(self.account_id)
            if o.created_session == pd.Timestamp(session).normalize()
        )

    # ------------------------------------------------------------- the path

    def run_session(
        self,
        session: pd.Timestamp,
        data: DataView,
        actions=(),
    ) -> BotSessionResult:
        """Advance this bot through one session. The one code path."""
        session = pd.Timestamp(session).normalize()

        bars = self._bars_for(session, data)
        broker_result = self.broker.process_session(session, bars, actions)
        marks = {t: b.close for t, b in bars.items()}

        if self.status == KILLED:
            return BotSessionResult(
                session=session, broker=broker_result,
                skipped="bot is killed", killed=True,
            )
        if self.status == PAUSED:
            return BotSessionResult(
                session=session, broker=broker_result, skipped="bot is paused"
            )
        if not self.enabled:
            return BotSessionResult(
                session=session, broker=broker_result, skipped="bot is disabled"
            )

        portfolio = self.portfolio_view(session, marks)

        # drawdown is judged before the strategy is even consulted
        equity_curve = self.ledger.equity_curve(self.account_id)
        requests = self.strategy.on_bar(session, data, portfolio)
        sized = self._size(requests, data, portfolio)

        decision = self._risk_manager.check(
            sized, data, portfolio, equity_curve=equity_curve,
            orders_today=self.orders_today(session),
        )
        decision.log(self.name)

        if decision.kill:
            self._kill(session, decision.kill_reason, data, portfolio)
            return BotSessionResult(
                session=session, broker=broker_result, requested=len(requests),
                blocked=tuple((b.request.describe(), b.reason) for b in decision.blocked),
                killed=True, kill_reason=decision.kill_reason,
            )

        submitted = 0
        for request in decision.allowed:
            order = self._submit(session, request)
            if order is not None and order.status is OrderStatus.PENDING:
                submitted += 1

        return BotSessionResult(
            session=session,
            broker=broker_result,
            requested=len(requests),
            submitted=submitted,
            blocked=tuple((b.request.describe(), b.reason) for b in decision.blocked),
        )

    def _bars_for(self, session: pd.Timestamp, data: DataView) -> dict:
        from engine.sim_broker import Bar

        bars: dict[str, Bar] = {}
        for ticker in data.tickers:
            row = data.bar(ticker)
            if row is None:
                continue
            bars[ticker] = Bar.from_row(row)
        return bars

    def _size(
        self, requests: list[OrderRequest], data: DataView, portfolio: PortfolioView
    ) -> list[OrderRequest]:
        """Fill in quantities the strategy left open. A request that cannot be
        sized is dropped here rather than being sent to be rejected."""
        out: list[OrderRequest] = []
        for request in requests:
            if request.qty is not None:
                out.append(request)
                continue
            if request.side is Side.SELL:
                out.append(request.with_qty(portfolio.shares(request.ticker)))
                continue
            qty = self.sizing.shares_for(
                request.ticker, data, portfolio, self.risk.max_positions
            )
            if qty <= 0:
                log.debug(
                    "%s: %s not affordable at current buying power",
                    self.name, request.ticker,
                )
                continue
            out.append(request.with_qty(qty))
        return out

    def _submit(self, session: pd.Timestamp, request: OrderRequest) -> Order | None:
        try:
            order = Order(
                account_id=self.account_id,
                created_session=session,
                ticker=request.ticker,
                side=request.side,
                qty=request.qty,
                type=request.type,
                limit_price=request.limit_price,
                stop_price=request.stop_price,
                tif=request.tif,
                reason=f"{self.strategy.name}: {request.reason}",
            )
        except Exception as e:  # malformed request — a strategy bug, not a rejection
            log.warning("%s: could not build order %s: %s", self.name, request.describe(), e)
            return None
        return self.broker.submit(order)

    def _kill(
        self, session: pd.Timestamp, reason: str, data: DataView, portfolio: PortfolioView
    ) -> None:
        """Stop the bot. Cancels resting orders; liquidates only if configured.

        Holding by default is deliberate — see bots/risk.py.

        Order of operations matters and is easy to get wrong: resting orders are
        cancelled FIRST (a resting sell reserves its shares, which would make
        the liquidation look like an oversell), the liquidation is submitted
        SECOND, and the account is marked killed LAST — because the broker
        refuses orders from a killed account, so marking it first would silently
        reject the very orders this is trying to place.
        """
        for order in self.broker.open_orders():
            self.broker.cancel(order.id)

        if self.risk.liquidate_on_kill:
            placed = 0
            for ticker, qty in portfolio.positions.items():
                if qty <= 0:
                    continue
                order = self._submit(
                    session,
                    OrderRequest(
                        ticker=ticker, side=Side.SELL, qty=qty,
                        reason=f"liquidate on kill: {reason}",
                    ),
                )
                if order is not None and order.status is OrderStatus.PENDING:
                    placed += 1
            log.warning(
                "%s killed (%s); liquidating %d position(s) on the next session",
                self.name, reason, placed,
            )
        else:
            log.warning("%s killed (%s); positions held", self.name, reason)

        self.ledger.set_account_status(self.account_id, KILLED)

    # ----------------------------------------------------------------- admin

    def pause(self) -> None:
        self.ledger.set_account_status(self.account_id, PAUSED)

    def resume(self) -> None:
        self.ledger.set_account_status(self.account_id, ACTIVE)

    def kill(self, reason: str = "manual") -> None:
        self.ledger.set_account_status(self.account_id, KILLED)
        for order in self.broker.open_orders():
            self.broker.cancel(order.id)
        log.warning("%s killed manually: %s", self.name, reason)

    def describe(self) -> str:
        return (
            f"{self.name}: {self.strategy.describe()} | {self.sizing.describe()} | "
            f"{self.risk.describe()}"
        )
