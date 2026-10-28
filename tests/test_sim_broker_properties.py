"""Property tests: random order sequences on random-walk prices.

The named tests in test_sim_broker.py check situations I thought of. These
check the invariants that must hold in situations I did not — which is where
an accounting bug actually hides.

After every session, for every account, all of the following must be true:

  * cash >= 0                       (no overdraft, ever)
  * every share count >= 0          (no accidental shorting)
  * equity == cash + Σ(qty × close) (the books balance)
  * rebuilding from the ledger reproduces the same state

Seeds are fixed, so a failure is reproducible rather than a story about a run
that happened once.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd
import pytest

from engine.costs import CostModel, ZERO_COSTS
from engine.ledger import DIVIDEND, SPLIT, Ledger
from engine.orders import Order, OrderError, OrderType, Side, TimeInForce
from engine.sim_broker import Bar, SimBroker

TICKERS = ("AAA", "BBB", "CCC")
SESSIONS = 60
START_CASH = 50_000.0
CENT = 0.01


@dataclass(frozen=True)
class Action:
    ticker: str
    ex_date: pd.Timestamp
    type: str
    value: float


def random_walk(rng: random.Random, n: int, start: float = 100.0) -> list[Bar]:
    """A price path with realistic intrabar ranges, so limit and stop orders
    actually get a chance to trigger."""
    bars: list[Bar] = []
    price = start
    for _ in range(n):
        open_ = price * (1.0 + rng.gauss(0.0, 0.01))
        close = open_ * (1.0 + rng.gauss(0.0, 0.02))
        high = max(open_, close) * (1.0 + abs(rng.gauss(0.0, 0.01)))
        low = min(open_, close) * (1.0 - abs(rng.gauss(0.0, 0.01)))
        low = max(low, 0.01)
        bars.append(Bar(open=open_, high=high, low=low, close=close, volume=1e6))
        price = close
    return bars


def random_order(rng: random.Random, broker: SimBroker, session, ticker: str,
                 last_close: float) -> Order | None:
    """A plausible order — sometimes affordable, sometimes not, which is the
    point: the broker must refuse the impossible ones cleanly."""
    side = Side.BUY if rng.random() < 0.55 else Side.SELL
    qty = rng.randint(1, 60)
    kind = rng.choice([OrderType.MARKET, OrderType.LIMIT, OrderType.STOP])
    kwargs: dict = {"type": kind}
    if kind is OrderType.LIMIT:
        kwargs["limit_price"] = round(last_close * rng.uniform(0.90, 1.10), 2)
    elif kind is OrderType.STOP:
        kwargs["stop_price"] = round(last_close * rng.uniform(0.90, 1.10), 2)
    try:
        return Order(
            account_id=broker.account_id, created_session=session, ticker=ticker,
            side=side, qty=qty,
            tif=TimeInForce.GTC if rng.random() < 0.5 else TimeInForce.DAY,
            reason="property test", **kwargs,
        )
    except OrderError:
        return None


def check_invariants(broker: SimBroker, result, closes: dict[str, float]) -> None:
    cash = broker.cash()
    assert cash >= -CENT, f"cash went negative: {cash}"

    positions = broker.positions()
    for ticker, position in positions.items():
        assert position.qty >= 0, f"{ticker}: negative share count {position.qty}"
        assert position.avg_cost >= 0, f"{ticker}: negative average cost"

    expected = cash + sum(
        p.qty * closes.get(t, p.avg_cost) for t, p in positions.items()
    )
    assert result.equity == pytest.approx(expected, abs=CENT), (
        f"equity {result.equity} != cash {cash} + holdings on {result.session.date()}"
    )


def run_simulation(seed: int, costs: CostModel, with_actions: bool = False) -> dict:
    rng = random.Random(seed)
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp()) / "ledger.db"
    ledger = Ledger(tmp)
    account = ledger.create_account(f"prop-{seed}", START_CASH)
    broker = SimBroker(ledger, account.id, costs=costs)

    paths = {t: random_walk(rng, SESSIONS, start=rng.uniform(20.0, 300.0))
             for t in TICKERS}
    sessions = pd.bdate_range("2026-01-05", periods=SESSIONS)

    for t, series in paths.items():
        broker.set_mark(t, series[0].open)

    stats = {"submitted": 0, "rejected": 0, "fills": 0, "expired": 0}

    for i, session in enumerate(sessions):
        bars = {t: paths[t][i] for t in TICKERS}

        actions = []
        if with_actions and i > 0:
            for t in TICKERS:
                roll = rng.random()
                if roll < 0.02:
                    actions.append(Action(t, session, SPLIT, rng.choice([2.0, 3.0])))
                elif roll < 0.08:
                    actions.append(
                        Action(t, session, DIVIDEND, round(rng.uniform(0.05, 0.8), 2))
                    )

        result = broker.process_session(session, bars, actions)
        stats["fills"] += len(result.fills)
        stats["expired"] += len(result.expired)
        check_invariants(broker, result, {t: b.close for t, b in bars.items()})

        # queue tomorrow's orders using only what today's bar showed
        for _ in range(rng.randint(0, 4)):
            ticker = rng.choice(TICKERS)
            order = random_order(rng, broker, session, ticker, bars[ticker].close)
            if order is None:
                continue
            stats["submitted"] += 1
            placed = broker.submit(order)
            if placed.status.value == "rejected":
                stats["rejected"] += 1
                assert placed.reject_reason, "a rejection must say why"

    # rebuilding from the ledger alone must reproduce the same state
    fresh = SimBroker(ledger, account.id, costs=costs)
    assert fresh.cash() == pytest.approx(broker.cash(), abs=CENT)
    assert fresh.positions().keys() == broker.positions().keys()
    for ticker, position in broker.positions().items():
        assert fresh.positions()[ticker].qty == pytest.approx(position.qty)
        assert fresh.positions()[ticker].avg_cost == pytest.approx(
            position.avg_cost, abs=CENT
        )

    stats["final_cash"] = broker.cash()
    stats["positions"] = len(broker.positions())
    ledger.close()
    return stats


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11, 42, 99, 1234])
def test_invariants_hold_on_random_sequences(seed):
    stats = run_simulation(seed, ZERO_COSTS)
    assert stats["submitted"] > 0, "the simulation did nothing"


@pytest.mark.parametrize("seed", [5, 17, 2026])
def test_invariants_hold_with_corporate_actions(seed):
    """Splits and dividends in the middle of random trading."""
    stats = run_simulation(seed, ZERO_COSTS, with_actions=True)
    assert stats["submitted"] > 0


@pytest.mark.parametrize("seed", [3, 23])
def test_invariants_hold_with_real_costs(seed):
    """Commission and slippage must not be able to push cash negative."""
    stats = run_simulation(seed, CostModel(), with_actions=True)
    assert stats["final_cash"] >= -CENT


def test_the_simulation_actually_exercises_the_broker():
    """A property test that never fills anything proves nothing — this makes
    the coverage visible rather than assumed."""
    stats = run_simulation(42, ZERO_COSTS, with_actions=True)
    assert stats["fills"] > 10, f"too few fills to be meaningful: {stats}"
    assert stats["rejected"] > 0, "no rejections: the cash checks were never tested"
    assert stats["expired"] > 0, "no expiries: DAY handling was never tested"


def test_two_bots_on_one_ledger_never_affect_each_other(tmp_path):
    """Phase 4 runs several bots side by side; their accounts must be
    completely isolated."""
    ledger = Ledger(tmp_path / "ledger.db")
    a = ledger.create_account("bot-a", 10_000.0)
    b = ledger.create_account("bot-b", 10_000.0)
    broker_a = SimBroker(ledger, a.id, costs=ZERO_COSTS)
    broker_b = SimBroker(ledger, b.id, costs=ZERO_COSTS)
    for broker in (broker_a, broker_b):
        broker.set_mark("AAA", 100.0)

    d1, d2 = pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")
    broker_a.submit(Order(account_id=a.id, created_session=d1, ticker="AAA",
                          side=Side.BUY, qty=50))
    bars = {"AAA": Bar(100.0, 100.0, 100.0, 100.0)}
    broker_a.process_session(d2, bars)
    broker_b.process_session(d2, bars)

    assert broker_a.positions()["AAA"].qty == 50
    assert broker_b.positions() == {}
    assert broker_a.cash() == pytest.approx(5_000.0)
    assert broker_b.cash() == pytest.approx(10_000.0)
    assert broker_b.open_orders() == []
    ledger.close()
