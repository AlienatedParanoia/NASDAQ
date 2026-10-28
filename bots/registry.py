"""Bot definitions, loaded from bots.yaml.

Bots live in config rather than code so adding one, changing its risk limits or
disabling it is an edit to a data file — reviewable, diffable, and not a reason
to touch the engine.

Every error here names the offending bot and says what was expected. A typo in
a strategy name must not silently produce a bot that never trades: that failure
mode looks exactly like a strategy that simply found no signals, and it would
be invisible for weeks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import paths
from bots.bot import Bot
from bots.risk import RiskLimits, build_risk
from bots.sizing import SizingPolicy, build_sizing
from bots.strategies.mean_reversion import MeanReversionStrategy
from bots.strategies.trend import TrendStrategy
from bots.strategy import Strategy
from data.symbols import normalize_all
from engine.costs import DEFAULT_COSTS, CostModel
from engine.ledger import Ledger

log = logging.getLogger(__name__)

#: Name used in bots.yaml -> strategy class.
STRATEGIES: dict[str, type[Strategy]] = {
    "trend": TrendStrategy,
    "mean_reversion": MeanReversionStrategy,
}

DEFAULT_STARTING_CASH = 100_000.0


class ConfigError(ValueError):
    """bots.yaml says something that cannot be acted on."""


@dataclass(slots=True)
class BotSpec:
    """One bot as configured — not yet bound to a ledger."""

    name: str
    strategy_name: str
    params: dict[str, Any] = field(default_factory=dict)
    universe: tuple[str, ...] = ()
    indexes: tuple[str, ...] = ()
    sizing: SizingPolicy = field(default_factory=build_sizing)
    risk: RiskLimits = field(default_factory=RiskLimits)
    starting_cash: float = DEFAULT_STARTING_CASH
    enabled: bool = True

    def build_strategy(self) -> Strategy:
        cls = STRATEGIES[self.strategy_name]
        try:
            return cls(**self.params)
        except TypeError as e:
            raise ConfigError(
                f"bot {self.name!r}: bad params for strategy "
                f"{self.strategy_name!r}: {e}"
            ) from e

    def resolve_universe(self) -> tuple[str, ...]:
        """Explicit tickers if given, otherwise the named indexes, otherwise
        the whole default universe."""
        if self.universe:
            return self.universe
        from data.universe import get_universe

        return tuple(get_universe(self.indexes or None))

    def build(
        self,
        ledger: Ledger,
        costs: CostModel = DEFAULT_COSTS,
        kind: str = "forward",
        universe: tuple[str, ...] | None = None,
    ) -> Bot:
        return Bot.create(
            name=self.name,
            strategy=self.build_strategy(),
            ledger=ledger,
            starting_cash=self.starting_cash,
            universe=universe if universe is not None else self.resolve_universe(),
            sizing=self.sizing,
            risk=self.risk,
            costs=costs,
            kind=kind,
            enabled=self.enabled,
        )

    def describe(self) -> str:
        state = "enabled" if self.enabled else "disabled"
        return (
            f"{self.name} [{state}]: {self.strategy_name} | "
            f"${self.starting_cash:,.0f} | {self.sizing.describe()} | "
            f"{self.risk.describe()}"
        )


def config_path() -> Path:
    return paths.resource_dir() / "bots.yaml"


def _require_mapping(value: Any, where: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def parse_spec(name: str, block: Any) -> BotSpec:
    block = _require_mapping(block, f"bot {name!r}")

    strategy_name = block.get("strategy")
    if strategy_name is None:
        raise ConfigError(f"bot {name!r}: no strategy named")
    if strategy_name not in STRATEGIES:
        raise ConfigError(
            f"bot {name!r}: unknown strategy {strategy_name!r}; "
            f"expected one of {sorted(STRATEGIES)}"
        )

    universe = block.get("universe") or ()
    if isinstance(universe, str):
        universe = [universe]
    indexes = block.get("indexes") or ()
    if isinstance(indexes, str):
        indexes = [indexes]

    starting_cash = float(block.get("starting_cash", DEFAULT_STARTING_CASH))
    if starting_cash <= 0:
        raise ConfigError(f"bot {name!r}: starting_cash must be positive")

    try:
        sizing = build_sizing(block.get("sizing"))
        risk = build_risk(_require_mapping(block.get("risk"), f"bot {name!r} risk"))
    except ValueError as e:
        raise ConfigError(f"bot {name!r}: {e}") from e

    return BotSpec(
        name=name,
        strategy_name=strategy_name,
        params=_require_mapping(block.get("params"), f"bot {name!r} params"),
        universe=tuple(normalize_all(universe)) if universe else (),
        indexes=tuple(indexes),
        sizing=sizing,
        risk=risk,
        starting_cash=starting_cash,
        enabled=bool(block.get("enabled", True)),
    )


def load_specs(path: Path | None = None) -> list[BotSpec]:
    """Read every bot definition. Raises ConfigError with the bot named."""
    import yaml

    path = path or config_path()
    if not path.exists():
        raise ConfigError(
            f"{path} is missing — define at least one bot (see the README)"
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from e

    bots_block = _require_mapping(raw.get("bots"), "top-level 'bots'")
    if not bots_block:
        raise ConfigError(f"{path} defines no bots under a 'bots:' key")

    return [parse_spec(name, block) for name, block in bots_block.items()]


def enabled_specs(path: Path | None = None) -> list[BotSpec]:
    return [spec for spec in load_specs(path) if spec.enabled]


def load_bots(
    ledger: Ledger,
    path: Path | None = None,
    costs: CostModel = DEFAULT_COSTS,
    kind: str = "forward",
    only_enabled: bool = True,
) -> list[Bot]:
    specs = enabled_specs(path) if only_enabled else load_specs(path)
    return [spec.build(ledger, costs=costs, kind=kind) for spec in specs]
