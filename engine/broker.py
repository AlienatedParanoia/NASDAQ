"""The Broker interface.

This is the seam that keeps the app broker-agnostic. Bots import `Broker` and
nothing below it; the only implementation in this repository is the simulated
one. A real-broker adapter could be written against this interface later
without touching a line of bot code — but see the hard rule in the project
spec: **there is no real-money trading code here, and none is planned.**

Deliberately narrow. Every method a bot needs and nothing more: no "get me the
raw API client", no vendor-shaped objects leaking through. If a bot can only
say these seven things, a bot cannot depend on simulator internals.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from engine.orders import AccountSummary, Order, Position


class Broker(ABC):
    """Everything a bot is allowed to ask of its account."""

    @abstractmethod
    def submit(self, order: Order) -> Order:
        """Accept or reject an order.

        Returns the same order with `id` assigned and `status` set. A rejection
        is NOT an exception — it is a returned order with status REJECTED and a
        human-readable `reject_reason`. Bots must cope with rejection as a
        normal outcome, because at a real broker it is one.
        """

    @abstractmethod
    def cancel(self, order_id: int) -> Order:
        """Cancel a resting order. Cancelling an already-terminal order is an
        error, not a silent no-op — it means the caller's view of the world is
        stale."""

    @abstractmethod
    def open_orders(self) -> list[Order]:
        """Orders still working, oldest first."""

    @abstractmethod
    def positions(self) -> dict[str, Position]:
        """Current holdings, derived from fills and corporate actions."""

    @abstractmethod
    def cash(self) -> float:
        """Settled cash. Never negative."""

    @abstractmethod
    def equity(self, mark_prices: dict[str, float]) -> float:
        """cash + the market value of every holding at the supplied prices.

        The caller supplies prices rather than the broker fetching them, so
        marking is explicit and a replay can mark against the same bar it is
        processing.
        """

    @abstractmethod
    def account_summary(self, mark_prices: dict[str, float] | None = None) -> AccountSummary:
        """One object with cash, equity, buying power, positions and open
        order count — what an account screen needs."""
