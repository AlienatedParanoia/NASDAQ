"""pytest configuration.

The default `pytest` run is fully offline and uses synthetic fixtures. The
handful of checks that genuinely need Yahoo (verifying our corporate-action
handling against the real NVDA split, our adjusted series against yfinance's
own) are marked `network` and skipped unless `--network` is passed, so the
suite stays fast and usable without a connection.
"""
from __future__ import annotations

import pytest


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--network",
        action="store_true",
        default=False,
        help="also run the tests that hit Yahoo Finance",
    )


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "network: requires live Yahoo Finance access")


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--network"):
        return
    skip = pytest.mark.skip(reason="needs --network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
