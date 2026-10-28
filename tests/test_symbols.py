"""Ticker normalisation. Offline."""
from __future__ import annotations

import pytest

from data.symbols import normalize, normalize_all


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("aapl", "AAPL"),
        ("  msft  ", "MSFT"),
        ("BRK.B", "BRK-B"),
        ("brk.b", "BRK-B"),
        ("BF.B", "BF-B"),
        ("BRK-B", "BRK-B"),  # already normalised, unchanged
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_normalize_is_idempotent():
    assert normalize(normalize("brk.b")) == normalize("brk.b")


@pytest.mark.parametrize("bad", ["", "   ", "."])
def test_empty_symbol_is_rejected(bad):
    # a blank ticker reaching the fetcher becomes a confusing Yahoo 404 several
    # layers from the real mistake, so it is refused at the boundary
    with pytest.raises(ValueError):
        normalize(bad)


def test_normalize_all_dedupes_and_keeps_order():
    assert normalize_all(["AAPL", "brk.b", "aapl", "BRK-B", "MSFT"]) == [
        "AAPL",
        "BRK-B",
        "MSFT",
    ]
