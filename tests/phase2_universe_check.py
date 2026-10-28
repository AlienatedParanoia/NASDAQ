"""Phase 2 check (network): fetch the whole S&P 500 + NASDAQ-100 union and prove
the caching holds.

Two passes. The first is the cold fetch — batched and rate-limited — over every
unique symbol in both indexes. The second re-reads the same universe immediately
and must be almost entirely cache hits; if it is not, the freshness logic is
broken and every launch would re-hammer Yahoo.

Offline structural checks (sizes, normalisation, duplicates) live in
tests/test_universe.py. Run from the project root:
python -m tests.phase2_universe_check
"""
from __future__ import annotations

import logging
import time

import paths
from data.fetcher import DataFetcher, configure_yfinance
from data.universe import (
    DEFAULT_INDEXES,
    describe,
    get_constituents,
    get_universe,
)

# 503 + 101 - 86 overlapping = 518. The original spec said "~600", which no
# arrangement of these two indexes produces; this band allows reconstitution
# drift without going quiet if a file collapses.
MIN_UNION, MAX_UNION = 500, 540

# yfinance cannot serve a handful of symbols at any given time (recent listings,
# ticker changes). More than this means something systemic.
MAX_FAILURES = 15


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    sp = get_universe(("sp500",))
    nd = get_universe(("nasdaq100",))
    universe = get_universe(DEFAULT_INDEXES)
    overlap = set(sp) & set(nd)

    print(f"S&P 500:     {len(sp)} symbols")
    print(f"NASDAQ-100:  {len(nd)} symbols")
    print(f"overlap:     {len(overlap)} symbols in both")
    print(f"union:       {len(universe)} unique symbols  ({describe()})")

    assert len(universe) == len(set(sp) | set(nd)), "union is not de-duplicated"
    assert MIN_UNION <= len(universe) <= MAX_UNION, (
        f"union is {len(universe)}, expected {MIN_UNION}-{MAX_UNION}"
    )
    assert len(universe) == len(set(universe)), "duplicates survived"
    for t in universe:
        assert t and t == t.upper() and "." not in t, f"malformed symbol {t!r}"
    print("universe structure OK: de-duplicated, normalised, no malformed symbols")

    # sector metadata comes along for the ride (phase 6 and the ML phase use it)
    sectors = {c.sector for c in get_constituents("sp500") if c.sector}
    print(f"S&P 500 GICS sectors present: {len(sectors)}")

    configure_yfinance(paths.data_dir())

    # ---- pass 1: cold-ish fetch over the whole union, batched ----
    fetcher = DataFetcher(paths.db_path())
    t0 = time.perf_counter()
    data = fetcher.get_prices(universe)
    cold_s = time.perf_counter() - t0
    cold_calls = fetcher.network_fetches
    missing = sorted(set(universe) - set(data))
    print(
        f"\npass 1: {len(data)}/{len(universe)} tickers in {cold_s:.1f}s "
        f"using {cold_calls} network call(s)"
    )
    if missing:
        print(f"  unavailable ({len(missing)}): {', '.join(missing)}")
    fetcher.close()

    assert len(missing) <= MAX_FAILURES, f"too many unavailable tickers: {missing}"
    for ticker, stock in data.items():
        assert not stock.prices.empty, f"{ticker}: empty frame"
        assert stock.prices.index.is_monotonic_increasing, f"{ticker}: unsorted"

    # ---- pass 2: same day, must be served from cache ----
    f2 = DataFetcher(paths.db_path())
    t0 = time.perf_counter()
    again = f2.get_prices(universe)
    warm_s = time.perf_counter() - t0
    warm_calls = f2.network_fetches
    print(
        f"pass 2: {len(again)} tickers in {warm_s:.1f}s "
        f"using {warm_calls} network call(s)"
    )
    f2.close()

    assert warm_calls == 0, (
        f"a same-day re-read made {warm_calls} network call(s); the freshness "
        f"check is not holding"
    )
    assert len(again) >= len(data), "cached pass returned fewer tickers than the fetch"
    print(f"caching OK: second pass was {cold_s / max(warm_s, 1e-6):.0f}x faster, 0 calls")

    print("\nPHASE 2 UNIVERSE CHECKS PASSED")


if __name__ == "__main__":
    main()
