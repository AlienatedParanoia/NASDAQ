"""Phase 1 sanity check (network): the data layer end-to-end against Yahoo.

The detailed, offline coverage lives in `tests/test_fetcher.py` (pytest). This
script is the hand-runnable version that exercises the real pipeline against the
real app cache: universe integrity, a batched fetch, as-traded prices, corporate
actions, and a zero-network cached re-read.

Run from the project root: python -m tests.phase1_data_check
"""
from __future__ import annotations

import logging
import time

import pandas as pd

import paths
from data.fetcher import DIVIDEND, SPLIT, DataFetcher, FetchError, configure_yfinance
from data.symbols import normalize
from data.universe import NASDAQ_100, get_universe

TICKERS = ["AAPL", "MSFT", "NVDA", "KHC"]
BOGUS = "FAKETICKERXYZ"

NVDA_SPLIT_DATE = pd.Timestamp("2024-06-10")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    universe = get_universe()
    assert len(NASDAQ_100) == 101, f"universe should have 101 symbols, got {len(NASDAQ_100)}"
    assert len(universe) == 101, "normalisation changed the universe size"
    assert all(t == normalize(t) for t in universe), "universe is not normalised"
    print(f"universe OK: {len(universe)} unique normalised symbols")

    configure_yfinance(paths.data_dir())
    fetcher = DataFetcher(paths.db_path())

    # one batched request for the whole set, not one per ticker
    t0 = time.perf_counter()
    data = fetcher.get_prices(TICKERS)
    print(
        f"batched fetch of {len(TICKERS)} tickers in {time.perf_counter() - t0:.1f}s "
        f"using {fetcher.network_fetches} network call(s)"
    )
    assert set(data) == set(TICKERS), f"missing: {set(TICKERS) - set(data)}"

    for t in TICKERS:
        px = data[t].prices
        n = len(px)
        assert 2400 <= n <= 2600, f"{t}: unexpected row count {n} (expected ~2515 for 10y)"
        assert (px["close"] > 0).all(), f"{t}: non-positive closes"
        assert px.index.is_monotonic_increasing, f"{t}: index not ascending"
        assert px.index.tz is None, f"{t}: index still tz-aware"
        assert not px.index.has_duplicates, f"{t}: duplicate dates"
        actions = fetcher.actions(t)
        splits = [a for a in actions if a.type == SPLIT]
        divs = [a for a in actions if a.type == DIVIDEND]
        print(
            f"{t}: {n} bars, close[-1]=${float(px['close'].iloc[-1]):,.2f} "
            f"on {px.index[-1].date()}, {len(splits)} split(s), {len(divs)} dividend(s)"
        )

    # as-traded, not split-adjusted: NVDA traded near $1,200 before its 10:1
    nvda = data["NVDA"].prices["close"]
    before = float(nvda.loc[nvda.index < NVDA_SPLIT_DATE].iloc[-1])
    after = float(nvda.loc[nvda.index >= NVDA_SPLIT_DATE].iloc[0])
    assert 900.0 < before < 1500.0, (
        f"NVDA close before the split is ${before:,.2f} — still split-adjusted?"
    )
    print(f"as-traded prices OK: NVDA ${before:,.2f} -> ${after:,.2f} across its 10:1")

    nvda_splits = [a for a in fetcher.actions("NVDA") if a.type == SPLIT]
    assert any(
        a.ex_date == NVDA_SPLIT_DATE and abs(a.value - 10.0) < 1e-9 for a in nvda_splits
    ), f"NVDA 10:1 on 2024-06-10 not stored; got {[(a.ex_date, a.value) for a in nvda_splits]}"
    print("corporate actions OK: NVDA 10:1 stored with ratio 10 on its ex-date")

    # the adjusted helper removes that discontinuity for indicator use
    adj = fetcher.adjusted_closes("NVDA")
    jump = abs(
        float(adj.loc[adj.index >= NVDA_SPLIT_DATE].iloc[0])
        / float(adj.loc[adj.index < NVDA_SPLIT_DATE].iloc[-1])
        - 1.0
    )
    assert jump < 0.25, f"adjusted series still jumps {jump:.1%} across the split"
    print(f"adjusted helper OK: split discontinuity reduced to {jump:.2%}")

    try:
        fetcher.get_stock(BOGUS)
        raise AssertionError("bogus ticker should have raised FetchError")
    except FetchError as e:
        print(f"bogus ticker correctly skipped: {e}")
    fetcher.close()

    f2 = DataFetcher(paths.db_path())
    t0 = time.perf_counter()
    f2.get_prices(TICKERS)
    dt = time.perf_counter() - t0
    assert f2.network_fetches == 0, (
        f"expected 0 network fetches on cached run, got {f2.network_fetches}"
    )
    print(f"cached re-read of {len(TICKERS)} tickers in {dt:.2f}s, 0 network calls")
    f2.close()

    print("PHASE 1 CHECKS PASSED")


if __name__ == "__main__":
    main()
