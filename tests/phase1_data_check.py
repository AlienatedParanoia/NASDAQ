"""Phase 1 sanity check: universe integrity, 4 real tickers + 1 bogus,
fundamentals unit conversion, and cache round-trip with zero network calls.

Run from the project root: python -m tests.phase1_data_check
"""
from __future__ import annotations

import logging
import time

import paths
from data.fetcher import DataFetcher, FetchError, configure_yfinance
from data.universe import NASDAQ_100

TICKERS = ["AAPL", "MSFT", "NVDA", "KHC"]
BOGUS = "FAKETICKERXYZ"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    assert len(NASDAQ_100) == 101, f"universe should have 101 symbols, got {len(NASDAQ_100)}"
    assert len(set(NASDAQ_100)) == 101, "universe contains duplicates"
    assert all(t == t.upper() and t.isalnum() for t in NASDAQ_100), "non-plain symbol"
    print(f"universe OK: {len(NASDAQ_100)} unique symbols")

    configure_yfinance(paths.data_dir())
    fetcher = DataFetcher(paths.db_path())

    for t in TICKERS:
        sd = fetcher.get_stock(t)
        n = len(sd.prices)
        assert 2400 <= n <= 2600, f"{t}: unexpected row count {n} (expected ~2515 for 10y)"
        assert (sd.prices["close"] > 0).all(), f"{t}: non-positive closes"
        assert sd.prices.index.is_monotonic_increasing, f"{t}: index not ascending"
        assert sd.prices.index.tz is None, f"{t}: index still tz-aware"
        assert not sd.stale, f"{t}: unexpectedly served stale data"
        f = sd.fundamentals
        assert f is not None, f"{t}: fundamentals missing"
        print(
            f"{t}: {n} bars, close[-1]={sd.prices['close'].iloc[-1]:.2f}, "
            f"pe={f.pe}, peg={f.peg}, rev_g={f.revenue_growth}, "
            f"earn_g={f.earnings_growth}, d/e={f.debt_to_equity}, "
            f"margins={f.profit_margins}, beta={f.beta}, sector={f.sector}"
        )

    aapl = fetcher.get_stock("AAPL").fundamentals
    assert aapl is not None and aapl.debt_to_equity is not None
    assert 0.5 < aapl.debt_to_equity < 5.0, (
        f"AAPL debt/equity {aapl.debt_to_equity} looks unconverted (missing /100?)"
    )
    print(f"AAPL debt/equity {aapl.debt_to_equity:.2f} -> unit conversion OK")

    try:
        fetcher.get_stock(BOGUS)
        raise AssertionError("bogus ticker should have raised FetchError")
    except FetchError as e:
        print(f"bogus ticker correctly skipped: {e}")
    fetcher.close()

    f2 = DataFetcher(paths.db_path())
    t0 = time.perf_counter()
    for t in TICKERS:
        f2.get_stock(t)
    dt = time.perf_counter() - t0
    assert f2.network_fetches == 0, (
        f"expected 0 network fetches on cached run, got {f2.network_fetches}"
    )
    print(f"cached re-read of {len(TICKERS)} tickers in {dt:.2f}s, 0 network calls")
    f2.close()

    print("PHASE 1 CHECKS PASSED")


if __name__ == "__main__":
    main()
