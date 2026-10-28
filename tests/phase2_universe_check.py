"""Phase 2 sanity check, part 2 (network): fetch the full NASDAQ-100 and report
what came back.

Reduced to a data check in phase 0, when the classifier it used to exercise was
removed; phase 2 rewrites it around the S&P 500 + NASDAQ-100 union.

The first run performs the cold fetch (~5-8 min at rate-limit-safe pacing);
subsequent runs hit the SQLite cache. Run from the project root:
python -m tests.phase2_universe_check
"""
from __future__ import annotations

import logging

import paths
from data.fetcher import DataFetcher, FetchError, configure_yfinance
from data.universe import NASDAQ_100

MIN_BARS = 200  # anything thinner than ~10 months is a broken fetch, not a young listing


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    configure_yfinance(paths.data_dir())
    fetcher = DataFetcher(paths.db_path())

    total = len(NASDAQ_100)
    bars: dict[str, int] = {}
    failures: list[str] = []
    thin: list[tuple[str, int]] = []
    for i, t in enumerate(NASDAQ_100, 1):
        try:
            data = fetcher.get_stock(t)
        except FetchError as e:
            failures.append(t)
            print(f"[{i:3d}/{total}] {t}: SKIPPED ({e})", flush=True)
            continue
        prices = data.prices
        bars[t] = len(prices)
        if len(prices) < MIN_BARS:
            thin.append((t, len(prices)))
        print(
            f"[{i:3d}/{total}] {t}: {len(prices)} bars, "
            f"last close ${float(prices['close'].iloc[-1]):,.2f} "
            f"on {prices.index[-1].date()}",
            flush=True,
        )
    fetcher.close()

    print()
    print(f"fetched {len(bars)}/{total}; failures ({len(failures)}): {failures}")
    if thin:
        print(f"short histories (recent listings, worth a look): {thin}")

    assert len(bars) + len(failures) == total
    assert len(failures) <= 6, f"too many fetch failures: {failures}"
    assert len(thin) <= 10, f"too many short histories: {thin}"

    print("PHASE 2 UNIVERSE CHECKS PASSED")


if __name__ == "__main__":
    main()
