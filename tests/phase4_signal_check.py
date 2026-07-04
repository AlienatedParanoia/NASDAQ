"""Phase 4 sanity check: headless scan of the full universe (cache-warm),
signal distribution, and reasoning-string coherence.

Run from the project root: python -m tests.phase4_signal_check
"""
from __future__ import annotations

import logging
from collections import Counter

import paths
from data.fetcher import DataFetcher, configure_yfinance
from data.universe import NASDAQ_100
from signals.signal_generator import scan_universe


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    configure_yfinance(paths.data_dir())
    fetcher = DataFetcher(paths.db_path())
    rows = scan_universe(
        NASDAQ_100,
        fetcher,
        progress=lambda i, n, t: print(f"[{i:3d}/{n}] {t}", flush=True)
        if i % 20 == 0 else None,
    )
    fetcher.close()

    assert len(rows) == 101, f"expected 101 rows, got {len(rows)}"
    errors = [r for r in rows if r.error]
    assert len(errors) <= 6, f"too many error rows: {[r.ticker for r in errors]}"

    classified = [r for r in rows if r.signal is not None]
    assert len(classified) >= 30, f"suspiciously few classified rows: {len(classified)}"
    for r in classified:
        assert r.reasoning and str(r.bucket).upper().split("-")[0] in r.reasoning.upper(), (
            f"{r.ticker}: reasoning missing bucket name: {r.reasoning[:80]}"
        )
        assert "Sharpe" in r.reasoning, f"{r.ticker}: reasoning missing backtest stats"
        assert "Exit cue:" in r.reasoning, f"{r.ticker}: reasoning missing exit cue"
        assert r.total_return is not None and r.sharpe is not None and r.max_dd is not None
        assert r.cagr is not None, f"{r.ticker}: missing CAGR"

    dist = Counter(str(r.signal) for r in classified)
    print(f"\nsignal distribution over {len(classified)} classified: {dict(dist)}")
    assert max(dist.values()) <= 0.95 * len(classified), f"degenerate distribution: {dist}"

    bucket_dist = Counter(str(r.bucket) for r in rows)
    print(f"bucket distribution: {dict(bucket_dist)}")
    print(f"errors: {[r.ticker for r in errors]}")

    print("\nsample reasoning strings:")
    shown = set()
    for r in classified:
        b = str(r.bucket)
        if b not in shown:
            shown.add(b)
            print(f"\n  {r.ticker}: {r.reasoning}")
        if len(shown) == 3:
            break

    # window slicing spot check: same cached data, different analysis windows
    import re

    from data.fetcher import DataFetcher as DF
    from signals.signal_generator import generate_signal

    f2 = DF(paths.db_path())
    data = f2.get_stock("AAPL")
    f2.close()
    r10 = generate_signal(data, window_years=10)
    r2 = generate_signal(data, window_years=2)
    assert r2.prices is not None and r10.prices is not None
    assert len(r2.prices) < len(r10.prices), "2y window should hold fewer bars than 10y"
    m = re.search(r"Backtest (\d+\.\d)y", r2.reasoning)
    assert m and 1.5 <= float(m.group(1)) <= 2.1, f"2y window span looks wrong: {r2.reasoning[:120]}"
    assert r2.bucket == r10.bucket, "bucket must not depend on the window"
    print(f"\nwindow slicing OK: 10y={len(r10.prices)} bars, 2y={len(r2.prices)} bars, "
          f"2y span reported {m.group(1)}y")

    print("\nPHASE 4 CHECKS PASSED")


if __name__ == "__main__":
    main()
