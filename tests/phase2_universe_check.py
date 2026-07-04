"""Phase 2 sanity check, part 2 (network): classify the full NASDAQ-100 and
print the bucket distribution.

The first run performs the cold fetch (~5-8 min at rate-limit-safe pacing);
subsequent runs hit the SQLite cache. Run from the project root:
python -m tests.phase2_universe_check
"""
from __future__ import annotations

import logging

import paths
from classifier.classify import Bucket, classify, compute_metrics
from data.fetcher import DataFetcher, FetchError, configure_yfinance
from data.universe import NASDAQ_100


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    configure_yfinance(paths.data_dir())
    fetcher = DataFetcher(paths.db_path())

    buckets: dict[Bucket, list[str]] = {b: [] for b in Bucket}
    failures: list[str] = []
    for i, t in enumerate(NASDAQ_100, 1):
        try:
            data = fetcher.get_stock(t)
        except FetchError as e:
            failures.append(t)
            print(f"[{i:3d}/101] {t}: SKIPPED ({e})", flush=True)
            continue
        bucket, reason = classify(compute_metrics(data))
        buckets[bucket].append(t)
        print(f"[{i:3d}/101] {t}: {bucket} — {reason}", flush=True)
    fetcher.close()

    print()
    counts = {str(b): len(v) for b, v in buckets.items()}
    print(f"distribution: {counts}")
    print(f"failures ({len(failures)}): {failures}")
    for b, members in buckets.items():
        print(f"{b}: {', '.join(members) if members else '(none)'}")

    assert sum(len(v) for v in buckets.values()) + len(failures) == 101
    assert len(failures) <= 6, f"too many fetch failures: {failures}"
    for b, members in buckets.items():
        assert len(members) < 90, f"bucket {b} absurdly dominant: {len(members)}/101"
    assert len(buckets[Bucket.UNCLASSIFIED]) < 60, "Unclassified holds the majority"
    for b, members in buckets.items():
        if not members:
            print(f"WARNING: bucket {b} is empty — worth a look, not necessarily a bug")

    for hot in ("PLTR", "MSTR"):
        assert hot not in buckets[Bucket.LONG_TERM], (
            f"{hot} classified Long-term — its vol/beta should have excluded it"
        )

    print("PHASE 2 UNIVERSE CHECKS PASSED")


if __name__ == "__main__":
    main()
