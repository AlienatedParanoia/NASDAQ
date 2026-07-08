"""Phase 8 checks: the daily top-picks report (signals/daily_report.py).

The ranking/rendering tests are pure and offline. A final integration check
runs the real pipeline over a handful of cached tickers to prove the report
composes with live ScanRows. Run from the project root:
python -m tests.phase8_daily_report_check
"""
from __future__ import annotations

import paths
from classifier.classify import Bucket
from data.fetcher import DataFetcher
from signals.daily_report import (
    REPORT_DISCLAIMER,
    buy_counts,
    daily_top_picks,
    render_report,
    score_candidates,
)
from signals.signal_generator import (
    SHARPE_BUG_TRIPWIRE,
    ScanRow,
    Signal,
    generate_signal,
)


def _row(ticker, signal, cagr, sharpe, max_dd, bucket=Bucket.TREND) -> ScanRow:
    return ScanRow(
        ticker=ticker, bucket=bucket, signal=signal,
        total_return=cagr, cagr=cagr, sharpe=sharpe, max_dd=max_dd,
        bh_return=cagr, bh_cagr=cagr, reasoning=f"{ticker} reasoning",
    )


def only_buys_are_ranked() -> None:
    rows = [
        _row("AAA", Signal.BUY, 0.20, 1.2, -0.15),
        _row("HHH", Signal.HOLD, 0.90, 4.0, -0.05),   # great numbers, not a Buy
        _row("VVV", Signal.AVOID, 0.50, 2.0, -0.10),
        _row("ERR", Signal.BUY, 0.30, 1.5, -0.20),
    ]
    rows[3] = ScanRow(  # a Buy row carrying an error must be excluded
        ticker="ERR", bucket=Bucket.TREND, signal=Signal.BUY,
        total_return=None, cagr=None, sharpe=None, max_dd=None,
        bh_return=None, bh_cagr=None, reasoning="broken", error="boom",
    )
    picks = daily_top_picks(rows)
    assert [p.row.ticker for p in picks] == ["AAA"], [p.row.ticker for p in picks]
    print("only Buy-rated rows with complete numbers are ranked")


def ranking_is_risk_adjusted() -> None:
    # A: best Sharpe + shallowest drawdown, modest return
    # B: best return but worst Sharpe and deepest drawdown
    # C: middling on everything
    rows = [
        _row("A", Signal.BUY, 0.20, 2.0, -0.10),
        _row("B", Signal.BUY, 0.50, 1.0, -0.50),
        _row("C", Signal.BUY, 0.30, 1.5, -0.30),
    ]
    order = [r.ticker for r, _ in score_candidates(rows)]
    assert order == ["A", "C", "B"], order
    top = daily_top_picks(rows, 2)
    assert [p.rank for p in top] == [1, 2]
    assert [p.row.ticker for p in top] == ["A", "C"]
    # the flashy high-return-but-deep-drawdown name must NOT win
    assert order[0] == "A", "risk-adjusted ranking should not reward raw return alone"
    print(f"risk-adjusted order {order}: Sharpe+low-DD beats raw return")


def top_n_and_ties() -> None:
    rows = [_row(t, Signal.BUY, 0.2, 1.0, -0.2) for t in ("ZZZ", "AAA", "MMM")]
    # identical metrics -> neutral 0.5 each -> deterministic tie-break by ticker
    order = [r.ticker for r, _ in score_candidates(rows)]
    assert order == ["AAA", "MMM", "ZZZ"], order
    assert len(daily_top_picks(rows, 5)) == 3, "never invents picks beyond candidates"
    assert daily_top_picks(rows, 0) == []
    print("top-N cap, fewer-than-N, and deterministic ties OK")


def implausible_sharpe_is_excluded() -> None:
    # a freshly listed name with a tripwire-flagged Sharpe must never be a
    # recommendation, even though its raw numbers look spectacular
    good = _row("GOOD", Signal.BUY, 0.25, 1.3, -0.20)
    bogus = _row("BOGUS", Signal.BUY, 20.0, SHARPE_BUG_TRIPWIRE + 1.0, -0.05)
    rows = [good, bogus]
    picks = daily_top_picks(rows)
    assert [p.row.ticker for p in picks] == ["GOOD"], [p.row.ticker for p in picks]
    n_buys, n_excluded = buy_counts(rows)
    assert (n_buys, n_excluded) == (2, 1), (n_buys, n_excluded)
    text = render_report(
        picks, as_of="2026-07-07", window_years=10,
        n_scanned=2, n_buys=n_buys, n_excluded=n_excluded,
    )
    assert "1 Buy-rated excluded for implausible" in text
    assert "of 1 eligible Buys" in text
    print("implausible-Sharpe Buys are excluded from picks and noted in the report")


def render_no_buys() -> None:
    rows = [_row("AAA", Signal.HOLD, 0.2, 1.0, -0.2)]
    text = render_report(
        daily_top_picks(rows), as_of="2026-07-07", window_years=10,
        n_scanned=1, n_buys=0,
    )
    assert "No Buy-rated opportunities today." in text
    assert "2026-07-07" in text and REPORT_DISCLAIMER in text
    print("empty-day report renders a clear 'nothing qualifies' message")


def render_with_picks() -> None:
    rows = [
        _row("A", Signal.BUY, 0.20, 2.0, -0.10),
        _row("B", Signal.BUY, 0.50, 1.0, -0.50),
    ]
    picks = daily_top_picks(rows)
    text = render_report(
        picks, as_of="2026-07-07", window_years=5, n_scanned=101, n_buys=2,
    )
    assert "Daily Report for 2026-07-07" in text
    assert "window: 5y" in text and "101 stocks" in text
    assert "#1  A" in text and "#2  B" in text
    assert "A reasoning" in text  # the plain-English rationale is carried through
    assert REPORT_DISCLAIMER in text
    print("populated report is dated, ranked, and carries each pick's reasoning")


def live_pipeline_integration(fetcher: DataFetcher) -> None:
    tickers = ["AAPL", "MSFT", "NVDA", "COST", "GILD", "SBUX"]
    rows = [generate_signal(fetcher.get_stock(t), 10) for t in tickers]
    picks = daily_top_picks(rows)
    assert len(picks) <= 5
    for p in picks:
        assert p.row.signal is Signal.BUY and p.row.error is None
        assert p.row.sharpe <= SHARPE_BUG_TRIPWIRE, "implausible Sharpe leaked in"
        assert 0.0 <= p.score <= 1.0
    scores = [p.score for p in picks]
    assert scores == sorted(scores, reverse=True), "picks must be best-first"
    ranks = [p.rank for p in picks]
    assert ranks == list(range(1, len(picks) + 1))
    buys = [r.ticker for r in rows if r.signal is Signal.BUY]
    print(f"live integration OK: {len(buys)} Buys among {tickers} -> "
          f"picks {[p.row.ticker for p in picks]}")


def main() -> None:
    only_buys_are_ranked()
    ranking_is_risk_adjusted()
    top_n_and_ties()
    implausible_sharpe_is_excluded()
    render_no_buys()
    render_with_picks()
    fetcher = DataFetcher(paths.db_path())
    try:
        live_pipeline_integration(fetcher)
    finally:
        fetcher.close()
    print("PHASE 8 DAILY REPORT CHECKS PASSED")


if __name__ == "__main__":
    main()
