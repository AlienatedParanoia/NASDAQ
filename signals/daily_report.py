"""Daily "top picks" report: rank the day's Buy-rated stocks and render a
dated, human-readable digest.

This is pure ranking + formatting on top of what ``scan_universe`` already
produces (a list of ``ScanRow``). No network, no UI, no ``date.today()`` — the
caller supplies the scanned rows and the as-of date so the whole module is
deterministic and unit-testable.

Ranking is risk-adjusted. Only stocks with a current **Buy** signal are
eligible (a Buy is already gated on backtest quality: Sharpe >= 0.5 and
positive return). Among that day's Buys, each of three existing ``ScanRow``
numbers is min-max normalized to [0, 1] and weighted:

  Sharpe 50%  +  CAGR 30%  +  shallower max-drawdown 20%

so a name only rises to the top by pairing return with risk control. Weights
live in named constants and are trivial to retune.
"""
from __future__ import annotations

from dataclasses import dataclass

from signals.signal_generator import SHARPE_BUG_TRIPWIRE, ScanRow, Signal

# ranking weights — must sum to 1.0; tune freely
WEIGHT_SHARPE = 0.50
WEIGHT_CAGR = 0.30
WEIGHT_DRAWDOWN = 0.20

DEFAULT_TOP_N = 5

REPORT_DISCLAIMER = (
    "Informational only. This ranks the app's own backtest-derived signals over "
    "the selected window — no transaction costs or slippage, ~15-min-delayed "
    "data, and long windows are survivorship-flattered. It is not a prediction, "
    "a guarantee, or personalized financial advice."
)


@dataclass(frozen=True, slots=True)
class Pick:
    rank: int          # 1-based
    row: ScanRow
    score: float       # composite 0-1 (higher is better)


def _is_implausible(r: ScanRow) -> bool:
    """True when the backtest Sharpe trips the pipeline's own sanity alarm.
    Such rows carry a 'verify pipeline' warning in their reasoning — usually a
    freshly listed name with a handful of trades over a short window — so it
    would be irresponsible to surface them as curated recommendations."""
    return r.sharpe is not None and r.sharpe > SHARPE_BUG_TRIPWIRE


def _is_candidate(r: ScanRow) -> bool:
    """A Buy-rated row with the numbers the score needs and a plausible
    Sharpe. A Buy always comes from a completed backtest, so the numbers are
    present; the guards keep the function total against hand-built or future
    row shapes, and drop tripwire-flagged rows from the recommendations."""
    return (
        r.signal is Signal.BUY
        and r.error is None
        and r.sharpe is not None
        and r.cagr is not None
        and r.max_dd is not None
        and not _is_implausible(r)
    )


def buy_counts(rows: list[ScanRow]) -> tuple[int, int]:
    """(total Buy-rated rows, of which excluded as implausible) — for the
    report header, so the exclusion is transparent rather than silent."""
    buys = [r for r in rows if r.signal is Signal.BUY and r.error is None]
    excluded = sum(1 for r in buys if _is_implausible(r))
    return len(buys), excluded


def _normalize(values: list[float]) -> list[float]:
    """Min-max onto [0, 1] where higher input -> higher output. A degenerate
    range (all equal, or a single candidate) maps everything to a neutral 0.5
    so no component dominates by accident."""
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 0:
        return [0.5] * len(values)
    return [(v - lo) / span for v in values]


def score_candidates(rows: list[ScanRow]) -> list[tuple[ScanRow, float]]:
    """All Buy-rated rows paired with their composite score, best first.

    max_dd is negative (0 = no drawdown), so normalizing it directly already
    rewards the shallower drawdown — no sign flip needed. Ties break on raw
    Sharpe, then ticker, so the ordering is fully deterministic."""
    candidates = [r for r in rows if _is_candidate(r)]
    if not candidates:
        return []
    ns = _normalize([r.sharpe for r in candidates])
    nc = _normalize([r.cagr for r in candidates])
    nd = _normalize([r.max_dd for r in candidates])
    scored = [
        (
            r,
            WEIGHT_SHARPE * s + WEIGHT_CAGR * c + WEIGHT_DRAWDOWN * d,
        )
        for r, s, c, d in zip(candidates, ns, nc, nd)
    ]
    scored.sort(key=lambda t: (-t[1], -t[0].sharpe, t[0].ticker))
    return scored


def daily_top_picks(rows: list[ScanRow], n: int = DEFAULT_TOP_N) -> list[Pick]:
    """The top ``n`` Buy-rated stocks for the day (fewer if fewer qualify)."""
    if n <= 0:
        return []
    scored = score_candidates(rows)
    return [Pick(i + 1, r, sc) for i, (r, sc) in enumerate(scored[:n])]


def _pct(v: float | None, spec: str = "{:+.1%}") -> str:
    return spec.format(v) if v is not None else "n/a"


def render_report(
    picks: list[Pick],
    *,
    as_of: str,
    window_years: int,
    n_scanned: int,
    n_buys: int,
    n_excluded: int = 0,
) -> str:
    """A dated plain-text digest of the picks (shown in the tab's report view
    and easy to assert on in tests). ``n_excluded`` is the count of Buy-rated
    rows dropped for an implausible Sharpe, reported for transparency."""
    header = [
        f"NASDAQ Advisor — Daily Report for {as_of}",
        f"Backtest window: {window_years}y   |   "
        f"Scanned {n_scanned} stocks   |   {n_buys} Buy-rated",
        "",
    ]
    if n_excluded:
        header.insert(
            2,
            f"({n_excluded} Buy-rated excluded for implausible backtest stats "
            f"— Sharpe over {SHARPE_BUG_TRIPWIRE:.0f}; see the Scanner tab.)",
        )
    eligible = n_buys - n_excluded
    if not picks:
        body = [
            "No Buy-rated opportunities today.",
            "",
            "Nothing clears the Buy bar with trustworthy backtest stats — every "
            "scanned stock is Hold, Avoid, unclassified, or flagged implausible. "
            "Often the right answer is to wait.",
        ]
        return "\n".join(header + body + ["", REPORT_DISCLAIMER])

    body = [f"Top {len(picks)} risk-adjusted picks (of {eligible} eligible Buys):", ""]
    for p in picks:
        r = p.row
        body.append(f"#{p.rank}  {r.ticker}  ({r.bucket})  —  BUY")
        body.append(
            f"      Ret/yr {_pct(r.cagr)}   Sharpe {r.sharpe:.2f}   "
            f"Max DD {_pct(r.max_dd)}   score {p.score:.2f}"
        )
        body.append(f"      {r.reasoning}")
        body.append("")
    return "\n".join(header + body + [REPORT_DISCLAIMER])
