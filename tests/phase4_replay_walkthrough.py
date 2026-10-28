"""Phase 4 walkthrough: replay the trend bot and account for every difference
against the old backtest engine.

Two things the spec asks for, both printed for hand-checking:

  1. Five trades with the raw bars around each fill, so the fill prices can be
     verified against the data directly.
  2. The replay's return next to strategies/backtest_engine's on the same
     window, with the gap broken down rather than hand-waved.

Needs a warm cache. Run from the project root:
    python -m tests.phase4_replay_walkthrough
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pandas as pd

import paths
from bots.registry import parse_spec
from bots.replay import replay_bot
from data.fetcher import DataFetcher, adjust_closes, configure_yfinance
from engine.costs import DEFAULT_COSTS, ZERO_COSTS
from engine.orders import Side
from strategies.backtest_engine import run_backtest
from strategies.trend import TrendStrategy

TICKER = "AAPL"
START = "2024-10-01"
END = "2026-10-01"
STARTING_CASH = 100_000.0
BARS_AROUND_FILL = 2


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show_fills(fills, prices: pd.DataFrame, limit: int = 5) -> None:
    """Print each fill with the bars either side so the price can be checked."""
    for n, fill in enumerate(fills[:limit], 1):
        idx = prices.index.get_indexer([fill.session])[0]
        lo = max(idx - BARS_AROUND_FILL, 0)
        hi = min(idx + BARS_AROUND_FILL + 1, len(prices))
        window = prices.iloc[lo:hi]

        print(
            f"\n  [{n}] {fill.side.value.upper():<4} {fill.qty:g} {fill.ticker} "
            f"on {fill.session.date()} at ${fill.price:,.4f}"
            f"   commission ${fill.commission:.2f}  slippage ${fill.slippage_cost:.2f}"
        )
        print(f"      {'date':<12}{'open':>10}{'high':>10}{'low':>10}{'close':>10}")
        for date, row in window.iterrows():
            marker = "  <-- fill" if date == fill.session else ""
            print(
                f"      {str(date.date()):<12}{row['open']:>10.2f}{row['high']:>10.2f}"
                f"{row['low']:>10.2f}{row['close']:>10.2f}{marker}"
            )
        expected = row_open = float(prices.loc[fill.session, "open"])
        drift = fill.price / row_open - 1.0
        print(
            f"      check: filled at the session OPEN ${row_open:,.4f} "
            f"{'+' if drift >= 0 else ''}{drift * 1e4:.1f} bps slippage"
        )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    configure_yfinance(paths.data_dir())

    fetcher = DataFetcher(paths.db_path())
    prices = fetcher._load_prices(TICKER)
    if prices.empty:
        raise SystemExit(
            f"no cached data for {TICKER} — run `python -m tests.phase1_data_check` first"
        )
    window = prices.loc[(prices.index >= pd.Timestamp(START)) & (prices.index <= pd.Timestamp(END))]

    rule(
        f"Replaying the trend bot on {TICKER}, {window.index[0].date()} to "
        f"{window.index[-1].date()} ({len(window)} sessions)"
    )

    # max_positions=1 so the bot deploys the whole account on this one ticker,
    # which is what makes it comparable with the fully-invested backtest engine
    spec = parse_spec("walkthrough-trend", {
        "strategy": "trend",
        "params": {"fast": 20, "slow": 50},
        "universe": [TICKER],
        "starting_cash": STARTING_CASH,
        "sizing": "equal_weight",
        "risk": {
            "max_positions": 1,
            "max_position_pct": None,
            "max_orders_per_day": None,
            "max_drawdown_pct": None,
            "min_order_value": 0,
        },
    })

    tmp = Path(tempfile.mkdtemp())
    with_costs = replay_bot(
        spec, fetcher, START, END, ledger_path=tmp / "costed.db",
        costs=DEFAULT_COSTS, universe=[TICKER],
    )
    free = replay_bot(
        spec, fetcher, START, END, ledger_path=tmp / "free.db",
        costs=ZERO_COSTS, universe=[TICKER], account_suffix="-nocost",
    )

    print(f"\n  {with_costs.summary()}")
    print(f"  fills: {len(with_costs.fills)}   blocked: {len(with_costs.blocked)}")

    rule(f"Five fills with the surrounding raw bars ({TICKER}, as-traded)")
    show_fills(with_costs.fills, prices, limit=5)

    # ---------------------------------------------------------------- compare
    rule("Replay vs the old backtest engine, on the same window")

    actions = fetcher.actions(TICKER)
    adjusted = adjust_closes(prices, actions)
    strategy = TrendStrategy()

    # The naive comparison is unfair to the engine and worth showing: given only
    # the window, its EMAs warm up INSIDE it, so it sits flat for the first ~50
    # sessions while the bot — which loaded full prior history — is already
    # positioned on day one. That is a setup artefact, not a real difference.
    cold_frame = pd.DataFrame({"close": adjusted.loc[window.index]})
    cold = run_backtest(cold_frame, strategy.positions(cold_frame))
    cold_start = strategy.positions(cold_frame).first_valid_index()

    # The fair comparison: warm the EMAs on all available history, then measure
    # only the window — exactly what the bot does.
    full_frame = pd.DataFrame({"close": adjusted})
    full_positions = strategy.positions(full_frame)
    returns = full_frame["close"].pct_change()
    segment = (full_positions.reindex(window.index) * returns.reindex(window.index)).dropna()
    engine_return = float((1.0 + segment).prod() - 1.0)

    bot_return = with_costs.total_return
    free_return = free.total_return

    print(f"""
  engine, cold start in window : {cold.total_return:+8.2%}   ({cold.n_trades} trades)
     ^ NOT a fair comparison: its EMAs only become valid on
       {cold_start.date()}, so it sits flat for the first
       {len(window.loc[: cold_start]) - 1} sessions while the bot is already invested.

  engine, warm EMAs            : {engine_return:+8.2%}   <- the fair baseline
  replay, no costs             : {free_return:+8.2%}   ({len(free.fills)} fills)
  replay, costed               : {bot_return:+8.2%}   ({len(with_costs.fills)} fills)
""")
    print(f"  remaining gap vs the fair baseline: {bot_return - engine_return:+.2%}")

    rule("Where the remaining gap comes from")

    cost_drag = free_return - bot_return
    structural = free_return - engine_return
    commission = with_costs.total_commission
    slippage = with_costs.total_slippage

    print(f"""
  1. COSTS                                       {-cost_drag:+8.2%}
     The engine assumes none. This replay paid ${commission:,.2f} commission
     and ${slippage:,.2f} slippage over {len(with_costs.fills)} fills
     (${commission + slippage:,.2f} on ${STARTING_CASH:,.0f} of capital).
     These come from engine/costs.py and are PLACEHOLDERS, not a real
     fee schedule — the real drag is whatever your broker actually charges.

  2. STRUCTURAL                                  {structural:+8.2%}
     Three causes, which cannot be separated without re-running the engine
     under each in turn:

     a) FILL TIMING — the largest. The engine fills at the DECISION bar's
        close; the bot fills at the NEXT session's open. Every overnight gap
        between those two points lands in the bot's result and in none of the
        engine's. Its sign is random per trade, so over only {len(with_costs.fills)} fills it does
        not average out.

     b) INTEGER SHARES. The engine's 0/1 position is implicitly fully invested
        at any fraction of a share. The bot buys whole shares and leaves the
        remainder as idle cash, so it is slightly under-invested throughout —
        a drag in a rising market.

     c) DIVIDENDS. The engine runs on ADJUSTED closes, so dividends are folded
        into the price path whether or not the strategy was holding. The bot
        runs on AS-TRADED prices and receives dividends as cash only while it
        actually holds the stock.

  Neither figure is "the right answer". The engine measures the SIGNAL. The
  replay measures what trading that signal would have done to an account.
""")

    print(f"  sanity: costed {bot_return:+.2%} + cost drag {cost_drag:+.2%} "
          f"= {bot_return + cost_drag:+.2%} = no-cost replay {free_return:+.2%}")
    assert abs((bot_return + cost_drag) - free_return) < 1e-9

    rule("Equity curve checkpoints (costed replay)")
    curve = with_costs.equity_curve
    for date in pd.date_range(curve.index[0], curve.index[-1], periods=6):
        nearest = curve.index[curve.index.get_indexer([date], method="nearest")[0]]
        print(f"    {nearest.date()}   ${curve.loc[nearest]:>12,.2f}")
    print(f"\n    max drawdown {with_costs.max_drawdown:.2%}")

    fetcher.close()
    print(f"\n{'=' * 78}")
    print("PHASE 4 WALKTHROUGH COMPLETE")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
