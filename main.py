"""nasdaq-advisor entry point."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

import paths
from data.fetcher import configure_yfinance


def _selftest() -> int:
    """Headless packaging smoke test (--selftest): fetch one ticker end-to-end
    through the real pipeline. Exercises curl_cffi TLS + certifi inside a
    frozen build, where import-time success proves nothing. Also replays the
    five Real Investment P&L scenarios — pure arithmetic, no network — so a
    build cannot ship with the cash ledger broken. Result goes to the log file
    (a --windowed exe has no stdout) and the exit code.

    The trading calendar is exercised here too, not merely imported:
    exchange_calendars loads data files at first use, so a packaging mistake
    shows up as a crash the first time a session is looked up — long after
    import succeeded, and only in the frozen build."""
    import logging

    from data import calendar as market_calendar
    from data.fetcher import DataFetcher
    from data.universe import describe, get_universe
    from tests.phase11_real_pnl_check import run_scenarios

    log = logging.getLogger("selftest")
    try:
        run_scenarios()
        log.info("selftest OK: real-investment P&L scenarios pass")

        # Thanksgiving 2025 is closed and the next session is a half-day: a
        # cheap assertion that the calendar's data files really shipped
        assert not market_calendar.is_trading_day("2025-11-27")
        assert market_calendar.is_early_close("2025-11-28")
        session = market_calendar.latest_complete_session()
        log.info("selftest OK: XNYS calendar loaded, latest complete session %s", session)

        # the constituent CSVs are bundled data, not code: a packaging miss
        # leaves the exe importable and the universe empty
        universe = get_universe()
        assert len(universe) > 400, f"universe looks truncated: {len(universe)}"
        log.info("selftest OK: universe loaded, %s", describe())

        _selftest_broker(log)

        fetcher = DataFetcher(paths.db_path())
        data = fetcher.get_stock("AAPL", force=True)
        fetcher.close()
        log.info(
            "selftest OK: %s fetched, %d bars through %s",
            data.ticker, len(data.prices), data.prices.index[-1].date(),
        )
        return 0
    except Exception:
        log.exception("selftest FAILED")
        return 1


def _selftest_broker(log) -> None:
    """Round-trip one order through a SimBroker on synthetic bars.

    Entirely offline and throwaway: it proves the engine imports, the ledger
    schema applies, and an order can actually fill inside a frozen build —
    none of which import-time success demonstrates.
    """
    import tempfile
    from pathlib import Path

    import pandas as pd

    from engine.costs import CostModel
    from engine.ledger import Ledger
    from engine.orders import Order, OrderStatus, Side
    from engine.sim_broker import Bar, SimBroker

    with Ledger(Path(tempfile.mkdtemp()) / "selftest.db") as ledger:
        account = ledger.create_account("selftest", 10_000.0)
        broker = SimBroker(ledger, account.id, costs=CostModel(0.0, 0.0, 1.0, 5.0))
        broker.set_mark("TEST", 100.0)
        d1 = pd.Timestamp("2026-01-05")
        d2 = pd.Timestamp("2026-01-06")

        order = broker.submit(
            Order(account_id=account.id, created_session=d1, ticker="TEST",
                  side=Side.BUY, qty=10, reason="selftest")
        )
        assert order.status is OrderStatus.PENDING, order.reject_reason
        result = broker.process_session(
            d2, {"TEST": Bar(100.0, 101.0, 99.0, 100.0)}
        )
        assert len(result.fills) == 1, "selftest order did not fill"
        assert broker.cash() >= 0
        expected = result.cash + 10 * 100.0
        assert abs(result.equity - expected) < 0.01, (
            f"equity identity broken: {result.equity} != {expected}"
        )
        log.info(
            "selftest OK: broker filled %d share(s) at %.4f, cash %.2f, equity %.2f",
            result.fills[0].qty, result.fills[0].price, result.cash, result.equity,
        )


def main() -> None:
    paths.setup_logging()
    configure_yfinance(paths.data_dir())
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    app = QApplication(sys.argv)
    app.setApplicationName("NASDAQ Advisor")

    from ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
