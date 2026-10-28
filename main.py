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
