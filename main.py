"""nasdaq-advisor entry point."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

import paths
from data.fetcher import configure_yfinance


def _selftest() -> int:
    """Headless packaging smoke test (--selftest): fetch one ticker end-to-end
    through the real pipeline. Exercises curl_cffi TLS + certifi inside a
    frozen build, where import-time success proves nothing. Result goes to the
    log file (a --windowed exe has no stdout) and the exit code."""
    import logging

    from data.fetcher import DataFetcher
    from signals.signal_generator import generate_signal

    log = logging.getLogger("selftest")
    try:
        fetcher = DataFetcher(paths.db_path())
        row = generate_signal(fetcher.get_stock("AAPL", force=True))
        fetcher.close()
        log.info(
            "selftest OK: %s bucket=%s signal=%s bars=%d",
            row.ticker, row.bucket, row.signal, len(row.prices),
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
