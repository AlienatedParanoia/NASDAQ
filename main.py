"""nasdaq-advisor entry point."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

import paths
from data.fetcher import configure_yfinance


def main() -> None:
    paths.setup_logging()
    configure_yfinance(paths.data_dir())
    app = QApplication(sys.argv)
    app.setApplicationName("NASDAQ Advisor")

    from ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
