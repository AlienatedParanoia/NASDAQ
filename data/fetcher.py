"""yfinance wrapper with a SQLite cache.

Prices: up to ten years of daily OHLCV per ticker (auto-adjusted close, so
dividends are folded in); younger listings get whatever history exists. Fundamentals: the valuation/growth/balance-sheet snapshot the
classifier and long-term scorer need. Both are cached in SQLite and refreshed
only when the cached copy is more than a day old.

Yahoo specifics this module encodes (verified against yfinance 1.5.1):
  - requests must be paced (~1/s) and not parallelized, or Yahoo rate-limits;
  - ``debtToEquity`` arrives as a percentage (150.0 means 1.5x);
  - PEG lives under ``trailingPegRatio`` (``pegRatio`` was dropped by Yahoo in
    mid-2025 and is simply absent);
  - a transient failure can return a *partial* info dict, so an all-None
    payload is treated as a failed fetch rather than as data.
"""
from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

log = logging.getLogger(__name__)

PRICE_COLUMNS = ("open", "high", "low", "close", "volume")
PRICE_PERIOD = "10y"
MAX_AGE = timedelta(days=1)
RATE_LIMIT_BACKOFF_S = 60.0
RATE_LIMIT_COOLDOWN_S = 300.0  # after a failed retry, fail fast this long
CACHE_VERSION = 2  # bump when the schema or fetch span changes; wipes the cache

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker  TEXT NOT NULL,
    date    TEXT NOT NULL,
    open REAL, high REAL, low REAL,
    close REAL,
    volume INTEGER,
    PRIMARY KEY (ticker, date)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS fundamentals (
    ticker           TEXT PRIMARY KEY,
    pe               REAL,
    peg              REAL,
    revenue_growth   REAL,
    earnings_growth  REAL,
    debt_to_equity   REAL,
    profit_margins   REAL,
    beta             REAL,
    sector           TEXT
);

CREATE TABLE IF NOT EXISTS fetch_meta (
    ticker     TEXT NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN ('prices','fundamentals')),
    fetched_at TEXT NOT NULL,
    status     TEXT NOT NULL,
    error      TEXT,
    PRIMARY KEY (ticker, kind)
);
"""


class FetchError(Exception):
    """No usable price data for a ticker (fresh fetch failed and cache is empty)."""


@dataclass(frozen=True, slots=True)
class Fundamentals:
    ticker: str
    pe: float | None
    peg: float | None
    revenue_growth: float | None
    earnings_growth: float | None
    debt_to_equity: float | None  # ratio (Yahoo's percentage already / 100)
    profit_margins: float | None
    beta: float | None
    sector: str | None


@dataclass(slots=True)
class StockData:
    ticker: str
    prices: pd.DataFrame  # DatetimeIndex ascending; open/high/low/close/volume
    fundamentals: Fundamentals | None
    stale: bool = False  # served from an out-of-date cache after a failed refetch


def _num(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def configure_yfinance(data_dir: Path) -> None:
    """Point yfinance's tz cache at a writable dir and enable its retries."""
    try:
        yf.set_tz_cache_location(str(data_dir / "yf_tz"))
    except Exception:
        log.debug("could not set yfinance tz cache location", exc_info=True)
    try:
        yf.config.network.retries = 2
    except Exception:
        log.debug("yfinance config API unavailable", exc_info=True)


class DataFetcher:
    """Cached access to prices and fundamentals.

    Not thread-safe: sqlite3 connections are thread-affine, so create one
    DataFetcher per thread (the UI scan worker builds its own).
    """

    # shared across ALL fetcher instances (scanner + portfolio workers can run
    # concurrently): one Yahoo request at a time, one pacing clock, and one
    # rate-limit circuit breaker for the whole process
    _net_lock = threading.Lock()
    _shared_last_net = 0.0
    _shared_cooldown_until = 0.0

    def __init__(
        self,
        db_path: Path,
        sleep_between: float = 1.0,
        should_stop: Callable[[], bool] | None = None,
    ):
        self.sleep_between = sleep_between
        self.should_stop = should_stop  # lets the UI cancel mid-backoff
        self.network_fetches = 0  # exposed so tests can prove cache hits
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version != CACHE_VERSION:
            log.info("cache version %s -> %s: resetting cache", version, CACHE_VERSION)
            self._conn.executescript(
                "DROP TABLE IF EXISTS prices;"
                "DROP TABLE IF EXISTS fundamentals;"
                "DROP TABLE IF EXISTS fetch_meta;"
            )
            self._conn.execute(f"PRAGMA user_version = {CACHE_VERSION}")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ cache

    def _is_fresh(self, ticker: str, kind: str) -> bool:
        row = self._conn.execute(
            "SELECT fetched_at, status FROM fetch_meta WHERE ticker=? AND kind=?",
            (ticker, kind),
        ).fetchone()
        if row is None or row[1] != "ok":
            return False
        return datetime.now(timezone.utc) - datetime.fromisoformat(row[0]) < MAX_AGE

    def _mark(self, ticker: str, kind: str, status: str, error: str | None) -> None:
        self._conn.execute(
            "INSERT INTO fetch_meta (ticker, kind, fetched_at, status, error)"
            " VALUES (?,?,?,?,?)"
            " ON CONFLICT(ticker, kind) DO UPDATE SET"
            " fetched_at=excluded.fetched_at, status=excluded.status, error=excluded.error",
            (ticker, kind, datetime.now(timezone.utc).isoformat(), status, error),
        )

    def _load_prices(self, ticker: str) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM prices"
            " WHERE ticker=? ORDER BY date",
            self._conn,
            params=(ticker,),
            index_col="date",
            parse_dates=["date"],
        )
        df.index.name = "date"
        return df

    def _store_prices(self, ticker: str, df: pd.DataFrame) -> None:
        rows = [
            (
                ticker,
                idx.strftime("%Y-%m-%d"),
                _num(r.open),
                _num(r.high),
                _num(r.low),
                _num(r.close),
                int(r.volume) if not math.isnan(r.volume) else 0,
            )
            for idx, r in zip(df.index, df.itertuples())
        ]
        with self._conn:
            self._conn.execute("DELETE FROM prices WHERE ticker=?", (ticker,))
            self._conn.executemany(
                "INSERT INTO prices (ticker, date, open, high, low, close, volume)"
                " VALUES (?,?,?,?,?,?,?)",
                rows,
            )
            self._mark(ticker, "prices", "ok", None)

    def _load_fundamentals(self, ticker: str) -> Fundamentals | None:
        row = self._conn.execute(
            "SELECT pe, peg, revenue_growth, earnings_growth, debt_to_equity,"
            " profit_margins, beta, sector FROM fundamentals WHERE ticker=?",
            (ticker,),
        ).fetchone()
        if row is None:
            return None
        return Fundamentals(ticker, *row)

    def _store_fundamentals(self, f: Fundamentals) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO fundamentals (ticker, pe, peg, revenue_growth,"
                " earnings_growth, debt_to_equity, profit_margins, beta, sector)"
                " VALUES (?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(ticker) DO UPDATE SET"
                " pe=excluded.pe, peg=excluded.peg,"
                " revenue_growth=excluded.revenue_growth,"
                " earnings_growth=excluded.earnings_growth,"
                " debt_to_equity=excluded.debt_to_equity,"
                " profit_margins=excluded.profit_margins,"
                " beta=excluded.beta, sector=excluded.sector",
                (
                    f.ticker, f.pe, f.peg, f.revenue_growth, f.earnings_growth,
                    f.debt_to_equity, f.profit_margins, f.beta, f.sector,
                ),
            )
            self._mark(f.ticker, "fundamentals", "ok", None)

    # ---------------------------------------------------------------- network

    def _cancelled(self) -> bool:
        return self.should_stop is not None and self.should_stop()

    def _backoff(self, seconds: float) -> None:
        """Sleep in small slices so a cancel request interrupts the wait —
        an uninterruptible 60s sleep would freeze shutdown and, on Windows,
        end in Qt fatally destroying a still-running thread."""
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if self._cancelled():
                raise FetchError("cancelled during rate-limit backoff")
            time.sleep(min(1.0, remaining))

    def _attempt(self, fn):
        """One paced network attempt. The lock serializes requests across all
        fetcher instances so combined traffic never exceeds ~1 req/s."""
        with DataFetcher._net_lock:
            wait = self.sleep_between - (time.monotonic() - DataFetcher._shared_last_net)
            if wait > 0:
                time.sleep(wait)
            self.network_fetches += 1
            try:
                return fn()
            finally:
                DataFetcher._shared_last_net = time.monotonic()

    def _network_call(self, fn):
        remaining = DataFetcher._shared_cooldown_until - time.monotonic()
        if remaining > 0:
            # circuit breaker: during a sustained 429 episode, fail fast to
            # cached/stale data instead of paying ~60s per ticker for hours
            raise FetchError(
                f"Yahoo rate-limit cooldown active for another {remaining:.0f}s"
            )
        if self._cancelled():
            raise FetchError("cancelled")
        try:
            return self._attempt(fn)
        except YFRateLimitError:
            log.warning("rate limited by Yahoo; backing off %.0fs", RATE_LIMIT_BACKOFF_S)
            self._backoff(RATE_LIMIT_BACKOFF_S)
            try:
                return self._attempt(fn)
            except YFRateLimitError:
                DataFetcher._shared_cooldown_until = (
                    time.monotonic() + RATE_LIMIT_COOLDOWN_S
                )
                log.warning(
                    "still rate limited; cooling down %.0fs — subsequent tickers "
                    "fall back to cache", RATE_LIMIT_COOLDOWN_S,
                )
                raise

    def _fetch_prices(self, ticker: str) -> pd.DataFrame:
        df = self._network_call(
            lambda: yf.Ticker(ticker).history(
                period=PRICE_PERIOD, interval="1d", auto_adjust=True
            )
        )
        if df is None or df.empty:
            raise FetchError(f"{ticker}: no price data returned (delisted or invalid)")
        df = df.rename(columns=str.lower)
        missing = set(PRICE_COLUMNS) - set(df.columns)
        if missing:
            raise FetchError(f"{ticker}: price data missing columns {sorted(missing)}")
        df = df[list(PRICE_COLUMNS)].copy()
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.dropna(subset=["close"])
        df["volume"] = df["volume"].fillna(0)
        return df

    _NUMERIC_FIELDS = (
        "pe", "peg", "revenue_growth", "earnings_growth",
        "debt_to_equity", "profit_margins", "beta",
    )

    def _fetch_fundamentals(self, ticker: str) -> Fundamentals:
        info = self._network_call(lambda: yf.Ticker(ticker).get_info()) or {}
        dte = _num(info.get("debtToEquity"))
        earnings_growth = _num(info.get("earningsGrowth"))
        if earnings_growth is None:
            earnings_growth = _num(info.get("earningsQuarterlyGrowth"))
        sector = info.get("sector")
        f = Fundamentals(
            ticker=ticker,
            pe=_num(info.get("trailingPE")),
            peg=_num(info.get("trailingPegRatio")),
            revenue_growth=_num(info.get("revenueGrowth")),
            earnings_growth=earnings_growth,
            debt_to_equity=dte / 100.0 if dte is not None else None,
            profit_margins=_num(info.get("profitMargins")),
            beta=_num(info.get("beta")),
            sector=sector if isinstance(sector, str) else None,
        )
        if all(getattr(f, name) is None for name in self._NUMERIC_FIELDS):
            raise FetchError(
                f"{ticker}: fundamentals payload empty (partial/failed Yahoo response)"
            )
        return f

    # ----------------------------------------------------------------- public

    def get_stock(self, ticker: str, force: bool = False) -> StockData:
        """Return cached-or-fresh data. Raises FetchError only when no usable
        price data exists at all; a failed fundamentals fetch degrades to the
        cached row (or None)."""
        stale = False
        price_error: str | None = None

        if force or not self._is_fresh(ticker, "prices"):
            try:
                self._store_prices(ticker, self._fetch_prices(ticker))
            except Exception as e:
                log.warning("%s: price fetch failed: %s", ticker, e)
                price_error = str(e)
                with self._conn:
                    self._mark(ticker, "prices", "failed", price_error)
                stale = True
        prices = self._load_prices(ticker)
        if prices.empty:
            # include the cause so callers can tell "bad symbol" from
            # transient failures (cooldown/cancelled/timeouts)
            raise FetchError(
                f"{ticker}: no usable price data "
                f"({price_error or 'fetch failed, cache empty'})"
            )

        if force or not self._is_fresh(ticker, "fundamentals"):
            try:
                self._store_fundamentals(self._fetch_fundamentals(ticker))
            except Exception as e:
                log.warning("%s: fundamentals fetch failed: %s", ticker, e)
                with self._conn:
                    self._mark(ticker, "fundamentals", "failed", str(e))
                stale = True
        fundamentals = self._load_fundamentals(ticker)

        return StockData(ticker=ticker, prices=prices, fundamentals=fundamentals, stale=stale)
