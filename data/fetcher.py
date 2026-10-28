"""yfinance wrapper with a SQLite cache, built so a permanent trade ledger can
sit on top of it.

The governing requirement: **a stored bar must never change**. Phase 3's ledger
records that a bot bought N shares at price P on session t, and that record has
to stay true forever. That rules out yfinance's default `auto_adjust=True`,
which rewrites every historical price after each dividend.

What this module stores, and why it is not simply "auto_adjust=False":

  yfinance's `auto_adjust=False` removes only the *dividend* adjustment. Prices
  come back still **split back-adjusted** — NVDA's 2024-06-07 bar reads $119.77
  even though it traded at ~$1,197 that day, because of the 10:1 split three
  sessions later. Storing that would mean (a) every new split silently rewrites
  history underneath the ledger, and (b) historical fills happen at prices that
  encode a future event, which is look-ahead bias wearing a disguise.

  So raw Yahoo output is **un-adjusted back to as-traded prices** on the way in
  (see `_unadjust_splits`), and corporate actions are stored as first-class rows.
  A bar, once written, is final. SimBroker then applies splits the same way in
  replay and in forward trading, because the data means the same thing in both.

  Strategies that want a continuous series call `adjusted_closes()`, which
  rebuilds the total-return series from as-traded prices + actions.

Yahoo specifics this module encodes (verified against yfinance 1.5.1):
  - requests must be paced (~1/s) and not parallelized, or Yahoo rate-limits;
  - `history(actions=True)` and `download(actions=True)` return Dividends and
    Stock Splits *inside the price frame*, indexed by ex-date, so no separate
    actions request is needed;
  - `download(group_by="ticker")` returns a 2-level column MultiIndex even for a
    single ticker, with a tz-naive index (`history` returns tz-aware).
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
from typing import Callable, Iterable, Sequence

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from data.symbols import normalize, normalize_all

log = logging.getLogger(__name__)

PRICE_COLUMNS = ("open", "high", "low", "close", "volume")
PRICE_PERIOD = "10y"
MAX_AGE = timedelta(days=1)
RATE_LIMIT_BACKOFF_S = 60.0
RATE_LIMIT_COOLDOWN_S = 300.0  # after a failed retry, fail fast this long
CACHE_VERSION = 3  # bump when the schema or fetch span changes; wipes the cache

# how many sessions of already-cached history to re-request on an incremental
# update. Yahoo revises recent bars (late prints, corrections), so the last few
# days are refetched and upserted rather than trusted.
OVERLAP_SESSIONS = 5

# tickers per yf.download call. Large enough that the universe is a handful of
# requests, small enough that one failure does not cost the whole fetch.
CHUNK_SIZE = 50

SPLIT = "split"
DIVIDEND = "dividend"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker  TEXT NOT NULL,
    date    TEXT NOT NULL,
    open REAL, high REAL, low REAL,
    close REAL,
    volume INTEGER,
    PRIMARY KEY (ticker, date)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS corporate_actions (
    ticker  TEXT NOT NULL,
    ex_date TEXT NOT NULL,
    type    TEXT NOT NULL CHECK (type IN ('split','dividend')),
    value   REAL NOT NULL,
    PRIMARY KEY (ticker, ex_date, type)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS fetch_meta (
    ticker     TEXT NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN ('prices')),
    fetched_at TEXT NOT NULL,
    status     TEXT NOT NULL,
    error      TEXT,
    PRIMARY KEY (ticker, kind)
);
"""


class FetchError(Exception):
    """No usable price data for a ticker (fresh fetch failed and cache is empty)."""


@dataclass(frozen=True, slots=True)
class CorporateAction:
    ticker: str
    ex_date: pd.Timestamp
    type: str            # 'split' | 'dividend'
    value: float         # split: ratio (10.0 for 10:1). dividend: cash per share


@dataclass(slots=True)
class StockData:
    ticker: str
    prices: pd.DataFrame  # DatetimeIndex ascending; as-traded open/high/low/close/volume
    stale: bool = False   # served from an out-of-date cache after a failed refetch


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


# ------------------------------------------------------------------ adjustments


def _later_product(index: pd.DatetimeIndex, factors: pd.Series) -> pd.Series:
    """For each bar t, the product of `factors` at every date strictly after t."""
    aligned = pd.Series(1.0, index=index)
    if factors is not None and len(factors):
        present = factors.reindex(index).fillna(1.0)
        aligned = present.where(present > 0, 1.0)
    return aligned[::-1].cumprod()[::-1].shift(-1).fillna(1.0)


def _split_factors(index: pd.DatetimeIndex, splits: pd.Series) -> pd.Series:
    """For each bar t, the product of every split ratio with ex-date > t.

    A split is effective at the open of its ex-date, so the ex-date bar itself
    is already post-split and its own ratio is excluded.
    """
    return _later_product(index, splits)


def _unadjust_splits(prices: pd.DataFrame, splits: pd.Series) -> pd.DataFrame:
    """Yahoo's split back-adjusted prices -> true as-traded prices.

    The exact inverse of the back-adjustment: a bar before an r:1 split has been
    divided by r by Yahoo, so multiply it back by r. Share counts move the other
    way, so volume is divided by the same factor.
    """
    factor = _split_factors(prices.index, splits)
    out = prices.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col] * factor
    out["volume"] = out["volume"] / factor
    return out


def adjust_closes(
    prices: pd.DataFrame, actions: Sequence[CorporateAction]
) -> pd.Series:
    """As-traded closes + corporate actions -> a total-return (adjusted) series.

    Undoes both effects that make an as-traded series discontinuous: splits (a
    10:1 drops the price 90% overnight) and dividends (the price gaps down by
    the payout on the ex-date). This is what indicators should see; the ledger
    must never use it.

    Matches yfinance's own `auto_adjust=True` output to within rounding.
    """
    close = prices["close"].astype(float) if "close" in prices else pd.Series(dtype=float)
    if close.empty:
        return close

    splits = pd.Series(
        {a.ex_date: a.value for a in actions if a.type == SPLIT}, dtype=float
    ).sort_index()
    dividends = pd.Series(
        {a.ex_date: a.value for a in actions if a.type == DIVIDEND}, dtype=float
    ).sort_index()

    # undo splits first, so dividend ratios are measured against a continuous
    # price (a payout must be compared with the price on the same share basis)
    adjusted = close / _split_factors(close.index, splits)

    if len(dividends):
        # each dividend scales every PRIOR bar by (1 - d / close before ex-date),
        # the convention Yahoo uses
        ratios = pd.Series(1.0, index=close.index)
        prev_close = adjusted.shift(1)
        for ex_date, amount in dividends.items():
            if ex_date not in close.index:
                continue
            base = prev_close.get(ex_date)
            if base is None or not math.isfinite(float(base)) or float(base) <= 0:
                continue
            ratios.loc[ex_date] = 1.0 - float(amount) / float(base)
        adjusted = adjusted * _later_product(close.index, ratios)

    return adjusted


# --------------------------------------------------------------------- fetching


def _default_download(tickers: Sequence[str], **kwargs) -> pd.DataFrame:
    """The real network call. Injectable so tests can run fully offline."""
    return yf.download(list(tickers), **kwargs)


def _extract(frame: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """One ticker's sub-frame out of a group_by='ticker' download result."""
    if frame is None or frame.empty:
        return None
    if isinstance(frame.columns, pd.MultiIndex):
        if ticker not in frame.columns.get_level_values(0):
            return None
        sub = frame[ticker]
    else:
        sub = frame
    return None if sub.empty else sub


def _clean(ticker: str, sub: pd.DataFrame) -> tuple[pd.DataFrame, list[CorporateAction]]:
    """A raw yfinance sub-frame -> (as-traded OHLCV, corporate actions)."""
    df = sub.rename(columns=str.lower)
    missing = set(PRICE_COLUMNS) - set(df.columns)
    if missing:
        raise FetchError(f"{ticker}: price data missing columns {sorted(missing)}")

    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    if df.empty:
        raise FetchError(f"{ticker}: no usable bars after cleaning")

    def _events(column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series(dtype=float)
        series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        return series[series > 0]

    splits = _events("stock splits")
    dividends = _events("dividends")

    prices = df[list(PRICE_COLUMNS)].copy()
    prices["volume"] = prices["volume"].fillna(0)
    prices = _unadjust_splits(prices, splits)

    actions = [
        CorporateAction(ticker, ex_date, SPLIT, float(v)) for ex_date, v in splits.items()
    ] + [
        CorporateAction(ticker, ex_date, DIVIDEND, float(v))
        for ex_date, v in dividends.items()
    ]
    return prices, actions


class DataFetcher:
    """Cached access to as-traded prices and corporate actions.

    Not thread-safe: sqlite3 connections are thread-affine, so create one
    DataFetcher per thread (the UI worker builds its own).
    """

    # shared across ALL fetcher instances (watchlist + portfolio workers can run
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
        download: Callable[..., pd.DataFrame] | None = None,
    ):
        self.sleep_between = sleep_between
        self.should_stop = should_stop  # lets the UI cancel mid-backoff
        self.network_fetches = 0  # exposed so tests can prove cache hits
        self._download = download or _default_download
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version != CACHE_VERSION:
            log.info("cache version %s -> %s: resetting cache", version, CACHE_VERSION)
            self._conn.executescript(
                "DROP TABLE IF EXISTS prices;"
                "DROP TABLE IF EXISTS fundamentals;"
                "DROP TABLE IF EXISTS corporate_actions;"
                "DROP TABLE IF EXISTS fetch_meta;"
            )
            self._conn.execute(f"PRAGMA user_version = {CACHE_VERSION}")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ cache

    def _is_fresh(self, ticker: str) -> bool:
        row = self._conn.execute(
            "SELECT fetched_at, status FROM fetch_meta WHERE ticker=? AND kind='prices'",
            (ticker,),
        ).fetchone()
        if row is None or row[1] != "ok":
            return False
        return datetime.now(timezone.utc) - datetime.fromisoformat(row[0]) < MAX_AGE

    def _mark(self, ticker: str, status: str, error: str | None) -> None:
        self._conn.execute(
            "INSERT INTO fetch_meta (ticker, kind, fetched_at, status, error)"
            " VALUES (?,'prices',?,?,?)"
            " ON CONFLICT(ticker, kind) DO UPDATE SET"
            " fetched_at=excluded.fetched_at, status=excluded.status, error=excluded.error",
            (ticker, datetime.now(timezone.utc).isoformat(), status, error),
        )

    def _last_cached_date(self, ticker: str) -> pd.Timestamp | None:
        row = self._conn.execute(
            "SELECT MAX(date) FROM prices WHERE ticker=?", (ticker,)
        ).fetchone()
        return pd.Timestamp(row[0]) if row and row[0] else None

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

    def _store(
        self, ticker: str, prices: pd.DataFrame, actions: Sequence[CorporateAction]
    ) -> None:
        """Upsert, never delete-and-reinsert: an incremental fetch overlaps the
        cached tail, and a stored bar must survive that untouched unless Yahoo
        genuinely revised it."""
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
            for idx, r in zip(prices.index, prices.itertuples())
        ]
        action_rows = [
            (a.ticker, a.ex_date.strftime("%Y-%m-%d"), a.type, a.value) for a in actions
        ]
        with self._conn:
            self._conn.executemany(
                "INSERT INTO prices (ticker, date, open, high, low, close, volume)"
                " VALUES (?,?,?,?,?,?,?)"
                " ON CONFLICT(ticker, date) DO UPDATE SET"
                " open=excluded.open, high=excluded.high, low=excluded.low,"
                " close=excluded.close, volume=excluded.volume",
                rows,
            )
            if action_rows:
                self._conn.executemany(
                    "INSERT INTO corporate_actions (ticker, ex_date, type, value)"
                    " VALUES (?,?,?,?)"
                    " ON CONFLICT(ticker, ex_date, type) DO UPDATE SET"
                    " value=excluded.value",
                    action_rows,
                )
            self._mark(ticker, "ok", None)

    def actions(self, ticker: str) -> list[CorporateAction]:
        ticker = normalize(ticker)
        rows = self._conn.execute(
            "SELECT ex_date, type, value FROM corporate_actions"
            " WHERE ticker=? ORDER BY ex_date",
            (ticker,),
        ).fetchall()
        return [
            CorporateAction(ticker, pd.Timestamp(ex_date), kind, float(value))
            for ex_date, kind, value in rows
        ]

    def adjusted_closes(self, ticker: str) -> pd.Series:
        """Split- and dividend-adjusted closes, for indicators."""
        ticker = normalize(ticker)
        return adjust_closes(self._load_prices(ticker), self.actions(ticker))

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

    def _download_chunk(self, tickers: Sequence[str], start: pd.Timestamp | None):
        """One paced download covering `tickers`. `start=None` means full history."""
        kwargs = dict(
            interval="1d",
            auto_adjust=False,   # see module docstring: still split-adjusted
            actions=True,        # splits/dividends arrive inside the frame
            group_by="ticker",
            threads=False,       # our pacing and circuit breaker assume serial
            progress=False,
        )
        if start is None:
            kwargs["period"] = PRICE_PERIOD
        else:
            kwargs["start"] = start.strftime("%Y-%m-%d")
        return self._network_call(lambda: self._download(list(tickers), **kwargs))

    def _fetch_into_cache(self, tickers: Sequence[str]) -> dict[str, str]:
        """Fetch and store `tickers`, batching by whether each needs full
        history or only an incremental top-up. Returns {ticker: error} for the
        ones that failed."""
        full: list[str] = []
        incremental: list[str] = []
        for t in tickers:
            (incremental if self._last_cached_date(t) is not None else full).append(t)

        errors: dict[str, str] = {}

        if incremental:
            # one window for the batch: the oldest tail among them, minus the
            # revision overlap. Fresher tickers simply receive a few extra bars,
            # which upsert harmlessly — cheaper than one request per start date.
            oldest = min(self._last_cached_date(t) for t in incremental)
            start = oldest - pd.Timedelta(f"{OVERLAP_SESSIONS * 2 + 5}D")
            for i in range(0, len(incremental), CHUNK_SIZE):
                errors.update(self._run_chunk(incremental[i:i + CHUNK_SIZE], start))

        for i in range(0, len(full), CHUNK_SIZE):
            errors.update(self._run_chunk(full[i:i + CHUNK_SIZE], None))

        return errors

    def _run_chunk(
        self, chunk: Sequence[str], start: pd.Timestamp | None
    ) -> dict[str, str]:
        errors: dict[str, str] = {}
        try:
            frame = self._download_chunk(chunk, start)
        except Exception as e:
            if len(chunk) == 1:
                with self._conn:
                    self._mark(chunk[0], "failed", str(e))
                return {chunk[0]: str(e)}
            # a whole-chunk failure must not blank 50 tickers; retry singly so
            # one bad symbol only costs itself
            log.warning("chunk of %d failed (%s); retrying individually", len(chunk), e)
            for t in chunk:
                errors.update(self._run_chunk([t], start))
            return errors

        for t in chunk:
            try:
                sub = _extract(frame, t)
                if sub is None:
                    raise FetchError(f"{t}: no price data returned (delisted or invalid)")
                prices, actions = _clean(t, sub)
                self._store(t, prices, actions)
            except Exception as e:
                log.warning("%s: price fetch failed: %s", t, e)
                errors[t] = str(e)
                with self._conn:
                    self._mark(t, "failed", str(e))
        return errors

    # ----------------------------------------------------------------- public

    def refresh(self, tickers: Iterable[str], force: bool = False) -> dict[str, str]:
        """Bring `tickers` up to date in one batched pass. Returns {ticker:
        error} for the failures; a ticker absent from that map is cached."""
        wanted = normalize_all(tickers)
        due = [t for t in wanted if force or not self._is_fresh(t)]
        if not due:
            return {}
        log.info("refreshing %d of %d tickers", len(due), len(wanted))
        return self._fetch_into_cache(due)

    def get_prices(
        self, tickers: Iterable[str], force: bool = False
    ) -> dict[str, StockData]:
        """Batched counterpart to get_stock. Tickers with no usable data are
        omitted from the result rather than raising, so one dead symbol never
        costs the caller the rest of the batch."""
        wanted = normalize_all(tickers)
        errors = self.refresh(wanted, force=force)
        out: dict[str, StockData] = {}
        for t in wanted:
            prices = self._load_prices(t)
            if prices.empty:
                log.warning(
                    "%s: no usable price data (%s)", t, errors.get(t, "cache empty")
                )
                continue
            out[t] = StockData(ticker=t, prices=prices, stale=t in errors)
        return out

    def get_stock(self, ticker: str, force: bool = False) -> StockData:
        """Return cached-or-fresh data for one ticker. Raises FetchError only
        when no usable price data exists at all."""
        ticker = normalize(ticker)
        errors = self.refresh([ticker], force=force)
        prices = self._load_prices(ticker)
        if prices.empty:
            # include the cause so callers can tell "bad symbol" from
            # transient failures (cooldown/cancelled/timeouts)
            raise FetchError(
                f"{ticker}: no usable price data "
                f"({errors.get(ticker, 'fetch failed, cache empty')})"
            )
        return StockData(ticker=ticker, prices=prices, stale=ticker in errors)


# --------------------------------------------------------------- what to fetch


def _no_held_tickers() -> set[str]:
    return set()


# Phase 3 points this at the trade ledger. Until then no bot holds anything, so
# the universe is the whole story.
held_tickers: Callable[[], set[str]] = _no_held_tickers


def tickers_to_fetch(universe: Iterable[str] | None = None) -> list[str]:
    """Universe ∪ everything a bot still holds.

    A position does not stop existing when its company leaves the index, so a
    held ticker must keep updating or its bot's equity silently freezes at the
    last cached close.
    """
    if universe is None:
        from data.universe import get_universe

        universe = get_universe()
    return normalize_all([*universe, *held_tickers()])
