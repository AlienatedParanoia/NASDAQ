"""Data layer: as-traded storage, corporate actions, adjustment maths, caching
and batching. Fully offline — the network is replaced by a fake downloader that
serves synthetic frames and records how it was called.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.fetcher import (
    CHUNK_SIZE,
    DIVIDEND,
    SPLIT,
    CorporateAction,
    DataFetcher,
    FetchError,
    adjust_closes,
    tickers_to_fetch,
)
import data.fetcher as fetcher_module

YF_COLUMNS = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits",
]


def make_frame(
    closes,
    start="2024-01-01",
    splits: dict[str, float] | None = None,
    dividends: dict[str, float] | None = None,
) -> pd.DataFrame:
    """A yfinance-shaped single-ticker frame: raw-ish OHLCV plus action columns.

    Values are taken as already split-back-adjusted, exactly as Yahoo returns
    them, so the fetcher's un-adjustment is what gets exercised.
    """
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    df = pd.DataFrame(
        {
            "Open": closes * 0.99,
            "High": closes * 1.02,
            "Low": closes * 0.98,
            "Close": closes,
            "Adj Close": closes,
            "Volume": np.full(len(closes), 1_000_000.0),
            "Dividends": np.zeros(len(closes)),
            "Stock Splits": np.zeros(len(closes)),
        },
        index=idx,
        columns=YF_COLUMNS,
    )
    df.index.name = "Date"
    for date, ratio in (splits or {}).items():
        df.loc[pd.Timestamp(date), "Stock Splits"] = ratio
    for date, amount in (dividends or {}).items():
        df.loc[pd.Timestamp(date), "Dividends"] = amount
    return df


class FakeYahoo:
    """Stands in for yf.download. Records every call so tests can assert on the
    request that was made, not just on the data that came back."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames
        self.calls: list[dict] = []

    def __call__(self, tickers, **kwargs):
        self.calls.append({"tickers": list(tickers), **kwargs})
        known = [t for t in tickers if t in self.frames]
        if not known:
            return pd.DataFrame()
        pieces = {}
        for t in known:
            frame = self.frames[t]
            if "start" in kwargs:
                frame = frame.loc[frame.index >= pd.Timestamp(kwargs["start"])]
            pieces[t] = frame
        # yfinance returns a 2-level (ticker, field) MultiIndex even for one ticker
        return pd.concat(pieces, axis=1)

    @property
    def periods(self) -> list[str | None]:
        return [c.get("period") for c in self.calls]

    @property
    def starts(self) -> list[str | None]:
        return [c.get("start") for c in self.calls]


@pytest.fixture
def db(tmp_path):
    return tmp_path / "cache.db"


def build(db, frames, **kwargs) -> tuple[DataFetcher, FakeYahoo]:
    fake = FakeYahoo(frames)
    return DataFetcher(db, sleep_between=0.0, download=fake, **kwargs), fake


# ----------------------------------------------------------------- as-traded


class TestAsTradedStorage:
    """The spec's core data decision: stored prices are what actually traded."""

    def test_split_is_undone_so_prices_are_as_traded(self, db):
        # Yahoo shows 100 before and after a 10:1 split; the pre-split bars
        # actually traded at 1000
        frames = {
            "NVDA": make_frame([100.0] * 6, splits={"2024-01-04": 10.0}),
        }
        f, _ = build(db, frames)
        prices = f.get_stock("NVDA").prices
        pre = prices.loc[: pd.Timestamp("2024-01-03"), "close"]
        post = prices.loc[pd.Timestamp("2024-01-04"):, "close"]
        assert pre.tolist() == pytest.approx([1000.0] * 3), "pre-split bars un-adjusted"
        assert post.tolist() == pytest.approx([100.0] * 3), "ex-date on is as-traded"
        f.close()

    def test_volume_moves_opposite_to_price(self, db):
        frames = {"NVDA": make_frame([100.0] * 6, splits={"2024-01-04": 10.0})}
        f, _ = build(db, frames)
        prices = f.get_stock("NVDA").prices
        # 10x fewer shares existed before the split
        assert prices.loc[pd.Timestamp("2024-01-03"), "volume"] == pytest.approx(100_000)
        assert prices.loc[pd.Timestamp("2024-01-04"), "volume"] == pytest.approx(1_000_000)
        f.close()

    def test_ohlc_all_scale_together(self, db):
        frames = {"X": make_frame([100.0] * 6, splits={"2024-01-04": 2.0})}
        f, _ = build(db, frames)
        row = f.get_stock("X").prices.loc[pd.Timestamp("2024-01-02")]
        assert row["open"] == pytest.approx(198.0)
        assert row["high"] == pytest.approx(204.0)
        assert row["low"] == pytest.approx(196.0)
        assert row["close"] == pytest.approx(200.0)
        f.close()

    def test_no_split_leaves_prices_untouched(self, db):
        f, _ = build(db, {"X": make_frame([10.0, 11.0, 12.0])})
        assert list(f.get_stock("X").prices["close"]) == pytest.approx([10.0, 11.0, 12.0])
        f.close()

    def test_consecutive_splits_compound(self, db):
        frames = {
            "X": make_frame(
                [100.0] * 8, splits={"2024-01-03": 2.0, "2024-01-05": 5.0}
            )
        }
        f, _ = build(db, frames)
        close = f.get_stock("X").prices["close"]
        assert close.loc[pd.Timestamp("2024-01-01")] == pytest.approx(1000.0)  # 2*5
        assert close.loc[pd.Timestamp("2024-01-03")] == pytest.approx(500.0)   # 5
        assert close.loc[pd.Timestamp("2024-01-05")] == pytest.approx(100.0)   # none
        f.close()


class TestCorporateActions:
    def test_split_stored_with_its_ratio(self, db):
        f, _ = build(db, {"NVDA": make_frame([100.0] * 6, splits={"2024-01-04": 10.0})})
        f.get_stock("NVDA")
        actions = f.actions("NVDA")
        splits = [a for a in actions if a.type == SPLIT]
        assert len(splits) == 1
        assert splits[0].ex_date == pd.Timestamp("2024-01-04")
        assert splits[0].value == pytest.approx(10.0)
        f.close()

    def test_dividends_stored_with_their_ex_dates(self, db):
        f, _ = build(
            db,
            {"MSFT": make_frame([100.0] * 8, dividends={"2024-01-03": 0.75,
                                                        "2024-01-09": 0.83})},
        )
        f.get_stock("MSFT")
        divs = [a for a in f.actions("MSFT") if a.type == DIVIDEND]
        assert [(a.ex_date, a.value) for a in divs] == [
            (pd.Timestamp("2024-01-03"), pytest.approx(0.75)),
            (pd.Timestamp("2024-01-09"), pytest.approx(0.83)),
        ]
        f.close()

    def test_zero_rows_are_not_stored_as_actions(self, db):
        f, _ = build(db, {"X": make_frame([100.0] * 5)})
        f.get_stock("X")
        assert f.actions("X") == []
        f.close()

    def test_actions_survive_a_refetch_without_duplicating(self, db):
        frames = {"NVDA": make_frame([100.0] * 6, splits={"2024-01-04": 10.0})}
        f, _ = build(db, frames)
        f.get_stock("NVDA", force=True)
        f.get_stock("NVDA", force=True)
        assert len(f.actions("NVDA")) == 1, "upsert, not append"
        f.close()


class TestImmutability:
    """The property the whole ledger depends on."""

    def test_stored_prices_are_identical_after_a_refetch(self, db):
        frames = {"X": make_frame([10.0, 11.0, 12.0, 13.0, 14.0])}
        f, _ = build(db, frames)
        first = f.get_stock("X").prices.copy()
        f.get_stock("X", force=True)
        second = f.get_stock("X").prices
        pd.testing.assert_frame_equal(first, second)
        f.close()

    def test_a_later_dividend_does_not_rewrite_earlier_bars(self, db):
        """The exact failure auto_adjust=True caused: yesterday's recorded fill
        price must not move because a dividend went ex today."""
        frames = {"X": make_frame([100.0] * 5)}
        f, fake = build(db, frames)
        before = f.get_stock("X").prices["close"].copy()

        fake.frames["X"] = make_frame([100.0] * 6, dividends={"2024-01-08": 2.5})
        f.get_stock("X", force=True)
        after = f.get_stock("X").prices["close"]

        pd.testing.assert_series_equal(before, after.loc[before.index])
        assert any(a.type == DIVIDEND for a in f.actions("X"))
        f.close()


class TestAdjustedCloses:
    def test_split_discontinuity_is_removed(self, db):
        f, _ = build(db, {"X": make_frame([100.0] * 6, splits={"2024-01-04": 10.0})})
        f.get_stock("X")
        adjusted = f.adjusted_closes("X")
        # as-traded jumps 1000 -> 100; adjusted must be flat
        assert adjusted.std() == pytest.approx(0.0, abs=1e-9)
        f.close()

    def test_dividend_is_folded_into_earlier_bars(self, db):
        f, _ = build(db, {"X": make_frame([100.0] * 5, dividends={"2024-01-03": 1.0})})
        f.get_stock("X")
        adjusted = f.adjusted_closes("X")
        # bars before the ex-date are scaled by (1 - 1/100)
        assert adjusted.loc[pd.Timestamp("2024-01-02")] == pytest.approx(99.0)
        assert adjusted.loc[pd.Timestamp("2024-01-03")] == pytest.approx(100.0)
        f.close()

    def test_no_actions_means_adjusted_equals_raw(self, db):
        f, _ = build(db, {"X": make_frame([10.0, 11.0, 12.0])})
        f.get_stock("X")
        pd.testing.assert_series_equal(
            f.adjusted_closes("X"), f.get_stock("X").prices["close"], check_names=False
        )
        f.close()

    def test_pure_function_handles_an_empty_frame(self):
        empty = pd.DataFrame({"close": pd.Series(dtype=float)})
        assert adjust_closes(empty, []).empty

    def test_pure_function_ignores_actions_outside_the_price_window(self):
        prices = pd.DataFrame(
            {"close": [10.0, 11.0]},
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
        )
        stray = [CorporateAction("X", pd.Timestamp("2030-01-01"), DIVIDEND, 1.0)]
        pd.testing.assert_series_equal(
            adjust_closes(prices, stray), prices["close"], check_names=False
        )


# ------------------------------------------------------------- fetch behaviour


class TestIncrementalFetching:
    def test_first_fetch_asks_for_full_history(self, db):
        f, fake = build(db, {"X": make_frame([10.0] * 5)})
        f.get_stock("X")
        assert fake.periods == ["10y"]
        assert fake.starts == [None]
        f.close()

    def test_an_up_to_date_ticker_makes_no_request_at_all(self, db):
        f, fake = build(db, {"X": make_frame([10.0] * 5)})
        f.get_stock("X")
        calls_after_first = len(fake.calls)
        f.get_stock("X")
        f.get_stock("X")
        assert len(fake.calls) == calls_after_first
        assert f.network_fetches == calls_after_first
        f.close()

    def test_a_stale_ticker_tops_up_instead_of_refetching_everything(self, db):
        """The spec's done-condition: no full-history request once cached."""
        f, fake = build(db, {"X": make_frame([10.0] * 5)})
        f.get_stock("X")
        # expire the freshness marker without touching the price rows
        f._conn.execute(
            "UPDATE fetch_meta SET fetched_at='2000-01-01T00:00:00+00:00'"
        )
        f._conn.commit()
        f.get_stock("X")
        assert fake.periods == ["10y", None], "second call must not ask for 10y"
        assert fake.starts[1] is not None, "second call must be windowed"
        f.close()

    def test_the_overlap_window_starts_before_the_cached_tail(self, db):
        f, fake = build(db, {"X": make_frame([10.0] * 10)})
        f.get_stock("X")
        last = f._last_cached_date("X")
        f._conn.execute("UPDATE fetch_meta SET fetched_at='2000-01-01T00:00:00+00:00'")
        f._conn.commit()
        f.get_stock("X")
        assert pd.Timestamp(fake.starts[1]) < last, "overlap must re-request recent bars"
        f.close()

    def test_a_revised_bar_is_corrected_by_the_overlap(self, db):
        f, fake = build(db, {"X": make_frame([10.0] * 5)})
        f.get_stock("X")
        revised = make_frame([10.0] * 5)
        revised.loc[revised.index[-1], "Close"] = 99.0
        fake.frames["X"] = revised
        f.get_stock("X", force=True)
        assert f.get_stock("X").prices["close"].iloc[-1] == pytest.approx(99.0)
        f.close()


class TestBatching:
    def test_many_tickers_cost_one_request(self, db):
        frames = {t: make_frame([10.0] * 5) for t in ("A", "B", "C", "D")}
        f, fake = build(db, frames)
        data = f.get_prices(["A", "B", "C", "D"])
        assert set(data) == {"A", "B", "C", "D"}
        assert len(fake.calls) == 1, "batched, not one request per ticker"
        f.close()

    def test_requests_are_chunked(self, db):
        tickers = [f"T{i}" for i in range(CHUNK_SIZE + 10)]
        f, fake = build(db, {t: make_frame([10.0] * 3) for t in tickers})
        f.get_prices(tickers)
        assert len(fake.calls) == 2
        assert len(fake.calls[0]["tickers"]) == CHUNK_SIZE
        f.close()

    def test_downloads_are_never_threaded(self, db):
        """Our pacing, 60s backoff and circuit breaker all assume one request
        in flight at a time."""
        f, fake = build(db, {"A": make_frame([10.0] * 3)})
        f.get_prices(["A"])
        assert all(c["threads"] is False for c in fake.calls)
        assert all(c["auto_adjust"] is False for c in fake.calls)
        assert all(c["actions"] is True for c in fake.calls)
        f.close()

    def test_one_dead_symbol_does_not_cost_the_batch(self, db):
        frames = {"GOOD1": make_frame([10.0] * 5), "GOOD2": make_frame([20.0] * 5)}
        f, _ = build(db, frames)  # DEAD has no frame
        data = f.get_prices(["GOOD1", "DEAD", "GOOD2"])
        assert set(data) == {"GOOD1", "GOOD2"}
        f.close()

    def test_a_chunk_level_failure_falls_back_to_single_requests(self, db):
        frames = {"A": make_frame([10.0] * 3), "B": make_frame([20.0] * 3)}
        fake = FakeYahoo(frames)
        calls = {"n": 0}
        original = fake.__call__

        def flaky(tickers, **kwargs):
            calls["n"] += 1
            if len(tickers) > 1:
                raise RuntimeError("chunk blew up")
            return original(tickers, **kwargs)

        f = DataFetcher(db, sleep_between=0.0, download=flaky)
        data = f.get_prices(["A", "B"])
        assert set(data) == {"A", "B"}, "both tickers recovered via single requests"
        assert calls["n"] == 3, "one failed batch, then one request each"
        f.close()

    def test_batch_results_match_single_fetches(self, db, tmp_path):
        frames = {"A": make_frame([10.0, 11.0, 12.0], splits={"2024-01-03": 2.0})}
        f1, _ = build(db, frames)
        batched = f1.get_prices(["A"])["A"].prices
        f1.close()
        f2, _ = build(tmp_path / "other.db", frames)
        single = f2.get_stock("A").prices
        pd.testing.assert_frame_equal(batched, single)
        f2.close()


class TestNormalisationAndErrors:
    def test_ticker_spelling_is_normalised_everywhere(self, db):
        f, fake = build(db, {"BRK-B": make_frame([10.0] * 5)})
        data = f.get_stock("brk.b")
        assert data.ticker == "BRK-B"
        assert fake.calls[0]["tickers"] == ["BRK-B"]
        # and a second spelling hits the same cache row, not a new fetch
        n = len(fake.calls)
        f.get_stock("BRK.B")
        assert len(fake.calls) == n
        f.close()

    def test_unknown_ticker_raises_with_a_reason(self, db):
        f, _ = build(db, {})
        with pytest.raises(FetchError, match="no usable price data"):
            f.get_stock("NOPE")
        f.close()

    def test_a_failed_refetch_serves_stale_cache_rather_than_raising(self, db):
        f, fake = build(db, {"X": make_frame([10.0] * 5)})
        f.get_stock("X")
        fake.frames.clear()  # Yahoo now returns nothing for X
        data = f.get_stock("X", force=True)
        assert data.stale is True
        assert len(data.prices) == 5, "cached bars still served"
        f.close()

    def test_missing_columns_are_rejected(self, db):
        broken = make_frame([10.0] * 3).drop(columns=["Low"])
        f, _ = build(db, {"X": broken})
        with pytest.raises(FetchError):
            f.get_stock("X")
        f.close()


class TestTickersToFetch:
    def test_defaults_to_the_universe_alone(self):
        assert tickers_to_fetch(["AAPL", "MSFT"]) == ["AAPL", "MSFT"]

    def test_held_tickers_are_added_even_outside_the_universe(self, monkeypatch):
        monkeypatch.setattr(fetcher_module, "held_tickers", lambda: {"DELISTED"})
        assert set(tickers_to_fetch(["AAPL"])) == {"AAPL", "DELISTED"}

    def test_a_held_ticker_already_in_the_universe_is_not_duplicated(self, monkeypatch):
        monkeypatch.setattr(fetcher_module, "held_tickers", lambda: {"AAPL"})
        assert tickers_to_fetch(["AAPL", "MSFT"]) == ["AAPL", "MSFT"]

    def test_held_tickers_are_normalised_too(self, monkeypatch):
        monkeypatch.setattr(fetcher_module, "held_tickers", lambda: {"brk.b"})
        assert set(tickers_to_fetch(["AAPL"])) == {"AAPL", "BRK-B"}


class TestCacheVersioning:
    def test_a_version_bump_wipes_the_cache(self, db, monkeypatch):
        f, _ = build(db, {"X": make_frame([10.0] * 5)})
        f.get_stock("X")
        f.close()
        monkeypatch.setattr(fetcher_module, "CACHE_VERSION", 999)
        f2, fake2 = build(db, {"X": make_frame([10.0] * 5)})
        assert f2._load_prices("X").empty, "old rows must not survive a schema change"
        f2.close()
