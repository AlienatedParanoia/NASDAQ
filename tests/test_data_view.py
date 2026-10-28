"""The look-ahead barrier. Offline.

Rule 5 says a decision made on bar t may only execute on t+1 or later, enforced
in code. These tests cover the observation half: a strategy must be unable to
see the future at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bots.data_view import DataView, LookAheadError, PortfolioView

DATES = pd.bdate_range("2026-01-05", periods=10)
MID = DATES[4]


def frame(n: int = 10, start: float = 100.0) -> pd.DataFrame:
    closes = start + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": closes - 0.5, "high": closes + 1.0, "low": closes - 1.0,
            "close": closes, "volume": np.full(n, 1e6),
        },
        index=DATES[:n],
    )


@pytest.fixture
def view() -> DataView:
    return DataView(
        session=MID,
        frames={"AAA": frame(), "BBB": frame(start=50.0)},
        adjusted_closes={"AAA": frame()["close"] * 0.9},
    )


class TestTruncation:
    def test_history_stops_at_the_session(self, view):
        history = view.history("AAA")
        assert history.index.max() == MID
        assert len(history) == 5

    def test_adjusted_stops_at_the_session(self, view):
        assert view.adjusted("AAA").index.max() == MID

    def test_close_is_the_session_close_not_the_latest(self, view):
        # the frame runs to 109; standing on bar 5 the close must be 104
        assert view.close("AAA") == pytest.approx(104.0)

    def test_bar_returns_this_sessions_bar(self, view):
        bar = view.bar("AAA")
        assert bar is not None and bar["close"] == pytest.approx(104.0)

    def test_bars_available_counts_only_the_past(self, view):
        assert view.bars_available("AAA") == 5


class TestLookAheadRaises:
    def test_asking_for_a_later_date_raises(self, view):
        with pytest.raises(LookAheadError, match="rule 5"):
            view.history("AAA", upto=DATES[7])

    def test_adjusted_raises_too(self, view):
        with pytest.raises(LookAheadError):
            view.adjusted("AAA", upto=DATES[9])

    def test_the_session_itself_is_allowed(self, view):
        assert not view.history("AAA", upto=MID).empty

    def test_an_earlier_date_is_allowed(self, view):
        assert len(view.history("AAA", upto=DATES[2])) == 3

    def test_the_error_names_both_dates(self, view):
        with pytest.raises(LookAheadError) as excinfo:
            view.history("AAA", upto=DATES[8])
        message = str(excinfo.value)
        assert str(DATES[8].date()) in message and str(MID.date()) in message


class TestIsolation:
    def test_history_is_a_copy_not_a_slice(self, view):
        """One bot mutating its view must not corrupt the cache the next bot
        reads in the same run."""
        history = view.history("AAA")
        history.loc[history.index[0], "close"] = -999.0
        assert view.history("AAA")["close"].iloc[0] == pytest.approx(100.0)

    def test_adjusted_is_a_copy_too(self, view):
        series = view.adjusted("AAA")
        series.iloc[0] = -999.0
        assert view.adjusted("AAA").iloc[0] != -999.0

    def test_bar_is_a_copy(self, view):
        bar = view.bar("AAA")
        bar["close"] = -1.0
        assert view.bar("AAA")["close"] == pytest.approx(104.0)


class TestMissingData:
    def test_an_unknown_ticker_gives_an_empty_frame(self, view):
        assert view.history("NOPE").empty
        assert view.close("NOPE") is None
        assert view.bar("NOPE") is None
        assert not view.has("NOPE")

    def test_a_ticker_with_no_adjusted_series_falls_back_to_as_traded(self, view):
        # BBB has no adjusted entry; as-traded is correct when there are no
        # corporate actions anyway
        assert view.adjusted("BBB").iloc[-1] == pytest.approx(view.close("BBB"))

    def test_a_session_with_no_bar_reports_none(self):
        view = DataView(session=pd.Timestamp("2026-01-10"), frames={"AAA": frame(5)})
        assert view.bar("AAA") is None
        assert view.close("AAA") is not None, "the last known close is still usable"


class TestForSession:
    def test_it_shares_data_without_recopying(self, view):
        later = view.for_session(DATES[7])
        assert later.session == DATES[7]
        assert len(later.history("AAA")) == 8
        assert view.frames is later.frames, "frames are shared, not duplicated"

    def test_walking_forward_reveals_data_progressively(self, view):
        lengths = [view.for_session(d).bars_available("AAA") for d in DATES]
        assert lengths == list(range(1, 11))


class TestPortfolioView:
    def test_it_reports_holdings(self):
        pv = PortfolioView(
            session=MID, cash=1000.0, equity=5000.0, buying_power=1000.0,
            positions={"AAA": 10.0, "BBB": 0.0},
        )
        assert pv.holds("AAA") and not pv.holds("BBB")
        assert pv.shares("AAA") == 10.0
        assert pv.position_count == 1

    def test_open_orders_are_visible_so_a_strategy_does_not_double_up(self):
        pv = PortfolioView(
            session=MID, cash=1.0, equity=1.0, buying_power=1.0,
            open_order_tickers=frozenset({"AAA"}),
        )
        assert pv.has_open_order("AAA") and not pv.has_open_order("BBB")

    def test_bars_held_counts_sessions_since_entry(self):
        pv = PortfolioView(
            session=DATES[9], cash=0.0, equity=0.0, buying_power=0.0,
            positions={"AAA": 5.0}, entry_sessions={"AAA": DATES[2]},
        )
        assert pv.bars_held("AAA", frame()) == 7

    def test_bars_held_is_none_without_an_entry(self):
        pv = PortfolioView(session=MID, cash=0.0, equity=0.0, buying_power=0.0)
        assert pv.bars_held("AAA", frame()) is None

    def test_it_is_immutable(self):
        pv = PortfolioView(session=MID, cash=1.0, equity=1.0, buying_power=1.0)
        with pytest.raises(AttributeError):
            pv.cash = 999.0  # type: ignore[misc]
