"""NYSE trading calendar. Offline — exchange_calendars ships its own rules.

The spec's done-conditions for the calendar live here: Thanksgiving 2025 is not
a trading day, and the day after it is an early close.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data import calendar as cal

THANKSGIVING_2025 = "2025-11-27"
BLACK_FRIDAY_2025 = "2025-11-28"


def test_thanksgiving_is_not_a_trading_day():
    assert not cal.is_trading_day(THANKSGIVING_2025)


def test_day_after_thanksgiving_is_a_trading_day_and_an_early_close():
    assert cal.is_trading_day(BLACK_FRIDAY_2025)
    assert cal.is_early_close(BLACK_FRIDAY_2025)


def test_a_normal_session_is_not_an_early_close():
    assert cal.is_trading_day("2025-11-26")
    assert not cal.is_early_close("2025-11-26")


def test_early_close_is_three_hours_before_a_normal_close():
    normal = cal.session_close("2025-11-26")
    early = cal.session_close(BLACK_FRIDAY_2025)
    assert normal.hour == 21 and early.hour == 18  # 16:00 vs 13:00 ET, in UTC
    assert normal.tz is not None and early.tz is not None


def test_weekends_and_holidays_are_skipped():
    assert not cal.is_trading_day("2025-11-29")  # Saturday
    assert not cal.is_trading_day("2025-11-30")  # Sunday
    assert not cal.is_trading_day("2025-12-25")  # Christmas


def test_next_and_previous_session_step_over_the_holiday():
    # Wed 26th -> (Thanksgiving closed) -> Fri 28th
    assert cal.next_session("2025-11-26") == pd.Timestamp(BLACK_FRIDAY_2025)
    assert cal.previous_session(BLACK_FRIDAY_2025) == pd.Timestamp("2025-11-26")


def test_next_session_is_strictly_after_its_argument():
    assert cal.next_session("2025-11-26") > pd.Timestamp("2025-11-26")
    assert cal.previous_session("2025-11-26") < pd.Timestamp("2025-11-26")


def test_trading_days_is_inclusive_and_excludes_the_holiday():
    days = [d.date().isoformat() for d in cal.trading_days("2025-11-24", "2025-11-28")]
    assert days == ["2025-11-24", "2025-11-25", "2025-11-26", "2025-11-28"]


def test_sessions_are_tz_naive_midnight():
    day = cal.trading_days("2025-11-24", "2025-11-24")[0]
    assert day.tz is None and day.hour == 0 and day.minute == 0


def test_session_close_rejects_a_non_session():
    with pytest.raises(ValueError):
        cal.session_close(THANKSGIVING_2025)


class TestLatestCompleteSession:
    """The daily runner must never process a bar that is still forming."""

    def test_mid_session_returns_the_previous_day(self):
        # 2025-11-26 15:00 UTC == 10:00 ET, market open
        assert cal.latest_complete_session("2025-11-26T15:00:00Z") == pd.Timestamp(
            "2025-11-25"
        )

    def test_after_the_close_returns_that_day(self):
        # 21:00 UTC == 16:00 ET, the bell
        assert cal.latest_complete_session("2025-11-26T21:00:00Z") == pd.Timestamp(
            "2025-11-26"
        )

    def test_early_close_counts_as_complete_in_the_afternoon(self):
        # 19:00 UTC == 14:00 ET: a normal day is still open, but the half-day
        # closed an hour ago, so its bar is final
        assert cal.latest_complete_session(
            f"{BLACK_FRIDAY_2025}T19:00:00Z"
        ) == pd.Timestamp(BLACK_FRIDAY_2025)

    def test_before_an_early_close_falls_back(self):
        assert cal.latest_complete_session(
            f"{BLACK_FRIDAY_2025}T17:00:00Z"
        ) == pd.Timestamp("2025-11-26")

    def test_weekend_returns_fridays_session(self):
        assert cal.latest_complete_session("2025-11-29T12:00:00Z") == pd.Timestamp(
            BLACK_FRIDAY_2025
        )

    def test_naive_input_is_read_as_utc(self):
        assert cal.latest_complete_session(
            "2025-11-26T21:00:00"
        ) == cal.latest_complete_session("2025-11-26T21:00:00Z")
