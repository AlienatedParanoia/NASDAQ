"""NYSE trading calendar (XNYS), wrapped so the rest of the app never touches
exchange_calendars directly.

Why a real calendar rather than "weekdays minus a holiday list": the simulator
decides what a *session* is, and a wrong answer there is not cosmetic. Queueing
an order for a day the market never opened silently delays every fill by a day;
treating a half-day as a full one makes the daily runner process a bar that does
not exist yet.

Everything here speaks tz-naive midnight Timestamps, matching how price frames
are indexed throughout the app. Session *closes* are the one exception — they
are real instants, returned tz-aware in UTC, because "has this session finished"
is only answerable against a clock.
"""
from __future__ import annotations

import datetime as _dt
import logging
from functools import lru_cache

import pandas as pd

log = logging.getLogger(__name__)

EXCHANGE = "XNYS"

DateLike = "_dt.date | _dt.datetime | str | pd.Timestamp"


@lru_cache(maxsize=1)
def _calendar():
    """Built once and cached: constructing an exchange_calendars calendar
    parses decades of holiday rules and costs ~0.5s."""
    import exchange_calendars as xcals

    return xcals.get_calendar(EXCHANGE)


def _day(d) -> pd.Timestamp:
    """Anything date-shaped -> tz-naive midnight Timestamp."""
    ts = pd.Timestamp(d)
    if ts.tz is not None:
        ts = ts.tz_convert(None) if ts.tzinfo is not None else ts.tz_localize(None)
    return ts.normalize()


def trading_days(start, end) -> pd.DatetimeIndex:
    """All sessions in [start, end], inclusive."""
    return _calendar().sessions_in_range(_day(start), _day(end))


def is_trading_day(d) -> bool:
    return bool(_calendar().is_session(_day(d)))


def next_session(d) -> pd.Timestamp:
    """The first session strictly after `d` (so passing a session returns the
    one following it, not `d` itself)."""
    return _day(_calendar().next_session(_day(d)))


def previous_session(d) -> pd.Timestamp:
    """The last session strictly before `d`."""
    return _day(_calendar().previous_session(_day(d)))


def session_close(d) -> pd.Timestamp:
    """The closing instant of session `d`, tz-aware UTC. Raises if `d` is not
    a session."""
    day = _day(d)
    if not is_trading_day(day):
        raise ValueError(f"{day.date()} is not an XNYS trading day")
    return _calendar().closes.loc[day]


def is_early_close(d) -> bool:
    """True for half-days (the afternoons before/after some holidays), which
    close at 13:00 ET instead of 16:00."""
    day = _day(d)
    return bool(is_trading_day(day) and day in _calendar().early_closes)


def latest_complete_session(now=None) -> pd.Timestamp:
    """The most recent session whose closing bell has already rung.

    This is the only honest answer to "which bar can I trade on": while the
    market is open the current session's OHLC is still moving, so the daily
    runner (phase 5) must never process it. Early closes are respected, since
    the calendar's own close times are used.

    `now` is interpreted as UTC when naive, so callers in any timezone get the
    same answer.
    """
    ts = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

    cal = _calendar()
    # look back far enough to clear the longest market closure on record
    candidates = cal.sessions_in_range(
        _day(ts - pd.Timedelta("14D")), _day(ts + pd.Timedelta("1D"))
    )
    for day in reversed(candidates):
        if cal.closes.loc[day] <= ts:
            return _day(day)
    raise RuntimeError(f"no completed XNYS session in the 14 days before {ts}")
