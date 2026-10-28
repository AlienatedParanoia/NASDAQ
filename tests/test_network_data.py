"""The data checks that genuinely need Yahoo, kept apart from the offline suite.

These verify our handling against reality rather than against a fixture we wrote:
the real NVDA split, real MSFT dividends, and our adjusted series compared with
yfinance's own. Run with:

    python -m pytest tests/test_network_data.py --network
"""
from __future__ import annotations

import pandas as pd
import pytest

import paths
from data.fetcher import DIVIDEND, SPLIT, DataFetcher, configure_yfinance

pytestmark = pytest.mark.network

NVDA_SPLIT_DATE = pd.Timestamp("2024-06-10")
NVDA_SPLIT_RATIO = 10.0


@pytest.fixture(scope="module")
def fetcher(tmp_path_factory):
    """A throwaway cache, so these tests neither depend on nor disturb the
    app's real one."""
    configure_yfinance(paths.data_dir())
    f = DataFetcher(tmp_path_factory.mktemp("netcache") / "cache.db")
    yield f
    f.close()


def test_nvda_split_is_stored_with_ratio_ten(fetcher):
    fetcher.get_stock("NVDA")
    splits = [a for a in fetcher.actions("NVDA") if a.type == SPLIT]
    match = [a for a in splits if a.ex_date == NVDA_SPLIT_DATE]
    assert match, f"2024-06-10 10:1 split missing; found {[a.ex_date for a in splits]}"
    assert match[0].value == pytest.approx(NVDA_SPLIT_RATIO)


def test_nvda_pre_split_prices_are_as_traded(fetcher):
    """The whole point of un-adjusting: NVDA traded near $1,200 before the
    split, not near $120."""
    prices = fetcher.get_stock("NVDA").prices
    before = prices.loc[prices.index < NVDA_SPLIT_DATE, "close"]
    assert not before.empty
    last_before = float(before.iloc[-1])
    assert 900.0 < last_before < 1500.0, (
        f"close before the split is {last_before:.2f} — looks still split-adjusted"
    )
    first_after = float(prices.loc[prices.index >= NVDA_SPLIT_DATE, "close"].iloc[0])
    assert 50.0 < first_after < 300.0
    assert last_before / first_after == pytest.approx(NVDA_SPLIT_RATIO, rel=0.15)


def test_msft_dividends_have_plausible_ex_dates_and_amounts(fetcher):
    fetcher.get_stock("MSFT")
    divs = [a for a in fetcher.actions("MSFT") if a.type == DIVIDEND]
    assert len(divs) >= 20, f"MSFT pays quarterly; only {len(divs)} found over 10y"
    assert all(0.0 < a.value < 10.0 for a in divs), "implausible dividend amount"
    dates = [a.ex_date for a in divs]
    assert dates == sorted(dates)
    assert len(set(dates)) == len(dates), "duplicate ex-dates"
    # quarterly cadence: ~4 a year, so gaps cluster around 90 days
    gaps = pd.Series(dates).diff().dropna().dt.days
    assert 60 <= gaps.median() <= 120, f"median gap {gaps.median()} is not quarterly"


@pytest.mark.parametrize("ticker", ["MSFT", "NVDA", "AAPL"])
def test_adjusted_helper_matches_yfinance_own_adjusted_series(fetcher, ticker):
    """Our reconstruction from as-traded prices + actions must agree with
    yfinance's auto_adjust=True output."""
    import yfinance as yf

    fetcher.get_stock(ticker)
    ours = fetcher.adjusted_closes(ticker)

    theirs = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=True)
    theirs = theirs["Close"]
    theirs.index = pd.DatetimeIndex(theirs.index).tz_localize(None).normalize()

    common = ours.index.intersection(theirs.index)
    assert len(common) > 200, f"only {len(common)} overlapping bars"
    a, b = ours.loc[common], theirs.loc[common]

    # compare shape, not level: the two series can differ by a constant factor
    # because they anchor their adjustment at different start dates
    ratio = (a / b).dropna()
    assert ratio.std() / ratio.mean() < 0.001, (
        f"{ticker}: adjusted series diverge in shape (ratio cv "
        f"{ratio.std() / ratio.mean():.5f})"
    )


def test_refetching_an_up_to_date_ticker_makes_no_request(fetcher):
    fetcher.get_stock("AAPL")
    before = fetcher.network_fetches
    fetcher.get_stock("AAPL")
    assert fetcher.network_fetches == before


def test_batched_fetch_covers_many_tickers_in_one_request(tmp_path):
    configure_yfinance(paths.data_dir())
    f = DataFetcher(tmp_path / "cache.db")
    tickers = ["AAPL", "MSFT", "KO", "XEL", "PEP"]
    data = f.get_prices(tickers)
    assert set(data) == set(tickers), f"missing: {set(tickers) - set(data)}"
    assert f.network_fetches == 1, f"expected one batched call, made {f.network_fetches}"
    for t in tickers:
        assert len(data[t].prices) > 2000, f"{t}: only {len(data[t].prices)} bars"
    f.close()
