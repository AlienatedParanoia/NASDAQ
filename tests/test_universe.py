"""Index membership loaded from the shipped CSVs. Offline.

These assert against the real data files rather than fixtures, because the
files are the thing that can rot: a bad scrape, a Wikipedia restructure, or a
normalisation slip would land here.
"""
from __future__ import annotations

import csv

import pytest

import paths
from data.symbols import normalize
from data.universe import (
    DEFAULT_INDEXES,
    INDEXES,
    Constituent,
    UnknownIndex,
    describe,
    get_constituents,
    get_universe,
    index_label,
)

# The true union is 503 + 101 - 86 overlapping = 518. (The original spec said
# "~600", which no arrangement of these two indexes produces.) The band allows
# for ordinary reconstitution drift without going quiet about a collapse.
MIN_UNION, MAX_UNION = 500, 540


class TestSizes:
    def test_sp500_has_about_five_hundred_members(self):
        assert 495 <= len(get_constituents("sp500")) <= 515

    def test_nasdaq100_has_about_a_hundred_members(self):
        # 101 symbols for 100 companies: Alphabet lists GOOG and GOOGL
        assert 99 <= len(get_constituents("nasdaq100")) <= 105

    def test_union_is_the_sum_minus_the_overlap(self):
        sp = set(get_universe(("sp500",)))
        nd = set(get_universe(("nasdaq100",)))
        union = get_universe(("sp500", "nasdaq100"))
        assert len(union) == len(sp | nd)
        assert len(union) == len(sp) + len(nd) - len(sp & nd)

    def test_union_is_in_the_expected_band(self):
        assert MIN_UNION <= len(get_universe()) <= MAX_UNION

    def test_the_indexes_genuinely_overlap(self):
        sp = set(get_universe(("sp500",)))
        nd = set(get_universe(("nasdaq100",)))
        # most big NASDAQ names are also in the S&P 500; if this collapsed to
        # zero, one of the two files is wrong
        assert 50 <= len(sp & nd) <= 101


class TestSymbolHygiene:
    @pytest.mark.parametrize("index", sorted(INDEXES))
    def test_every_ticker_is_already_normalised(self, index):
        bad = [c.ticker for c in get_constituents(index) if c.ticker != normalize(c.ticker)]
        assert not bad, f"{index}: un-normalised symbols {bad}"

    @pytest.mark.parametrize("index", sorted(INDEXES))
    def test_no_duplicate_tickers(self, index):
        tickers = [c.ticker for c in get_constituents(index)]
        assert len(tickers) == len(set(tickers))

    @pytest.mark.parametrize("index", sorted(INDEXES))
    def test_no_malformed_symbols(self, index):
        for c in get_constituents(index):
            assert c.ticker, "empty ticker"
            assert 1 <= len(c.ticker) <= 8, f"implausible symbol {c.ticker!r}"
            assert c.ticker == c.ticker.upper()
            assert "." not in c.ticker, f"{c.ticker}: dot should be a hyphen"
            assert " " not in c.ticker
            assert c.ticker.replace("-", "").isalnum(), f"odd symbol {c.ticker!r}"

    def test_class_share_symbols_use_yahoos_hyphen(self):
        sp = get_universe(("sp500",))
        assert "BRK-B" in sp and "BRK.B" not in sp
        assert "BF-B" in sp and "BF.B" not in sp

    @pytest.mark.parametrize("index", sorted(INDEXES))
    def test_every_member_has_a_name(self, index):
        missing = [c.ticker for c in get_constituents(index) if not c.name]
        assert not missing, f"{index}: no company name for {missing}"


class TestSelection:
    def test_each_index_returns_only_its_own_members(self):
        for index in INDEXES:
            expected = {c.ticker for c in get_constituents(index)}
            assert set(get_universe((index,))) == expected

    def test_default_is_both_indexes(self):
        assert get_universe() == get_universe(DEFAULT_INDEXES)
        assert set(DEFAULT_INDEXES) == set(INDEXES)

    def test_order_is_stable_across_calls(self):
        assert get_universe() == get_universe()

    def test_a_duplicate_index_does_not_duplicate_tickers(self):
        once = get_universe(("sp500",))
        twice = get_universe(("sp500", "sp500"))
        assert once == twice

    def test_unknown_index_raises_rather_than_returning_nothing(self):
        # a typo in a UI selector must be loud, not a silently smaller universe
        with pytest.raises(UnknownIndex):
            get_universe(("sp5000",))
        with pytest.raises(UnknownIndex):
            get_constituents("nope")
        with pytest.raises(UnknownIndex):
            index_label("nope")

    def test_empty_selection_raises(self):
        with pytest.raises(UnknownIndex):
            get_universe(())


class TestMetadata:
    def test_constituents_are_frozen_dataclasses(self):
        c = get_constituents("sp500")[0]
        assert isinstance(c, Constituent)
        with pytest.raises(AttributeError):
            c.ticker = "X"  # type: ignore[misc]

    def test_sp500_supplies_sectors_and_dates(self):
        members = get_constituents("sp500")
        assert sum(1 for c in members if c.sector) > len(members) * 0.95
        assert sum(1 for c in members if c.date_added) > len(members) * 0.8

    def test_nasdaq100_supplies_sectors_but_no_dates(self):
        # the source table has no "date added" column; empty is honest, and
        # better than inventing one
        members = get_constituents("nasdaq100")
        assert sum(1 for c in members if c.sector) > len(members) * 0.95
        assert all(c.date_added == "" for c in members)

    def test_describe_names_the_indexes_and_counts(self):
        text = describe(("nasdaq100",))
        assert "NASDAQ-100" in text and str(len(get_universe(("nasdaq100",)))) in text
        assert "S&P 500" in describe() and "NASDAQ-100" in describe()


class TestCsvFiles:
    @pytest.mark.parametrize("index", sorted(INDEXES))
    def test_file_exists_with_the_documented_header(self, index):
        path = paths.constituents_dir() / f"{index}.csv"
        assert path.exists(), f"{path} missing"
        with path.open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        assert header == ["ticker", "name", "sector", "date_added"]

    def test_the_shipped_files_are_what_the_loader_returns(self):
        """Guards against the loader silently dropping rows."""
        for index in INDEXES:
            path = paths.constituents_dir() / f"{index}.csv"
            with path.open(newline="", encoding="utf-8") as fh:
                rows = [r for r in csv.DictReader(fh) if (r.get("ticker") or "").strip()]
            assert len(rows) == len(get_constituents(index))
