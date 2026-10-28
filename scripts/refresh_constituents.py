"""Rebuild data/constituents/*.csv from Wikipedia, showing what changed.

A maintenance tool, not part of the app: the running program only ever reads the
CSVs, so index membership changes land in a reviewable commit rather than
mutating under a live process.

    python scripts/refresh_constituents.py --dry-run   # show the diff only
    python scripts/refresh_constituents.py             # write if it looks sane
    python scripts/refresh_constituents.py --force     # write regardless

Two things worth knowing about the sources:

  * The Nasdaq-100 constituents are NOT on /wiki/Nasdaq-100 any more — that page
    lost its Components section. They live in a separate article,
    /wiki/List_of_NASDAQ-100_companies. Both pages happen to expose their table
    as id="constituents", which is the hook used here.
  * The two indexes do not describe themselves the same way. The S&P 500 table
    carries GICS sectors and a "Date added" column; the NASDAQ-100 table carries
    ICB industries and no date at all. `sector` is therefore comparable *within*
    an index, not across the two, and `date_added` is simply empty for the
    NASDAQ-100.

Parsing uses beautifulsoup4 with Python's stdlib html.parser, which adds no
runtime dependency (pandas.read_html would need lxml or html5lib, neither of
which this project installs).
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402
from data.symbols import normalize  # noqa: E402

# Wikipedia asks automated readers to identify themselves.
USER_AGENT = "nasdaq-advisor/1.0 (personal trading simulator; constituents refresh)"

TIMEOUT_S = 30

# A real reconstitution moves a handful of names. Losing more than this means the
# page was restructured and we are about to overwrite a good file with rubbish.
MAX_SHRINK = 0.10


@dataclass(frozen=True, slots=True)
class Source:
    index: str
    url: str
    ticker_col: int
    name_col: int
    sector_col: int
    date_col: int | None
    expected: int        # roughly how many members to expect
    sector_scheme: str


SOURCES = (
    Source(
        index="sp500",
        url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        ticker_col=0, name_col=1, sector_col=2, date_col=5,
        expected=503, sector_scheme="GICS",
    ),
    Source(
        index="nasdaq100",
        url="https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
        ticker_col=0, name_col=1, sector_col=3, date_col=None,
        expected=101, sector_scheme="ICB",
    ),
)

FIELDS = ("ticker", "name", "sector", "date_added")


def fetch_rows(source: Source) -> list[dict[str, str]]:
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(
        source.url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table", id="constituents")
    if table is None:
        raise SystemExit(
            f"{source.index}: no table#constituents at {source.url} — the page "
            f"layout changed; fix this script rather than the CSV"
        )

    rows: list[dict[str, str]] = []
    for tr in table.find_all("tr")[1:]:
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) <= source.ticker_col:
            continue
        raw = cells[source.ticker_col]
        if not raw:
            continue
        rows.append(
            {
                "ticker": normalize(raw),
                "name": _cell(cells, source.name_col),
                "sector": _cell(cells, source.sector_col),
                "date_added": _cell(cells, source.date_col),
            }
        )

    # de-duplicate while preserving source order (a company with two share
    # classes appears twice in some tables)
    seen: dict[str, dict[str, str]] = {}
    for row in rows:
        seen.setdefault(row["ticker"], row)
    return sorted(seen.values(), key=lambda r: r["ticker"])


def _cell(cells: list[str], index: int | None) -> str:
    if index is None or index >= len(cells):
        return ""
    return cells[index]


def read_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def diff(old: list[dict[str, str]], new: list[dict[str, str]]) -> tuple[list, list, list]:
    old_map = {r["ticker"]: r for r in old}
    new_map = {r["ticker"]: r for r in new}
    added = sorted(set(new_map) - set(old_map))
    removed = sorted(set(old_map) - set(new_map))
    renamed = sorted(
        t for t in set(old_map) & set(new_map)
        if old_map[t].get("name", "") != new_map[t].get("name", "")
    )
    return added, removed, renamed


def report(source: Source, old: list, new: list) -> bool:
    """Print the diff. Returns True when the new file looks safe to write."""
    added, removed, renamed = diff(old, new)
    print(f"\n=== {source.index} ({source.sector_scheme} sectors) ===")
    print(f"{source.url}")
    print(f"members: {len(old)} -> {len(new)}")

    if added:
        print(f"  added ({len(added)}):")
        for t in added:
            name = next(r["name"] for r in new if r["ticker"] == t)
            print(f"    + {t:<8} {name}")
    if removed:
        print(f"  removed ({len(removed)}):")
        for t in removed:
            name = next(r["name"] for r in old if r["ticker"] == t)
            print(f"    - {t:<8} {name}")
    if renamed:
        print(f"  renamed ({len(renamed)}):")
        for t in renamed:
            before = next(r["name"] for r in old if r["ticker"] == t)
            after = next(r["name"] for r in new if r["ticker"] == t)
            print(f"    ~ {t:<8} {before!r} -> {after!r}")
    if not (added or removed or renamed):
        print("  no changes")

    if not new:
        print("  REFUSING: scrape produced no rows")
        return False
    if old and len(new) < len(old) * (1 - MAX_SHRINK):
        print(
            f"  REFUSING: membership fell {1 - len(new) / len(old):.0%} "
            f"(>{MAX_SHRINK:.0%}); this looks like a page restructure, not a "
            f"reconstitution. Re-run with --force if it is genuine."
        )
        return False
    if abs(len(new) - source.expected) > source.expected * 0.1:
        print(f"  WARNING: expected ~{source.expected} members, got {len(new)}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="show the diff only")
    parser.add_argument(
        "--force", action="store_true", help="write even if the diff looks wrong"
    )
    parser.add_argument(
        "--index", choices=[s.index for s in SOURCES], help="refresh just one index"
    )
    args = parser.parse_args(argv)

    sources = [s for s in SOURCES if args.index in (None, s.index)]
    out_dir = paths.constituents_dir()
    failures = 0

    for source in sources:
        path = out_dir / f"{source.index}.csv"
        try:
            new = fetch_rows(source)
        except Exception as e:
            print(f"{source.index}: fetch failed: {e}", file=sys.stderr)
            failures += 1
            continue

        ok = report(source, read_existing(path), new)
        if args.dry_run:
            print(f"  (dry run — {path} not written)")
            continue
        if not ok and not args.force:
            failures += 1
            continue
        write_csv(path, new)
        print(f"  wrote {path} ({len(new)} rows)")

    if failures:
        print(f"\n{failures} index/indexes not refreshed", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
