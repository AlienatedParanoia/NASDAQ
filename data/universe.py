"""NASDAQ-100 constituents.

Hardcoded list as of 2026-07-04, cross-verified against Wikipedia and
Slickcharts (identical 101-symbol sets), with every membership change since
the December 2025 annual reconstitution confirmed via Nasdaq press releases:

  - 2025-12-22 annual reconstitution: +ALNY +FER +INSM +MPWR +STX +WDC,
    -BIIB -CDW -GFS -LULU -ON -TTD
  - 2026-01-20: WMT replaced AZN
  - 2026-04-20: SNDK replaced TEAM
  - 2026-05-18: LITE replaced CSGP
  - 2026-06-22 quarterly rebalance: +ALAB +CRWV +NBIS +RKLB +TER,
    -CHTR -CTSH -INSM -VRSK -ZS

101 symbols = 100 companies (Alphabet lists both GOOGL and GOOG). All symbols
are plain alphanumeric and identical in Yahoo Finance format.
"""

NASDAQ_100: tuple[str, ...] = (
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "ALAB", "ALNY",
    "AMAT", "AMD", "AMGN", "AMZN", "APP", "ARM", "ASML", "AVGO", "AXON",
    "BKNG", "BKR", "CCEP", "CDNS", "CEG", "CMCSA", "COST", "CPRT", "CRWD",
    "CRWV", "CSCO", "CSX", "CTAS", "DASH", "DDOG", "DXCM", "EA", "EXC",
    "FANG", "FAST", "FER", "FTNT", "GEHC", "GILD", "GOOG", "GOOGL", "HON",
    "IDXX", "INTC", "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LITE",
    "LRCX", "MAR", "MCHP", "MDLZ", "MELI", "META", "MNST", "MPWR", "MRVL",
    "MSFT", "MSTR", "MU", "NBIS", "NFLX", "NVDA", "NXPI", "ODFL", "ORLY",
    "PANW", "PAYX", "PCAR", "PDD", "PEP", "PLTR", "PYPL", "QCOM", "REGN",
    "RKLB", "ROP", "ROST", "SBUX", "SHOP", "SNDK", "SNPS", "STX", "TER",
    "TMUS", "TRI", "TSLA", "TTWO", "TXN", "VRTX", "WBD", "WDAY", "WDC",
    "WMT", "XEL",
)


def get_universe() -> list[str]:
    return list(NASDAQ_100)
