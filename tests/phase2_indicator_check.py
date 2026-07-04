"""Phase 2 sanity check, part 1 (offline): indicator correctness.

Vectorized implementations are compared against straightforward reference
loops (independent implementations of the same Wilder recurrences), plus
behavioural checks on synthetic series.

Run from the project root: python -m tests.phase2_indicator_check
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies import indicators as ind


def _ref_rsi(close: pd.Series, period: int = 14) -> float:
    """Textbook Wilder RSI via explicit loop; returns the last value."""
    delta = close.diff().to_numpy()[1:]
    gains = np.clip(delta, 0, None)
    losses = np.clip(-delta, 0, None)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _ref_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Textbook Wilder ADX via explicit ewm-equivalent smoothing loop."""
    h, l, c = high.to_numpy(), low.to_numpy(), close.to_numpy()
    n = len(c)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    alpha = 1.0 / period

    def smooth(x):
        s = x[1]
        out = [s]
        for v in x[2:]:
            s = s + alpha * (v - s)
            out.append(s)
        return np.array(out)

    s_tr, s_p, s_m = smooth(tr), smooth(plus_dm), smooth(minus_dm)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_di = 100 * s_p / s_tr
        m_di = 100 * s_m / s_tr
        dx = 100 * np.abs(p_di - m_di) / (p_di + m_di)
    dx = np.nan_to_num(dx)
    adx = dx[0]
    for v in dx[1:]:
        adx = adx + alpha * (v - adx)
    return float(adx)


def main() -> None:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2024-01-01", periods=400)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 400))), index=idx)
    high = close * (1 + np.abs(rng.normal(0, 0.01, 400)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, 400)))

    # vectorized vs reference implementations on a random walk
    rsi_vec = float(ind.rsi_wilder(close, 14).iloc[-1])
    rsi_ref = _ref_rsi(close, 14)
    assert abs(rsi_vec - rsi_ref) < 0.5, f"RSI mismatch: {rsi_vec} vs ref {rsi_ref}"
    print(f"RSI vectorized {rsi_vec:.2f} vs reference {rsi_ref:.2f} OK")

    adx_vec = float(ind.adx_wilder(high, low, close, 14).iloc[-1])
    adx_ref = _ref_adx(high, low, close, 14)
    assert abs(adx_vec - adx_ref) < 1.0, f"ADX mismatch: {adx_vec} vs ref {adx_ref}"
    print(f"ADX vectorized {adx_vec:.2f} vs reference {adx_ref:.2f} OK")

    # behavioural checks
    rising = pd.Series(np.arange(1.0, 101.0), index=pd.bdate_range("2024-01-01", periods=100))
    assert float(ind.rsi_wilder(rising).iloc[-1]) > 99, "RSI of monotonic rise should be ~100"

    flat = pd.Series(50.0, index=rising.index)
    bb = ind.bollinger(flat)
    assert float(bb["upper"].iloc[-1]) == float(bb["lower"].iloc[-1]) == 50.0
    assert float(ind.annualized_vol(flat, 50).iloc[-1]) == 0.0
    assert float(ind.rsi_wilder(flat).iloc[-1]) == 50.0, "flat RSI convention is 50"

    trending_high = rising * 1.01
    trending_low = rising * 0.99
    adx_trend = float(ind.adx_wilder(trending_high, trending_low, rising).iloc[-1])
    assert adx_trend > 25, f"ADX of a clean monotonic trend should be high, got {adx_trend}"

    ema_flat = ind.ema(flat, 20)
    assert abs(float(ema_flat.iloc[-1]) - 50.0) < 1e-9

    a = pd.Series([1, 2, 3, 4], dtype=float)
    b = pd.Series([3, 3, 3, 3], dtype=float)
    x = ind.crossover(a, b)
    assert list(x) == [False, False, False, True], f"crossover wrong: {list(x)}"

    print("PHASE 2 INDICATOR CHECKS PASSED")


if __name__ == "__main__":
    main()
