"""Unit tests for the deterministic support/resistance levels layer.

Synthetic OHLCV frames -> known level/score outputs. No yfinance.
"""

import pandas as pd

from gpt_investor.data.levels import (
    find_pivots,
    cluster_levels,
    build_levels,
    nearest_levels,
    score_levels,
    format_levels,
    chip_label,
    _round_number,
)


def _frame(highs, lows, closes=None, volumes=None) -> pd.DataFrame:
    """Build an OHLCV DataFrame from high/low series; close defaults mid-range."""
    n = len(highs)
    closes = closes if closes is not None else [(h + l) / 2 for h, l in zip(highs, lows)]
    volumes = volumes if volumes is not None else [1_000_000] * n
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=idx,
    )


def _oscillating(n=250, low=90.0, high=110.0):
    """A clean sawtooth between `low` and `high` — repeated pivots at both ends."""
    highs, lows, closes = [], [], []
    for i in range(n):
        up = (i // 5) % 2 == 0
        base = high if up else low
        highs.append(base + 1.0)
        lows.append(base - 1.0)
        closes.append(base)
    return _frame(highs, lows, closes)


# --- pivots ---------------------------------------------------------------

def test_find_pivots_empty_on_short_history():
    df = _frame([100] * 5, [99] * 5)
    assert find_pivots(df) == []
    assert find_pivots(None) == []


def test_find_pivots_detects_isolated_high_and_low():
    highs = [10, 10, 10, 10, 10, 20, 10, 10, 10, 10, 10]
    lows = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    df = _frame(highs, lows)
    kinds = {(p["kind"], p["price"]) for p in find_pivots(df, lookback=5)}
    assert ("high", 20) in kinds


def test_find_pivots_low_detection():
    highs = [11] * 11
    lows = [10, 10, 10, 10, 10, 2, 10, 10, 10, 10, 10]
    df = _frame(highs, lows)
    kinds = {(p["kind"], p["price"]) for p in find_pivots(df, lookback=5)}
    assert ("low", 2) in kinds


# --- clustering -----------------------------------------------------------

def test_cluster_merges_nearby_pivots():
    pivots = [
        {"price": 100.0, "kind": "high", "volume": 1.0},
        {"price": 100.5, "kind": "high", "volume": 1.0},   # within 2% -> merge
        {"price": 130.0, "kind": "high", "volume": 1.0},   # separate
    ]
    levels = cluster_levels(pivots)
    assert len(levels) == 2
    merged = min(levels, key=lambda x: x["price"])
    assert merged["touches"] == 2
    assert 100.0 <= merged["price"] <= 100.5


def test_cluster_strength_rises_with_touches():
    many = [{"price": 100.0 + 0.1 * i, "kind": "low", "volume": 1.0} for i in range(4)]
    one = [{"price": 200.0, "kind": "high", "volume": 1.0}]
    levels = cluster_levels(many + one)
    strong = max(levels, key=lambda x: x["strength"])
    weak = min(levels, key=lambda x: x["strength"])
    assert strong["touches"] > weak["touches"]
    assert strong["strength"] > weak["strength"]


def test_cluster_empty():
    assert cluster_levels([]) == []


# --- build_levels ---------------------------------------------------------

def test_build_levels_includes_smas_and_extremes():
    df = _oscillating()
    levels = build_levels(df, price=100.0)
    sources = " ".join(lv.get("source", "") for lv in levels)
    assert "SMA50" in sources and "SMA200" in sources
    assert "52w high" in sources and "52w low" in sources
    assert levels == sorted(levels, key=lambda x: x["price"])


def test_build_levels_empty_on_thin_history():
    df = _frame([100] * 30, [99] * 30)
    assert build_levels(df) == []


# --- nearest_levels -------------------------------------------------------

def test_nearest_levels_picks_closest_each_side():
    levels = [
        {"price": 90.0, "strength": 3.0},
        {"price": 95.0, "strength": 2.0},
        {"price": 105.0, "strength": 4.0},
        {"price": 120.0, "strength": 1.0},
    ]
    near = nearest_levels(levels, 100.0)
    assert near["support"]["price"] == 95.0
    assert near["resistance"]["price"] == 105.0
    assert round(near["pct_to_support"], 3) == 0.05
    assert round(near["pct_to_resistance"], 3) == 0.05


def test_nearest_levels_handles_missing_side():
    levels = [{"price": 50.0, "strength": 2.0}]
    near = nearest_levels(levels, 100.0)
    assert near["resistance"] is None
    assert near["pct_to_resistance"] is None
    assert near["support"]["price"] == 50.0


# --- score_levels ---------------------------------------------------------

def test_score_no_levels_is_neutral():
    df = _frame([100] * 30, [99] * 30)   # too thin -> no structure
    scored = score_levels(df)
    assert scored["score"] == 5.0
    assert "no clear levels" in scored["flags"]


def test_score_good_reward_to_risk_beats_poor():
    df = _oscillating()
    # just above support (near 90) with lots of room up to resistance (near 110)
    good = score_levels(df, price=92.0)
    # just under resistance (near 110) with a long drop to support
    poor = score_levels(df, price=108.0)
    assert good["reward_to_risk"] is not None and poor["reward_to_risk"] is not None
    assert good["reward_to_risk"] > poor["reward_to_risk"]
    assert good["score"] > poor["score"]


def test_score_pinned_under_resistance_flagged():
    df = _oscillating()   # top ~111 touched many times -> strong resistance
    scored = score_levels(df, price=109.5)
    assert scored["resistance"] is not None
    assert "pinned under resistance" in scored["flags"]
    assert scored["score"] <= 5.0


def test_score_stays_in_bounds():
    df = _oscillating()
    for p in (85.0, 92.0, 100.0, 108.0, 115.0):
        s = score_levels(df, price=p)["score"]
        assert 0.0 <= s <= 10.0


def test_score_no_support_below_penalised():
    df = _oscillating()
    scored = score_levels(df, price=80.0)   # below the whole range
    assert scored["support"] is None
    assert "no support below" in scored["flags"]


# --- round numbers & formatting ------------------------------------------

def test_round_number_scales_with_magnitude():
    assert _round_number(7.3) == 7.5
    assert _round_number(103.0) == 105.0
    assert _round_number(742.0) == 740.0


def test_format_and_chip_render():
    scored = score_levels(_oscillating(), price=100.0)
    md = format_levels(scored)
    assert "Price levels:" in md and "Reward-to-risk" in md
    chip = chip_label(scored)
    assert chip.startswith("S ") and "/ R " in chip


def test_chip_handles_missing_sides():
    scored = {"support": None, "resistance": None}
    assert chip_label(scored) == "S — / R —"
