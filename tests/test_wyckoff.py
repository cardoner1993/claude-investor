"""Unit tests for the deterministic Wyckoff timing layer.

Two styles, both yfinance-free:
  - hand-crafted signal dicts -> known phase / score (classify_phase, score_wyckoff)
  - small synthetic OHLCV DataFrames -> feature extraction (compute_signals)
"""

import pandas as pd
import pytest

from gpt_investor.data.wyckoff import (
    compute_signals,
    classify_phase,
    score_wyckoff,
    format_wyckoff,
)


# --- signal-dict fixtures --------------------------------------------------

def _base_sig(**over) -> dict:
    """A neutral, fully-populated feature dict; override what a test needs."""
    sig = {
        "n_bars": 250,
        "last_close": 100.0, "sma50": 100.0, "sma200": 100.0,
        "pct_above_200d": 0.0, "trend": "flat",
        "dist_52w_high": -0.20, "dist_52w_low": 0.20,
        "vol_ratio": 1.0, "vol_surge": False, "up_down_vol_bias": 0.0,
        "range_pct": 0.18, "consolidating": False, "expanding": False,
        "breakout_up": False, "breakout_down": False, "making_new_low": False,
    }
    sig.update(over)
    return sig


# --- classify_phase --------------------------------------------------------

def test_phase_markup():
    sig = _base_sig(trend="up", pct_above_200d=0.10, dist_52w_high=-0.03, up_down_vol_bias=0.30)
    assert classify_phase(sig) == "markup"


def test_phase_markup_via_breakout():
    sig = _base_sig(trend="up", pct_above_200d=0.05, dist_52w_high=-0.15,
                    breakout_up=True, vol_surge=True, up_down_vol_bias=0.1)
    assert classify_phase(sig) == "markup"


def test_phase_distribution_negative_volume_at_highs():
    sig = _base_sig(trend="up", pct_above_200d=0.08, consolidating=True, up_down_vol_bias=-0.30)
    assert classify_phase(sig) == "distribution"


def test_phase_distribution_via_breakdown_from_highs():
    sig = _base_sig(trend="up", pct_above_200d=0.06, breakout_down=True, up_down_vol_bias=-0.05)
    assert classify_phase(sig) == "distribution"


def test_phase_markdown():
    sig = _base_sig(trend="down", pct_above_200d=-0.15, up_down_vol_bias=-0.40,
                    dist_52w_low=0.02, making_new_low=True)
    assert classify_phase(sig) == "markdown"


def test_phase_accumulation():
    sig = _base_sig(trend="flat", pct_above_200d=-0.05, consolidating=True,
                    up_down_vol_bias=0.0, making_new_low=False)
    assert classify_phase(sig) == "accumulation"


def test_accumulation_not_markdown_when_holding_above_low():
    # below 200d and flat, but range is tight and not making new lows -> base, not falling knife
    sig = _base_sig(trend="flat", pct_above_200d=-0.08, consolidating=True,
                    up_down_vol_bias=-0.05, making_new_low=False)
    assert classify_phase(sig) == "accumulation"


def test_phase_neutral_thin_history():
    assert classify_phase(_base_sig(n_bars=30)) == "neutral"


def test_phase_neutral_no_sma200():
    assert classify_phase(_base_sig(sma200=None)) == "neutral"


def test_phase_neutral_no_rule_fires():
    sig = _base_sig(trend="flat", pct_above_200d=0.0, consolidating=False)
    assert classify_phase(sig) == "neutral"


# --- score_wyckoff ---------------------------------------------------------

def test_score_markup_confirmed_is_strong():
    sig = _base_sig(trend="up", pct_above_200d=0.12, dist_52w_high=-0.02,
                    up_down_vol_bias=0.3, breakout_up=True, vol_surge=True)
    out = score_wyckoff(sig)
    assert out["phase"] == "markup"
    assert out["score"] >= 8.0
    assert out["tier"] == "Strong"


def test_score_markup_overextended_penalised():
    sig = _base_sig(trend="up", pct_above_200d=0.40, dist_52w_high=-0.01, up_down_vol_bias=0.3)
    out = score_wyckoff(sig)
    assert out["phase"] == "markup"
    assert "overextended" in out["flags"]
    assert out["score"] < 8.0          # 8 base - 2 overextended


def test_score_markdown_is_avoid():
    sig = _base_sig(trend="down", pct_above_200d=-0.20, up_down_vol_bias=-0.5,
                    dist_52w_low=0.01, making_new_low=True)
    out = score_wyckoff(sig)
    assert out["phase"] == "markdown"
    assert out["tier"] == "Avoid"
    assert "below 200d" in out["flags"]
    assert "making new lows" in out["flags"]


def test_score_accumulation_flags_no_volume():
    sig = _base_sig(trend="flat", pct_above_200d=-0.05, consolidating=True,
                    up_down_vol_bias=0.0, vol_surge=False)
    out = score_wyckoff(sig)
    assert out["phase"] == "accumulation"
    assert "no volume confirmation" in out["flags"]


def test_score_thin_history_capped_at_neutral():
    # would otherwise be markup ~9, but thin history can't confirm -> capped at 5
    sig = _base_sig(n_bars=80, trend="up", pct_above_200d=0.12, dist_52w_high=-0.02,
                    up_down_vol_bias=0.3, breakout_up=True, vol_surge=True)
    out = score_wyckoff(sig, phase="markup")
    assert out["score"] <= 5.0
    assert "thin history" in out["flags"]


def test_score_clamped_to_range():
    out = score_wyckoff(_base_sig(), phase="markdown")
    assert 0.0 <= out["score"] <= 10.0


def test_score_explicit_phase_overrides_classification():
    sig = _base_sig()  # would classify neutral
    assert score_wyckoff(sig, phase="markup")["phase"] == "markup"


# --- compute_signals on synthetic OHLCV ------------------------------------

def _ohlcv(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    vols = volumes if volumes is not None else [1_000_000.0] * n
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open":   closes,
            "High":   [c * 1.01 for c in closes],
            "Low":    [c * 0.99 for c in closes],
            "Close":  closes,
            "Volume": vols,
        },
        index=idx,
    )


def test_compute_signals_uptrend():
    closes = [100.0 + i * 0.5 for i in range(260)]      # steady climb
    sig = compute_signals(_ohlcv(closes))
    assert sig["n_bars"] == 260
    assert sig["sma50"] is not None and sig["sma200"] is not None
    assert sig["trend"] == "up"
    assert sig["pct_above_200d"] > 0
    assert sig["dist_52w_high"] is not None and sig["dist_52w_high"] <= 0


def test_compute_signals_downtrend():
    closes = [200.0 - i * 0.5 for i in range(260)]      # steady decline
    sig = compute_signals(_ohlcv(closes))
    assert sig["trend"] == "down"
    assert sig["pct_above_200d"] < 0
    assert sig["making_new_low"] is True


def test_compute_signals_volume_surge_and_bias():
    closes = [100.0 + i * 0.3 for i in range(260)]
    vols = [1_000_000.0] * 255 + [5_000_000.0] * 5    # recent surge on up days
    sig = compute_signals(_ohlcv(closes, vols))
    assert sig["vol_ratio"] is not None and sig["vol_ratio"] > 1.5
    assert sig["vol_surge"] is True
    assert sig["up_down_vol_bias"] is not None and sig["up_down_vol_bias"] > 0


def test_compute_signals_empty_df_safe():
    sig = compute_signals(None)
    assert sig["n_bars"] == 0
    assert classify_phase(sig) == "neutral"


def test_compute_signals_short_history_no_sma200():
    sig = compute_signals(_ohlcv([100.0 + i for i in range(60)]))
    assert sig["sma200"] is None
    assert classify_phase(sig) == "neutral"


# --- format ----------------------------------------------------------------

def test_format_wyckoff_contains_phase_and_tier():
    out = score_wyckoff(_base_sig(trend="up", pct_above_200d=0.1, dist_52w_high=-0.02,
                                  up_down_vol_bias=0.3))
    block = format_wyckoff(out)
    assert "Wyckoff timing" in block
    assert out["phase"] in block
    assert out["tier"] in block
