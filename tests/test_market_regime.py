"""Deterministic regime classification + formatting.

All tests feed synthetic indicator dicts to `classify_regime` / `format_regime`.
No yfinance traffic — wire-level fetch is exercised by a network-marked smoke test.
"""

import pytest

from gpt_investor.data.market_regime import (
    classify_regime,
    format_regime,
    get_market_regime,
)


def _ind(value, delta_5d=0.0, direction="flat"):
    return {"value": value, "delta_5d": delta_5d, "direction": direction}


# --- classify_regime ------------------------------------------------------

def test_panic_opportunity_when_vix_extreme_and_inverted():
    indicators = {
        "vix":  _ind(42.0),
        "hyg":  _ind(70.0, direction="falling"),
    }
    assert classify_regime(indicators, curve=-0.5) == "panic-opportunity"


def test_recession_warning_when_inverted_credit_weakening():
    indicators = {
        "vix":  _ind(24.0),
        "hyg":  _ind(72.0, direction="falling"),
    }
    assert classify_regime(indicators, curve=-0.3) == "recession-warning"


def test_late_cycle_when_flat_curve_elevated_vix():
    indicators = {
        "vix":  _ind(22.0),
        "hyg":  _ind(78.0, direction="rising"),  # credit OK
    }
    assert classify_regime(indicators, curve=0.2) == "late-cycle-caution"


def test_late_cycle_also_triggers_on_mild_inversion_with_vol():
    # Inverted but vix isn't extreme enough for recession-warning AND
    # hyg isn't falling, so recession-warning rule doesn't fire. Late cycle does.
    indicators = {
        "vix":  _ind(22.0),
        "hyg":  _ind(78.0, direction="flat"),
    }
    assert classify_regime(indicators, curve=-0.1) == "late-cycle-caution"


def test_risk_on_bull_when_calm_and_steep_and_credit_rising():
    indicators = {
        "vix":  _ind(13.0),
        "hyg":  _ind(80.0, direction="rising"),
    }
    assert classify_regime(indicators, curve=1.4) == "risk-on-bull"


def test_mixed_when_calm_but_curve_flat():
    indicators = {
        "vix":  _ind(13.0),
        "hyg":  _ind(80.0, direction="rising"),
    }
    assert classify_regime(indicators, curve=0.3) == "mixed"


def test_mixed_when_calm_but_credit_not_rising():
    indicators = {
        "vix":  _ind(12.0),
        "hyg":  _ind(80.0, direction="flat"),
    }
    assert classify_regime(indicators, curve=1.4) == "mixed"


def test_mixed_when_required_inputs_missing():
    assert classify_regime({"vix": _ind(None), "hyg": _ind(80.0)}, curve=1.0) == "mixed"
    assert classify_regime({"vix": _ind(15.0), "hyg": _ind(80.0)}, curve=None) == "mixed"


def test_panic_takes_priority_over_recession_warning():
    indicators = {
        "vix":  _ind(40.0),
        "hyg":  _ind(60.0, direction="falling"),
    }
    # Both panic and recession rules would match; panic should win (ordered first).
    assert classify_regime(indicators, curve=-1.0) == "panic-opportunity"


# --- format_regime --------------------------------------------------------

def test_format_regime_renders_all_indicators_and_arrows():
    regime = {
        "as_of": "2026-05-24",
        "label": "risk-on-bull",
        "indicators": {
            "vix":  _ind(13.2, -0.4, "falling"),
            "tnx":  _ind(42.1, 0.1, "rising"),
            "irx":  _ind(40.5, -0.05, "falling"),
            "dxy":  _ind(104.1, -0.3, "falling"),
            "hyg":  _ind(78.4, 0.2, "rising"),
            "gold": _ind(2421.0, 5.0, "rising"),
        },
        "curve": 0.16,
        "curve_direction": "rising",
        "summary": "ignored by formatter",
    }
    out = format_regime(regime)
    assert "**Market Regime**: risk-on-bull" in out
    assert "VIX**: 13.20" in out
    assert "↓" in out  # falling vix arrow
    assert "10y−3m spread**: +0.16pp" in out
    assert "HY credit (HYG)" in out
    assert "USD index (DXY)" in out
    assert "Gold" in out


def test_format_regime_handles_missing_values():
    regime = {
        "label": "mixed",
        "indicators": {
            "vix":  _ind(None),
            "tnx":  _ind(None),
            "irx":  _ind(None),
            "dxy":  _ind(None),
            "hyg":  _ind(None),
            "gold": _ind(None),
        },
        "curve": None,
        "curve_direction": "unknown",
    }
    out = format_regime(regime)
    assert "n/a" in out
    assert "**Market Regime**: mixed" in out


# --- live wire test (skipped offline) -------------------------------------

@pytest.mark.network
def test_get_market_regime_live_smoke():
    regime = get_market_regime()
    assert regime["label"] in {
        "panic-opportunity", "recession-warning", "late-cycle-caution",
        "risk-on-bull", "mixed",
    }
    assert "vix" in regime["indicators"]
    assert "summary" in regime
