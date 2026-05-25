"""Unit tests for the deterministic fundamental scorer.

Hand-crafted metric dicts -> known score outputs. No yfinance.
"""

import pytest

from gpt_investor.data.fundamentals import (
    score_fundamentals,
    format_fundamentals,
    _score_pe,
    _score_growth,
    _score_roe,
    _score_debt_equity,
    _score_fcf_margin,
)


def _solid_company() -> dict:
    """A boring high-quality compounder: cheap-ish, growing, profitable, cash-generative, low debt."""
    return {
        "trailing_pe":      16.0,
        "forward_pe":       14.0,
        "price_to_book":    2.5,
        "ev_ebitda":        11.0,
        "revenue_growth":   0.12,
        "earnings_growth":  0.18,
        "roe":              0.22,
        "operating_margin": 0.20,
        "profit_margin":    0.15,
        "gross_margin":     0.45,
        "free_cashflow":    8_000_000_000,
        "revenue":          50_000_000_000,   # 16% FCF margin
        "debt_to_equity":   45.0,             # 0.45x
        "trailing_eps":     3.5,
        "market_cap":       300_000_000_000,
    }


def _stretched_growth() -> dict:
    """Expensive, fast-growing, FCF-negative, leveraged. Classic stretched growth."""
    return {
        "trailing_pe":      80.0,
        "forward_pe":       60.0,
        "price_to_book":    9.0,
        "ev_ebitda":        45.0,
        "revenue_growth":   0.40,
        "earnings_growth":  -0.20,
        "roe":              -0.05,
        "operating_margin": -0.10,
        "profit_margin":    -0.15,
        "gross_margin":     0.55,
        "free_cashflow":    -500_000_000,
        "revenue":          5_000_000_000,
        "debt_to_equity":   250.0,
        "trailing_eps":     -1.8,
        "market_cap":       50_000_000_000,
    }


# --- per-dimension boundaries ---------------------------------------------

@pytest.mark.parametrize("pe,expected", [
    (None, 0.0),
    (-5,   0.0),
    (10,   10.0),
    (15,   8.0),
    (22,   6.0),
    (30,   4.0),
    (45,   2.0),
    (80,   0.0),
])
def test_score_pe_buckets(pe, expected):
    assert _score_pe(pe) == expected


@pytest.mark.parametrize("g,expected", [
    (0.30,  10.0),
    (0.20,  8.0),
    (0.10,  6.0),
    (0.03,  4.0),
    (-0.03, 2.0),
    (-0.10, 0.0),
    (None,  5.0),
])
def test_score_growth_buckets(g, expected):
    assert _score_growth(g) == expected


@pytest.mark.parametrize("roe,expected", [
    (0.30,  10.0),
    (0.20,  8.0),
    (0.14,  6.0),
    (0.07,  4.0),
    (0.01,  2.0),
    (-0.05, 0.0),
    (None,  5.0),
])
def test_score_roe_buckets(roe, expected):
    assert _score_roe(roe) == expected


def test_score_debt_equity_normalises_percentage():
    # 45 in yfinance == 0.45x ratio -> "<0.6" bucket = 8.0
    score, ratio = _score_debt_equity(45.0)
    assert score == 8.0
    assert ratio == pytest.approx(0.45)


def test_score_debt_equity_handles_missing():
    score, ratio = _score_debt_equity(None)
    assert score == 5.0
    assert ratio is None


def test_score_fcf_margin_negative_fcf_zero():
    score, m = _score_fcf_margin(-1_000_000, 10_000_000)
    assert score == 0.0
    assert m == pytest.approx(-0.1)


def test_score_fcf_margin_missing_revenue_zero():
    score, m = _score_fcf_margin(1_000_000, None)
    assert score == 0.0
    assert m is None


# --- composite ------------------------------------------------------------

def test_solid_company_lands_in_solid_or_strong_tier():
    s = score_fundamentals(_solid_company())
    assert s["score"] >= 6.0
    assert s["tier"] in ("Solid", "Strong")
    assert s["flags"] == []  # nothing to warn about


def test_stretched_growth_drops_below_average():
    s = score_fundamentals(_stretched_growth())
    assert s["score"] < 5.0
    assert s["tier"] in ("Average", "Weak", "Avoid")
    # should flag at least these issues
    assert "negative FCF" in s["flags"]
    assert "premium valuation" in s["flags"]
    assert "high leverage" in s["flags"]
    assert "earnings loss" in s["flags"]


def test_score_dimensions_present_and_in_range():
    s = score_fundamentals(_solid_company())
    for dim in ("valuation", "growth", "profitability", "cash", "balance"):
        assert dim in s["dimensions"]
        v = s["dimensions"][dim]["score"]
        assert 0.0 <= v <= 10.0


def test_score_handles_all_none_metrics():
    s = score_fundamentals({})
    # composite must still be a number, not crash
    assert isinstance(s["score"], float)
    assert 0.0 <= s["score"] <= 10.0
    assert s["tier"] in ("Strong", "Solid", "Average", "Weak", "Avoid")


def test_format_fundamentals_is_markdown():
    s = score_fundamentals(_solid_company())
    out = format_fundamentals(s)
    assert "Fundamental score:" in out
    assert "Valuation" in out
    assert "Growth" in out
    assert "Profitability" in out
    assert "Cash" in out
    assert "Balance" in out


def test_format_fundamentals_renders_flags_when_present():
    s = score_fundamentals(_stretched_growth())
    out = format_fundamentals(s)
    assert "Flags:" in out
    assert "negative FCF" in out
