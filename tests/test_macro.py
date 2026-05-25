"""Deterministic macro snapshot — stance classification, delta math, formatting.

Live API fetches (FRED, ECB SDW, PBOC scrape) are gated by `@pytest.mark.network`.
"""

import pytest

from gpt_investor.data.macro import (
    _classify_stance,
    _delta_90d,
    format_liquidity,
    get_liquidity_snapshot,
    snapshot_is_complete,
)


# --- _classify_stance -----------------------------------------------------

@pytest.mark.parametrize("delta,expected", [
    (-0.50, "easing"),
    (-0.06, "easing"),
    (-0.05, "neutral"),   # exactly at eps → neutral (strict-less-than)
    (0.0,   "neutral"),
    (0.05,  "neutral"),
    (0.06,  "tightening"),
    (1.25,  "tightening"),
    (None,  "unknown"),
])
def test_classify_stance(delta, expected):
    assert _classify_stance(delta) == expected


# --- _delta_90d -----------------------------------------------------------

def test_delta_90d_basic():
    # desc-sorted: latest first
    series = [
        ("2026-05-20", 4.30),
        ("2026-04-20", 4.50),  # 30d prior — should NOT match
        ("2026-02-15", 5.00),  # ~95d prior — should match
        ("2026-01-01", 5.25),
    ]
    assert _delta_90d(series) == pytest.approx(-0.70)


def test_delta_90d_no_old_enough_observation():
    series = [
        ("2026-05-20", 4.30),
        ("2026-05-10", 4.32),
        ("2026-04-25", 4.40),  # only 25d back
    ]
    assert _delta_90d(series) is None


def test_delta_90d_empty():
    assert _delta_90d([]) is None


def test_delta_90d_bad_dates_skipped():
    series = [
        ("2026-05-20", 4.30),
        ("not-a-date", 4.40),
        ("2026-01-15", 5.00),
    ]
    assert _delta_90d(series) == pytest.approx(-0.70)


# --- format_liquidity -----------------------------------------------------

def _leg(bank, region, rate, stance, delta, url, as_of="2026-05-20"):
    return {
        "bank": bank,
        "region": region,
        "rate_pct": rate,
        "rate_label": f"{rate:.2f}%" if rate is not None else "n/a",
        "delta_90d_pp": delta,
        "stance": stance,
        "as_of": as_of,
        "source_url": url,
    }


def test_format_liquidity_full_snapshot():
    snap = {
        "as_of": "2026-05-24",
        "banks": [
            _leg("Fed", "US", 4.33, "easing", -0.50, "https://fred.stlouisfed.org/series/DFF"),
            _leg("ECB", "EU", 2.40, "neutral", 0.0, "https://data.ecb.europa.eu/data/datasets/FM"),
            _leg("PBOC", "China", 3.10, "easing", -0.20, "https://www.chinamoney.com.cn/english/bmkilrlpr/"),
        ],
    }
    out = format_liquidity(snap)
    assert "**Global Liquidity Snapshot**" in out
    assert "Fed (US)**: 4.33% — easing (-0.50pp 90d)" in out
    assert "ECB (EU)**: 2.40% — neutral" in out
    assert "(+0.00pp 90d)" in out
    assert "PBOC (China)**: 3.10% — easing (-0.20pp 90d)" in out
    assert "https://fred.stlouisfed.org/series/DFF" in out


def test_format_liquidity_handles_missing_leg():
    snap = {
        "as_of": "2026-05-24",
        "banks": [
            _leg("Fed", "US", None, "unknown", None,
                 "https://fred.stlouisfed.org/series/DFF", as_of=None),
        ],
    }
    out = format_liquidity(snap)
    # rate_label was set from `rate_pct: None` via the `_leg` helper, so it would
    # crash; emulate the real fetcher path that sets "n/a" instead.
    snap["banks"][0]["rate_label"] = "n/a"
    out = format_liquidity(snap)
    assert "Fed (US)**: n/a — unknown" in out
    # No "(... 90d)" suffix when delta is None
    assert "pp 90d" not in out


# --- snapshot_is_complete ------------------------------------------------

def test_complete_snapshot_passes():
    snap = {"banks": [
        _leg("Fed",  "US",    4.33, "easing",     -0.5, "u1"),
        _leg("ECB",  "EU",    2.40, "neutral",     0.0, "u2"),
        _leg("PBOC", "China", 3.10, "easing",     -0.2, "u3"),
    ]}
    assert snapshot_is_complete(snap) is True


def test_snapshot_with_none_rate_is_incomplete():
    snap = {"banks": [
        _leg("Fed",  "US",    None, "unknown",  None, "u1"),
        _leg("ECB",  "EU",    2.40, "neutral",   0.0, "u2"),
        _leg("PBOC", "China", 3.10, "easing",   -0.2, "u3"),
    ]}
    assert snapshot_is_complete(snap) is False


def test_snapshot_with_nan_rate_is_incomplete():
    snap = {"banks": [
        _leg("Fed",  "US",    float("nan"), "unknown", None, "u1"),
        _leg("ECB",  "EU",    2.40,         "neutral",  0.0, "u2"),
        _leg("PBOC", "China", 3.10,         "easing",  -0.2, "u3"),
    ]}
    assert snapshot_is_complete(snap) is False


def test_empty_snapshot_is_incomplete():
    assert snapshot_is_complete({"banks": []}) is False
    assert snapshot_is_complete({}) is False


# --- live smoke (skipped offline) -----------------------------------------

@pytest.mark.network
def test_get_liquidity_snapshot_live_smoke():
    snap = get_liquidity_snapshot()
    assert "banks" in snap
    assert {leg["bank"] for leg in snap["banks"]} == {"Fed", "ECB", "PBOC"}
    md = format_liquidity(snap)
    assert "Global Liquidity Snapshot" in md
