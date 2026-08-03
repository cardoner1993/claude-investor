from datetime import date

from gpt_investor.data import signals as s


def test_score_short_interest():
    assert s.score_short_interest(0.20)["crowded"] is True
    assert s.score_short_interest(0.05)["crowded"] is False
    assert s.score_short_interest(None) == {"short_pct": None, "crowded": False}
    assert s.score_short_interest(float("nan"))["short_pct"] is None


def test_days_to_earnings_and_banner():
    today = date(2026, 8, 3)
    dates = [date(2026, 7, 1), date(2026, 8, 6), date(2026, 11, 1)]  # past + near + far
    assert s.days_to_earnings(dates, today) == 3
    assert s.days_to_earnings([date(2026, 7, 1)], today) is None  # all past
    assert s.days_to_earnings([], today) is None
    assert s.earnings_banner(3) == "EARNINGS IN 3D"
    assert s.earnings_banner(0) == "EARNINGS TODAY"
    assert s.earnings_banner(10) == ""
    assert s.earnings_banner(None) == ""


def test_net_insider_flow():
    today = date(2026, 8, 3)
    rows = [
        {"value": 1_000_000, "date": date(2026, 8, 1), "transaction": "Purchase"},
        {"value": 400_000, "date": date(2026, 7, 20), "transaction": "Sale"},
        {"value": 100_000, "date": date(2026, 6, 1), "transaction": "Purchase"},  # inside 90d, outside 30d
        {"value": 500_000, "date": date(2026, 8, 2), "transaction": "Gift"},      # unsigned
    ]
    assert s.net_insider_flow(rows, today, 30) == 600_000.0   # +1M buy - 400k sell
    assert s.net_insider_flow(rows, today, 90) == 700_000.0   # + the Jun buy
    assert s.net_insider_flow([], today, 30) is None
    # a future-dated row is ignored
    assert s.net_insider_flow([{"value": 1, "date": date(2026, 9, 1), "transaction": "Purchase"}], today, 30) is None


def test_cagr():
    assert s.cagr(100, 200, 1) == 1.0            # doubled in a year
    assert s.cagr(100, 100, 3) == 0.0
    assert s.cagr(None, 200, 2) is None
    assert s.cagr(-5, 200, 2) is None            # undefined through zero/negative
    assert s.cagr(100, 200, 0) is None


def test_margin_slope():
    assert s.margin_slope([0.1, 0.2, 0.3]) > 0    # expanding
    assert s.margin_slope([0.3, 0.2, 0.1]) < 0    # contracting
    assert s.margin_slope([0.2]) is None
    assert s.margin_slope([]) is None


def test_peer_relative():
    r = s.peer_relative(10, [20, 30, 40])   # own cheaper than median 30
    assert r["median"] == 30
    assert r["ratio"] < 1.0
    assert r["cheaper"] is True
    assert s.peer_relative(None, [10, 20]) == {"median": None, "ratio": None, "cheaper": None}
    assert s.peer_relative(10, [])["median"] is None


def test_row_helpers_handle_nan_alignment():
    import pandas as pd

    # newest-first columns; Operating Income has a NaN in the middle period
    df = pd.DataFrame(
        {"2025": [100.0, 20.0], "2024": [90.0, float("nan")], "2023": [80.0, 16.0]},
        index=["Total Revenue", "Operating Income"],
    )
    assert s._row(df, "Total Revenue") == [100.0, 90.0, 80.0]
    assert s._row(df, "Operating Income") == [20.0, 16.0]        # NaN dropped
    assert s._raw_row(df, "Operating Income") == [20.0, None, 16.0]  # NaN kept as None
    assert s._row(df, "Missing Line") is None
    # column-aligned pairing skips only the NaN period, not a shifted year
    oi_raw, rev_raw = s._raw_row(df, "Operating Income"), s._raw_row(df, "Total Revenue")
    pairs = [(oi, rv) for oi, rv in zip(oi_raw, rev_raw) if oi is not None and rv]
    assert pairs == [(20.0, 100.0), (16.0, 80.0)]


def test_format_signals_smoke():
    sig = {
        "short": {"short_pct": 0.18, "crowded": True},
        "earnings_days": 3, "earnings_banner": "EARNINGS IN 3D",
        "insider_30d": 600000.0, "insider_90d": -200000.0,
        "rev_cagr": 0.12, "fcf_cagr": 0.08, "op_margin_slope": 0.01,
        "peers": {"pe_median": 22.0, "n": 6, "pe_rel": {"ratio": 0.8, "cheaper": True}},
    }
    out = s.format_signals(sig)
    assert "crowded short" in out
    assert "EARNINGS IN 3D" in out
    assert "Peer valuation" in out
