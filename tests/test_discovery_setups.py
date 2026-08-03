import pytest

from gpt_investor.data import discovery as d


def test_is_equity_symbol():
    assert d._is_equity_symbol("AAPL")
    assert d._is_equity_symbol("BRK-B")   # dash-class equities allowed
    assert not d._is_equity_symbol("GC=F")   # futures
    assert not d._is_equity_symbol("^GSPC")  # index
    assert not d._is_equity_symbol("")


def test_is_equity_quote():
    assert d._is_equity_quote({"quoteType": "EQUITY", "symbol": "MSFT"})
    assert not d._is_equity_quote({"quoteType": "CRYPTOCURRENCY", "symbol": "BTC-USD"})
    assert not d._is_equity_quote({"quoteType": "EQUITY", "symbol": "^VIX"})


def test_dollar_volume_prefers_regular_then_avg():
    assert d._dollar_volume({"regularMarketPrice": 10, "regularMarketVolume": 100}) == 1000
    assert d._dollar_volume({"regularMarketPrice": 10, "averageDailyVolume3Month": 50}) == 500
    assert d._dollar_volume({}) == 0


def test_prefilter_ranks_by_dollar_volume_and_includes_trending():
    quotes = {
        "AAA": {"symbol": "AAA", "regularMarketPrice": 1, "regularMarketVolume": 10},      # 10
        "BBB": {"symbol": "BBB", "regularMarketPrice": 100, "regularMarketVolume": 1000},  # 100000
        "CCC": {"symbol": "CCC", "regularMarketPrice": 5, "regularMarketVolume": 5},        # 25
    }
    out = d._prefilter_by_dollar_volume(quotes, trending=["ZZZ"], top=4)
    assert out[0] == "ZZZ"        # trending force-included first
    assert out[1] == "BBB"        # then highest dollar-volume
    assert out[-1] == "AAA"       # lowest dollar-volume last
    # cap respected: top=2 keeps trending + best quote, drops the rest
    assert d._prefilter_by_dollar_volume(quotes, trending=["ZZZ"], top=2) == ["ZZZ", "BBB"]


def test_regime_fit_rewards_alignment():
    # markup in a bull regime is the best fit; markdown the worst
    assert d._regime_fit("Solid", "markup", "risk-on-bull") > 0
    assert d._regime_fit("Solid", "markdown", "risk-on-bull") < 0
    # accumulation is favoured in a washout
    assert d._regime_fit("Solid", "accumulation", "panic-opportunity") > 0
    # unknown regime falls back to the "mixed" table, never raises
    assert d._regime_fit("Solid", "markup", None) == d._REGIME_FIT["mixed"]["markup"]


def _mk(fund_score, tier, wy_score, phase):
    return {"score": fund_score, "tier": tier}, {"score": wy_score, "phase": phase}


def test_setup_score_and_ranking():
    strong_accum = _mk(8.0, "Strong", 7.0, "accumulation")
    weak_markdown = _mk(2.0, "Weak", 1.0, "markdown")
    s1 = d._setup_score(*strong_accum, "risk-on-bull")
    s2 = d._setup_score(*weak_markdown, "risk-on-bull")
    assert s1 > s2
    # soft gate adds +1 for Solid/Strong AND accumulation/markup
    ungated = d._setup_score({"score": 8.0, "tier": "Average"}, {"score": 7.0, "phase": "accumulation"}, "risk-on-bull")
    gated = d._setup_score({"score": 8.0, "tier": "Strong"}, {"score": 7.0, "phase": "accumulation"}, "risk-on-bull")
    assert gated - ungated == pytest.approx(1.0)


def test_why_chip_format():
    fund, wy = _mk(8.0, "Strong", 8.0, "markup")
    why = d._why_chip(fund, wy, "risk-on-bull")
    assert why == "Strong • markup • +regime"
    assert d._why_chip(fund, {"score": 1.0, "phase": "markdown"}, "risk-on-bull").endswith("-regime")


@pytest.mark.network
def test_get_setup_candidates_smoke():
    out = d.get_setup_candidates(num=2)
    assert isinstance(out, dict)
    for _, entry in out.items():
        assert {"status", "fund", "wyckoff", "setup_score", "why"} <= set(entry)
