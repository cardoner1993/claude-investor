"""Ticker resolver: scoring + dot-ticker support.

Live yfinance call is marked `network`; offline tests exercise the pure
scoring function against synthetic search results.
"""

import pytest

from gpt_investor.data.discovery import _score_quote, resolve_ticker_verbose, resolve_ticker


def _q(symbol, longname=None, shortname=None, quotetype="EQUITY", exchange="NMS"):
    out = {"symbol": symbol, "quoteType": quotetype, "exchange": exchange}
    if longname:
        out["longname"] = longname
    if shortname:
        out["shortname"] = shortname
    return out


# --- _score_quote ---------------------------------------------------------

def test_score_exact_ticker_match_dominates():
    s = _score_quote("AAPL", _q("AAPL", longname="Apple Inc."))
    assert s >= 1.0


def test_score_dot_ticker_base_match_strong():
    # Searching "ACS" should reward ACS.MC strongly via the base-symbol bonus
    s_match = _score_quote("ACS", _q("ACS.MC", longname="ACS Actividades de Construccion y Servicios SA"))
    s_other = _score_quote("ACS", _q("GGAL", longname="Grupo Financiero Galicia SA"))
    assert s_match > s_other


def test_score_company_name_full_text_wins_over_unrelated_dotless():
    """Long-form company name should beat a random dotless equity."""
    s_match = _score_quote(
        "ACS Actividades de Construccion",
        _q("ACS.MC", longname="ACS Actividades de Construccion y Servicios SA"),
    )
    s_random = _score_quote(
        "ACS Actividades de Construccion",
        _q("GGAL", longname="Grupo Financiero Galicia SA"),
    )
    assert s_match > s_random


def test_score_empty_quote_returns_zero():
    assert _score_quote("AAPL", _q("", quotetype="EQUITY")) == 0.0


# --- resolve_ticker_verbose (mocked yfinance) -----------------------------

@pytest.fixture
def fake_search(monkeypatch):
    """Replace yfinance.Search with a stub that returns predetermined quotes."""
    captured = {}

    class _FakeSearch:
        def __init__(self, query, max_results=10):
            captured["query"] = query
            captured["max_results"] = max_results
            self.quotes = captured.get("quotes", [])

    import gpt_investor.data.discovery as d
    monkeypatch.setattr(d.yf, "Search", _FakeSearch)
    return captured


def test_resolve_picks_best_name_match_not_first_dotless(fake_search):
    fake_search["quotes"] = [
        _q("GGAL", longname="Grupo Financiero Galicia SA"),                # dotless but irrelevant
        _q("ACS.MC", longname="ACS Actividades de Construccion y Servicios SA"),  # dot but right match
        _q("ACS-OLD", longname="ACS Holdings (delisted)"),
    ]
    info = resolve_ticker_verbose("ACS Actividades de Construccion")
    assert info is not None
    assert info["symbol"] == "ACS.MC"
    assert "ACS" in info["name"]
    # GGAL should appear in alternatives, not as top pick
    alt_syms = [s for s, _, _ in info["alternatives"]]
    assert "GGAL" in alt_syms


def test_resolve_returns_none_on_empty_yfinance(fake_search):
    fake_search["quotes"] = []
    assert resolve_ticker_verbose("nonexistent-zzz") is None


def test_resolve_falls_back_to_non_equity_when_no_equities(fake_search):
    fake_search["quotes"] = [
        _q("BTC-USD", longname="Bitcoin USD", quotetype="CRYPTOCURRENCY"),
    ]
    info = resolve_ticker_verbose("bitcoin")
    assert info is not None
    assert info["symbol"] == "BTC-USD"


def test_resolve_ticker_back_compat_returns_symbol(fake_search):
    fake_search["quotes"] = [_q("AAPL", longname="Apple Inc.")]
    assert resolve_ticker("AAPL") == "AAPL"


def test_resolve_ticker_back_compat_returns_none(fake_search):
    fake_search["quotes"] = []
    assert resolve_ticker("zzz") is None


# --- live smoke -----------------------------------------------------------

@pytest.mark.network
def test_live_resolve_acs_mc_direct_works():
    """User can always type the full ticker (`ACS.MC`) when yfinance's
    fuzzy Search misses the home-exchange listing — the confirm panel exists
    precisely so they can catch and correct that."""
    info = resolve_ticker_verbose("ACS.MC")
    assert info is not None
    assert info["symbol"] == "ACS.MC"
    assert "ACS" in info["name"]
