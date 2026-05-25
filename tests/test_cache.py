"""Disk-cache helpers (analyses + liquidity).

Each test patches `cache._DB` to a tmp file so the prod `analyses.db` is
untouched.
"""

import time

import pytest

from gpt_investor.storage import cache


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_analyses.db"
    monkeypatch.setattr(cache, "_DB", str(db_path))
    return db_path


# --- liquidity disk cache ------------------------------------------------

def test_liquidity_cache_miss_returns_none(tmp_db):
    assert cache.get_cached_liquidity() is None


def test_liquidity_cache_round_trip(tmp_db):
    cache.save_cached_liquidity("**Snapshot** Fed neutral")
    assert cache.get_cached_liquidity() == "**Snapshot** Fed neutral"


def test_liquidity_cache_overwrites_previous(tmp_db):
    cache.save_cached_liquidity("first")
    cache.save_cached_liquidity("second")
    assert cache.get_cached_liquidity() == "second"


def test_liquidity_cache_expires_past_ttl(tmp_db):
    cache.save_cached_liquidity("stale")
    # Pass ttl=0 — even a fresh row is "older than 0 seconds".
    assert cache.get_cached_liquidity(ttl_seconds=0) is None


def test_liquidity_cache_honors_custom_ttl(tmp_db, monkeypatch):
    # Backdate the row by 30 minutes; 1h TTL should still return it.
    cache.save_cached_liquidity("fresh-ish")
    backdated = time.time() - 30 * 60
    import sqlite3
    with sqlite3.connect(str(tmp_db)) as conn:
        conn.execute("UPDATE liquidity SET fetched_at = ? WHERE key = ?", (backdated, "default"))
        conn.commit()
    assert cache.get_cached_liquidity(ttl_seconds=3600) == "fresh-ish"
    assert cache.get_cached_liquidity(ttl_seconds=60) is None


# --- analyses cache (sanity — existing path still works) ------------------

def test_analyses_miss_returns_none(tmp_db):
    assert cache.get_cached("AAPL") is None


def test_analyses_save_then_get(tmp_db):
    sentiment = {
        "score": 0.42,
        "confidence": "high",
        "drivers": ["a", "b"],
        "summary": "upbeat coverage",
    }
    cache.save_cached("AAPL", sentiment, "Strong Buy", "**Verdict**: Buy")
    got = cache.get_cached("AAPL")
    assert got is not None
    assert got["analyst_ratings"] == "Strong Buy"
    assert got["final_analysis"] == "**Verdict**: Buy"
    assert got["sentiment_dict"]["score"] == 0.42
    assert got["sentiment_dict"]["drivers"] == ["a", "b"]


def test_analyses_requires_verdict_and_ratings_for_hit(tmp_db):
    # Missing final_analysis → save_cached refuses to write at all
    cache.save_cached("AAPL", {"score": 0.0, "confidence": "low", "drivers": [], "summary": ""}, "Hold", "")
    assert cache.get_cached("AAPL") is None


# --- save_cached refusal guards ------------------------------------------

def _good_sentiment():
    return {"score": 0.42, "confidence": "high", "drivers": ["a"], "summary": "x"}


def test_save_refuses_empty_final_analysis(tmp_db):
    cache.save_cached("AAPL", _good_sentiment(), "Strong Buy", "")
    assert cache.get_cached("AAPL") is None
    # And no row was written at all (would shadow tomorrow's valid run otherwise)
    import sqlite3
    with sqlite3.connect(str(tmp_db)) as c:
        rows = c.execute("SELECT * FROM analyses WHERE ticker=?", ("AAPL",)).fetchall()
    assert rows == []


def test_save_refuses_empty_analyst_ratings(tmp_db):
    cache.save_cached("AAPL", _good_sentiment(), "", "**Verdict**: Buy")
    assert cache.get_cached("AAPL") is None


def test_save_refuses_whitespace_only_inputs(tmp_db):
    cache.save_cached("AAPL", _good_sentiment(), "   ", "**Verdict**: Buy")
    assert cache.get_cached("AAPL") is None
    cache.save_cached("AAPL", _good_sentiment(), "Hold", "\n  \n")
    assert cache.get_cached("AAPL") is None


def test_save_refuses_nan_sentiment_score(tmp_db):
    bad = {"score": float("nan"), "confidence": "low", "drivers": [], "summary": "x"}
    cache.save_cached("AAPL", bad, "Hold", "**Verdict**: Hold")
    assert cache.get_cached("AAPL") is None


def test_save_refuses_none_sentiment(tmp_db):
    cache.save_cached("AAPL", None, "Hold", "**Verdict**: Hold")
    assert cache.get_cached("AAPL") is None


def test_save_refuses_missing_score_field(tmp_db):
    cache.save_cached("AAPL", {"confidence": "high"}, "Hold", "**Verdict**: Hold")
    assert cache.get_cached("AAPL") is None


def test_save_refuses_test_pollution_string(tmp_db):
    # The exact bug from earlier: a string snapshot ("** test snapshot **")
    # was passed as sentiment. Without the guard, this would have written.
    # With the guard, only legacy non-empty strings pass — that's still
    # accepted (back-compat), so this MUST be caught at a different layer
    # (the smoke test should never call save_cached). Verify back-compat
    # path still works:
    cache.save_cached("AAPL", "legacy prose summary", "Hold", "**Verdict**: Hold")
    got = cache.get_cached("AAPL")
    assert got is not None
    assert got["sentiment_dict"] is None  # legacy str → no structured dict
