import pytest

from gpt_investor.infra import resilience as r


def test_safe_get_nested_and_missing():
    obj = {"a": {"b": [{"url": "x"}, "prose"]}}
    assert r.safe_get(obj, "a", "b", 0, "url") == "x"
    assert r.safe_get(obj, "a", "b", 1, "url", default="d") == "d"   # index 1 is a str
    assert r.safe_get(obj, "a", "missing", default=None) is None
    assert r.safe_get(obj, "a", "b", 9, default="oob") == "oob"       # index out of range
    assert r.safe_get(None, "a", default=7) == 7


def test_first_dict():
    assert r.first_dict([{"a": 1}, "prose"]) == {"a": 1}
    assert r.first_dict(["prose", {"b": 2}]) == {"b": 2}
    assert r.first_dict([]) == {}
    assert r.first_dict("nope") == {}
    assert r.first_dict({"already": 1}) == {"already": 1}


def test_with_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    out = r.with_retry(flaky, tries=3, _sleep=lambda _: None)
    assert out == "ok"
    assert calls["n"] == 3


def test_with_retry_reraises_after_exhaustion():
    def always_fail():
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        r.with_retry(always_fail, tries=2, _sleep=lambda _: None)


def test_resilient_serves_last_good_and_tracks_degradation():
    r.reset_health()
    state = {"fail": False}

    def leg(t):
        if state["fail"]:
            raise ConnectionError("yahoo down")
        return f"price:{t}"

    # first call succeeds and caches last-good
    assert r.resilient("price", leg, "AAPL", key="AAPL", _sleep=lambda _: None) == "price:AAPL"
    assert r.degraded_legs() == []

    # now the source fails — last-good is served, leg marked degraded
    state["fail"] = True
    assert r.resilient("price", leg, "AAPL", key="AAPL", tries=2, _sleep=lambda _: None) == "price:AAPL"
    assert "price" in r.degraded_legs()


def test_resilient_reraises_without_cached_fallback():
    r.reset_health()

    def leg(_t):
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        r.resilient("fundamentals", leg, "NOPE", key="NOPE", tries=2, _sleep=lambda _: None)
    assert "fundamentals" in r.degraded_legs()


def test_reset_health_clears():
    r.reset_health()
    assert r.degraded_legs() == []
