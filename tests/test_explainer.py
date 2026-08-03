from gpt_investor.llm import explainer
from gpt_investor.llm.schemas import PROMPT_VERSION
from gpt_investor.storage import cache


def test_explain_verdict_short_circuits_without_verdict(monkeypatch):
    called = False

    def _boom(*a, **k):
        nonlocal called
        called = True
        return "x"

    monkeypatch.setattr(explainer, "call_claude", _boom)
    assert explainer.explain_verdict("fund", "sent", "wyck", "macro", "") == ""
    assert called is False


def test_explain_verdict_builds_message_from_blocks(monkeypatch):
    captured = {}

    def _fake(system, user, model="haiku", tools=True):
        captured["system"] = system
        captured["user"] = user
        captured["tools"] = tools
        captured["model"] = model
        return "  plain english out  "

    monkeypatch.setattr(explainer, "call_claude", _fake)
    out = explainer.explain_verdict(
        "FUND-BLOCK", "SENT-BLOCK", "WYCK-BLOCK", "MACRO-BLOCK", "VERDICT-MD"
    )
    assert out == "plain english out"
    assert captured["tools"] is False
    assert captured["model"] == "haiku"
    for token in ("FUND-BLOCK", "SENT-BLOCK", "WYCK-BLOCK", "MACRO-BLOCK", "VERDICT-MD"):
        assert token in captured["user"]


def test_explain_verdict_swallows_errors(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("cli down")

    monkeypatch.setattr(explainer, "call_claude", _boom)
    assert explainer.explain_verdict("f", "s", "w", "m", "verdict") == ""


def test_explainer_cache_roundtrip_and_version_isolation():
    assert cache.get_cached_explainer("EXPL", PROMPT_VERSION) is None
    cache.save_cached_explainer("EXPL", PROMPT_VERSION, "hello world")
    assert cache.get_cached_explainer("EXPL", PROMPT_VERSION) == "hello world"
    # a different prompt version is a cache miss (bump recomputes)
    assert cache.get_cached_explainer("EXPL", "v-other") is None


def test_explainer_cache_ignores_empty():
    cache.save_cached_explainer("EMPTYX", PROMPT_VERSION, "   ")
    assert cache.get_cached_explainer("EMPTYX", PROMPT_VERSION) is None
