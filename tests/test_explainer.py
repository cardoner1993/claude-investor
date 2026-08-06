from gpt_investor.llm import explainer
from gpt_investor.llm.explainer import ExplanationLLM
from gpt_investor.llm.schemas import PROMPT_VERSION
from gpt_investor.storage import cache


def test_explain_verdict_short_circuits_without_verdict(monkeypatch):
    called = False

    def _boom(*a, **k):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(explainer, "call_claude_structured", _boom)
    assert explainer.explain_verdict("fund", "sent", "wyck", "macro", "") == ""
    assert called is False


def test_explain_verdict_builds_message_and_validates(monkeypatch):
    captured = {}

    def _fake(schema, system, user, model="sonnet", tools=True):
        captured.update(schema=schema, user=user, tools=tools, model=model)
        return ExplanationLLM(explanation="  a validated plain-english synthesis paragraph.  ")

    monkeypatch.setattr(explainer, "call_claude_structured", _fake)
    out = explainer.explain_verdict(
        "FUND-BLOCK", "SENT-BLOCK", "WYCK-BLOCK", "MACRO-BLOCK", "VERDICT-MD"
    )
    assert out == "a validated plain-english synthesis paragraph."   # stripped
    assert captured["schema"] is ExplanationLLM                       # schema-enforced
    assert captured["tools"] is False and captured["model"] == "sonnet"
    for token in ("FUND-BLOCK", "SENT-BLOCK", "WYCK-BLOCK", "MACRO-BLOCK", "VERDICT-MD"):
        assert token in captured["user"]


def test_explain_verdict_returns_empty_on_invalid_output(monkeypatch):
    monkeypatch.setattr(explainer, "call_claude_structured", lambda *a, **k: None)
    assert explainer.explain_verdict("f", "s", "w", "m", "VERDICT") == ""


def test_explain_verdict_defangs_html(monkeypatch):
    payload = ExplanationLLM(explanation="Solid setup <script>alert(1)</script> with upside now.")
    monkeypatch.setattr(explainer, "call_claude_structured", lambda *a, **k: payload)
    out = explainer.explain_verdict("f", "s", "w", "m", "VERDICT")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_explain_verdict_swallows_errors(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("cli down")

    monkeypatch.setattr(explainer, "call_claude_structured", _boom)
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
