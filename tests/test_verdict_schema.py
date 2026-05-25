"""VerdictLLM schema validation + markdown rendering."""

import pytest
from pydantic import ValidationError

from gpt_investor.llm.schemas import VerdictLLM, render_verdict_markdown


def _valid_payload(**overrides):
    base = {
        "verdict": "Buy",
        "confidence": "high",
        "price_target": 175.0,
        "thesis": "Solid tier fundamentals plus easing macro support a multi-quarter run higher.",
        "positives": ["fast revenue growth", "margins expanding"],
        "risks": ["concentration in top customer", "valuation stretched"],
        "fundamentals_addressed": "Solid 6.8 tier validates pricing power.",
        "sentiment_addressed": "+0.42 high confirms market read.",
        "industry_addressed": "Sector tailwind from AI capex cycle.",
        "macro_addressed": "Easing Fed reduces equity discount rate.",
    }
    base.update(overrides)
    return base


def test_valid_payload_parses():
    v = VerdictLLM.model_validate(_valid_payload())
    assert v.verdict == "Buy"
    assert v.confidence == "high"
    assert v.price_target == 175.0


def test_verdict_must_be_in_literal_set():
    with pytest.raises(ValidationError):
        VerdictLLM.model_validate(_valid_payload(verdict="Strong Buy"))


def test_confidence_must_be_in_literal_set():
    with pytest.raises(ValidationError):
        VerdictLLM.model_validate(_valid_payload(confidence="medium"))


def test_price_target_nullable():
    v = VerdictLLM.model_validate(_valid_payload(price_target=None))
    assert v.price_target is None


def test_thesis_min_length_enforced():
    with pytest.raises(ValidationError):
        VerdictLLM.model_validate(_valid_payload(thesis="short"))


def test_positives_min_length_enforced():
    with pytest.raises(ValidationError):
        VerdictLLM.model_validate(_valid_payload(positives=["only one"]))


def test_addressed_fields_cannot_be_empty():
    for field in ("fundamentals_addressed", "sentiment_addressed",
                  "industry_addressed", "macro_addressed"):
        with pytest.raises(ValidationError):
            VerdictLLM.model_validate(_valid_payload(**{field: ""}))


def test_addressed_fields_accept_no_impact_literal():
    v = VerdictLLM.model_validate(_valid_payload(macro_addressed="no impact"))
    assert v.macro_addressed == "no impact"


# --- render_verdict_markdown ---------------------------------------------

def test_render_includes_all_sections():
    v = VerdictLLM.model_validate(_valid_payload())
    md = render_verdict_markdown(v, current_price=160.0)
    assert "**Verdict**: Buy (high confidence)" in md
    assert "**Price Target**: $175.00" in md
    assert "(current: $160.00)" in md
    assert "**Thesis**:" in md
    assert "- fast revenue growth" in md
    assert "- concentration in top customer" in md
    assert "**Input audit**:" in md
    assert "_Fundamentals_: Solid 6.8 tier" in md
    assert "_Sentiment_: +0.42 high" in md
    assert "_Industry_: Sector tailwind" in md
    assert "_Macro_: Easing Fed" in md


def test_render_handles_null_price_target():
    v = VerdictLLM.model_validate(_valid_payload(price_target=None))
    md = render_verdict_markdown(v, current_price=160.0)
    assert "**Price Target**: n/a" in md


def test_render_handles_no_impact_macro():
    v = VerdictLLM.model_validate(_valid_payload(macro_addressed="no impact"))
    md = render_verdict_markdown(v, current_price=100.0)
    assert "_Macro_: no impact" in md
