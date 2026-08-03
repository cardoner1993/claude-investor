from gpt_investor.llm.schemas import VerdictLLM, render_verdict_markdown
from gpt_investor.llm.verdict import (
    PROMPT_VERSION,
    parse_verdict,
    parse_analyst_grade,
    analyst_grade_to_score,
)
from gpt_investor.storage import cache


def _make_verdict(**over) -> VerdictLLM:
    base = dict(
        verdict="Buy", confidence="high", price_target=150.0,
        thesis="Solid tier fundamentals with confirmed markup timing support entry.",
        positives=["growing revenue", "strong FCF"],
        risks=["stretched valuation", "competition"],
        fundamentals_addressed="Solid tier drove the constructive stance",
        sentiment_addressed="positive coverage reinforced it",
        industry_addressed="tailwinds in the sector",
        macro_addressed="no impact",
        technical_addressed="markup phase confirmed timing",
        prob_up=0.55, prob_flat=0.30, prob_down=0.15,
        premortem="competition compresses margins faster than modeled",
    )
    base.update(over)
    return VerdictLLM(**base)


def test_parse_verdict_roundtrips_rendered_markdown():
    md = render_verdict_markdown(_make_verdict(), current_price=120.0)
    parsed = parse_verdict(md)
    assert parsed["verdict"] == "Buy"
    assert parsed["confidence"] == "high"
    assert parsed["price_target"] == 150.0
    assert parsed["prob_up"] == 0.55
    assert parsed["prob_flat"] == 0.30
    assert parsed["prob_down"] == 0.15


def test_parse_verdict_handles_na_target_and_sell():
    md = render_verdict_markdown(
        _make_verdict(verdict="Sell", confidence="low", price_target=None), 42.0
    )
    parsed = parse_verdict(md)
    assert parsed["verdict"] == "Sell"
    assert parsed["confidence"] == "low"
    assert parsed["price_target"] is None


def test_parse_verdict_empty():
    assert parse_verdict("") == {
        "verdict": None, "confidence": None, "price_target": None,
        "prob_up": None, "prob_flat": None, "prob_down": None,
    }


def test_parse_analyst_grade():
    text = "Latest analyst rating for AAPL:\nFirm: MS\nTo Grade: Strong Buy\nAction: up"
    assert parse_analyst_grade(text) == "Strong Buy"
    assert parse_analyst_grade("To Grade: N/A") is None
    assert parse_analyst_grade("No analyst ratings available.") is None


def test_analyst_grade_to_score():
    assert analyst_grade_to_score("Strong Buy") == 1.0
    assert analyst_grade_to_score("Buy") == 0.6
    assert analyst_grade_to_score("Hold") == 0.0
    assert analyst_grade_to_score("Underperform") == -0.6
    assert analyst_grade_to_score("Sell") == -1.0
    assert analyst_grade_to_score("gibberish") is None
    assert analyst_grade_to_score(None) is None


def test_record_and_read_verdict_roundtrip():
    cache.record_verdict("TESTX", PROMPT_VERSION, {
        "price": 100.0, "fund_score": 6.8, "fund_tier": "Solid",
        "sentiment_score": 0.4, "sentiment_conf": "high",
        "verdict": "Buy", "confidence": "high", "price_target": 150.0,
        "sonnet_text": "**Verdict**: Buy", "spy_at_capture": 500.0,
        "wyckoff_phase": "markup", "wyckoff_score": 8.0,
    })
    rows = [r for r in cache.all_verdicts() if r["ticker"] == "TESTX"]
    assert len(rows) == 1
    assert rows[0]["fund_tier"] == "Solid"
    assert rows[0]["price_7d"] is None


def test_record_verdict_idempotent_per_day():
    cache.record_verdict("TESTY", PROMPT_VERSION, {"price": 1.0, "verdict": "Buy"})
    cache.record_verdict("TESTY", PROMPT_VERSION, {"price": 2.0, "verdict": "Sell"})
    rows = [r for r in cache.all_verdicts() if r["ticker"] == "TESTY"]
    assert len(rows) == 1
    assert rows[0]["price"] == 1.0  # first write wins


def test_verdict_outcome_fill():
    cache.record_verdict("TESTZ", PROMPT_VERSION, {"price": 100.0, "spy_at_capture": 500.0})
    row_id = [r for r in cache.all_verdicts() if r["ticker"] == "TESTZ"][0]["id"]
    needing = {r["id"] for r in cache.verdicts_needing_outcome("price_30d", "spy_30d")}
    assert row_id in needing
    cache.set_verdict_outcomes(row_id, {"price_30d": 110.0, "spy_30d": 505.0, "bogus": 1})
    row = [r for r in cache.all_verdicts() if r["id"] == row_id][0]
    assert row["price_30d"] == 110.0
    assert row["spy_30d"] == 505.0
    assert "bogus" not in row
