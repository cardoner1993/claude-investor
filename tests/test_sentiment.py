"""Unit tests for the quantified sentiment module.

VADER scoring uses real text but the lexicon is deterministic, so the tests
are stable. LLM JSON parsing + combiner are pure logic.
"""

import pytest

from gpt_investor.data.sentiment import (
    score_articles_with_vader,
    parse_llm_json,
    combine_sentiment,
    chip_label,
    chip_color,
)


# --- VADER scoring --------------------------------------------------------

def _article(title: str, summary: str = "") -> dict:
    return {"content": {"title": title, "summary": summary}}


def test_vader_positive_news_scores_positive():
    arts = [
        _article("Company X smashes earnings, raises guidance", "Record profit and strong demand."),
        _article("Stellar quarter for Company X", "Beat all expectations."),
    ]
    score, n = score_articles_with_vader(arts)
    assert score > 0.3
    assert n == 2


def test_vader_negative_news_scores_negative():
    arts = [
        _article("Company X plunges on dismal results", "Massive losses, worst quarter ever."),
        _article("CEO resigns amid scandal", "Investigation reveals fraud."),
    ]
    score, n = score_articles_with_vader(arts)
    assert score < -0.3
    assert n == 2


def test_vader_empty_articles_returns_zero():
    score, n = score_articles_with_vader([])
    assert score == 0.0
    assert n == 0


def test_vader_skips_articles_with_no_text():
    arts = [_article("", ""), _article("Solid beat", "")]
    score, n = score_articles_with_vader(arts)
    assert n == 1
    assert score > 0  # the one valid article is positive


# --- LLM JSON parsing -----------------------------------------------------

def test_parse_fenced_json_block():
    text = 'Some prose.\n```json\n{"score": 0.5, "drivers": ["a", "b"], "summary": "ok"}\n```\nTrailing.'
    obj = parse_llm_json(text)
    assert obj == {"score": 0.5, "drivers": ["a", "b"], "summary": "ok"}


def test_parse_bare_json_object():
    text = 'Here is the analysis: {"score": -0.2, "drivers": ["x"], "summary": "neg"} and more text.'
    obj = parse_llm_json(text)
    assert obj is not None
    assert obj["score"] == -0.2


def test_parse_pure_json_string():
    obj = parse_llm_json('{"score": 0.0, "drivers": [], "summary": ""}')
    assert obj is not None
    assert obj["score"] == 0.0


def test_parse_returns_none_for_garbage():
    assert parse_llm_json("This is just prose with no JSON.") is None
    assert parse_llm_json("") is None
    assert parse_llm_json("{not valid json at all}") is None


def test_parse_requires_score_field():
    # JSON exists but lacks `score`, so it's not a sentiment payload.
    assert parse_llm_json('{"summary": "no score here"}') is None


# --- combiner -------------------------------------------------------------

def test_combine_high_confidence_when_scores_agree():
    out = combine_sentiment(0.5, n_articles=10, llm_data={"score": 0.6, "drivers": [], "summary": ""})
    assert out["confidence"] == "high"
    assert 0.5 < out["score"] < 0.65
    assert out["components"]["disagreement"] < 0.2


def test_combine_low_confidence_when_scores_diverge():
    out = combine_sentiment(-0.4, n_articles=10, llm_data={"score": 0.6, "drivers": [], "summary": ""})
    assert out["confidence"] == "low"
    assert out["components"]["disagreement"] > 0.4


def test_combine_low_confidence_when_few_articles():
    # scores agree but only 2 articles -> still low
    out = combine_sentiment(0.5, n_articles=2, llm_data={"score": 0.5, "drivers": [], "summary": ""})
    assert out["confidence"] == "low"


def test_combine_falls_back_to_vader_when_llm_missing():
    out = combine_sentiment(0.3, n_articles=10, llm_data=None)
    assert out["score"] == 0.3
    assert out["confidence"] == "low"
    assert out["components"]["llm_score"] is None


def test_combine_clamps_extreme_llm_scores():
    out = combine_sentiment(0.0, n_articles=10, llm_data={"score": 5.0, "drivers": [], "summary": ""})
    # 5.0 clamped to 1.0, then blended 0.4*0 + 0.6*1.0 = 0.6
    assert -1.0 <= out["score"] <= 1.0
    assert out["components"]["llm_score"] == 1.0


def test_combine_drops_garbage_llm_score():
    out = combine_sentiment(0.4, n_articles=10, llm_data={"score": "not a number", "drivers": [], "summary": ""})
    assert out["components"]["llm_score"] is None
    assert out["score"] == 0.4  # vader-only fallback
    assert out["confidence"] == "low"


def test_combine_caps_drivers_at_five():
    out = combine_sentiment(0.0, n_articles=5, llm_data={
        "score": 0.0, "drivers": ["a", "b", "c", "d", "e", "f", "g"], "summary": "",
    })
    assert len(out["drivers"]) == 5


# --- chip formatting -------------------------------------------------------

def test_chip_label_format():
    assert chip_label(0.42, "high") == "+0.42 high"
    assert chip_label(-0.15, "med") == "-0.15 med"
    assert chip_label(0.0, "low") == "+0.00 low"


def test_chip_color_low_confidence_is_gray():
    assert chip_color(0.5, "low") == "gray"
    assert chip_color(-0.5, "low") == "gray"


def test_chip_color_positive_high_confidence_is_green():
    assert chip_color(0.4, "high") == "green"


def test_chip_color_negative_high_confidence_is_red():
    assert chip_color(-0.4, "high") == "red"


def test_chip_color_near_neutral_is_amber():
    assert chip_color(0.05, "high") == "amber"
    assert chip_color(-0.05, "med") == "amber"
