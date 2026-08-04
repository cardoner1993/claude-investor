from gpt_investor.llm import audit
from gpt_investor.llm.verdict import PROMPT_VERSION
from gpt_investor.storage import cache


def test_worst_label_and_combine():
    assert audit.worst_label("agree", "caution") == "caution"
    assert audit.worst_label("caution", "disagree") == "disagree"
    assert audit.worst_label("agree", "agree") == "agree"
    combined = audit.combine_audits({"label": "agree", "note": "ok"}, {"label": "disagree", "note": "no"})
    assert combined["label"] == "disagree"
    assert "Financial (agree)" in combined["text"]
    assert "Sentiment (disagree)" in combined["text"]


def test_enough_history_and_format_cases():
    assert audit.enough_history([{}] * 5) is True
    assert audit.enough_history([{}] * 4) is False
    assert "No comparable" in audit.format_cases([])
    txt = audit.format_cases([
        {"ticker": "AAA", "fund_tier": "Solid", "wyckoff_phase": "markup",
         "regime_label": "risk-on-bull", "verdict": "Buy", "ret": 0.1, "win": True}
    ])
    assert "AAA" in txt and "WIN" in txt


def _seed(ticker, sector, tier, price, fwd_30d):
    cache.record_verdict(ticker, PROMPT_VERSION, {
        "price": price, "fund_tier": tier, "sector": sector, "verdict": "Buy",
        "regime_label": "risk-on-bull", "wyckoff_phase": "markup",
    })
    row_id = [r for r in cache.all_verdicts() if r["ticker"] == ticker][0]["id"]
    cache.set_verdict_outcomes(row_id, {"price_30d": fwd_30d})


def test_get_similar_past_filters_balances_and_requires_outcome():
    # matching sector, mix of wins/losses, one without outcome (excluded)
    _seed("SIMW1", "Tech", "Solid", 100, 120)   # +20% win
    _seed("SIMW2", "Tech", "Weak", 100, 110)    # +10% win (matches on sector)
    _seed("SIML1", "Tech", "Solid", 100, 80)    # -20% loss
    _seed("SIML2", "Tech", "Avoid", 100, 90)    # -10% loss (matches on sector)
    cache.record_verdict("SIMNONE", PROMPT_VERSION, {"price": 100, "sector": "Tech"})  # no outcome
    # a non-matching row (different sector/tier/regime) is excluded
    cache.record_verdict("OTHER", PROMPT_VERSION, {
        "price": 100, "fund_tier": "Strong", "sector": "Energy", "regime_label": "mixed",
    })
    oid = [r for r in cache.all_verdicts() if r["ticker"] == "OTHER"][0]["id"]
    cache.set_verdict_outcomes(oid, {"price_30d": 200})

    cases = audit.get_similar_past("Tech", "Solid", "risk-on-bull", horizon=30, limit=4)
    tickers = {c["ticker"] for c in cases}
    assert "SIMNONE" not in tickers   # no realised outcome
    assert "OTHER" not in tickers     # matches nothing
    assert any(c["win"] for c in cases) and any(not c["win"] for c in cases)  # balanced
    assert len(cases) <= 4
