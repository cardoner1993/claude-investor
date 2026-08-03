"""Audit agents — advisory critique of a verdict vs similar past outcomes.

Two specialist haiku agents look at the just-made verdict through disjoint
lenses (financial, sentiment) plus a balanced sample of *similar past cases
and how they actually turned out*. Purely advisory — they never mutate the
verdict; they surface an `agree / caution / disagree` chip and a note.

Gated: the caller skips the audits entirely when fewer than 5 similar past
cases with realised outcomes exist (nothing to learn from yet), so this layer
only activates once verdict_history has accumulated filled outcomes.
"""

from __future__ import annotations

from loguru import logger

from gpt_investor.llm.claude import call_claude_structured
from gpt_investor.llm.schemas import AuditLLM
from gpt_investor.storage.cache import all_verdicts

# worst-of ordering for combining the two agents' labels
_LABEL_RANK = {"agree": 0, "caution": 1, "disagree": 2}
_MIN_SIMILAR = 5


def _ret(price, fwd) -> float | None:
    if not price or fwd is None or price == 0:
        return None
    return (fwd - price) / price


def _match_score(v: dict, sector, fund_tier, regime_label) -> int:
    """Count aligned attributes (sector / tier / regime). A None on either side
    never counts — an unknown sector must not match every NULL-sector row, and
    regime alone (market-wide, identical all run) must not qualify a case."""
    score = 0
    if sector and v.get("sector") == sector:
        score += 1
    if fund_tier and v.get("fund_tier") == fund_tier:
        score += 1
    if regime_label and v.get("regime_label") == regime_label:
        score += 1
    return score


def get_similar_past(sector, fund_tier, regime_label, horizon: int = 30, limit: int = 6,
                     min_match: int = 2) -> list[dict]:
    """Balanced win/loss sample of past verdicts similar to this one.

    Similarity requires at least `min_match` of {sector, fund_tier, regime_label}
    to align (regime-only or None-only matches don't qualify). Only rows with a
    realised `price_{horizon}d` outcome count. Returns up to `limit` cases,
    balanced between wins and losses so an agent can't just pattern-match "it
    always goes up".
    """
    price_col = f"price_{horizon}d"
    wins: list[dict] = []
    losses: list[dict] = []
    for v in all_verdicts():
        r = _ret(v.get("price"), v.get(price_col))
        if r is None:
            continue
        if _match_score(v, sector, fund_tier, regime_label) < min_match:
            continue
        case = {
            "ticker": v.get("ticker"),
            "verdict": v.get("verdict"),
            "fund_tier": v.get("fund_tier"),
            "sector": v.get("sector"),
            "regime_label": v.get("regime_label"),
            "wyckoff_phase": v.get("wyckoff_phase"),
            "ret": round(r, 4),
            "win": r > 0,
        }
        (wins if r > 0 else losses).append(case)

    half = max(1, limit // 2)
    balanced = wins[:half] + losses[:half]
    return balanced[:limit]


def format_cases(cases: list[dict]) -> str:
    if not cases:
        return "No comparable past cases."
    lines = ["Similar past verdicts and realised outcomes:"]
    for c in cases:
        outcome = "WIN" if c["win"] else "LOSS"
        lines.append(
            f"- {c['ticker']} [{c.get('fund_tier')}/{c.get('wyckoff_phase')}/{c.get('regime_label')}] "
            f"verdict={c.get('verdict')} → {c['ret']:+.1%} ({outcome})"
        )
    return "\n".join(lines)


def _run_agent(role: str, focus_block: str, verdict_md: str, cases: list[dict], lens: str) -> dict:
    system_prompt = (
        f"You are a skeptical {role} auditor. You see ONLY the {lens} evidence and the "
        f"proposed verdict — not the other layers. Using the realised outcomes of similar "
        f"past cases, judge whether the verdict is well-supported from your lens. Be "
        f"willing to disagree. Emit agree / caution / disagree with a one-two sentence note."
    )
    user_message = (
        f"Proposed verdict:\n{verdict_md}\n\n"
        f"{lens.capitalize()} evidence:\n{focus_block}\n\n"
        f"{format_cases(cases)}"
    )
    parsed = call_claude_structured(AuditLLM, system_prompt, user_message, model="haiku", tools=False)
    if parsed is None:
        return {"label": "caution", "note": "audit unavailable (parse failed)"}
    return {"label": parsed.label, "note": parsed.note}


def audit_financial(verdict_md: str, fund_block: str, cases: list[dict]) -> dict:
    return _run_agent("financial", fund_block, verdict_md, cases, "financial")


def audit_sentiment(verdict_md: str, sentiment_block: str, cases: list[dict]) -> dict:
    return _run_agent("sentiment", sentiment_block, verdict_md, cases, "sentiment")


def worst_label(*labels: str) -> str:
    return max(labels, key=lambda label: _LABEL_RANK.get(label, 0))


def combine_audits(fin: dict, sent: dict) -> dict:
    """Merge the two agents into `{label, text}`; label is the worst of two."""
    label = worst_label(fin["label"], sent["label"])
    text = (
        f"**Audit: {label}**\n\n"
        f"- _Financial ({fin['label']})_: {fin['note']}\n"
        f"- _Sentiment ({sent['label']})_: {sent['note']}"
    )
    return {"label": label, "text": text}


def enough_history(cases: list[dict]) -> bool:
    return len(cases) >= _MIN_SIMILAR


def log_gate(ticker: str, n: int) -> None:
    logger.bind(ticker=ticker).info(
        "audit gate: {} balanced similar cases w/ outcomes (need {})", n, _MIN_SIMILAR
    )
