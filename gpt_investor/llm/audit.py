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
    """Fractional forward return from entry to a realised later price.

    `(fwd - price) / price` — e.g. entry 100, later 120 → 0.20 (+20%). Used to
    label a past verdict as a win (return > 0) or loss when sampling similar
    cases. Returns None when it can't be computed (missing price, or a zero
    entry that would divide by zero) so unusable rows are skipped.

    Parameters
    ----------
    price : float | None
        Entry (capture) price. Missing/zero yields None.
    fwd : float | None
        Realised forward price at the horizon.

    Returns
    -------
    float | None
        Fractional return (0.20 = +20%), or None when an input is missing/zero.
    """
    if not price or fwd is None or price == 0:
        return None
    return (fwd - price) / price


def _match_score(v: dict, sector, fund_tier, regime_label) -> int:
    """Count aligned attributes between a past verdict and the current one.

    A None on either side never counts — an unknown sector must not match every
    NULL-sector row, and regime alone (market-wide, identical every run) must not
    qualify a case.

    Parameters
    ----------
    v : dict
        Past verdict row with `sector`, `fund_tier`, `regime_label` keys.
    sector : str | None
        Current verdict's sector.
    fund_tier : str | None
        Current verdict's fundamental tier.
    regime_label : str | None
        Current market-regime label.

    Returns
    -------
    int
        Number of aligned attributes (0–3).
    """
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
    realised `price_{horizon}d` outcome count. The sample is balanced between
    wins and losses so an agent can't just pattern-match "it always goes up".

    Parameters
    ----------
    sector : str | None
        Current verdict's sector.
    fund_tier : str | None
        Current verdict's fundamental tier.
    regime_label : str | None
        Current market-regime label.
    horizon : int, optional
        Forward-return window in days, default 30 (selects the `price_{horizon}d`
        column).
    limit : int, optional
        Max cases returned, default 6.
    min_match : int, optional
        Minimum aligned attributes required, default 2.

    Returns
    -------
    list[dict]
        Up to `limit` case dicts, each with `ticker`, `verdict`, `fund_tier`,
        `sector`, `regime_label`, `wyckoff_phase`, `ret` (float) and `win` (bool),
        drawn evenly from wins and losses.
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
    """Render similar past cases as a markdown bullet list for the prompt.

    Parameters
    ----------
    cases : list[dict]
        Case dicts from `get_similar_past`.

    Returns
    -------
    str
        One line per case (ticker, tier/phase/regime, verdict, realised return,
        WIN/LOSS), or a "No comparable past cases." placeholder when empty.
    """
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
    """Run one single-lens haiku auditor against the verdict and past cases.

    The agent sees only its own lens's evidence, never the other layers, so the
    two audits stay disjoint.

    Parameters
    ----------
    role : str
        Auditor role label (e.g. "financial").
    focus_block : str
        Lens-specific evidence shown to the agent.
    verdict_md : str
        Proposed verdict markdown under review.
    cases : list[dict]
        Similar past cases with realised outcomes.
    lens : str
        Lens name used in prompt phrasing.

    Returns
    -------
    dict
        `{label, note}`; falls back to `{"caution", "audit unavailable ..."}`
        when structured parsing fails.
    """
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
    """Audit the verdict through the financial lens only.

    Parameters
    ----------
    verdict_md : str
        Proposed verdict markdown.
    fund_block : str
        Fundamental evidence block.
    cases : list[dict]
        Similar past cases with realised outcomes.

    Returns
    -------
    dict
        `{label, note}` from the financial auditor.
    """
    return _run_agent("financial", fund_block, verdict_md, cases, "financial")


def audit_sentiment(verdict_md: str, sentiment_block: str, cases: list[dict]) -> dict:
    """Audit the verdict through the sentiment lens only.

    Parameters
    ----------
    verdict_md : str
        Proposed verdict markdown.
    sentiment_block : str
        Sentiment evidence block.
    cases : list[dict]
        Similar past cases with realised outcomes.

    Returns
    -------
    dict
        `{label, note}` from the sentiment auditor.
    """
    return _run_agent("sentiment", sentiment_block, verdict_md, cases, "sentiment")


def worst_label(*labels: str) -> str:
    """Pick the most cautious of several audit labels.

    Ranks agree < caution < disagree and returns the highest — a single
    disagreement dominates.

    Parameters
    ----------
    *labels : str
        One or more of "agree" / "caution" / "disagree".

    Returns
    -------
    str
        The worst-ranked label.
    """
    return max(labels, key=lambda label: _LABEL_RANK.get(label, 0))


def combine_audits(fin: dict, sent: dict) -> dict:
    """Merge the two agents into one label plus a rendered note.

    Combined label is the worst of the two (a single disagreement dominates).

    Parameters
    ----------
    fin : dict
        Financial audit `{label, note}`.
    sent : dict
        Sentiment audit `{label, note}`.

    Returns
    -------
    dict
        `{label, text}` where `text` is markdown summarising both lenses.
    """
    label = worst_label(fin["label"], sent["label"])
    text = (
        f"**Audit: {label}**\n\n"
        f"- _Financial ({fin['label']})_: {fin['note']}\n"
        f"- _Sentiment ({sent['label']})_: {sent['note']}"
    )
    return {"label": label, "text": text}


def enough_history(cases: list[dict]) -> bool:
    """Gate: whether enough similar past cases exist to run the audits.

    Parameters
    ----------
    cases : list[dict]
        Balanced similar cases from `get_similar_past`.

    Returns
    -------
    bool
        True when at least `_MIN_SIMILAR` cases are available.
    """
    return len(cases) >= _MIN_SIMILAR


def log_gate(ticker: str, n: int) -> None:
    """Log how many balanced similar cases the audit gate found.

    Parameters
    ----------
    ticker : str
        Ticker being audited.
    n : int
        Count of balanced similar cases with realised outcomes.
    """
    logger.bind(ticker=ticker).info(
        "audit gate: {} balanced similar cases w/ outcomes (need {})", n, _MIN_SIMILAR
    )
