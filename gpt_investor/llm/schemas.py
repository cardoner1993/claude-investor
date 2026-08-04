"""Pydantic schemas for structured LLM outputs.

All LLM calls that need a typed payload go through `call_claude_structured`
in `claude.py` and validate against a model defined here.
"""

from typing import Literal

from pydantic import BaseModel, Field

from gpt_investor.llm.probability import render_probability_block

# Bumped whenever `get_final_analysis`'s prompt / input set changes meaningfully.
# Phase 2's verdict_history filters on this so calibration never mixes contracts;
# verdict.py (P2) imports this constant rather than redefining it.
#   v1 — fundamentals + sentiment + industry + macro/regime
#   v2 — adds the Wyckoff timing layer + technical_addressed audit field (PW)
#   v3 — adds probabilistic distribution (up/flat/down) + pre-mortem (P5)
PROMPT_VERSION = "v3"


class SentimentLLM(BaseModel):
    """Finance-aware sentiment emitted by the LLM (pre-combination with VADER)."""

    score: float = Field(ge=-1.0, le=1.0, description="Net sentiment in [-1, +1]")
    drivers: list[str] = Field(
        min_length=1,
        max_length=5,
        description="3 short bullets citing strongest positive or negative signals",
    )
    summary: str = Field(
        min_length=1,
        max_length=800,
        description="2-3 sentence paragraph for the human reader",
    )


class VerdictLLM(BaseModel):
    """Final Buy/Hold/Sell verdict from sonnet.

    Every `*_addressed` field is a one-sentence explanation of how that input
    informed the verdict — or the literal string `"no impact"` if it did not.
    This forces the model to acknowledge each input; downstream we log all of
    these so we can audit whether the model is silently ignoring data.
    """

    verdict: Literal["Buy", "Hold", "Sell"]
    confidence: Literal["low", "med", "high"]
    price_target: float | None = Field(
        default=None,
        description="Target price in USD, or null if unwilling to commit",
    )
    thesis: str = Field(
        min_length=20,
        max_length=500,
        description="2-3 sentences, must reference the fundamental tier",
    )
    positives: list[str] = Field(min_length=2, max_length=4)
    risks: list[str] = Field(min_length=2, max_length=4)

    fundamentals_addressed: str = Field(
        min_length=4,
        max_length=300,
        description="How the deterministic fundamental score informed the verdict, or 'no impact'",
    )
    sentiment_addressed: str = Field(
        min_length=4,
        max_length=300,
        description="How news sentiment informed the verdict, or 'no impact'",
    )
    industry_addressed: str = Field(
        min_length=4,
        max_length=300,
        description="How industry/sector context informed the verdict, or 'no impact'",
    )
    macro_addressed: str = Field(
        min_length=4,
        max_length=300,
        description="How macro/liquidity/regime context informed the verdict, or 'no impact'",
    )
    technical_addressed: str = Field(
        min_length=4,
        max_length=300,
        description="How the Wyckoff price/volume phase informed the verdict, or 'no impact'",
    )

    # P5 — probabilistic verdict. The three probabilities should roughly sum to
    # 1; `normalize_probs` renormalises downstream so we don't hard-fail on a
    # 0.55/0.30/0.20 that sums to 1.05.
    prob_up: float = Field(ge=0.0, le=1.0, description="P(price up > +2% over the horizon)")
    prob_flat: float = Field(ge=0.0, le=1.0, description="P(price within ±2%)")
    prob_down: float = Field(ge=0.0, le=1.0, description="P(price down < -2%)")
    premortem: str = Field(
        min_length=10,
        max_length=400,
        description="Pre-mortem: assume the verdict is wrong 6 months out — the single most likely reason why",
    )


class AuditLLM(BaseModel):
    """A specialist audit agent's advisory take on a verdict, informed by
    similar past cases and their realised outcomes."""

    label: Literal["agree", "caution", "disagree"]
    note: str = Field(
        min_length=4,
        max_length=400,
        description="One-two sentence critique referencing the past-case win/loss pattern",
    )


def render_verdict_markdown(v: VerdictLLM, current_price: float) -> str:
    """Render a `VerdictLLM` into the markdown shape callers + cache already expect."""
    pt = f"${v.price_target:.2f}" if v.price_target is not None else "n/a"
    positives = "\n".join(f"- {p}" for p in v.positives)
    risks = "\n".join(f"- {r}" for r in v.risks)
    return (
        f"**Verdict**: {v.verdict} ({v.confidence} confidence)\n\n"
        f"**Price Target**: {pt}  (current: ${current_price:.2f})\n\n"
        f"**Thesis**: {v.thesis}\n\n"
        f"**Positives**:\n{positives}\n\n"
        f"**Risks**:\n{risks}\n\n"
        f"**Input audit**:\n"
        f"- _Fundamentals_: {v.fundamentals_addressed}\n"
        f"- _Sentiment_: {v.sentiment_addressed}\n"
        f"- _Industry_: {v.industry_addressed}\n"
        f"- _Macro_: {v.macro_addressed}\n"
        # `technical_addressed` defaults on verdicts produced before PW shipped.
        f"- _Technical_: {getattr(v, 'technical_addressed', 'no impact')}\n\n"
        + render_probability_block(
            v.prob_up, v.prob_flat, v.prob_down, v.premortem, current_price, v.price_target
        )
    )
