"""Pydantic schemas for structured LLM outputs.

All LLM calls that need a typed payload go through `call_claude_structured`
in `claude.py` and validate against a model defined here.
"""

from typing import Literal

from pydantic import BaseModel, Field


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
        f"- _Macro_: {v.macro_addressed}"
    )
