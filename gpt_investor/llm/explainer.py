"""Click-time plain-English explainer.

When the user opens a ticker dialog, a single haiku call synthesises the
already-computed layers (fundamentals, sentiment, Wyckoff timing, macro) and
the verdict into a short plain-English walkthrough of *why* the verdict is what
it is. Synthesis only — no tools, no new analysis, no new numbers. The result
is cached in SQLite keyed by (ticker, date, prompt_version) so re-opening a
dialog is free and a prompt-version bump recomputes.
"""

from loguru import logger
from pydantic import BaseModel, Field

from gpt_investor.llm.claude import call_claude_structured
from gpt_investor.llm.schemas import PROMPT_VERSION


class ExplanationLLM(BaseModel):
    """Schema the explainer LLM must satisfy.

    A single prose field with hard length bounds. Routing through
    `call_claude_structured` means the model's output is validated (present,
    non-trivial, not runaway) before we ever cache or render it — we never push
    unverified free text into SQLite.
    """

    explanation: str = Field(
        min_length=40,
        max_length=1500,
        description="2-3 paragraph plain-English walkthrough. Prose only, no markup.",
    )

# The explainer caches under its OWN version so editing _SYSTEM_PROMPT below
# busts stale prose, independently of the verdict's PROMPT_VERSION. Bump the
# `-expN` suffix whenever _SYSTEM_PROMPT changes; the verdict version is folded
# in so a verdict-contract change also invalidates the explanation of it.
EXPLAINER_VERSION = f"{PROMPT_VERSION}-exp1"

_SYSTEM_PROMPT = (
    "You explain an investment verdict to a smart non-expert. You are given the "
    "deterministic layer scores and the final verdict. Write a short plain-English "
    "walkthrough (2-3 tight paragraphs) of WHY the verdict landed where it did — how "
    "the fundamental tier, news sentiment, price/volume timing and macro backdrop "
    "combine into the call. Synthesise ONLY what is given: introduce no new facts, "
    "numbers, tickers or analysis. No headers, no preamble, no bullet lists — just prose."
)


def explain_verdict(
    fund_block: str,
    sentiment_block: str,
    wyckoff_block: str,
    macro_block: str,
    verdict_md: str,
) -> str:
    """Synthesise a plain-English explanation of the verdict.

    Synthesis only: reads the rendered markdown blocks already shown in the UI,
    never recomputes them and introduces no new facts. Returns "" on failure.

    Parameters
    ----------
    fund_block : str
        Rendered fundamentals block; may be empty.
    sentiment_block : str
        Rendered news-sentiment block; may be empty.
    wyckoff_block : str
        Rendered price/volume timing (Wyckoff) block; may be empty.
    macro_block : str
        Rendered macro/liquidity block; may be empty.
    verdict_md : str
        Rendered final verdict markdown. Empty short-circuits to "".

    Returns
    -------
    str
        Validated, HTML-defanged plain-English walkthrough, or "" if the model
        produced nothing schema-valid. Missing input layers are logged; the
        partial-input warning is surfaced separately in the UI by the caller.
    """
    if not verdict_md:
        return ""

    blocks = {
        "Fundamentals": fund_block,
        "Sentiment": sentiment_block,
        "Technical / Wyckoff": wyckoff_block,
        "Macro": macro_block,
    }
    missing = [name for name, text in blocks.items() if not text]
    if missing:
        logger.warning("explain_verdict missing input blocks: {}", ", ".join(missing))

    parts = [f"{name}:\n{text}" for name, text in blocks.items() if text]
    parts.append(f"Verdict:\n{verdict_md}")
    user_message = "\n\n".join(parts)

    try:
        parsed = call_claude_structured(
            ExplanationLLM, _SYSTEM_PROMPT, user_message, model="haiku", tools=False
        )
    except Exception as e:
        logger.warning("explain_verdict failed: {}", e)
        return ""
    if parsed is None:
        logger.warning("explain_verdict: no schema-valid output — not caching")
        return ""
    return _defang(parsed.explanation.strip())


def _defang(text: str) -> str:
    """Neutralise raw HTML in LLM prose before it is markdown-rendered into the
    dialog via `dangerouslySetInnerHTML`. The explainer is prose only, so
    escaping angle brackets is lossless and blocks tag/script injection.

    Parameters
    ----------
    text : str
        Model-produced explanation text.

    Returns
    -------
    str
        Same text with `<`/`>` escaped to HTML entities.
    """
    return text.replace("<", "&lt;").replace(">", "&gt;")
