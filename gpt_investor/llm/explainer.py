"""Click-time plain-English explainer.

When the user opens a ticker dialog, a single haiku call synthesises the
already-computed layers (fundamentals, sentiment, Wyckoff timing, macro) and
the verdict into a short plain-English walkthrough of *why* the verdict is what
it is. Synthesis only — no tools, no new analysis, no new numbers. The result
is cached in SQLite keyed by (ticker, date, prompt_version) so re-opening a
dialog is free and a prompt-version bump recomputes.
"""

from loguru import logger

from gpt_investor.llm.claude import call_claude
from gpt_investor.llm.schemas import PROMPT_VERSION

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
        Plain-English walkthrough of why the verdict landed, or "" on failure.
    """
    if not verdict_md:
        return ""
    parts = [
        f"Fundamentals:\n{fund_block}" if fund_block else "",
        f"Sentiment:\n{sentiment_block}" if sentiment_block else "",
        f"Technical / Wyckoff:\n{wyckoff_block}" if wyckoff_block else "",
        f"Macro:\n{macro_block}" if macro_block else "",
        f"Verdict:\n{verdict_md}",
    ]
    user_message = "\n\n".join(p for p in parts if p)
    try:
        return call_claude(_SYSTEM_PROMPT, user_message, model="haiku", tools=False).strip()
    except Exception as e:
        logger.warning("explain_verdict failed: {}", e)
        return ""
