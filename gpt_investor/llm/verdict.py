"""Parse the rendered verdict + analyst grade back into structured fields.

`get_final_analysis` returns the markdown produced by
`schemas.render_verdict_markdown`; the verdict-history data layer needs the
verdict, confidence and price target back as typed values so it can record
them alongside the inputs. Rather than thread the raw `VerdictLLM` through the
pipeline, we re-parse the markdown here — the same text the cache already
stores, so a cached verdict records identically to a fresh one.

`PROMPT_VERSION` is imported from `schemas` (never redefined) so calibration
never mixes rows produced under different prompt contracts.
"""

import re

from gpt_investor.llm.schemas import PROMPT_VERSION  # re-exported for the data layer

__all__ = ["PROMPT_VERSION", "parse_verdict", "parse_analyst_grade", "analyst_grade_to_score"]

_VERDICT_RE = re.compile(r"\*\*Verdict\*\*:\s*(Buy|Hold|Sell)", re.IGNORECASE)
_CONF_RE = re.compile(r"\((low|med|high)\s+confidence\)", re.IGNORECASE)
_TARGET_RE = re.compile(r"\*\*Price Target\*\*:\s*\$([\d,]+(?:\.\d+)?)", re.IGNORECASE)


def parse_verdict(sonnet_text: str) -> dict:
    """Extract `{verdict, confidence, price_target}` from the rendered markdown.

    Returns Nones for any field that can't be found (e.g. an old cached row or
    a "$n/a" target) — the caller stores what it gets.
    """
    if not sonnet_text:
        return {"verdict": None, "confidence": None, "price_target": None}

    vm = _VERDICT_RE.search(sonnet_text)
    cm = _CONF_RE.search(sonnet_text)
    tm = _TARGET_RE.search(sonnet_text)

    verdict = vm.group(1).capitalize() if vm else None
    confidence = cm.group(1).lower() if cm else None
    price_target = float(tm.group(1).replace(",", "")) if tm else None

    return {"verdict": verdict, "confidence": confidence, "price_target": price_target}


# Grade phrases → normalised score in [-1, +1]. Checked longest-first so
# "strong buy" wins over "buy". yfinance grades are inconsistent across firms;
# this covers the common vocabulary and returns None on anything unrecognised.
_GRADE_SCORES: dict[str, float] = {
    "strong buy": 1.0,
    "conviction buy": 1.0,
    "buy": 0.6,
    "outperform": 0.6,
    "overweight": 0.6,
    "accumulate": 0.5,
    "add": 0.5,
    "positive": 0.5,
    "hold": 0.0,
    "neutral": 0.0,
    "market perform": 0.0,
    "sector perform": 0.0,
    "equal-weight": 0.0,
    "equal weight": 0.0,
    "in-line": 0.0,
    "peer perform": 0.0,
    "reduce": -0.5,
    "underperform": -0.6,
    "underweight": -0.6,
    "sell": -1.0,
    "strong sell": -1.0,
}


def parse_analyst_grade(analyst_ratings: str) -> str | None:
    """Pull the `To Grade:` value out of `get_analyst_ratings` output."""
    if not analyst_ratings:
        return None
    m = re.search(r"To Grade:\s*(.+)", analyst_ratings)
    if not m:
        return None
    grade = m.group(1).strip()
    if not grade or grade.upper() == "N/A":
        return None
    return grade


def analyst_grade_to_score(grade: str | None) -> float | None:
    """Map an analyst grade phrase to a [-1, +1] score, or None if unknown."""
    if not grade:
        return None
    g = grade.strip().lower()
    for phrase, score in sorted(_GRADE_SCORES.items(), key=lambda x: -len(x[0])):
        if phrase in g:
            return score
    return None
