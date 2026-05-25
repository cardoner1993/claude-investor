"""Quantified sentiment: VADER baseline + LLM finance-aware score.

Final sentiment is a hybrid of two independent scorers:
    * VADER (rule-based lexicon) over each article's title + summary.
      Deterministic, fast, finance-naive.
    * LLM-emitted score (in the same prose call). Finance-aware,
      not deterministic.

Disagreement between the two becomes the confidence signal:
    abs(vader - llm) > 0.4  -> low
    abs(vader - llm) > 0.2  -> med
    otherwise               -> high

Few articles (<3) also forces low confidence.

The final returned score is a weighted blend (LLM 0.6, VADER 0.4); the
LLM is weighted slightly higher because VADER misses finance idioms
("beat expectations", "guidance raised") but the disagreement check
prevents it from drifting unchecked.
"""

import json
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_vader = SentimentIntensityAnalyzer()

# Fenced JSON block extractor. We accept ```json ... ``` or a bare {...} block.
_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{[^{}]*\"score\"[^{}]*\}", re.DOTALL)


def score_articles_with_vader(articles: list[dict]) -> tuple[float, int]:
    """Mean VADER compound score across articles' (title + summary).

    Returns (mean_score in -1..+1, n_scored_articles).
    """
    scores: list[float] = []
    for a in articles:
        c = a.get("content", {}) if isinstance(a, dict) else {}
        title = c.get("title", "") or ""
        summary = c.get("summary", "") or ""
        text = (title + ". " + summary).strip()
        if not text or text == ".":
            continue
        s = _vader.polarity_scores(text)
        scores.append(s["compound"])
    if not scores:
        return 0.0, 0
    return round(sum(scores) / len(scores), 3), len(scores)


def parse_llm_json(text: str) -> dict | None:
    """Extract a JSON object from LLM output.

    The model sometimes wraps in ```json fences, sometimes inlines a bare
    object. We try fenced first, then bare, then any JSON-decodable block.
    Returns None if no valid score-bearing JSON found.
    """
    if not text:
        return None
    m = _JSON_BLOCK_RE.search(text)
    candidates: list[str] = []
    if m:
        candidates.append(m.group(1))
    m2 = _BARE_JSON_RE.search(text)
    if m2:
        candidates.append(m2.group(0))
    # last-resort: try the whole string
    candidates.append(text.strip())

    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "score" in obj:
            return obj
    return None


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def combine_sentiment(
    vader_score: float,
    n_articles: int,
    llm_data: dict | None,
) -> dict:
    """Merge VADER baseline and LLM emission into the canonical sentiment dict."""
    if llm_data is None:
        llm_data = {}
    raw_llm_score = llm_data.get("score")
    try:
        llm_score = _clamp(float(raw_llm_score)) if raw_llm_score is not None else None
    except (TypeError, ValueError):
        llm_score = None

    if llm_score is None:
        # LLM failed to emit a usable score -> fall back to VADER only, low confidence.
        final_score = vader_score
        confidence = "low"
        disagreement = None
    else:
        final_score = _clamp(0.4 * vader_score + 0.6 * llm_score)
        disagreement = abs(vader_score - llm_score)
        if n_articles < 3:
            confidence = "low"
        elif disagreement > 0.4:
            confidence = "low"
        elif disagreement > 0.2:
            confidence = "med"
        else:
            confidence = "high"

    drivers = llm_data.get("drivers") or []
    if not isinstance(drivers, list):
        drivers = []
    drivers = [str(d) for d in drivers[:5]]

    summary = llm_data.get("summary") or ""
    if not isinstance(summary, str):
        summary = ""

    return {
        "score": round(final_score, 3),
        "confidence": confidence,
        "drivers": drivers,
        "summary": summary.strip(),
        "components": {
            "vader_score": vader_score,
            "llm_score": llm_score,
            "n_articles": n_articles,
            "disagreement": round(disagreement, 3) if disagreement is not None else None,
        },
    }


# --- formatting helpers ----------------------------------------------------

def chip_label(score: float, confidence: str) -> str:
    """Compact card-chip label, e.g. '+0.42 high' or '-0.15 low'."""
    sign = "+" if score >= 0 else ""
    return f"{sign}{score:.2f} {confidence}"


def chip_color(score: float, confidence: str) -> str:
    """Radix color_scheme for the chip."""
    if confidence == "low":
        return "gray"
    if score >= 0.2:
        return "green"
    if score <= -0.2:
        return "red"
    return "amber"


def format_for_llm(sentiment: dict) -> str:
    """Render the sentiment dict as a markdown block for downstream LLM prompts."""
    s = sentiment
    lines = [
        f"**Sentiment**: score {s['score']:+.2f} ({s['confidence']} confidence) "
        f"— VADER {s['components']['vader_score']:+.2f}, "
        f"LLM {s['components'].get('llm_score'):+.2f}"
        if s['components'].get('llm_score') is not None else
        f"**Sentiment**: score {s['score']:+.2f} ({s['confidence']} confidence) "
        f"— VADER {s['components']['vader_score']:+.2f}, LLM unavailable",
    ]
    if s.get("summary"):
        lines.append("")
        lines.append(s["summary"])
    if s.get("drivers"):
        lines.append("")
        for d in s["drivers"]:
            lines.append(f"- {d}")
    return "\n".join(lines)
