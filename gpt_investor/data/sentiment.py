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

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_vader = SentimentIntensityAnalyzer()


def score_articles_with_vader(articles: list[dict]) -> tuple[float, int]:
    """Mean VADER compound score across articles' (title + summary).

    Parameters
    ----------
    articles : list[dict]
        yfinance news items; each may carry a `content` block with
        `title` and `summary`.

    Returns
    -------
    tuple[float, int]
        (mean compound score in -1..+1 rounded to 3dp, count of scored
        articles). Empty input yields (0.0, 0).
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


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp a value into a closed range.

    Parameters
    ----------
    x : float
        Value to clamp.
    lo : float, optional
        Lower bound. Default -1.0.
    hi : float, optional
        Upper bound. Default 1.0.

    Returns
    -------
    float
        `x` bounded to [lo, hi].
    """
    return max(lo, min(hi, x))


def combine_sentiment(
    vader_score: float,
    n_articles: int,
    llm_data: dict | None,
) -> dict:
    """Merge VADER baseline and LLM emission into the canonical sentiment dict.

    Final score is a 0.4/0.6 VADER/LLM blend; disagreement plus article
    count drive the confidence tier. A missing LLM score falls back to
    VADER only at low confidence.

    Parameters
    ----------
    vader_score : float
        Mean VADER compound score in -1..+1.
    n_articles : int
        Number of articles VADER scored.
    llm_data : dict or None
        LLM emission with optional `score`, `drivers`, `summary`.

    Returns
    -------
    dict
        Keys `score`, `confidence`, `drivers`, `summary`, and `components`
        (vader_score, llm_score, n_articles, disagreement).
    """
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
    """Compact card-chip label, e.g. '+0.42 high' or '-0.15 low'.

    Parameters
    ----------
    score : float
        Sentiment score in -1..+1.
    confidence : str
        Confidence tier ("low"/"med"/"high").

    Returns
    -------
    str
        Signed score with confidence suffix.
    """
    sign = "+" if score >= 0 else ""
    return f"{sign}{score:.2f} {confidence}"


def chip_color(score: float, confidence: str) -> str:
    """Radix color_scheme for the chip.

    Low confidence is always gray; otherwise green/red past +/-0.2, else amber.

    Parameters
    ----------
    score : float
        Sentiment score in -1..+1.
    confidence : str
        Confidence tier ("low"/"med"/"high").

    Returns
    -------
    str
        Radix color scheme name.
    """
    if confidence == "low":
        return "gray"
    if score >= 0.2:
        return "green"
    if score <= -0.2:
        return "red"
    return "amber"


def format_for_llm(sentiment: dict) -> str:
    """Render the sentiment dict as a markdown block for downstream LLM prompts.

    Parameters
    ----------
    sentiment : dict
        Output of `combine_sentiment()`.

    Returns
    -------
    str
        Newline-joined markdown: header line plus optional summary and drivers.
    """
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
