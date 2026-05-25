import yfinance as yf
from loguru import logger

from gpt_investor.storage.cache import save_cached_liquidity
from gpt_investor.llm.claude import call_claude, call_claude_structured
from gpt_investor.data.fundamentals import fetch_fundamentals, score_fundamentals, format_fundamentals
from gpt_investor.data.macro import get_liquidity_snapshot, format_liquidity, snapshot_is_complete
from gpt_investor.data.market_regime import format_regime
from gpt_investor.data.market_data import _fetch_article_text
from gpt_investor.llm.schemas import SentimentLLM, VerdictLLM, render_verdict_markdown
from gpt_investor.data.sentiment import (
    score_articles_with_vader,
    combine_sentiment,
    format_for_llm as format_sentiment_for_llm,
)


def get_sentiment_analysis(ticker, news) -> dict:
    """Return a quantified sentiment dict (see gpt_investor.data.sentiment.combine_sentiment).

    The LLM is asked to emit a JSON block with `score`, `drivers`, `summary`.
    VADER scores each article independently. The two are combined, with
    disagreement driving confidence.
    """
    article_parts = []
    sources = {"yf_summary": 0, "fetched": 0, "title_only": 0}
    for article in news:
        c = article.get("content", {})
        title = c.get("title", "")
        if not title:
            continue
        date = c.get("pubDate", "")[:10]
        body = c.get("summary", "").strip()
        if body:
            sources["yf_summary"] += 1
        else:
            url = c.get("canonicalUrl", {}).get("url", "")
            if url:
                body = _fetch_article_text(url)
                if body:
                    sources["fetched"] += 1
                else:
                    sources["title_only"] += 1
            else:
                sources["title_only"] += 1
        part = f"**{title}** ({date})"
        if body:
            part += f"\n{body[:500]}"
        article_parts.append(part)

    vader_score, n = score_articles_with_vader(news)
    tlog = logger.bind(ticker=ticker)
    tlog.info(
        "sentiment sources  yf_summary={}  fetched={}  title_only={}  vader={:+.2f} (n={})",
        sources["yf_summary"], sources["fetched"], sources["title_only"], vader_score, n,
    )

    system_prompt = (
        f"You are a finance sentiment analyst for {ticker}. "
        f"You are measured and skeptical."
    )

    user_message = (
        f"Analyse the sentiment of these {len(article_parts)} recent news items about {ticker}.\n\n"
        + "\n\n".join(article_parts)
    )

    parsed = call_claude_structured(
        SentimentLLM,
        system_prompt,
        user_message,
        tools=False,
    )
    if parsed is None:
        tlog.warning("sentiment LLM JSON parse FAILED — falling back to VADER only")
        llm_data = None
    else:
        llm_data = parsed.model_dump()

    sentiment = combine_sentiment(vader_score, n, llm_data)
    tlog.info(
        "sentiment final={:+.2f} ({}) vader={:+.2f} llm={}",
        sentiment["score"], sentiment["confidence"],
        sentiment["components"]["vader_score"], sentiment["components"]["llm_score"],
    )
    return sentiment


def get_industry_analysis(ticker):
    info = yf.Ticker(ticker).info
    industry = info.get("industry", "")
    sector = info.get("sector", "")

    if not industry and not sector:
        return "Industry analysis unavailable for this ticker."

    system_prompt = (
        f"You are an industry analysis assistant. Search for and analyse the current state of the "
        f"{industry} industry and {sector} sector. Be measured and discerning. You are a skeptical investor."
    )

    user_message = (
        f"Search for the latest information on the {industry} industry and {sector} sector. "
        f"Cover current trends, growth prospects, recent regulatory changes, and the competitive "
        f"landscape. Ground your analysis in recent sources."
    )

    return call_claude(system_prompt, user_message, require_tools=["WebSearch"])


def get_final_analysis(
    ticker,
    current_price,
    sentiment,
    analyst_ratings,
    industry_analysis,
    liquidity_context: str = "",
    fundamentals: dict | None = None,
    regime: dict | None = None,
):
    """Run final Buy/Hold/Sell verdict.

    `fundamentals` is the dict returned by `score_fundamentals`. If not
    passed, it is fetched + scored inline (kept for callers that haven't
    been updated yet). Callers that need the score for the UI should
    compute it once and pass it through.

    `sentiment` is the dict returned by `get_sentiment_analysis`. Backward
    compat: a plain string is rendered as-is.

    `regime` is the dict returned by `market_regime.get_market_regime()`.
    When supplied, it's rendered into the user message and the system
    prompt forces sonnet to address macro impact (or declare it immaterial).
    """
    if fundamentals is None:
        fundamentals = score_fundamentals(fetch_fundamentals(ticker))
    fundamentals_block = format_fundamentals(fundamentals)

    if isinstance(sentiment, dict):
        sentiment_block = format_sentiment_for_llm(sentiment)
    else:
        sentiment_block = str(sentiment)

    regime_block = format_regime(regime) if regime else ""

    system_prompt = (
        "You are a concise, opinionated financial analyst. "
        "Weigh the deterministic fundamental score heavily — it is not LLM-generated. "
        "Emit ONE JSON object matching the schema. "
        "Thesis must reference the fundamental tier explicitly. "
        "For every `*_addressed` field, write ONE sentence on how that input "
        "informed the verdict, or the literal phrase `no impact` if it did not. "
        "Never leave an `_addressed` field empty — every input must be acknowledged."
    )

    user_message = (
        f"Ticker: {ticker} | Current price: ${current_price:.2f}\n\n"
        f"{fundamentals_block}\n\n"
        f"Sentiment:\n{sentiment_block}\n\n"
        f"Analyst ratings:\n{analyst_ratings}\n\n"
        f"Industry context:\n{industry_analysis}\n\n"
        + (f"Macro liquidity context:\n{liquidity_context}\n\n" if liquidity_context else "")
        + (f"Market regime:\n{regime_block}\n\n" if regime_block else "")
        + "Give your investment recommendation as the structured verdict."
    )

    try:
        parsed = call_claude_structured(
            VerdictLLM,
            system_prompt,
            user_message,
            model="sonnet",
            tools=False,
        )
    except Exception:
        logger.bind(ticker=ticker).exception("final_analysis FAILED (subprocess)")
        return ""

    if parsed is None:
        logger.bind(ticker=ticker).warning("final_analysis schema validation failed")
        return ""

    logger.bind(ticker=ticker).info(
        "verdict={} conf={} target={}  fund='{}'  sent='{}'  ind='{}'  macro='{}'",
        parsed.verdict, parsed.confidence,
        f"${parsed.price_target:.2f}" if parsed.price_target is not None else "n/a",
        parsed.fundamentals_addressed[:60],
        parsed.sentiment_addressed[:60],
        parsed.industry_addressed[:60],
        parsed.macro_addressed[:60],
    )
    return render_verdict_markdown(parsed, current_price)


def _liquidity_commentary(snapshot_md: str) -> str:
    """One-paragraph equity implication, generated from the deterministic snapshot.

    The model only sees the rendered numbers — it cannot invent rates. Single
    haiku call, no tools. Failures fall back to empty string (snapshot still ships).
    """
    system_prompt = (
        "You are a macro analyst. Read the snapshot below and write ONE paragraph "
        "(max 40 words) on what this rate constellation implies for global equities. "
        "Reference specific stances. No URLs, no preamble, no markdown headers."
    )
    try:
        return call_claude(system_prompt, snapshot_md, model="haiku", tools=False).strip()
    except Exception as e:
        logger.warning("liquidity commentary failed: {}", e)
        return ""


def get_liquidity_context() -> str:
    """Deterministic snapshot from official APIs (FRED + ECB SDW + PBOC scrape),
    optionally augmented with a 1-paragraph LLM commentary on the numbers.

    Numbers are NEVER hallucinated — they come from the APIs. The LLM only
    reads the rendered snapshot and writes one prose sentence on implication.
    """
    snapshot = get_liquidity_snapshot()
    snapshot_md = format_liquidity(snapshot)
    # URLs now come from constants in macro.py, not LLM output, so the old
    # HEAD-verify (designed to catch hallucinated URLs) is no longer needed
    # and produces false positives against servers that block HEAD (FRED).

    commentary = _liquidity_commentary(snapshot_md) if snapshot_is_complete(snapshot) else ""
    text = snapshot_md + (f"\n\n**Macro backdrop**: {commentary}" if commentary else "")

    # Refuse to disk-cache partial / "n/a" snapshots — otherwise a transient
    # outage (missing FRED key, ChinaMoney 5xx, ECB SDW down) freezes the bad
    # value into the 6h cache. UI still gets the partial result this run.
    if snapshot_is_complete(snapshot):
        try:
            save_cached_liquidity(text)
        except Exception as e:
            logger.warning("liquidity disk-cache save failed: {}", e)
    else:
        missing = [leg["bank"] for leg in snapshot["banks"] if leg.get("rate_pct") is None]
        logger.warning("liquidity snapshot incomplete (missing: {}) — NOT caching", missing)

    return text
