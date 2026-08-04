import os
import subprocess
import threading
import json

import yfinance as yf
from cachetools import TTLCache
from loguru import logger

from gpt_investor.llm.claude import add_token_usage

MAX_TICKERS_TO_ANALYZE = int(os.getenv("MAX_TICKERS_TO_ANALYZE", 4))

# Yahoo Finance caches — shared across threads, protected by _yf_lock
_yf_lock = threading.Lock()
# (yf_key, yf_type) → list[str] of base ticker symbols; 4-hour TTL
_yf_company_cache: TTLCache = TTLCache(maxsize=64, ttl=4 * 3600)
# industry_query → Counter of news ticker mentions; 15-minute TTL
_yf_news_cache: TTLCache = TTLCache(maxsize=64, ttl=15 * 60)

# Maps user keywords → Yahoo Finance industry keys (yf.Industry)
# Keys must match what Yahoo Finance uses (lowercase, hyphen-separated)
_KEYWORD_TO_YF_INDUSTRY: dict[str, str] = {
    "semiconductor": "semiconductors",
    "chip": "semiconductors",
    "software application": "software-application",
    "software infrastructure": "software-infrastructure",
    "cloud": "software-infrastructure",
    "saas": "software-infrastructure",
    "consumer electronics": "consumer-electronics",
    "computer hardware": "computer-hardware",
    "hardware": "computer-hardware",
    "communication equipment": "communication-equipment",
    "oil gas integrated": "oil-gas-integrated",
    "oil gas": "oil-gas-integrated",
    "oil": "oil-gas-integrated",
    "petroleum": "oil-gas-integrated",
    "midstream": "oil-gas-midstream",
    "pipeline": "oil-gas-midstream",
    "oil gas ep": "oil-gas-e-p",
    "exploration": "oil-gas-e-p",
    "drug manufacturer": "drug-manufacturers-general",
    "pharma": "drug-manufacturers-general",
    "pharmaceutical": "drug-manufacturers-general",
    "biotech": "biotechnology",
    "biotechnology": "biotechnology",
    "genomic": "biotechnology",
    "medical device": "medical-devices",
    "medical devices": "medical-devices",
    "health plan": "healthcare-plans",
    "insurance": "insurance-diversified",
    "bank": "banks-diversified",
    "diversified bank": "banks-diversified",
    "regional bank": "banks-regional",
    "asset management": "asset-management",
    "fintech": "credit-services",
    "credit": "credit-services",
    "internet retail": "internet-retail",
    "e-commerce": "internet-retail",
    "ecommerce": "internet-retail",
    "auto": "auto-manufacturers",
    "car": "auto-manufacturers",
    "automobile": "auto-manufacturers",
    "restaurant": "restaurants",
    "aerospace": "aerospace-defense",
    "defense": "aerospace-defense",
    "gold": "gold",
    "silver": "silver",
    "copper": "copper",
    "steel": "steel",
    "chemical": "specialty-chemicals",
    "telecom": "telecom-services",
    "telecommunication": "telecom-services",
    "entertainment": "entertainment",
    "media": "entertainment",
    "streaming": "entertainment",
    "internet": "internet-content-information",
    "social media": "internet-content-information",
    "solar": "solar",
    "renewable": "utilities-renewable",
    "electric utility": "utilities-regulated-electric",
    "reit": "reit-diversified",
    "real estate investment": "reit-diversified",
}

# Falls back to sector-level when no industry match
_KEYWORD_TO_YF_SECTOR: dict[str, str] = {
    "tech": "technology", "technology": "technology", "ai": "technology",
    "cyber": "technology", "computing": "technology", "data": "technology",
    "energy": "energy", "gas": "energy", "lng": "energy", "coal": "energy",
    "health": "healthcare", "medical": "healthcare", "drug": "healthcare",
    "bank": "financial-services", "financ": "financial-services",
    "invest": "financial-services", "asset": "financial-services",
    "retail": "consumer-cyclical", "luxury": "consumer-cyclical",
    "travel": "consumer-cyclical", "hotel": "consumer-cyclical", "gaming": "consumer-cyclical",
    "food": "consumer-defensive", "beverage": "consumer-defensive", "tobacco": "consumer-defensive",
    "manufactur": "industrials", "transport": "industrials", "logistics": "industrials",
    "mining": "basic-materials", "metal": "basic-materials", "material": "basic-materials",
    "media": "communication-services", "telecom": "communication-services",
    "real estate": "real-estate", "reit": "real-estate", "property": "real-estate",
    "utility": "utilities", "utilities": "utilities", "solar": "utilities", "wind": "utilities",
}


def _yf_lookup(industry: str) -> tuple[str | None, str]:
    """Map an industry string to a Yahoo Finance key via keyword matching.

    Industry keys are checked longest-first (more specific wins), then sector
    keys as a fallback.

    Parameters
    ----------
    industry : str
        Free-text industry name.

    Returns
    -------
    tuple[str | None, str]
        ``(key, type)`` where type is ``'industry'`` or ``'sector'``, or
        ``(None, '')`` if no keyword matched.
    """
    lower = industry.lower()
    # Check industry keys longest-first (more specific wins)
    for keyword, key in sorted(_KEYWORD_TO_YF_INDUSTRY.items(), key=lambda x: -len(x[0])):
        if keyword in lower:
            return key, "industry"
    # Fall back to sector
    for keyword, key in sorted(_KEYWORD_TO_YF_SECTOR.items(), key=lambda x: -len(x[0])):
        if keyword in lower:
            return key, "sector"
    return None, ""


def _get_yf_tickers(industry: str, num: int, yf_key_override: str = "") -> list[str]:
    """Discover tickers for an industry from Yahoo Finance, news-reordered.

    Primary source is ``yf.Industry`` / ``yf.Sector`` ``top_companies`` (Yahoo's
    authoritative rankings), then reordered so the most-discussed in YF news
    appear first. Both the company list (4h) and news counts (15min) are cached
    to avoid redundant HTTP calls on repeat runs.

    Parameters
    ----------
    industry : str
        Free-text industry name, used for keyword lookup and news search.
    num : int
        Target number of tickers to return.
    yf_key_override : str, optional
        Explicit YF industry key that bypasses keyword matching. Default ``""``.

    Returns
    -------
    list[str]
        Up to ``num`` base ticker symbols, or ``[]`` if no YF match.
    """
    from collections import Counter

    if yf_key_override:
        yf_key, yf_type = yf_key_override, "industry"
    else:
        yf_key, yf_type = _yf_lookup(industry)

    # --- Primary: Yahoo Finance authoritative top companies (cached 4h) ---
    base_tickers: list[str] = []
    if yf_key:
        cache_key = (yf_key, yf_type)
        with _yf_lock:
            cached = _yf_company_cache.get(cache_key)
        if cached is not None:
            base_tickers = cached
            logger.info("yf_{} {} → {} companies (cached)", yf_type, yf_key, len(base_tickers))
        else:
            try:
                obj = yf.Industry(yf_key) if yf_type == "industry" else yf.Sector(yf_key)
                companies = obj.top_companies
                if companies is not None and not companies.empty:
                    base_tickers = [
                        sym for sym in companies.index
                        if isinstance(sym, str) and "." not in sym
                    ][:num * 3]
                    with _yf_lock:
                        _yf_company_cache[cache_key] = base_tickers
                    logger.info("yf_{} {} → {} companies (fetched)", yf_type, yf_key, len(base_tickers))
            except Exception as e:
                logger.warning("yf_{} {} failed: {}", yf_type, yf_key, e)

    if not base_tickers:
        return []

    # --- News reordering: which of those are most discussed? (cached 15min) ---
    with _yf_lock:
        news_counts = _yf_news_cache.get(industry)
    if news_counts is not None:
        logger.info("yf_news '{}' mentions (cached): {}", industry, news_counts.most_common(4))
    else:
        news_counts = Counter()
        try:
            for term in [industry, f"{industry} stocks"]:
                s = yf.Search(term, max_results=1, news_count=10)
                for article in s.news:
                    for t in article.get("relatedTickers", []):
                        if "." not in t and "=" not in t and "^" not in t and t.isupper() and len(t) <= 5:
                            news_counts[t] += 1
            with _yf_lock:
                _yf_news_cache[industry] = news_counts
            logger.info("yf_news '{}' mentions (fetched): {}", industry, news_counts.most_common(4))
        except Exception as e:
            logger.warning("yf_news failed: {}", e)

    base_set = set(base_tickers)
    news_first = [t for t, _ in news_counts.most_common() if t in base_set]
    seen = set(news_first)
    rest = [t for t in base_tickers if t not in seen]

    combined = (news_first + rest)[:num]
    logger.info("yf_combined → {}", combined)
    return combined


_TICKER_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "tickers": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["tickers"],
})


def _claude_tickers(industry: str, num_tickers: int) -> tuple[list[str], dict]:
    """Ask Claude (with WebSearch) for trending tickers in an industry.

    Runs the Claude CLI with a JSON schema and parses ``structured_output``.

    Parameters
    ----------
    industry : str
        Industry to search news for.
    num_tickers : int
        Number of ticker symbols to request.

    Returns
    -------
    tuple[list[str], dict]
        ``(tickers, model_usage)`` — stripped ticker symbols and the CLI's
        ``modelUsage`` dict for token accounting.
    """
    system_prompt = (
        f"You are a financial analyst assistant. Find the {num_tickers} most actively discussed "
        f"and newsworthy companies in the {industry} industry right now based on current news."
    )
    user_message = (
        f"Search financial news for the most trending companies in the {industry} industry "
        f"right now. Return exactly {num_tickers} ticker symbols."
    )
    result = subprocess.run(
        [
            "claude", "-p", user_message,
            "--system-prompt", system_prompt,
            "--model", "haiku",
            "--tools", "WebSearch,WebFetch",
            "--permission-mode", "bypassPermissions",
            "--no-session-persistence",
            "--output-format", "json",
            "--json-schema", _TICKER_SCHEMA,
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    ticker_list = data.get("structured_output", {}).get("tickers", [])
    return [t.strip() for t in ticker_list], data.get("modelUsage", {})


def generate_ticker_ideas(industry, num_tickers: int = MAX_TICKERS_TO_ANALYZE, yf_key_override: str = "") -> dict[str, str]:
    """Build the ticker set for an industry, filling YF gaps with Claude.

    Uses Yahoo Finance first; if it returns fewer than ``num_tickers``, asks
    Claude for the remainder and merges without duplicates (YF first).

    Parameters
    ----------
    industry : str
        Free-text industry name.
    num_tickers : int, optional
        Target ticker count. Default ``MAX_TICKERS_TO_ANALYZE``.
    yf_key_override : str, optional
        Explicit YF industry key that bypasses keyword matching. Default ``""``.

    Returns
    -------
    dict[str, str]
        ``{ticker: "pending"}`` for each selected ticker.
    """
    yf_tickers = _get_yf_tickers(industry, num_tickers, yf_key_override)

    if len(yf_tickers) >= num_tickers:
        logger.info("generate_tickers using Yahoo Finance data: {}", yf_tickers[:num_tickers])
        return {t: "pending" for t in yf_tickers[:num_tickers]}

    # Not enough from YF — fill remaining with Claude web search
    remaining = num_tickers - len(yf_tickers)
    logger.info("generate_tickers YF gave {}, asking Claude for {} more", len(yf_tickers), remaining)
    claude_tickers, model_usage = _claude_tickers(industry, remaining)

    call_input = sum(u.get("inputTokens", 0) for u in model_usage.values())
    call_output = sum(u.get("outputTokens", 0) for u in model_usage.values())
    call_cache_read = sum(u.get("cacheReadInputTokens", 0) for u in model_usage.values())
    add_token_usage(call_input, call_output, call_cache_read)
    logger.info("tokens generate_tickers in={:,} out={:,}", call_input, call_output)

    # Merge: YF first, then Claude fills gaps (no duplicates)
    seen = set(yf_tickers)
    combined = list(yf_tickers)
    for t in claude_tickers:
        if t not in seen:
            combined.append(t)
            seen.add(t)
        if len(combined) >= num_tickers:
            break

    return {t: "pending" for t in combined[:num_tickers]}


_TRENDING_SEARCH_TERMS = [
    "stocks to watch today",
    "market movers",
    "earnings today",
    "stock news today",
    "most active stocks",
]

# Trending caches — 30-minute TTL so repeated runs in a session stay fast
_yf_trending_cache: TTLCache = TTLCache(maxsize=4, ttl=30 * 60)
_TRENDING_CACHE_KEY = "trending"
_yf_trending_industries_cache: TTLCache = TTLCache(maxsize=4, ttl=30 * 60)
_TRENDING_INDUSTRIES_CACHE_KEY = "trending_industries"


def get_trending_tickers(num: int = MAX_TICKERS_TO_ANALYZE) -> dict[str, str]:
    """Rank the most-mentioned tickers across trending YF news searches.

    Cached 30min. Filters out non-equity symbols (dots, ``=``, ``^``, longer
    than 5 chars, non-uppercase).

    Parameters
    ----------
    num : int, optional
        Number of tickers to return. Default ``MAX_TICKERS_TO_ANALYZE``.

    Returns
    -------
    dict[str, str]
        ``{ticker: "pending"}`` for the top ``num`` tickers.
    """
    from collections import Counter

    with _yf_lock:
        cached = _yf_trending_cache.get(_TRENDING_CACHE_KEY)
    if cached is not None:
        logger.info("trending {} (cached)", cached[:num])
        return {t: "pending" for t in cached[:num]}

    counts: Counter = Counter()
    for term in _TRENDING_SEARCH_TERMS:
        try:
            s = yf.Search(term, max_results=1, news_count=15)
            for article in s.news:
                for t in article.get("relatedTickers", []):
                    if "." not in t and "=" not in t and "^" not in t and t.isupper() and len(t) <= 5:
                        counts[t] += 1
        except Exception as e:
            logger.warning("trending search '{}' failed: {}", term, e)

    top = [t for t, _ in counts.most_common(num * 2)][:num]
    logger.info("trending top {}: {}  counts={}", num, top, counts.most_common(num))

    with _yf_lock:
        _yf_trending_cache[_TRENDING_CACHE_KEY] = top

    return {t: "pending" for t in top}


def get_trending_industries(num: int = 5) -> list[tuple[str, str]]:
    """Rank trending industries from the most-mentioned tickers in YF news.

    Scans YF news for top tickers (same terms as trending tickers), looks up
    each ticker's ``industryKey`` in parallel, then ranks industries by weighted
    mention count. Cached 30min.

    Parameters
    ----------
    num : int, optional
        Number of industries to return. Default ``5``.

    Returns
    -------
    list[tuple[str, str]]
        ``[(display_name, yf_key), ...]`` up to ``num`` entries; ``[]`` if no
        tickers were found.
    """
    from collections import Counter

    with _yf_lock:
        cached = _yf_trending_industries_cache.get(_TRENDING_INDUSTRIES_CACHE_KEY)
    if cached is not None:
        logger.info("trending_industries (cached) {}", cached[:num])
        return cached[:num]

    counts: Counter = Counter()
    for term in _TRENDING_SEARCH_TERMS:
        try:
            s = yf.Search(term, max_results=1, news_count=15)
            for article in s.news:
                for t in article.get("relatedTickers", []):
                    if "." not in t and "=" not in t and "^" not in t and t.isupper() and len(t) <= 5:
                        counts[t] += 1
        except Exception as e:
            logger.warning("trending_industries search '{}' failed: {}", term, e)

    top_tickers = [t for t, _ in counts.most_common(20)]
    if not top_tickers:
        return []

    ticker_industries: dict[str, tuple[str, str]] = {}

    def _fetch_industry(ticker: str) -> None:
        """Look up a ticker's industry and record it in ``ticker_industries``.

        Silently skips tickers whose info lacks an industry key/display.

        Parameters
        ----------
        ticker : str
            Ticker symbol to resolve.
        """
        try:
            info = yf.Ticker(ticker).info
            key = info.get("industryKey", "")
            display = info.get("industry", "")
            if key and display:
                ticker_industries[ticker] = (display, key)
        except Exception:
            pass

    threads = [
        threading.Thread(target=_fetch_industry, args=(t,), daemon=True)
        for t in top_tickers
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    industry_counts: Counter = Counter()
    for ticker, (display, key) in ticker_industries.items():
        industry_counts[(display, key)] += counts[ticker]

    result = [(d, k) for (d, k), _ in industry_counts.most_common(num * 2)][:num]
    logger.info("trending_industries (fetched) {}", result)

    with _yf_lock:
        _yf_trending_industries_cache[_TRENDING_INDUSTRIES_CACHE_KEY] = result

    return result


# Ordered sector keys (YF's own taxonomy) with display names for the UI
_YF_SECTOR_ORDER: list[tuple[str, str]] = [
    ("technology",            "Technology"),
    ("energy",                "Energy"),
    ("utilities",             "Utilities"),
    ("healthcare",            "Healthcare"),
    ("financial-services",    "Financials"),
    ("consumer-cyclical",     "Consumer Cyclical"),
    ("consumer-defensive",    "Consumer Defensive"),
    ("industrials",           "Industrials"),
    ("basic-materials",       "Basic Materials"),
    ("communication-services","Communications"),
    ("real-estate",           "Real Estate"),
]


def _key_to_display(key: str) -> str:
    """Title-case a hyphenated YF key for display.

    E.g. ``'oil-gas-e-p'`` → ``'Oil Gas E-P'`` (capitalizes each word, keeps
    hyphens as spaces).

    Parameters
    ----------
    key : str
        Hyphen-separated YF key.

    Returns
    -------
    str
        Human-readable display name.
    """
    return " ".join(w.capitalize() for w in key.split("-"))


def get_yf_industry_groups() -> list[tuple[str, list[tuple[str, str]]]]:
    """Fetch YF's full industry taxonomy grouped by sector.

    All sectors are fetched in parallel with a 10s per-thread timeout.

    Returns
    -------
    list[tuple[str, list[tuple[str, str]]]]
        ``[(sector_display, [(industry_display, industry_key), ...]), ...]`` in
        ``_YF_SECTOR_ORDER``; ``[]`` if all sectors fail (caller should use a
        hardcoded fallback).
    """
    results: dict[str, tuple[str, list[tuple[str, str]]]] = {}

    def _fetch(sector_key: str, sector_display: str) -> None:
        """Fetch one sector's industries and record them in ``results``.

        Parameters
        ----------
        sector_key : str
            YF sector key to query.
        sector_display : str
            Display name stored alongside the industry pairs.
        """
        try:
            df = yf.Sector(sector_key).industries
            if df is not None and not df.empty:
                pairs = [(_key_to_display(idx), idx) for idx in df.index if isinstance(idx, str)]
                if pairs:
                    results[sector_key] = (sector_display, pairs)
                    logger.debug("yf_sectors {}: {} industries", sector_key, len(pairs))
        except Exception as e:
            logger.warning("yf_sectors {} failed: {}", sector_key, e)

    threads = [
        threading.Thread(target=_fetch, args=(key, display), daemon=True)
        for key, display in _YF_SECTOR_ORDER
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    return [results[key] for key, _ in _YF_SECTOR_ORDER if key in results]


def _name_of(q: dict) -> str:
    """Extract a display name from a yfinance Search quote.

    Tries long/short name variants in both casings.

    Parameters
    ----------
    q : dict
        A yfinance ``Search.quotes`` entry.

    Returns
    -------
    str
        First non-empty name field, or ``""``.
    """
    return (q.get("longname") or q.get("longName")
            or q.get("shortname") or q.get("shortName") or "")


def _score_quote(query: str, q: dict) -> float:
    """Rank a yfinance Search quote by how well it matches ``query``.

    Score is ``max(name_score, sym_score) + overlap`` where name_score is the
    ``SequenceMatcher`` ratio (0-1), sym_score is a symbol-match bonus (0-1),
    and overlap is a token-overlap bonus (0-0.3). Higher = better.

    Parameters
    ----------
    query : str
        User's raw query.
    q : dict
        A yfinance ``Search.quotes`` entry.

    Returns
    -------
    float
        Match score; higher is a better match.
    """
    import difflib

    q_low = query.lower().strip()
    q_words = q_low.split()
    q_first = q_words[0] if q_words else q_low
    name = _name_of(q).lower()
    sym = (q.get("symbol") or "").lower()
    sym_base = sym.split(".")[0]  # ACS.MC → "acs"

    name_score = difflib.SequenceMatcher(None, q_low, name).ratio() if name else 0.0
    if sym == q_low:
        sym_score = 1.0
    elif sym_base == q_low:  # query is bare ticker "ACS" → ACS.MC matches
        sym_score = 0.9
    elif sym_base == q_first:  # query is "ACS Actividades..." → ACS.MC matches first word
        sym_score = 0.4
    elif sym.startswith(q_low):
        sym_score = 0.5
    else:
        sym_score = 0.0

    q_tokens = {t for t in q_words if len(t) > 1}
    n_tokens = set(name.split())
    overlap = (len(q_tokens & n_tokens) / len(q_tokens)) * 0.3 if q_tokens else 0.0

    return max(name_score, sym_score) + overlap


def resolve_ticker_verbose(query: str) -> dict | None:
    """Resolve ``query`` to the best-matching ticker plus alternatives.

    Allows dot-tickers (``ACS.MC``, ``BMW.DE``) — picks whichever EQUITY scores
    highest by name match, not the first dotless one.

    Parameters
    ----------
    query : str
        Ticker or company name to resolve.

    Returns
    -------
    dict | None
        ``{symbol, name, exchange, score, alternatives}`` where ``alternatives``
        is a list of ``(symbol, name, score)`` tuples for the next-best matches;
        ``None`` if no candidates.
    """
    query = query.strip()
    if not query:
        return None
    try:
        quotes = yf.Search(query, max_results=10).quotes
    except Exception as e:
        logger.warning("resolve_ticker '{}' failed: {}", query, e)
        return None

    if not quotes:
        return None

    equities = [q for q in quotes if q.get("quoteType", "") == "EQUITY"]
    candidates = equities or quotes

    scored = sorted(
        ((q, _score_quote(query, q)) for q in candidates),
        key=lambda x: x[1],
        reverse=True,
    )
    top_q, top_score = scored[0]
    sym = top_q.get("symbol", "")
    if not sym:
        return None

    alternatives = [
        (q.get("symbol", ""), _name_of(q), round(score, 2))
        for q, score in scored[1:5]
        if q.get("symbol")
    ]
    logger.info(
        "resolve_ticker '{}' → {} (score={:.2f}, name={!r})  alts: {}",
        query, sym, top_score, _name_of(top_q),
        ", ".join(f"{s}({sc:.2f})" for s, _, sc in alternatives) or "none",
    )
    return {
        "symbol": sym,
        "name": _name_of(top_q),
        "exchange": top_q.get("exchange", ""),
        "score": round(top_score, 2),
        "alternatives": alternatives,
    }


def resolve_ticker(query: str) -> str | None:
    """Resolve ``query`` to a single top ticker symbol.

    Back-compat wrapper around :func:`resolve_ticker_verbose`.

    Parameters
    ----------
    query : str
        Ticker or company name to resolve.

    Returns
    -------
    str | None
        The top-scoring symbol, or ``None`` if unresolved.
    """
    info = resolve_ticker_verbose(query)
    return info["symbol"] if info else None
