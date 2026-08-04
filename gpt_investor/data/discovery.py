import os
import subprocess
import threading
import json

import requests
import yfinance as yf
from cachetools import TTLCache
from loguru import logger

from gpt_investor.llm.claude import add_token_usage
from gpt_investor.data.fundamentals import fetch_fundamentals, score_fundamentals
from gpt_investor.data.wyckoff import fetch_ohlcv_batch, compute_signals, score_wyckoff
from gpt_investor.data.market_regime import get_market_regime

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
    """Return (key, type) where type is 'industry' or 'sector', or (None, '') if no match."""
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
    """
    Yahoo Finance-centric ticker discovery:
    1. Primary: yf.Industry / yf.Sector top_companies (Yahoo's own authoritative rankings)
    2. News reordering: most-discussed in YF news appears first
    Both results are cached to avoid redundant HTTP calls on repeat runs.
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

# Trending-industries cache — 30-minute TTL so repeated runs in a session stay fast
_yf_trending_industries_cache: TTLCache = TTLCache(maxsize=4, ttl=30 * 60)
_TRENDING_INDUSTRIES_CACHE_KEY = "trending_industries"


# --- signal-driven setup discovery (PD') -----------------------------------
#
# Supersedes the old "what's trendy on Yahoo" mover funnel. Yahoo screens are
# demoted to a cheap candidate universe; the tool's OWN deterministic scores
# (fundamentals tier + Wyckoff phase + regime fit) decide what surfaces. So a
# day_loser basing in accumulation outranks a day_gainer topping in
# distribution — the chart sorts dip-buys from falling knives.

_SETUP_SCREENS = ("most_actives", "day_gainers", "day_losers")
_SCREEN_COUNT = 50          # quotes pulled per predefined screen
_PREFILTER_N = 60           # cap on names scored (bounds the fundamentals cost)
_TRENDING_MAX = 15          # trending-endpoint symbols folded into the universe
_GOOD_PHASES = frozenset({"accumulation", "markup"})

_yf_setup_cache: TTLCache = TTLCache(maxsize=4, ttl=15 * 60)
_SETUP_CACHE_KEY = "setups"

_NON_EQUITY_MARKERS = ("=", "^")


def _is_equity_symbol(sym: str) -> bool:
    """Test whether a symbol is a plain equity ticker.

    Rejects FX (`EURUSD=X`), futures (`GC=F`) and indices (`^GSPC`).

    Parameters
    ----------
    sym : str
        Ticker symbol to check.

    Returns
    -------
    bool
        True if non-empty and free of non-equity markers.
    """
    return bool(sym) and not any(m in sym for m in _NON_EQUITY_MARKERS)


def _is_equity_quote(q: dict) -> bool:
    """Test whether a yfinance quote is a plain equity.

    Parameters
    ----------
    q : dict
        yfinance quote with `quoteType` and `symbol` keys.

    Returns
    -------
    bool
        True if `quoteType` is EQUITY and the symbol passes `_is_equity_symbol`.
    """
    return q.get("quoteType") == "EQUITY" and _is_equity_symbol(q.get("symbol", ""))


def _dollar_volume(q: dict) -> float:
    """Compute traded dollar-volume for a quote.

    Falls back to 3-month average daily volume when live volume is absent.

    Parameters
    ----------
    q : dict
        yfinance quote with price and volume keys.

    Returns
    -------
    float
        Price times volume; 0.0 when either input is missing.
    """
    px = q.get("regularMarketPrice") or 0
    vol = q.get("regularMarketVolume") or q.get("averageDailyVolume3Month") or 0
    return float(px) * float(vol)


def _trending_endpoint() -> list[str]:
    """Fetch Yahoo's trending-tickers endpoint, best-effort.

    Only widens the candidate universe, so any failure is harmless.

    Returns
    -------
    list[str]
        Equity symbols from the trending feed; empty on any error.
    """
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/trending/US",
            params={"count": _TRENDING_MAX},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        r.raise_for_status()
        quotes = r.json().get("finance", {}).get("result", [{}])[0].get("quotes", [])
        syms = [q.get("symbol", "") for q in quotes]
        return [s for s in syms if _is_equity_symbol(s)]
    except Exception as e:
        logger.warning("trending endpoint failed: {}", e)
        return []


def _screen_universe() -> tuple[dict[str, dict], list[str]]:
    """Pull the candidate universe from Yahoo screens + trending endpoint.

    Yahoo only *lists* here — no ranking value is taken from the screen order.

    Returns
    -------
    tuple[dict[str, dict], list[str]]
        `(quotes_by_symbol, trending_symbols)`; the first maps symbol to its
        screen quote, the second is the trending-endpoint symbols.
    """
    quotes_by_sym: dict[str, dict] = {}
    for name in _SETUP_SCREENS:
        try:
            res = yf.screen(name, count=_SCREEN_COUNT)
            for q in res.get("quotes", []):
                sym = q.get("symbol")
                if sym and _is_equity_quote(q):
                    quotes_by_sym.setdefault(sym, q)
        except Exception as e:
            logger.warning("screen '{}' failed: {}", name, e)
    trending = _trending_endpoint()
    logger.info("setup universe: {} screened equities, {} trending", len(quotes_by_sym), len(trending))
    return quotes_by_sym, trending


def _prefilter_by_dollar_volume(quotes_by_sym: dict[str, dict], trending: list[str], top: int) -> list[str]:
    """Rank the universe by dollar-volume, forcing trending equities in.

    Trending symbols are prepended even without a screen quote to rank on, so
    they widen coverage. Bounds the downstream fundamentals cost.

    Parameters
    ----------
    quotes_by_sym : dict[str, dict]
        Symbol to screen quote.
    trending : list[str]
        Trending symbols to force-include, in order.
    top : int
        Cap on the returned list length.

    Returns
    -------
    list[str]
        Up to `top` symbols: trending first, then remaining by descending
        dollar-volume.
    """
    ranked = sorted(quotes_by_sym.values(), key=_dollar_volume, reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for sym in trending[:_TRENDING_MAX]:
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    for q in ranked:
        sym = q.get("symbol")
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
        if len(out) >= top:
            break
    return out[:top]


# --- pure scoring (testable without yfinance) ------------------------------

# Timing overlay: does the Wyckoff phase suit the macro regime? Bonus in pp
# added to the base setup score. First version — tuned by intuition, not fit.
_REGIME_FIT = {
    "risk-on-bull":       {"markup": 1.5, "accumulation": 1.0, "neutral": 0.0, "distribution": -1.0, "markdown": -1.5},
    "panic-opportunity":  {"accumulation": 1.5, "markup": 0.5, "neutral": 0.0, "distribution": -0.5, "markdown": -1.0},
    "late-cycle-caution": {"accumulation": 0.5, "markup": 0.0, "neutral": 0.0, "distribution": -1.0, "markdown": -1.5},
    "recession-warning":  {"accumulation": 0.0, "markup": -0.5, "neutral": -0.5, "distribution": -1.5, "markdown": -2.0},
    "mixed":              {"markup": 0.5, "accumulation": 0.5, "neutral": 0.0, "distribution": -0.5, "markdown": -1.0},
}


def _regime_fit(fund_tier: str, wyckoff_phase: str, regime_label: str | None) -> float:
    """Look up the timing bonus for a Wyckoff phase under a macro regime.

    Parameters
    ----------
    fund_tier : str
        Fundamentals tier (unused by the current table; kept for signature).
    wyckoff_phase : str
        Wyckoff phase, e.g. `accumulation`, `markup`.
    regime_label : str | None
        Macro-regime label; falls back to `mixed` when None or unknown.

    Returns
    -------
    float
        Bonus in percentage points; 0.0 for an unmapped phase.
    """
    table = _REGIME_FIT.get(regime_label or "mixed", _REGIME_FIT["mixed"])
    return table.get(wyckoff_phase, 0.0)


def _setup_score(fund: dict, wyckoff: dict, regime_label: str | None) -> float:
    """Blend fundamentals, Wyckoff timing, regime fit and a soft gate.

    Soft gate (+1) rewards the ideal setup — a Solid/Strong company in
    accumulation or markup — without hard-excluding anything else.

    Parameters
    ----------
    fund : dict
        Fundamentals result with `score` and `tier` keys.
    wyckoff : dict
        Wyckoff result with `score` and `phase` keys.
    regime_label : str | None
        Macro-regime label passed through to `_regime_fit`.

    Returns
    -------
    float
        Composite setup score, rounded to 2 decimals.
    """
    base = 0.55 * fund.get("score", 0.0) + 0.45 * wyckoff.get("score", 0.0)
    fit = _regime_fit(fund.get("tier", ""), wyckoff.get("phase", ""), regime_label)
    gate = 1.0 if (fund.get("tier") in ("Strong", "Solid") and wyckoff.get("phase") in _GOOD_PHASES) else 0.0
    return round(base + fit + gate, 2)


def _why_chip(fund: dict, wyckoff: dict, regime_label: str | None) -> str:
    """Build the short "why" label shown on a setup card.

    Parameters
    ----------
    fund : dict
        Fundamentals result with a `tier` key.
    wyckoff : dict
        Wyckoff result with a `phase` key.
    regime_label : str | None
        Macro-regime label used to derive the regime tag.

    Returns
    -------
    str
        `"<tier> • <phase> • <±/~regime>"`.
    """
    fit = _regime_fit(fund.get("tier", ""), wyckoff.get("phase", ""), regime_label)
    regime_tag = "+regime" if fit > 0 else "-regime" if fit < 0 else "~regime"
    return f"{fund.get('tier', '?')} • {wyckoff.get('phase', '?')} • {regime_tag}"


def get_setup_candidates(num: int = MAX_TICKERS_TO_ANALYZE) -> dict[str, dict]:
    """Run signal-driven discovery and return the top setups.

    Screen a universe, prefilter by liquidity, score each name with the tool's
    own deterministic signals, rank by composite setup score, truncate to
    `num`. 15-minute TTL cache; the cache holds the full ranking so a smaller
    `num` reslices without recomputing.

    Parameters
    ----------
    num : int, optional
        Number of setups to return; defaults to `MAX_TICKERS_TO_ANALYZE`.

    Returns
    -------
    dict[str, dict]
        Ordered `{ticker: {status, fund, wyckoff, setup_score, why}}` so
        `state.py` can reuse the scores on the cards without re-fetching;
        empty on an empty universe.
    """
    with _yf_lock:
        cached = _yf_setup_cache.get(_SETUP_CACHE_KEY)
    if cached is not None:
        logger.info("setups (cached) {}", list(cached)[:num])
        return dict(list(cached.items())[:num])

    quotes_by_sym, trending = _screen_universe()
    if not quotes_by_sym and not trending:
        logger.warning("setup discovery: empty universe")
        return {}
    prefiltered = _prefilter_by_dollar_volume(quotes_by_sym, trending, _PREFILTER_N)

    try:
        regime_label = get_market_regime().get("label")
    except Exception as e:
        logger.warning("setup discovery regime fetch failed: {}", e)
        regime_label = None

    ohlcv = fetch_ohlcv_batch(prefiltered)

    scored: dict[str, dict] = {}
    lock = threading.Lock()

    def _score_one(sym: str) -> None:
        """Score one symbol and store the entry under `lock`.

        Runs per-thread; swallows and logs any failure so one bad ticker
        never sinks the batch.

        Parameters
        ----------
        sym : str
            Ticker to score.
        """
        try:
            fund = score_fundamentals(fetch_fundamentals(sym))
            wyckoff = score_wyckoff(compute_signals(ohlcv.get(sym)))
            entry = {
                "status": "pending",
                "fund": fund,
                "wyckoff": wyckoff,
                "setup_score": _setup_score(fund, wyckoff, regime_label),
                "why": _why_chip(fund, wyckoff, regime_label),
            }
            with lock:
                scored[sym] = entry
        except Exception as e:
            logger.warning("setup score '{}' failed: {}", sym, e)

    threads = [threading.Thread(target=_score_one, args=(s,), daemon=True) for s in prefiltered]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    ranked = sorted(scored.items(), key=lambda kv: kv[1]["setup_score"], reverse=True)
    result = dict(ranked[:num])
    logger.info(
        "setups ranked (regime={}): {}",
        regime_label,
        [(s, e["setup_score"], e["why"]) for s, e in ranked[:num]],
    )

    with _yf_lock:
        _yf_setup_cache[_SETUP_CACHE_KEY] = dict(ranked)
    return result


def get_trending_industries(num: int = 5) -> list[tuple[str, str]]:
    """
    Find trending industries by:
    1. Scanning YF news for the most-mentioned tickers (same terms as trending tickers)
    2. Looking up each ticker's industryKey in parallel
    3. Ranking industries by weighted mention count
    Returns [(display_name, yf_key), ...] up to `num` entries.
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
    """'oil-gas-e-p' → 'Oil Gas E-P'  (title-case, preserve hyphens after first word)"""
    return " ".join(w.capitalize() for w in key.split("-"))


def get_yf_industry_groups() -> list[tuple[str, list[tuple[str, str]]]]:
    """
    Fetch YF's full industry taxonomy grouped by sector.
    All sectors are fetched in parallel; each has a 10s timeout.
    Returns [] if all sectors fail (caller should use a hardcoded fallback).
    """
    results: dict[str, tuple[str, list[tuple[str, str]]]] = {}

    def _fetch(sector_key: str, sector_display: str) -> None:
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
    return (q.get("longname") or q.get("longName")
            or q.get("shortname") or q.get("shortName") or "")


def _score_quote(query: str, q: dict) -> float:
    """Rank a yfinance Search quote by how well it matches `query`.

    Sum of:
      * `SequenceMatcher` ratio between query and longName/shortName  (0-1)
      * symbol-exact-match bonus                                       (0-1)
      * token-overlap bonus (query words present in name)              (0-0.3)
    Higher = better.
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
    """Resolve `query` to the best-matching ticker + return alternatives.

    Returns `{symbol, name, exchange, score, alternatives}` or None if no
    candidates. Allows dot-tickers (`ACS.MC`, `BMW.DE`) — picks whichever
    EQUITY scores highest by name match, not the first dotless one.

    `alternatives` is a list of `(symbol, name, score)` tuples for the
    next-best matches, useful for surfacing to the user when the top
    pick looks wrong.
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
    """Back-compat wrapper — returns only the top symbol or None."""
    info = resolve_ticker_verbose(query)
    return info["symbol"] if info else None
