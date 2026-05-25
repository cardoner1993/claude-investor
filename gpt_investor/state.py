import asyncio
import time
import markdown as md_lib

import reflex as rx
from loguru import logger

from gpt_investor.storage.cache import get_cached, save_cached, get_cached_liquidity
from gpt_investor.llm.analysis import (
    get_sentiment_analysis,
    get_industry_analysis,
    get_final_analysis,
    get_liquidity_context,
)
from gpt_investor.data.market_data import (
    get_current_price,
    get_company_name,
    get_news,
    get_analyst_ratings,
)
from gpt_investor.data.fundamentals import fetch_fundamentals, score_fundamentals, format_fundamentals
from gpt_investor.data.market_regime import get_market_regime
from gpt_investor.data.sentiment import chip_label, chip_color, format_for_llm as format_sentiment_for_llm
from gpt_investor.data.discovery import (
    MAX_TICKERS_TO_ANALYZE,
    generate_ticker_ideas,
    get_trending_tickers,
    get_trending_industries,
    resolve_ticker,
)
from gpt_investor.llm.claude import get_token_totals

# tier -> Radix color_scheme used for the fundamental-score badge on each card.
_TIER_COLORS = {
    "Strong":  "green",
    "Solid":   "jade",
    "Average": "amber",
    "Weak":    "orange",
    "Avoid":   "red",
}


def _log(ticker: str, step: str, elapsed: float | None = None):
    if elapsed is not None:
        logger.bind(ticker=ticker).info("{}  ({:.1f}s)", step, elapsed)
    else:
        logger.bind(ticker=ticker).info("{}", step)


def _resolve_single(query: str) -> dict[str, str]:
    """Resolve a company name or ticker to a single-entry pending dict."""
    ticker = resolve_ticker(query)
    if ticker:
        return {ticker: "pending"}
    logger.warning("single: could not resolve '{}'", query)
    return {}


async def _analyze_ticker(
    state,
    ticker: str,
    industry_task: "asyncio.Task[str] | None",
    liquidity_context: str = "",
    regime: dict | None = None,
):
    """`industry_task` is a Task whose result is the industry analysis string.
    Cached-path tickers don't need it (skipped). Live-path tickers await it
    just before the sonnet call so it runs in parallel with sentiment/news/etc.
    None means caller pre-determined no tickers needed it (all cached).
    """
    ticker_start = time.time()
    try:
        async with state:
            state.tickers[ticker] = "processing"

        cached = get_cached(ticker)
        if cached:
            _log(ticker, "cache hit — skipping pipeline")
            price, name, fund_raw = await asyncio.gather(
                asyncio.to_thread(get_current_price, ticker),
                asyncio.to_thread(get_company_name, ticker),
                asyncio.to_thread(fetch_fundamentals, ticker),
            )
            scored = score_fundamentals(fund_raw)
            fund_block = format_fundamentals(scored)
            sentiment_dict = cached.get("sentiment_dict")
            final_analysis = cached["final_analysis"]
            async with state:
                state.names[ticker] = name
                state.analyses[ticker] = final_analysis
                state.fund_summary[ticker] = f"{scored['tier']} {scored['score']}"
                state.fund_color[ticker] = _TIER_COLORS.get(scored["tier"], "gray")
                state.fund_block[ticker] = fund_block
                if sentiment_dict:
                    state.sent_summary[ticker] = chip_label(sentiment_dict["score"], sentiment_dict["confidence"])
                    state.sent_color[ticker] = chip_color(sentiment_dict["score"], sentiment_dict["confidence"])
                    state.sent_block[ticker] = format_sentiment_for_llm(sentiment_dict)
                state.tickers[ticker] = "cached"
                if state.selected_ticker == ticker:
                    state.selected_name = name
                    state.selected_analysis = final_analysis
                    state.selected_analysis_html = md_lib.markdown(
                        final_analysis, extensions=["nl2br", "sane_lists"]
                    )
            _log(ticker, f"DONE (cached) fund={scored['tier']} {scored['score']}", time.time() - ticker_start)
            return

        t = time.time()
        _log(ticker, "fetching news")
        news = await asyncio.to_thread(get_news, ticker)
        _log(ticker, f"got {len(news)} articles", time.time() - t)

        _log(ticker, "running sentiment + ratings + price + fundamentals (parallel)")
        t = time.time()
        sentiment, analyst_ratings, price, name, fund_raw = await asyncio.gather(
            asyncio.to_thread(get_sentiment_analysis, ticker, news),
            asyncio.to_thread(get_analyst_ratings, ticker),
            asyncio.to_thread(get_current_price, ticker),
            asyncio.to_thread(get_company_name, ticker),
            asyncio.to_thread(fetch_fundamentals, ticker),
        )
        scored = score_fundamentals(fund_raw)
        fund_block = format_fundamentals(scored)
        sent_block = format_sentiment_for_llm(sentiment)
        _log(
            ticker,
            f"all parallel done  price={price:.2f}  fund={scored['tier']} {scored['score']}  "
            f"sent={sentiment['score']:+.2f}({sentiment['confidence']})",
            time.time() - t,
        )

        # publish fast signals to the UI before the slow sonnet call runs
        async with state:
            state.fund_summary[ticker] = f"{scored['tier']} {scored['score']}"
            state.fund_color[ticker] = _TIER_COLORS.get(scored["tier"], "gray")
            state.fund_block[ticker] = fund_block
            state.sent_summary[ticker] = chip_label(sentiment["score"], sentiment["confidence"])
            state.sent_color[ticker] = chip_color(sentiment["score"], sentiment["confidence"])
            state.sent_block[ticker] = sent_block

        # Await the industry analysis only now (started in parallel by fetch_analyses).
        # On any error or if caller didn't start one (all-cached run interrupted by a
        # late miss — shouldn't happen but guard anyway), fall back to empty string.
        industry_analysis = ""
        if industry_task is not None:
            try:
                industry_analysis = await industry_task
            except Exception as e:
                _log(ticker, f"industry analysis failed: {e}")

        _log(ticker, "running final analysis (sonnet)")
        t = time.time()
        final_analysis = await asyncio.to_thread(
            get_final_analysis,
            ticker, price, sentiment, analyst_ratings,
            industry_analysis, liquidity_context, scored, regime,
        )
        _log(ticker, "final analysis done", time.time() - t)

        save_cached(ticker, sentiment, analyst_ratings, final_analysis)

        totals = get_token_totals()
        async with state:
            state.names[ticker] = name
            state.analyses[ticker] = final_analysis
            state.tickers[ticker] = "finished"
            state.input_tokens = totals["input"]
            state.output_tokens = totals["output"]
            state.cache_read_tokens = totals["cache_read"]
            if state.selected_ticker == ticker:
                state.selected_name = name
                state.selected_analysis = final_analysis
                state.selected_analysis_html = md_lib.markdown(
                    final_analysis, extensions=["nl2br", "sane_lists"]
                )

        _log(ticker, "DONE", time.time() - ticker_start)
    except Exception as e:
        _log(ticker, f"FAILED: {e}")
        async with state:
            state.tickers[ticker] = "error"


class State(rx.State):

    industry: str = ""
    industry_input: str = ""
    discovery_mode: str = "industry"
    direct_yf_key: str = ""
    company_query: str = ""
    trending_industries: list[list[str]] = []
    trending_industries_loading: bool = False
    expanded_sectors: list[str] = []
    tickers: dict[str, str]
    analyses: dict[str, str]
    names: dict[str, str]
    fund_summary: dict[str, str] = {}   # ticker -> "Solid 6.8"
    fund_color: dict[str, str] = {}     # ticker -> color_scheme name
    fund_block: dict[str, str] = {}     # ticker -> markdown block for dialog
    sent_summary: dict[str, str] = {}   # ticker -> "+0.42 high"
    sent_color: dict[str, str] = {}     # ticker -> Radix color
    sent_block: dict[str, str] = {}     # ticker -> markdown block for dialog
    liquidity_context: str = ""
    liquidity_html: str = ""
    liquidity_is_mock: bool = False
    error_message: str = ""
    stage: str = "stopped"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    selected_ticker: str = ""
    selected_name: str = ""
    selected_is_cached: bool = False
    selected_analysis: str = ""
    selected_analysis_html: str = ""
    selected_fund_html: str = ""
    selected_fund_summary: str = ""
    selected_fund_color: str = ""
    selected_sent_html: str = ""
    selected_sent_summary: str = ""
    selected_sent_color: str = ""

    @rx.var
    def all_done(self) -> bool:
        return (
            len(self.tickers) > 0
            and all(v in ("finished", "cached", "error") for v in self.tickers.values())
        )

    def load_mock_data(self):
        """Inject fake finished tickers to test the dialog without running analysis."""
        mock_analysis = (
            "**Verdict**: Buy — strong momentum with solid fundamentals.\n\n"
            "**Price Target**: $150  (current: $120.00)\n\n"
            "**Thesis**: The company is well-positioned in a growing market with expanding margins. "
            "Recent product launches have been well received and analyst sentiment is improving.\n\n"
            "**Positives**:\n- Revenue growing 20% YoY\n- Strong free cash flow\n- Market leader in its segment\n\n"
            "**Risks**:\n- Valuation is stretched at current levels\n- Competition intensifying\n- Macro headwinds could dampen demand"
        )
        self.industry = "test"
        self.industry_input = ""
        self.stage = "done"
        self.tickers = {"MOCK": "finished", "FAIL": "error", "WAIT": "processing"}
        self.analyses = {"MOCK": mock_analysis}
        self.names = {"MOCK": "Mock Corporation", "FAIL": "Failed Inc.", "WAIT": "Waiting Ltd."}
        self.fund_summary = {"MOCK": "Solid 6.8"}
        self.fund_color = {"MOCK": "jade"}
        self.fund_block = {"MOCK": "**Fundamental score: 6.8/10 (Solid)**\n\n- Valuation 7.0/10\n- Growth 7.0/10"}
        self.sent_summary = {"MOCK": "+0.42 high"}
        self.sent_color = {"MOCK": "green"}
        self.sent_block = {"MOCK": "**Sentiment**: score +0.42 (high confidence) — VADER +0.30, LLM +0.50\n\nUpbeat coverage on product launches and analyst upgrades.\n\n- Bull: new product ramp\n- Bull: margin expansion\n- Bear: regulator probe ongoing"}
        mock_liquidity = (
            "**Global Liquidity Snapshot**\n\n"
            "**Fed (US)**: 4.50% — neutral — Holding rates steady with gradual QT continuing.\n"
            "**ECB (EU)**: 2.50% — easing — Cut rates twice in 2025, further cuts expected.\n"
            "**PBOC (China)**: 9.50% RRR — easing — Cut RRR 25bps in Q1 2025 to support growth.\n\n"
            "**Macro backdrop**: Mixed global liquidity — EU and China easing offset by Fed caution; "
            "net slightly supportive for equities but upside capped by US tightness."
        )
        self.liquidity_context = mock_liquidity
        self.liquidity_html = md_lib.markdown(mock_liquidity, extensions=["nl2br", "sane_lists"])
        self.liquidity_is_mock = True

    def _reset_tickers(self):
        self.tickers = {}
        self.analyses = {}
        self.names = {}
        self.fund_summary = {}
        self.fund_color = {}
        self.fund_block = {}
        self.sent_summary = {}
        self.sent_color = {}
        self.sent_block = {}
        self.selected_ticker = ""

    def set_industry_input(self, value: str):
        self.industry_input = value

    def industry_pick(self, label: str, yf_key: str):
        self.industry = label
        self.industry_input = label
        self.discovery_mode = "industry"
        self.direct_yf_key = yf_key
        self._reset_tickers()
        self.stage = "analyzing"
        return State.fetch_analyses

    def trending_pick(self):
        self.industry = "Today's Trending"
        self.industry_input = ""
        self.discovery_mode = "trending"
        self.direct_yf_key = ""
        self._reset_tickers()
        self.stage = "analyzing"
        return State.fetch_analyses

    @rx.event(background=True)
    async def fetch_trending_industries(self):
        async with self:
            self.trending_industries_loading = True
            self.trending_industries = []
        industries = await asyncio.to_thread(get_trending_industries)
        async with self:
            self.trending_industries = [[d, k] for d, k in industries]
            self.trending_industries_loading = False

    def toggle_sector(self, sector: str):
        if sector in self.expanded_sectors:
            self.expanded_sectors.remove(sector)
        else:
            self.expanded_sectors.append(sector)

    def quick_pick(self, label: str):
        self.industry_input = label
        self.industry = label
        self.discovery_mode = "industry"
        self.direct_yf_key = ""
        self._reset_tickers()
        self.stage = "analyzing"
        return State.fetch_analyses

    def open_ticker(self, ticker: str):
        self.selected_ticker = ticker
        self.selected_name = self.names.get(ticker, "")
        self.selected_is_cached = self.tickers.get(ticker) == "cached"
        raw = self.analyses.get(ticker, "")
        self.selected_analysis = raw
        self.selected_analysis_html = md_lib.markdown(raw, extensions=["nl2br", "sane_lists"])
        fund_block = self.fund_block.get(ticker, "")
        self.selected_fund_html = (
            md_lib.markdown(fund_block, extensions=["nl2br", "sane_lists"]) if fund_block else ""
        )
        self.selected_fund_summary = self.fund_summary.get(ticker, "")
        self.selected_fund_color = self.fund_color.get(ticker, "")
        sent_block = self.sent_block.get(ticker, "")
        self.selected_sent_html = (
            md_lib.markdown(sent_block, extensions=["nl2br", "sane_lists"]) if sent_block else ""
        )
        self.selected_sent_summary = self.sent_summary.get(ticker, "")
        self.selected_sent_color = self.sent_color.get(ticker, "")

    def close_ticker(self):
        self.selected_ticker = ""
        self.selected_name = ""
        self.selected_is_cached = False
        self.selected_analysis = ""
        self.selected_analysis_html = ""
        self.selected_fund_html = ""
        self.selected_fund_summary = ""
        self.selected_fund_color = ""
        self.selected_sent_html = ""
        self.selected_sent_summary = ""
        self.selected_sent_color = ""

    def handle_submit(self, data: dict):
        industry = data.get("industry", "").strip()
        if industry:
            self.industry = industry
            self.industry_input = industry
            self.discovery_mode = "industry"
            self.direct_yf_key = ""
            self._reset_tickers()
            self.stage = "analyzing"
            return State.fetch_analyses

    def set_company_query(self, value: str):
        self.company_query = value

    def handle_company_submit(self, data: dict):
        query = data.get("company", "").strip()
        if query:
            self.industry = query
            self.company_query = query
            self.discovery_mode = "single"
            self.direct_yf_key = ""
            self._reset_tickers()
            self.stage = "analyzing"
            return State.fetch_analyses

    @rx.event(background=True)
    async def fetch_analyses(self):
        async with self:
            self.error_message = ""
        run_start = time.time()
        logger.info("=" * 50)
        if self.discovery_mode == "single":
            logger.info("run starting analysis for company: {}", self.company_query or self.industry)
        elif self.discovery_mode == "trending":
            logger.info("run starting analysis for trending tickers")
        else:
            logger.info("run starting analysis for industry: {}", self.industry)

        t = time.time()
        session_needs_liquidity = not self.liquidity_context or self.liquidity_is_mock
        liquidity_source = "session"
        disk_liq: str | None = None
        if session_needs_liquidity:
            disk_liq = await asyncio.to_thread(get_cached_liquidity)

        if self.discovery_mode == "trending":
            discover = asyncio.to_thread(get_trending_tickers)
        elif self.discovery_mode == "single":
            discover = asyncio.to_thread(_resolve_single, self.company_query)
        else:
            discover = asyncio.to_thread(generate_ticker_ideas, self.industry, MAX_TICKERS_TO_ANALYZE, self.direct_yf_key)

        # Market regime (VIX / yield curve / HY / DXY / gold) — cheap yfinance
        # bundle, ~2-5s. Run every analysis (no cache; intraday data matters).
        regime_task = asyncio.create_task(asyncio.to_thread(get_market_regime))

        need_fetch = session_needs_liquidity and disk_liq is None
        if need_fetch:
            tickers_dict, liquidity_context = await asyncio.gather(
                discover, asyncio.to_thread(get_liquidity_context),
            )
            liquidity_html = md_lib.markdown(liquidity_context, extensions=["nl2br", "sane_lists"])
            liquidity_source = "fetched"
        else:
            tickers_dict = await discover
            if disk_liq is not None:
                liquidity_context = disk_liq
                liquidity_html = md_lib.markdown(liquidity_context, extensions=["nl2br", "sane_lists"])
                liquidity_source = "disk-cache"
            else:
                liquidity_context = self.liquidity_context
                liquidity_html = self.liquidity_html
        need_liquidity = need_fetch  # preserved name for downstream state update
        logger.info("run tickers: {}  ({:.1f}s)", list(tickers_dict.keys()), time.time() - t)
        logger.info("run liquidity: {}", liquidity_source)

        if not tickers_dict:
            async with self:
                self.liquidity_context = liquidity_context
                self.liquidity_html = liquidity_html
                if need_liquidity:
                    self.liquidity_is_mock = False
                self.error_message = f"No company found for '{self.company_query}'"
                self.stage = "stopped"
            return

        async with self:
            self.tickers = tickers_dict
            self.liquidity_context = liquidity_context
            self.liquidity_html = liquidity_html
            if need_liquidity:
                self.liquidity_is_mock = False

        # Pre-check cache: if every ticker is already analyzed today, skip the
        # industry-analysis Claude call entirely (cached path doesn't use it).
        cache_hits = await asyncio.to_thread(
            lambda: {t: get_cached(t) is not None for t in tickers_dict}
        )
        any_live = not all(cache_hits.values())
        first_ticker = next(iter(tickers_dict))

        industry_task: "asyncio.Task[str] | None" = None
        if any_live:
            # Start industry analysis in parallel with per-ticker fan-out.
            # Live tickers await it just before their sonnet call.
            industry_task = asyncio.create_task(
                asyncio.to_thread(get_industry_analysis, first_ticker)
            )
            logger.info(
                "run industry analysis scheduled (parallel with tickers)  cached={}/{}",
                sum(cache_hits.values()), len(tickers_dict),
            )
        else:
            logger.info("run industry analysis SKIPPED  all {} tickers cached", len(tickers_dict))

        # Resolve market regime (started in parallel above). Skip-cached path
        # tickers don't need it (verdict already cached); live tickers consume it.
        regime: dict | None = None
        if any_live:
            try:
                regime = await regime_task
            except Exception as e:
                logger.warning("market regime fetch failed: {}", e)
        else:
            regime_task.cancel()

        totals = get_token_totals()
        async with self:
            self.input_tokens = totals["input"]
            self.output_tokens = totals["output"]
            self.cache_read_tokens = totals["cache_read"]

        t = time.time()
        await asyncio.gather(*[
            _analyze_ticker(self, ticker, industry_task, liquidity_context, regime)
            for ticker in tickers_dict
        ])
        logger.info("run all tickers done  ({:.1f}s)", time.time() - t)

        totals = get_token_totals()
        async with self:
            self.input_tokens = totals["input"]
            self.output_tokens = totals["output"]
            self.cache_read_tokens = totals["cache_read"]
            self.stage = "done"

        logger.info("run COMPLETE  total={:.1f}s", time.time() - run_start)
        logger.info("=" * 50)
