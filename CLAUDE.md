# Claude Investor — Project Context

AI-powered investment analysis app. User picks an industry (or trending news), gets Buy/Hold/Sell recommendations for the top companies with supporting analysis.

## Stack

- **Reflex** — Python full-stack web framework (backend state + React frontend)
- **yfinance** — market data, news, analyst ratings, fundamentals
- **Claude CLI via subprocess** — all LLM calls go through `claude -p ... --output-format stream-json --verbose` (not the Anthropic SDK), which bills against the user's Claude Code subscription, not a separate API key. Stream-json shape is parsed for tool-use audit. Do NOT switch to the SDK without confirming with the user.
- **SQLite** (`analyses.db`) — daily analysis cache keyed by `(ticker, date)`
- **beautifulsoup4 + requests** — article text fetching for sentiment analysis; HEAD-verification of central-bank source URLs
- **pytest** — `tests/` directory; `network` marker for live HTTP tests

## Key files

| File | Role |
|---|---|
| `gpt_investor/gpt_investor.py` | Entry point — `index()` page + `app` setup only (~55 lines) |
| `gpt_investor/state.py` | Reflex `State` class, all event handlers, async pipeline (`_analyze_ticker`), `_reset_tickers()` helper |
| `gpt_investor/components.py` | All UI component functions + `_YF_INDUSTRY_GROUPS_FALLBACK` + startup taxonomy fetch |
| `gpt_investor/claude.py` | `call_claude` with stream-json parser, `require_tools` enforcement + 1 retry, `get_last_call_meta()` audit, token globals |
| `gpt_investor/market_data.py` | Pure yfinance getters: price, news, ratings, company name, article fetch |
| `gpt_investor/fundamentals.py` | Deterministic 5-dimension scorer — `fetch_fundamentals`, `score_fundamentals`, `format_fundamentals`. No LLM. |
| `gpt_investor/analysis.py` | LLM-powered analysis: sentiment, industry, final analysis, liquidity context (with URL HEAD-verification) |
| `gpt_investor/discovery.py` | Ticker/industry discovery, YF taxonomy fetch, caches, keyword maps |
| `gpt_investor/cache.py` | SQLite cache — `get_cached` / `save_cached` |
| `tests/` | `pytest` suite — parser, fundamentals, URL verify (network-marked) |
| `rxconfig.py` | Reflex app config |
| `analyses.db` | Generated at runtime, gitignored |

## How to run

```bash
pyenv activate claude-investor
reflex run
# or with more tickers:
MAX_TICKERS_TO_ANALYZE=6 reflex run

# tests
pytest tests/                 # all (incl. live network checks)
pytest -m "not network"       # offline only
```

## Analysis pipeline (per ticker)

1. `generate_ticker_ideas` — Yahoo Finance `top_companies` + news mention reordering; Claude CLI fallback if YF returns too few
2. `get_liquidity_context` — single Claude call with `require_tools=["WebSearch"]`; fetches Fed/ECB/PBOC current stance with source URLs; URLs HEAD-verified (401/403 = alive); runs in parallel with ticker discovery
3. `get_industry_analysis` — Claude with `require_tools=["WebSearch"]` on first ticker's industry/sector
4. Per ticker (all parallel):
   - `get_news` — yfinance news metadata
   - `get_sentiment_analysis` — uses YF `summary` field; falls back to `requests` fetch; Claude call with `tools=False` (content already provided)
   - `get_analyst_ratings` — yfinance recommendations
   - `get_current_price` — yfinance 1-min history
   - `get_company_name` — yfinance `shortName`
   - `fetch_fundamentals` + `score_fundamentals` — deterministic 5-dim score (yfinance only, no LLM). Score published to UI before sonnet runs.
5. `get_final_analysis` — Claude sonnet with formatted fundamental score block + sentiment + ratings + industry + liquidity. System prompt instructs the model to weigh the deterministic score heavily and reference the tier in its thesis.
6. `save_cached` — stores sentiment, ratings, final analysis to SQLite (fundamental score re-fetched each run, not cached)

## Cache behaviour

- Key: `(ticker, date)` — expires daily at midnight
- Granularity: sentiment + analyst_ratings + final_analysis stored separately
- Cache hit path: skips sentiment/ratings/sonnet; still fetches fresh price + company name + **fundamentals** (so the score chip stays current even on cached verdicts), marks card as blue "Cached" badge
- Cache miss path: runs full pipeline, saves LLM outputs (fundamentals never cached — yfinance call is cheap)

## Liquidity context

- Fetched once per session, reused across industry changes (no wall-clock cost on subsequent runs)
- Stored in `State.liquidity_context` + `State.liquidity_html`
- `State.liquidity_is_mock: bool` — set to `True` by `load_mock_data`; forces a real re-fetch on the next real run even though `liquidity_context` is non-empty. Cleared after the real fetch completes.
- **WebSearch enforced** via `require_tools=["WebSearch"]` — `call_claude` retries once with a stricter system prompt if the model didn't actually invoke WebSearch. Logs `[tools] WebSearchxN ... satisfied=True/False`.
- **Source URLs verified** — `_url_alive()` HEADs every URL in the returned text; 401/403 counts as alive (PBOC and some Fed pages gate bots), 404/410/5xx/DNS-fail counts as dead. Dead URLs appended to the panel as `_Unverified source URLs:_ ...`.

## Ticker card statuses

`"pending"` → `"processing"` → `"finished"` (green) / `"cached"` (blue) / `"error"` (red)

The card also shows a colored **fundamental tier badge** (`Strong`/`Solid`/`Average`/`Weak`/`Avoid`) below the company name once the score is computed — usually visible ~10s before the sonnet verdict finishes.

## Fundamental scoring

Pure-Python, deterministic, runs against `yfinance.Ticker(t).info`. No LLM.

Five dimensions, each scored 0–10, then weighted into a composite 0–10:

| Dimension | Weight | Inputs |
|---|---|---|
| Valuation | 0.25 | forward P/E (fallback trailing), P/B, EV/EBITDA |
| Growth | 0.20 | `revenueGrowth`, `earningsGrowth` |
| Profitability | 0.20 | ROE, operating margin |
| Cash | 0.20 | FCF margin = `freeCashflow / totalRevenue` |
| Balance | 0.15 | `debtToEquity / 100` (yfinance reports %) |

Tier thresholds: `≥8 Strong / ≥6 Solid / ≥4 Average / ≥2 Weak / <2 Avoid`.

Flags surfaced in the dialog: `negative FCF`, `shrinking revenue`, `declining earnings`, `high leverage`, `earnings loss`, `premium valuation`, `no FCF data`.

The scored dict is passed into `get_final_analysis(... fundamentals=...)` so sonnet sees the score block rather than a raw P/E dump. System prompt instructs the model to weigh the deterministic score and reference the tier in its thesis.

**Known limitation:** valuation scoring is sector-agnostic. Asset-light tech megacaps (NVDA, GOOG) get punished on P/B because intangibles aren't on the balance sheet. Plan: add sector-relative valuation later.

Unit-tested against synthetic metric dicts in `tests/test_fundamentals.py` (32 cases, no yfinance dependency).

## LLM call conventions

- `call_claude(system, user, model="haiku", tools=True, require_tools=None, max_retries=1)` in `claude.py` — all LLM calls go here
- Uses `--output-format stream-json --verbose` + `--allowed-tools WebSearch,WebFetch`; parses NDJSON line-by-line in `_parse_stream_json` to extract: final `result` text, every `tool_use` block, structured search-result URLs from `tool_use_result.results[].content[].url`, and `modelUsage`
- `tools=False` for sentiment (content already supplied); `tools=True` (default) for industry, liquidity, final analysis
- `require_tools=["WebSearch"]` on liquidity + industry. If the model returned text without invoking any required tool, `call_claude` retries once with `YOU MUST call one of these tools before answering...` prepended to the system prompt.
- `get_last_call_meta()` returns `{tool_calls, urls, tool_counts, retried, satisfied}` from the most recent call. Used by `get_liquidity_context` to log warnings when WebSearch didn't fire.
- Logs per call: `[tools] WebSearchxN urls=K retried=True/False satisfied=True/False`
- Token totals tracked via `add_token_usage(input, output, cache_read)` + `get_token_totals()` — displayed bottom-right

### Parser gotchas (caught in prod)

- `tool_use_result.results` is heterogeneous: first element is a dict with `content` (list of `{title, url}`), but a trailing element is the model's prose summary as a bare string. Parser must `isinstance(r, dict)` check before `.get()`.
- `stream-json` requires `--verbose` or the CLI errors out.

## UI structure

```
hero
[Today's Trending]  [Trending Industries]
  └── Trending Industries → loads orange industry badges → click one → analysis
  └── Today's Trending → immediately analyzes top mentioned companies from YF news
sector accordion (collapsible per sector, shows industry count)
  └── click sector header → expands to show industry chips
  └── click industry chip → analysis
Custom industry... [Go]          ← free-text industry name
🔍 Ticker or company name [Analyse]  ← single-company mode (CEG or "Constellation Energy")
  └── [on run]
      status line
      liquidity panel (Global Liquidity Snapshot — Fed/ECB/PBOC)
      ticker grid (cards)
      analysis dialog (opens on card click)
```

## Discovery modes

- `discovery_mode = "industry"` — uses `generate_ticker_ideas` with `direct_yf_key` (set by sector/industry chips) or keyword lookup (set by custom text input)
- `discovery_mode = "trending"` — uses `get_trending_tickers` (YF news mention counts)
- `discovery_mode = "single"` — resolves one company via `resolve_ticker(query)` then runs the full pipeline on that single ticker
- `direct_yf_key` — when set by an industry chip click, bypasses keyword matching and calls `yf.Industry(key)` directly

## Single-company analysis

- Entry point: amber "Analyse" button with a search icon in the search form
- `State.company_query: str` — bound to the ticker/name input
- `handle_company_submit` sets `discovery_mode = "single"` and triggers `fetch_analyses`
- `_resolve_single(query)` calls `resolve_ticker(query)` → `yf.Search(query).quotes` → returns first EQUITY quote without a dot in the symbol. Falls back to first result of any type. Returns `{}` on failure.
- On resolution failure: `fetch_analyses` saves any fetched liquidity to state, sets `stage = "done"`, and shows "No companies found" in the tickers grid (no spinning UI left behind)
- On success: same full pipeline as any industry run — liquidity, `get_industry_analysis` (uses the ticker's own sector from yfinance), `_analyze_ticker`. Same card and dialog output.

## Industry taxonomy

- `get_yf_industry_groups()` in `discovery.py` — fetches all 11 YF sectors in parallel threads at startup, returns `[(sector_display, [(industry_display, yf_key), ...]), ...]`
- Falls back to `_YF_INDUSTRY_GROUPS_FALLBACK` in `components.py` if YF is unreachable at startup
- `State.expanded_sectors: list[str]` — tracks which sector headers are expanded in the accordion; `toggle_sector(sector)` adds/removes from the list

## Trending Industries

- `get_trending_industries(num)` in `discovery.py` — scans YF news for top mentioned tickers, fetches each ticker's `industryKey` in parallel, ranks industries by weighted mention count; 30-min TTL cache
- `State.fetch_trending_industries` — background event, sets `trending_industries_loading`, populates `State.trending_industries: list[list[str]]` (each item is `[display, yf_key]`)
- Result shown as orange clickable badges; clicking one calls `industry_pick(display, yf_key)`

## Concurrency guard — one analysis at a time

While `State.stage == "analyzing"`, all entry points that would trigger a new run are blocked:

| Element | Mechanism |
|---|---|
| Today's Trending button | `loading=(stage == "analyzing")` |
| Trending Industries button | `disabled=(stage == "analyzing")` |
| Trending industry badges (orange) | `on_click=rx.noop()` + `opacity=0.4` + `cursor=not-allowed` |
| Industry chips in accordion | same as above |
| Go (custom industry) button | `loading=(stage == "analyzing")` |
| Analyse (single company) button | `loading=(stage == "analyzing")` |

Sector accordion headers (`toggle_sector`) are intentionally NOT blocked — they only expand/collapse the UI, no analysis triggered.

## Key decisions made

- **subprocess over SDK**: keeps billing under Claude Code subscription. SDK would require a separate ANTHROPIC_API_KEY and pay-per-token billing.
- **SQLite over JSON cache**: thread-safe concurrent writes when multiple tickers finish simultaneously.
- **All intermediate steps cached**: lets you re-run just the final analysis after prompt tweaks without re-fetching news/sentiment.
- **Liquidity context fetched once per session**: parallel with ticker discovery on first run, reused thereafter. `liquidity_is_mock` flag prevents Test UI mock data from blocking the real fetch.
- **`tools=False` for sentiment**: YF summaries + requests fetch supply the content; removing tool access cuts ~15s per ticker.
- **Deterministic fundamental score (replaces `_get_financials`)**: yfinance numbers → 5-dim weighted score → tier. Sonnet gets the scored block, not raw P/E. Repeatable across runs, comparable across tickers, no hallucination on the numbers. `_get_financials` was deleted.
- **WebSearch enforcement via `require_tools`**: switched to `stream-json --verbose` parsing so we can see whether the model actually invoked WebSearch. Retries once with stricter prompt if it didn't. Liquidity URLs are additionally HEAD-verified (401/403 = alive). Stops `get_liquidity_context` / `get_industry_analysis` from silently hallucinating Fed/ECB/PBOC numbers.
- **Dynamic YF taxonomy**: sector/industry list fetched from `yf.Sector(key).industries` at startup rather than hardcoded, so it reflects YF's actual taxonomy. `direct_yf_key` bypasses keyword matching for chip-triggered runs.
- **Collapsible sector accordion**: dynamic fetch returns many industries per sector; accordion keeps the UI compact while exposing the full taxonomy.
- **Single-company mode via `yf.Search`**: `resolve_ticker` prefers EQUITY quotes without dots (avoids ADRs/ETFs), falls back to first result, returns `{}` on failure. `get_industry_analysis` uses `.get()` on yfinance info to avoid `KeyError` on unknown tickers. Empty `tickers_dict` triggers an early return that saves liquidity state and sets `stage = "done"` so no spinners are left running.
- **One analysis at a time**: all clickable entry points (buttons, chips, badges) are disabled or noop'd while `stage == "analyzing"`. Prevents concurrent `fetch_analyses` calls from clobbering shared state.

## Planned improvements (not yet implemented)

Ranked by investment-utility leverage (high → low):

1. **Quantified sentiment** — replace prose sentiment with JSON `{score: -1..+1, confidence: low/med/high, drivers: [3 bullets]}` so tickers are comparable. Prerequisite for the ranking table.
2. **Comparative ranking table** — after all tickers done, render one sortable table: `ticker | price | fund_score | sentiment_score | analyst_consensus | verdict | upside%`. Replaces reading N cards.
3. **Earnings calendar guard** — `yf.Ticker.calendar`; if next earnings <7d, banner the card "EARNINGS IN 3D".
4. **Divergence flag** — sentiment positive + analysts neutral = potential mispricing; sentiment negative + fundamentals strong = potential dip-buy. Colored chip on the card.
5. **Sector-relative valuation** — peer median P/E / EV/EBITDA from `yf.Industry(key).top_companies` to soften P/B punishment on asset-light megacaps (NVDA, GOOG).
6. **Catalyst timeline** — extract dated events (earnings, FDA decisions, product launches) from news.
7. **Discovery filters** — min market cap, min avg volume, exclude OTC — kills penny-stock noise.
8. **Verdict history** — store past verdicts in SQLite, render a time-series so you can backtest your own past calls vs actual price moves.
9. **Multi-source sentiment** — WebSearch Reuters/WSJ headlines in addition to YF news (avoids YF's press-release bias).
10. **Analyst consensus over time** — not just latest rating, but rolling 90d direction.

Lower-priority engineering items (defer until utility shape is right):

- `subprocess.run` timeout in `call_claude` — currently no timeout → CLI hang locks the pipeline
- `result.returncode` + stderr handling — silent failures on auth/rate-limit
- Parallel HEAD checks in `_url_alive` loop (3 workers, drops liquidity URL verify from ~9s to ~3s)
- Per-step progress on ticker cards (not just "Analysing")
- Ticker count UI slider (currently `MAX_TICKERS_TO_ANALYZE` env var)
- Liquidity TTL — currently session-long; goes stale across Fed decisions
- Surface `get_last_call_meta()["urls"]` as a "Sources searched: N" chip on liquidity + industry panels
