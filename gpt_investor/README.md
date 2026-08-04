# gpt_investor — package layout

What each file does. Deterministic Python does the scoring; the LLM only
synthesises a verdict over those scores.

## Entry / state / UI

| File | Role |
|---|---|
| `gpt_investor.py` | Entry point — builds the `index()` page and the Reflex `app`. |
| `state.py` | The `State` class: every event handler + `_analyze_ticker`, the async per-ticker pipeline (fetch → score → verdict → cache). The orchestrator `fetch_analyses` lives here. |
| `ui/components.py` | All UI components (cards, dialog, search form, accordion). Fetches the Yahoo industry taxonomy at import, with a hardcoded fallback. |

## LLM (`llm/`)

| File | Role |
|---|---|
| `claude.py` | The only place that calls the model. Runs the `claude` CLI via subprocess, parses stream-json, enforces `require_tools`, tracks tokens. `call_claude_structured` validates output against a Pydantic schema. |
| `schemas.py` | Pydantic schemas for structured output (`SentimentLLM`, `VerdictLLM`) + `render_verdict_markdown`. `PROMPT_VERSION` lives here. |
| `analysis.py` | The LLM steps: sentiment, industry analysis, the final Buy/Hold/Sell verdict, liquidity commentary. |

## Data / scoring (`data/`) — no LLM

| File | Role |
|---|---|
| `market_data.py` | Thin yfinance getters: price, news, analyst ratings, company name, article text fetch. |
| `fundamentals.py` | Deterministic 5-dimension fundamental score (valuation/growth/profitability/cash/balance) → 0–10 + tier. Pure `score_fundamentals`, split-out `fetch_fundamentals`. |
| `wyckoff.py` | Deterministic price/volume timing score. OHLCV → features → Wyckoff phase → 0–10 timing score. Pure core, testable without yfinance. |
| `market_regime.py` | One yfinance download of VIX / yield curve / HY credit / DXY / gold → a regime label (`risk-on-bull`, `recession-warning`, …). |
| `macro.py` | Central-bank liquidity snapshot from official APIs (FRED / ECB / PBOC). Numbers are fetched, never LLM-generated. |
| `sentiment.py` | Blends VADER (rule-based) with the LLM sentiment score; disagreement drives confidence. |
| `discovery.py` | Ticker + industry discovery: Yahoo `top_companies`, news-mention reorder, Claude fallback, `resolve_ticker` (name → symbol), YF taxonomy fetch. |

## Storage / infra

| File | Role |
|---|---|
| `storage/cache.py` | SQLite. Daily `(ticker, date)` analysis cache + the liquidity snapshot cache. |
| `infra/logging_config.py` | Loguru setup (console + rotating file, per-ticker binding). |

## Open PRs add

- `data/discovery.py` — `get_setup_candidates`, a tool-scored discovery funnel (#2)
- `llm/explainer.py` — click-time plain-English verdict summary (#3)
- `storage/cache.py` + `scripts/` — `verdict_history` + nightly fill + calibration (#4)
- `llm/audit.py` — advisory audit agents vs past outcomes (#5)
- `data/signals.py` — short interest, earnings, insider, trend, peers (#6)
- `llm/probability.py` — probabilistic verdicts + Kelly sizing (#7)
- `infra/resilience.py` — yfinance retry / serve-last-good (#8)
