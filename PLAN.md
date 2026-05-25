# Claude Investor — Roadmap

Living plan for evolving the tool from "LLM verdict on yfinance numbers" into "auditable multi-agent investment co-pilot with measurable edge".

Each phase is independently shippable. Don't start a phase until the one before it has produced visible value (or, for data-collection phases, has been running long enough to matter).

---

## Done

### Hardening LLM grounding
- [x] WebSearch enforcement (`require_tools=["WebSearch"]`) on `get_liquidity_context` + `get_industry_analysis`
- [x] `stream-json --verbose` parsing → tool-use audit in `get_last_call_meta()`
- [x] URL HEAD-verification of central-bank sources (401/403 = alive)
- [x] Retry once with stricter prompt if required tool didn't fire
- [x] Subprocess timeout (180s), returncode + stderr handling

### Deterministic fundamental scoring
- [x] `gpt_investor/fundamentals.py` — 5-dim weighted score (valuation, growth, profitability, cash, balance)
- [x] Tier output (Strong/Solid/Average/Weak/Avoid) + flags
- [x] Card chip + dialog block
- [x] Sonnet prompt updated to weigh deterministic score
- [x] Unit tests (32 cases against synthetic dicts)

### Quantified sentiment (hybrid VADER + LLM)
- [x] `gpt_investor/sentiment.py` — VADER baseline + LLM JSON emission + combiner
- [x] Disagreement between VADER and LLM → confidence (low/med/high)
- [x] Card chip + dialog block
- [x] Cache schema: `sentiment_json` column with idempotent migration
- [x] Unit tests (21 cases)

### Codebase cleanup
- [x] All package imports flipped to absolute (`from gpt_investor.x import ...`)
- [x] Removed stale `call_claude` import in `discovery.py`
- [x] Removed dead `prices` global from `state.py`
- [x] Bare `except` in `get_final_analysis` now logs full traceback
- [x] `_reset_tickers()` helper deduplicates 5 reset sites

Tests at this point: **67 passing** (3 modules: parser, fundamentals, sentiment, url_verify).

---

## Phase 0 — Market regime indicators (next; ~1h, no LLM)

**Goal:** add VIX / yield curve / DXY / HY credit / gold to the macro layer so verdicts are aware of the regime, not just central-bank rates.

### Build
- [ ] `gpt_investor/market_regime.py`
  - [ ] `get_market_regime() -> dict` — one yfinance `download()` call across `^VIX`, `^TNX`, `^IRX`, `DX-Y.NYB`, `HYG`, `GC=F`
  - [ ] 5-day deltas → "rising/falling" classification
  - [ ] Derived: `curve = TNX - IRX`
  - [ ] Regime label heuristic: risk-on bull / late-cycle caution / recession warning / panic-opportunity / mixed
  - [ ] `format_regime(regime: dict) -> str` markdown block
- [ ] Wire into `get_final_analysis` system prompt (between liquidity and verdict instructions)
- [ ] Wire into `liquidity_panel` UI (extend existing block; don't add a new panel)
- [ ] Cache per session like current `liquidity_context`; refresh on Test UI reset
- [ ] Tests: `tests/test_market_regime.py` — regime-label rules against synthetic dicts (no yfinance hit)

### Acceptance
- Verdict on Citi/META cards visibly references regime (yield curve, VIX) in thesis or risks
- Panel shows curve value with direction arrow
- Tests: regime label is deterministic per input

### Decisions locked
- Extend existing liquidity panel (no separate UI block)
- Indicator list: `^VIX`, `^TNX`, `^IRX`, `DX-Y.NYB`, `HYG`, `GC=F`
- Regime label classification: code-side (deterministic), not LLM

---

## Phase 1 — Click-time explainer agent (~3h, 1 new LLM agent)

**Goal:** when user opens a ticker dialog, a separate agent emits a plain-English walkthrough of the verdict (mimicking the chat-style explanations we did for NVDA/META/Citi cards).

### Build
- [ ] `gpt_investor/explainer.py`
  - [ ] `explain_verdict(fund, sentiment, regime, analyst_text, sonnet_text) -> str`
  - [ ] System prompt: "Translate quant signals into plain English for a non-quant reader. Walk through fund tier, sentiment, regime, then verdict. ~300 words. No new analysis — only synthesis."
  - [ ] Model: `haiku`, `tools=False`
- [ ] State: new field `explainer_html: dict[str, str]` (per-ticker)
- [ ] `open_ticker` fires explainer in background; sets a `loading` flag; populates state on return
- [ ] Dialog: new section "Plain English" above sonnet's verdict block, with spinner while loading
- [ ] Cache: store in SQLite `analyses` table; recompute only on cache miss or prompt-version bump

### Acceptance
- Click any ticker → "Plain English" section streams in within ~5-10s
- Explanation references the actual fund/sentiment/regime values for that card
- Cached on subsequent clicks of the same ticker

### Decisions locked
- Haiku model (cheap, fast)
- Triggered on click only (not in background pipeline) → no extra cost for unread cards

---

## Phase 2 — Verdict history table (~3h, no LLM)

**Goal:** every sonnet verdict is captured with inputs + outputs; nightly job fills in actual 7/30/90/365-day returns. Foundation for calibration.

### Build

#### A. Verdict parser
- [ ] `gpt_investor/verdict.py`
  - [ ] `parse_verdict(sonnet_text) -> {verdict, target}` (regex)
  - [ ] `parse_analyst_grade(analyst_text) -> str | None`
  - [ ] `analyst_grade_to_score(grade) -> float | None` ([-1, +1] normalised)
  - [ ] `PROMPT_VERSION` constant (bump on every prompt change)
- [ ] `tests/test_verdict.py` — sample sonnet outputs, edge cases (missing target, etc.)

#### B. Schema + capture
- [ ] Extend `cache.py` with `verdict_history` table:
  - inputs: `captured_at`, `ticker`, `sector`, `industry`, `price`, `fund_score`, `fund_tier`, `sentiment_score`, `sentiment_conf`, `analyst_grade`, `analyst_score`, `regime_label`
  - output: `verdict`, `price_target`, `sonnet_text`, `prompt_version`
  - outcomes (NULL until filled): `price_7d`, `price_30d`, `price_90d`, `price_365d`
  - benchmark: `spy_at_capture`, `spy_7d`, `spy_30d`, `spy_90d`, `spy_365d`
  - meta: `last_filled_at`, `audit_text` (Phase 3), `audit_label` (Phase 3)
- [ ] `record_verdict(...)` function
- [ ] Wire into `state.py::_analyze_ticker` after sonnet returns (miss path only — cached path has nothing new to record)
- [ ] `fetch_fundamentals()` extended to include `sector`, `industry` in return dict

#### C. Nightly outcome filler
- [ ] `scripts/__init__.py`
- [ ] `scripts/fill_outcomes.py`
  - [ ] `_get_close_near(ticker, target_date)` — yfinance lookup with ±5d window for weekends/holidays
  - [ ] Fill `price_Nd` and `spy_Nd` for any row where horizon has passed
  - [ ] Update `last_filled_at`
  - [ ] Idempotent — safe to re-run
- [ ] Cron / launchd job: nightly at 22:00 local

#### D. Calibration report
- [ ] `scripts/calibration.py`
  - [ ] Group by `fund_tier`, `verdict`, `sentiment_conf`, `regime_label`
  - [ ] Columns: N, mean 7/30/90d return, hit rate (>0), alpha vs SPY
  - [ ] Filter by `prompt_version` to avoid mixing contracts
  - [ ] CLI output (table). Reflex page later if useful.

### Acceptance
- Every ticker analysis writes one new `verdict_history` row
- Nightly job populates outcomes as horizons pass
- `python -m scripts.calibration` runs and prints summary tables (empty/sparse at first)
- `prompt_version` correctly partitions data

### Decisions locked
- Per-row SPY benchmark stored (for alpha measurement)
- Prompt-version pinned in each row
- No UI surfacing of history yet (silent collection)

---

## Phase 3 — Audit agents (financial + sentiment, ~4h, 2 new LLM agents)

**Goal:** after sonnet emits verdict, two specialist audit agents critique it independently, using verdict_history for similar past cases. Both surface in dialog with chips on card.

### Build
- [ ] `gpt_investor/audit.py`
  - [ ] `get_similar_past(fund_tier, sent_score, sector, k=5, balanced=True) -> list[row]` — query `verdict_history` for similar setups WITH outcomes; force at least one win + one loss in the k selected
  - [ ] `audit_financial(fund, sonnet_text, similar) -> dict` — haiku agent, sees fund + verdict + 5 similar with outcomes. Output: `{label: AGREE/SOFTEN/CONTRADICT, text: str}`
  - [ ] `audit_sentiment(sentiment, regime, sonnet_text, similar) -> dict` — same structure, different inputs
  - [ ] Gating: skip if N similar < 5 (cold start)
- [ ] Wire into `_analyze_ticker` AFTER sonnet returns, BEFORE save_cached
- [ ] Both audits run in parallel via `asyncio.gather`
- [ ] Store in `verdict_history.audit_text` (combined) and `audit_label` (worst of the two: CONTRADICT > SOFTEN > AGREE)
- [ ] UI: chip on card (`Audit ✓` / `Audit ~` / `Audit ✗` colours green / amber / red)
- [ ] Dialog: new section below verdict — show both audits side-by-side

### Acceptance
- Once verdict_history has ≥5 similar entries with outcomes, audits start firing
- Both audits run in parallel; total added latency ~10s
- Calibration script extended to measure audit label vs actual outcome (does SOFTEN actually correlate with worse returns?)

### Decisions locked
- Haiku model (cheap)
- Advisory only — verdicts on card unchanged; audits surface in dialog
- Specialist disjoint contexts: financial sees fund+verdict only; sentiment sees sent+regime+verdict only
- Domain isolation is the whole point — don't give either audit the other's data

---

## Phase 4 — Higher-signal data (~6-8h, mostly no LLM)

After Phases 0-3 are stable and verdict_history has 4+ weeks of data, add the cheap-yfinance signals ranked highest in the earlier analysis.

- [ ] **B1. Short interest** (`info["shortPercentOfFloat"]`) — flag if >15%
- [ ] **B2. Earnings calendar guard** (`yf.Ticker.calendar`) — banner card "EARNINGS IN Nd"; record `days_to_earnings` in verdict_history so calibration can measure
- [ ] **B3. Insider transactions** (`yf.Ticker.insider_transactions`) — net 30/90d dollar flow → new flag or sub-dimension
- [ ] **B4. Multi-year trend** — 4y revenue CAGR, FCF CAGR, margin slope from `income_stmt` + `cashflow`. Trajectory > snapshot.
- [ ] **B5. Peer comparison** — `yf.Industry(key).top_companies[:10]`, parallel info fetch, sector medians for P/E / P/B / EV/EBITDA / ROE. Fixes bank-and-megacap scoring unfairness.
- [ ] Each new signal gets a `verdict_history` column so calibration can measure its lift.

### Decisions locked (when started)
- Add columns to `verdict_history` not new tables — keep schema flat
- Prompt-version bumps when new signals enter sonnet's input set

---

## Phase 5 — Probabilistic verdicts + portfolio context (later)

After Phase 4 and 8+ weeks of calibration data.

- [ ] **C1. Probabilistic verdict JSON** — sonnet emits `{verdict, confidence, horizon_days, prob_up_20pct, prob_flat, prob_down_20pct, thesis_killer}`. Brier score / log-loss become measurable.
- [ ] **C2. Pre-mortem in prompt** — "Before listing positives, list 3 ways the thesis fails."
- [ ] **C3. Portfolio context** — `State.portfolio: dict[ticker, shares]` from a user-input form. Final prompt sees holdings: "User already owns X shares of Y" → trim/add/skip recommendation.
- [ ] **C4. Correlation aware** — compute pairwise 90d return correlation across user's holdings. Warn on `Buy X correlates 0.85 with held Y`.
- [ ] **C5. Position sizing** — fractional Kelly from the probabilistic verdict.
- [ ] **C6. Earnings transcript ingestion** — fetch transcript, summarise management tone shift vs prior quarter, re-run verdict.

---

## Cross-cutting principles

- **Prompt-version discipline**: bump `verdict.PROMPT_VERSION` every time `get_final_analysis` system prompt changes meaningfully. Calibration queries filter on it. Treat it like an API contract.
- **No code-side LLM-output adjustment**: keep LLM verdict raw in the data. Code-side calibration becomes a separate layer the user can compare against. If we adjust LLM output silently, we can't measure the LLM.
- **Specialist agents > generalist agents**: domain-isolated audits catch more than one big judge.
- **Cheap yfinance signals first**: insider buying, peer medians, earnings calendar — all free, all high-signal, all ignored today.
- **Validation before optimisation**: don't optimise prompts or weights until calibration data exists. Until then we're flying blind.

## Open questions to revisit

- Should the explainer (Phase 1) be cached per (ticker, date)? Currently planned yes; revisit if explanations feel stale.
- Should audit labels (Phase 3) ever override sonnet's verdict on the card? Currently planned advisory-only. May revisit once we've measured audit accuracy.
- When to add transcript ingestion (Phase 5 C6)? Depends on Seeking Alpha scrape feasibility / paid API budget.
- How to detect "this prompt change is material enough to bump prompt_version"? Currently subjective; consider adding a hash of the system prompt to verdict_history for auto-detection.

## Build sequencing (single-line)

```
P0 (regime indicators) → P1 (explainer agent) → P2 (verdict_history) → wait ~4 wks → P3 (audits) → P4 (data layers) → P5 (probabilistic + portfolio)
```

P0 + P1 + P2 can ship in one focused session (~7h). After that, the tool is data-accumulating and the next phase has actual ground truth to learn from.
