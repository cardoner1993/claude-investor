# Claude Investor — Roadmap

Living plan for evolving the tool from "LLM verdict on yfinance numbers" into "auditable multi-agent investment co-pilot with measurable edge".

Each phase is independently shippable. Don't start a phase until the one before it has produced visible value (or, for data-collection phases, has been running long enough to matter).

---

## The analysis pipeline (narrative framing)

Read top to bottom — this is the story one ticker travels through before a verdict, plus the loop that checks whether the verdict was right. Each stage maps to a module that already exists or one the phases below will build. The build backlog (Phases D, 0–5) is the *how*; this section is the *what and why*.

```
[0] OVERVIEW   what's moving right now, what regime are we in
       ↓
[1] MACRO      central-bank stance + market regime (top-down filter)
       ↓
[2] MICRO      fundamentals + sentiment + analyst + industry (bottom-up)
       ↓
[3] WYCKOFF    price/volume structure — is the chart confirming the thesis?
       ↓
[4] DECISION   Buy/Hold/Sell verdict that must address every layer above
       ↓
[5] DATA LAYER record verdict + inputs → fill real outcomes → calibrate
       ↑__________________________ feed back: "was the yes/no right?" ____|
```

### [0] Overview — *what is going on*

What's actually moving in the market right now, gated by fresh news, plus the macro backdrop the user is walking into.

| Piece | Module | Status |
|---|---|---|
| Signal-driven setups (tool-scored, not Yahoo-ranked) | `data/discovery.py` `get_setup_candidates` | **Phase PD' — not built** (today: weak `get_trending_tickers` "what's trendy on Yahoo") |
| Trending industries / sector accordion | `data/discovery.py` | exists |
| Liquidity + regime snapshot panel | `data/macro.py` + `data/market_regime.py` | exists |

Output of this stage = the candidate set the user picks from. **Key change (PD'):** the "Today's Trending" mode is being replaced by "Today's Setups" — Yahoo screens become a cheap *candidate universe*, and the tool's own deterministic scores (fundamentals + Wyckoff + regime fit) decide what surfaces. See the Phase PD' block below.

### [1] Macro — *top-down filter*

Deterministic, no hallucinated numbers. Two sub-layers, both already built and already fed into the verdict:

- **Central-bank stance** — `data/macro.py`: Fed (FRED) / ECB (SDW) / PBOC (ChinaMoney) policy rates, stance from 90-day delta. One LLM paragraph reads the *rendered* numbers only.
- **Market regime** — `data/market_regime.py`: VIX, 10y−3m curve, DXY, HY credit (HYG), gold → 5-day deltas → regime label (`risk-on-bull` … `panic-opportunity`). Pure code-side classification.

Status: **exists.** Refinements (regime tuning, more indicators) live in Phase 0.

### [2] Micro — *bottom-up read*

| Sub-layer | Module | Status |
|---|---|---|
| Fundamentals (5-dim deterministic score → tier) | `data/fundamentals.py` | exists |
| Quantified sentiment (VADER + LLM combiner) | `data/sentiment.py` | exists |
| Analyst ratings | `data/market_data.py` | exists |
| Industry/sector analysis (WebSearch-grounded) | `llm/analysis.py` `get_industry_analysis` | exists |

Status: **exists.** Higher-signal additions (short interest, insider flow, peer-relative valuation, multi-year trend) are Phase 4.

### [3] Wyckoff — *is the chart confirming the thesis?* — **BUILT (GH #1)**

The missing layer. Fundamentals say *what to own*; Wyckoff says *whether now*. Deterministic price/volume structure, modelled on `fundamentals.py` — pure Python, yfinance OHLCV only, no LLM, fully unit-testable.

> **Status:** `data/wyckoff.py` shipped — `compute_signals` / `classify_phase` / `score_wyckoff` / `format_wyckoff` + `fetch_ohlcv` / `fetch_ohlcv_batch` / `score_ticker`. Wired into `_analyze_ticker` (both cache paths), `get_final_analysis`, `VerdictLLM.technical_addressed`, card chip (outline badge) + dialog block. `PROMPT_VERSION` → `v2` in `llm/schemas.py`. 23 unit tests in `tests/test_wyckoff.py` (144 offline total, all green).

**Build — `gpt_investor/data/wyckoff.py`**

- [x] `fetch_ohlcv(ticker, period="1y", interval="1d") -> DataFrame` — single `yf.Ticker(t).history`; tolerate short history; log + return empty on failure.
- [x] `fetch_ohlcv_batch(tickers, period="1y") -> dict[str, DataFrame]` — multi-ticker `yf.download(group_by="ticker", threads=True)` (mirror `market_regime.py`). **Discovery (PD') scores ~60 names per refresh — a per-ticker loop = 60 history calls; the batch path collapses that to one.** Per-ticker `fetch_ohlcv` stays for single-company / card paths.
- [x] `compute_signals(df) -> dict` — deterministic features:
  - [x] trend: 50d vs 200d SMA (golden/death cross, % above/below 200d)
  - [x] momentum: price vs 20d high/low, distance from 52w high/low
  - [x] volume: 5d vs 50d average → `vol_surge` bool; up-volume vs down-volume bias
  - [x] range: Bollinger/ATR-style band width → `consolidating` vs `expanding`
  - [x] breakout: close breaching N-day range on a volume surge
- [x] `classify_phase(signals) -> Literal["accumulation","markup","distribution","markdown","neutral"]` — Wyckoff-flavoured rule mapping (e.g. flat range + volume drying + above 200d → accumulation; new highs + rising volume → markup; first match wins, fall back to `neutral`). Pure function, ordered rules — the testable core. **These 5 labels are canonical — PD' ranks on this exact vocabulary (no "early-markup"; an early-markup setup reads as `accumulation` breaking out or fresh `markup`).**
- [x] `score_wyckoff(signals, phase) -> dict` — `{phase, tier, score 0-10, flags}` where tier ∈ Strong/Solid/Average/Weak/Avoid (mirror fundamentals tiers so the card chip is consistent). Flags: `below 200d`, `distribution volume`, `overextended`, `no volume confirmation`, `thin history`.
- [x] `format_wyckoff(scored) -> str` — markdown block for the verdict prompt + dialog.
- [x] Tests `tests/test_wyckoff.py` — `classify_phase` + `score_wyckoff` against synthetic OHLCV/signal dicts (no yfinance hit), same style as `test_fundamentals.py`. **(23 cases; +`compute_signals` on synthetic DataFrames.)**

**Wire-in**

- [x] `state.py::_analyze_ticker`: add Wyckoff scoring (`score_ticker`) to the parallel `asyncio.gather` block in **both** the cache-hit and miss paths — publish a Wyckoff phase chip to the card alongside the fundamental tier chip.
- [x] `llm/analysis.py::get_final_analysis`: pass `wyckoff` block in the user message between micro and macro; system prompt frames it as the timing overlay on the fundamental case.
- [x] `llm/schemas.py::VerdictLLM`: add **required** `technical_addressed: str` field (hard audit parity with the other `_addressed` fields — no default). `render_verdict_markdown` still reads it via `getattr(..., 'no impact')` so old cached *rendered* verdicts (stored as markdown, never re-validated) display fine.
  - [x] **PROMPT_VERSION ordering fix:** constant introduced in `llm/schemas.py` now (not waiting for P2's `verdict.py`); PW bumped it `v1` → `v2`. P2's `verdict.py` will import it rather than redefine.
- [x] UI: Wyckoff phase chip on the card (outline badge) + block in the dialog (mirror the fundamental tier chip/block).

**Decisions locked**
- Deterministic only — no LLM in the Wyckoff module itself (the verdict LLM *reads* the rendered block, never invents the structure). Same contract as fundamentals/regime.
- Daily bars, 1y window — no intraday/multi-timeframe (keeps it one cheap yfinance call). Revisit if phase detection proves too coarse.
- Tier vocabulary reused from fundamentals (Strong…Avoid) so the card stays visually consistent.

**Open questions**
- Phase confidence score, or hard label only? (Lean: label + 0–10 score, like fundamentals.)
- Divergence flag when Wyckoff phase contradicts fundamentals (strong fund + distribution phase = "right company, wrong time")? Natural follow-up once both chips exist.

### [4] Decision — *invest in this trending company or not*

`llm/analysis.py::get_final_analysis` → `VerdictLLM`. Sonnet sees every layer above and must emit a structured verdict where each `*_addressed` field justifies how that input moved the call (or `no impact`). The deterministic scores are weighted heavily and can't be hallucinated.

Status: **exists + Wyckoff-aware** — `technical_addressed` (Stage 3) now forces the verdict to justify the chart phase, so all five inputs (fund, sentiment, industry, macro, technical) are audited. Probabilistic verdicts + pre-mortem are Phase 5.

### [5] Data layer — *did the yes/no actually work?* — **largely not built**

The point the user is driving at: today a verdict ships and we never learn if it was right. This stage closes the loop so both the user *and the LLM* can see, on the next similar setup, what happened last time.

| Step | Module | Status |
|---|---|---|
| Parse `{verdict, target}` out of the LLM output | `verdict.py` | Phase 2A — not built |
| Capture every verdict + all inputs + SPY benchmark | `verdict_history` table in `storage/cache.py` | Phase 2B — not built |
| Nightly fill real 7/30/90/365d returns | `scripts/fill_outcomes.py` | Phase 2C — not built |
| Calibration report (hit rate, alpha by tier/regime/phase) | `scripts/calibration.py` | Phase 2D — not built |
| Audit agents critique verdict vs *similar past outcomes* | `audit.py` | Phase 3 — not built |

**The feedback mechanism** (this is the "make the LLM understand what happened" part): once `verdict_history` has outcomes, the audit agents (Phase 3) and the verdict prompt itself retrieve the *k most similar past setups with their actual results* and inject them — "last 5 times fundamentals=Strong + regime=late-cycle + Wyckoff=distribution, the Buy verdicts averaged −4% vs SPY." The LLM now reasons against its own track record instead of in a vacuum. Wyckoff phase (Stage 3) becomes one of the similarity keys, so the data layer must add a `wyckoff_phase` + `wyckoff_score` column when Stage 3 ships.

**Discipline** (cross-cutting, already noted below): never code-adjust the LLM verdict — keep it raw in `verdict_history` so calibration measures the model honestly. `PROMPT_VERSION` partitions data across prompt changes.

> **Net: stages 0–2 and 4 exist today. The two pieces of net-new work this plan adds are [3] Wyckoff (the missing analytical layer) and [5] the data/feedback loop (so a "yes invest" / "no" becomes measurable and the LLM learns from it). Build sequencing for both is in the phases below.**

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

## Phase PD' — Signal-driven discovery (replaces "trendy on Yahoo"; ~5h, no LLM)

> **Supersedes the old Phase D (mover funnel).** Same data-fetch plumbing, *flipped ranking*. Phase D ranked candidates by **how much they moved** — still "what's hot on Yahoo." PD' ranks by **how good a setup the tool itself thinks they are**, using the deterministic layers already built (fundamentals) and planned (Wyckoff, regime). Yahoo's screens are demoted to a cheap *candidate universe* — they list tickers, they no longer decide which surface. **Depends on PW (Wyckoff) being built first** — Wyckoff phase is a ranking input.

### Why "trendy on Yahoo" is the wrong signal

Today `get_trending_tickers` (`discovery.py:273`) runs `yf.Search` over 5 fixed phrases and counts `relatedTickers` in the returned articles. The fundamental problem isn't *how* it measures trendiness — it's that **trendiness is the wrong target**. Surfacing what's already in the headlines means chasing moves that already happened. The tool has its own opinion (fundamentals tier, chart phase, regime fit) and should use it to surface *setups*, not echo the news cycle.

### The flip

```
OLD (Phase D):   Yahoo lists hot tickers  →  rank by movement  →  surface
NEW (PD'):       Yahoo lists a universe   →  score by the tool's OWN signals  →  surface setups
```

### Funnel design

```
1. UNIVERSE — Yahoo screens only LIST tickers (cheap, broad net, no ranking authority):
     yf.screen("most_actives") + yf.screen("day_gainers") + yf.screen("day_losers")
     + trending endpoint                              → ~100-200 raw symbols
     (day_losers now INCLUDED — accumulation/basing names live there; the
      Wyckoff filter sorts dip-buys from falling knives)
2. EQUITY FILTER — drop "." / "-USD" (crypto) / "=" (FX) / "^" (index); upper-case, len<=5
3. PRE-FILTER for cost — keep top ~60 by dollar-volume (price × volume) so we don't
     run full fundamentals + OHLCV on 200 names every refresh
4. SCORE each survivor — deterministic, NO LLM, reuse existing modules in parallel threads:
     - fundamentals tier   (data/fundamentals.py  — score_fundamentals)
     - wyckoff phase+score  (data/wyckoff.py       — score_wyckoff, Stage [3])
     - regime fit           (data/market_regime.py — does this name suit current regime?)
5. RANK by composite SETUP score, e.g.:
     fund_score (0-10) * w_f  +  wyckoff_score (0-10) * w_w  +  regime_fit_bonus
     with a soft gate: prefer fund >= Solid AND wyckoff in {accumulation, markup}
6. TRUNCATE to MAX_TICKERS_TO_ANALYZE top setups, ranked
7. Feed survivors into the existing per-ticker pipeline unchanged
     (the scores computed here are REUSED — no re-fetch for the cards)
```

### What "regime fit" means (new small helper)

A code-side rule, not an LLM call: given the current `market_regime.label`, prefer certain Wyckoff phases / fundamental profiles.
- `risk-on-bull` → favour markup phase, growth tilt
- `late-cycle-caution` / `recession-warning` → favour accumulation phase, strong balance sheet, low leverage
- `panic-opportunity` → favour deeply-based accumulation names (the dip-buy case)
First-pass: a small bonus/penalty lookup table. Tune later against calibration data (Phase 2).

### Build

- [ ] `gpt_investor/data/discovery.py`
  - [ ] `_screen_universe(count) -> list[dict]` — wrap `yf.screen` for `most_actives` + `day_gainers` + `day_losers`; normalize `{symbol, name, pct, volume, price}`; tolerate dict-or-list shape; log + return `[]` on failure
  - [ ] `_trending_endpoint(count) -> list[str]` — `requests.get` the trending REST URL (`query1.finance.yahoo.com/v1/finance/trending/US`) with `User-Agent` + timeout; parse `finance.result[0].quotes[].symbol`; `[]` on error
  - [ ] `_is_equity_symbol(sym) -> bool` — extract the existing inline predicate, extend for `-USD`
  - [ ] `_prefilter_by_dollar_volume(candidates, keep=60) -> list[dict]` — cheap liquidity gate before expensive scoring
  - [ ] `_regime_fit(fund_tier, wyckoff_phase, regime_label) -> float` — the lookup-table bonus above; pure function, unit-testable
  - [ ] `_setup_score(fund, wyckoff, regime_label) -> float` — composite rank; pure function
  - [ ] `get_setup_candidates(num=MAX_TICKERS_TO_ANALYZE) -> dict[str, dict]` — orchestrates the funnel; returns ordered `{ticker: {status:"pending", fund, wyckoff, setup_score, why}}`. Parallelize the per-ticker scoring with threads. The `fund`/`wyckoff` dicts are passed straight into `_analyze_ticker` so the cards don't re-fetch.
  - [ ] Cache: 15-min `TTLCache` (setups shift intraday but not every second). Reuse `_yf_lock`.
  - [ ] **Delete** `get_trending_tickers` + `_TRENDING_SEARCH_TERMS`.
- [ ] `gpt_investor/state.py`
  - [ ] `_analyze_ticker` accepts pre-computed `fund` / `wyckoff` dicts (skip re-scoring on the discovery path; still fetch fresh price + news + sentiment + sonnet). Keep them optional so single-company / industry modes still compute inline.
  - [ ] `fetch_analyses` `discovery_mode == "trending"` branch calls `get_setup_candidates` instead of `get_trending_tickers`. Rename the button label "Today's Trending" → **"Today's Setups"** (the mode is no longer about trendiness).
  - [ ] Update imports.
- [ ] Empty-result handling: 0 survivors (market closed, screens down, nothing passes the soft gate) → existing empty-`tickers_dict` early return; generalize the message to "No high-quality setups right now".
- [ ] UI: card shows the **why** chip — `"Strong • accumulation • +regime"` — so the user sees *why the tool picked it*, not a % move.
- [ ] Tests: `tests/test_discovery_setups.py`
  - [ ] `_is_equity_symbol` rules (NVDA yes; BTC-USD / EURUSD=X / ^GSPC / BRK.B no)
  - [ ] `_regime_fit` lookup determinism across (tier, phase, regime) combos
  - [ ] `_setup_score` + ranking determinism on synthetic fund/wyckoff dicts (no yfinance hit)
  - [ ] `_prefilter_by_dollar_volume` keeps the right top-N
  - [ ] `network`-marked: `get_setup_candidates()` returns non-empty during market hours (smoke)

### Acceptance

- "Today's Setups" surfaces tickers the **tool** rates well (fund tier + Wyckoff phase + regime fit), NOT whatever Yahoo ranks as hot
- Every card shows *why* it surfaced (the signal chip), not a % move
- `day_losers` names that are in accumulation + strong fundamentals can surface (dip-buys); falling knives (markdown phase, weak fund) are filtered out
- Scores computed in discovery are reused by the cards (no double yfinance fetch)
- Market-closed / weekend / all-filtered-out degrades to a clean empty state — no hanging spinner
- Crypto / FX / index never appear
- Offline tests deterministic; one network smoke test

### Decisions locked

- **Yahoo screens = candidate universe only.** They list tickers; the tool's deterministic scores decide what surfaces. This is the whole point of the rewrite.
- Include `day_losers` in the universe (the Wyckoff filter makes losers a *feature* — accumulation/dip-buy source — not noise).
- Scoring is **deterministic, no LLM** — reuses `fundamentals.py` + `wyckoff.py` + `market_regime.py`. The verdict LLM still only runs per-surfaced-ticker, unchanged.
- Pre-filter to ~60 by dollar-volume before full scoring — bounds the yfinance cost (the main downside vs the old funnel).
- Reuse the discovery-computed scores on the cards (pass through `_analyze_ticker`) so we pay the fetch once.
- **Depends on PW (Wyckoff).** Build order: `PW → PD'`.

### Open questions

- Composite weights `w_f` / `w_w` / regime bonus — hand-set at first; tune against calibration (Phase 2) once outcomes exist. (Don't optimise before data — see cross-cutting principles.)
- Hard gate vs soft gate on `fund >= Solid AND wyckoff ∈ {accumulation, markup}`? Soft (rank penalty) first, so a great-fundamentals name in neutral phase can still surface. Revisit.
- Universe size: is `most_actives + gainers + losers + trending` broad enough, or add a sector-rotation screen / 52-week-high screen later?
- Cost ceiling: ~60 full scores per refresh × yfinance latency — measure wall-clock; if too slow, cache per-ticker fund/wyckoff scores across discovery runs (not just the candidate list).

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
P0 (regime indicators) → PW (Wyckoff layer) → PD' (signal-driven discovery) → P1 (explainer agent) → P2 (verdict_history) → wait ~4 wks → P3 (audits) → P4 (data layers) → P5 (probabilistic + portfolio)
```

PW (Wyckoff — Stage [3] in the narrative pipeline above) ships **before** both PD' and P2: PD' needs the Wyckoff phase as a ranking input, and `verdict_history` needs to record `wyckoff_phase` + `wyckoff_score` from the first row or calibration can't measure the chart layer's lift. PW's full build spec lives in the "[3] Wyckoff" section above rather than as a separate numbered phase block.

PD' (signal-driven discovery) **supersedes the old Phase D mover funnel** — same Yahoo data plumbing, but the ranking is flipped from "what moved most" to "what the tool scores as the best setup." It depends on PW, so it can no longer ship first; it slots after the Wyckoff layer exists.

P0 + P1 + P2 can ship in one focused session (~7h). After that, the tool is data-accumulating and the next phase has actual ground truth to learn from.
