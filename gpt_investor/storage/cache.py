import json
import math
import os
import sqlite3
import time
from datetime import date

from loguru import logger

# Path resolved at call-time via env var so tests (conftest.py) can redirect
# every cache write to a tmp DB without monkeypatching the module attribute.
_DB = os.getenv("ANALYSES_DB", "analyses.db")

# 6 hours by default — central-bank stances change slowly; FOMC/ECB
# decisions land at predictable times and the disk cache only feeds the
# session-warmup path (a fresh `reload` still gets disk cache, not a
# stale in-memory one). Override with LIQUIDITY_TTL_SECONDS env var.
LIQUIDITY_TTL_SECONDS = int(os.getenv("LIQUIDITY_TTL_SECONDS", 6 * 3600))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            ticker          TEXT NOT NULL,
            date            TEXT NOT NULL,
            sentiment       TEXT,
            analyst_ratings TEXT,
            final_analysis  TEXT,
            sentiment_json  TEXT,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS liquidity (
            key         TEXT PRIMARY KEY,
            fetched_at  REAL NOT NULL,
            text        TEXT NOT NULL
        )
    """)
    # Verdict history — the feedback loop. One row per (ticker, date, prompt_version)
    # captured on the cache-MISS path only (a fresh verdict). Outcome + benchmark
    # columns start NULL; scripts/fill_outcomes.py backfills them once the horizon
    # has elapsed. Calibration groups these rows to measure edge. Inputs are stored
    # raw — never code-adjusted — so the LLM verdict stays auditable.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS verdict_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT NOT NULL,
            date            TEXT NOT NULL,
            created_at      REAL NOT NULL,
            prompt_version  TEXT NOT NULL,
            price           REAL,
            fund_score      REAL,
            fund_tier       TEXT,
            sentiment_score REAL,
            sentiment_conf  TEXT,
            analyst_grade   TEXT,
            analyst_score   REAL,
            regime_label    TEXT,
            wyckoff_phase   TEXT,
            wyckoff_score   REAL,
            sector          TEXT,
            industry        TEXT,
            verdict         TEXT,
            confidence      TEXT,
            price_target    REAL,
            sonnet_text     TEXT,
            spy_at_capture  REAL,
            price_7d        REAL,
            price_30d       REAL,
            price_90d       REAL,
            price_365d      REAL,
            spy_7d          REAL,
            spy_30d         REAL,
            spy_90d         REAL,
            spy_365d        REAL,
            UNIQUE (ticker, date, prompt_version)
        )
    """)
    # Idempotent migration for older DBs that pre-date sentiment_json.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(analyses)").fetchall()}
    if "sentiment_json" not in cols:
        conn.execute("ALTER TABLE analyses ADD COLUMN sentiment_json TEXT")
    conn.commit()
    return conn


def get_cached(ticker: str) -> dict | None:
    today = date.today().isoformat()
    with _conn() as conn:
        row = conn.execute(
            "SELECT sentiment, analyst_ratings, final_analysis, sentiment_json "
            "FROM analyses WHERE ticker=? AND date=?",
            (ticker, today),
        ).fetchone()
    if not row:
        return None
    sentiment_legacy, analyst_ratings, final_analysis, sentiment_json = row
    # Require at least the LLM verdict + analyst ratings to consider it a hit.
    if not (analyst_ratings and final_analysis):
        return None
    sentiment_dict: dict | None = None
    if sentiment_json:
        try:
            sentiment_dict = json.loads(sentiment_json)
        except json.JSONDecodeError:
            sentiment_dict = None
    return {
        "sentiment": sentiment_legacy,            # legacy prose, may be None for new rows
        "sentiment_dict": sentiment_dict,         # canonical structured form
        "analyst_ratings": analyst_ratings,
        "final_analysis": final_analysis,
    }


def _sentiment_is_clean(sentiment) -> bool:
    """Reject NaN scores, missing dict fields, completely empty payloads."""
    if sentiment is None:
        return False
    if isinstance(sentiment, dict):
        score = sentiment.get("score")
        try:
            f = float(score)
        except (TypeError, ValueError):
            return False
        if math.isnan(f) or math.isinf(f):
            return False
        return True
    return bool(str(sentiment).strip())


def save_cached(
    ticker: str,
    sentiment,
    analyst_ratings: str,
    final_analysis: str,
) -> None:
    """`sentiment` may be a dict (preferred) or legacy str.

    Refuses to write a degraded row (empty verdict, empty ratings, NaN
    sentiment score) — `get_cached` would skip it next time anyway, but a
    polluted row still consumes a `(ticker, date)` slot and makes audit
    queries noisy. Better to leave the slot empty so the next run retries.
    """
    if not (analyst_ratings and analyst_ratings.strip()):
        logger.warning("[{}] skip cache write: empty analyst_ratings", ticker)
        return
    if not (final_analysis and final_analysis.strip()):
        logger.warning("[{}] skip cache write: empty final_analysis", ticker)
        return
    if not _sentiment_is_clean(sentiment):
        logger.warning("[{}] skip cache write: bad sentiment payload ({!r})", ticker, sentiment)
        return

    today = date.today().isoformat()
    if isinstance(sentiment, dict):
        sentiment_legacy = sentiment.get("summary", "")
        sentiment_json = json.dumps(sentiment)
    else:
        sentiment_legacy = str(sentiment)
        sentiment_json = None
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO analyses "
            "(ticker, date, sentiment, analyst_ratings, final_analysis, sentiment_json) "
            "VALUES (?,?,?,?,?,?)",
            (ticker, today, sentiment_legacy, analyst_ratings, final_analysis, sentiment_json),
        )
        conn.commit()


def get_cached_liquidity(ttl_seconds: int = LIQUIDITY_TTL_SECONDS) -> str | None:
    """Return cached liquidity snapshot if fresher than `ttl_seconds`, else None."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT fetched_at, text FROM liquidity WHERE key=?", ("default",)
        ).fetchone()
    if not row:
        return None
    fetched_at, text = row
    if time.time() - fetched_at > ttl_seconds:
        return None
    return text


def save_cached_liquidity(text: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO liquidity (key, fetched_at, text) VALUES (?,?,?)",
            ("default", time.time(), text),
        )
        conn.commit()


# --- verdict history (feedback loop) ---------------------------------------

# Columns a caller may set on insert. Outcome/benchmark columns are filled later
# by the nightly job, not at capture time.
_VERDICT_INPUT_COLS = (
    "price", "fund_score", "fund_tier", "sentiment_score", "sentiment_conf",
    "analyst_grade", "analyst_score", "regime_label", "wyckoff_phase",
    "wyckoff_score", "sector", "industry", "verdict", "confidence",
    "price_target", "sonnet_text", "spy_at_capture",
)
_VERDICT_OUTCOME_COLS = (
    "price_7d", "price_30d", "price_90d", "price_365d",
    "spy_7d", "spy_30d", "spy_90d", "spy_365d",
)


def record_verdict(ticker: str, prompt_version: str, row: dict) -> None:
    """Insert one verdict-history row (cache-miss path only).

    Idempotent per (ticker, date, prompt_version) — a same-day re-run keeps the
    first capture. Never raises into the pipeline; logs and swallows on failure.

    Parameters
    ----------
    ticker : str
        Symbol the verdict is for.
    prompt_version : str
        Prompt contract identifier, stored so calibration never mixes contracts.
    row : dict
        Input columns to store; may contain any of `_VERDICT_INPUT_COLS`, and
        unknown keys are ignored.
    """
    today = date.today().isoformat()
    cols = ["ticker", "date", "created_at", "prompt_version"]
    vals: list = [ticker, today, time.time(), prompt_version]
    for c in _VERDICT_INPUT_COLS:
        if c in row:
            cols.append(c)
            vals.append(row[c])
    placeholders = ",".join("?" for _ in cols)
    try:
        with _conn() as conn:
            conn.execute(
                f"INSERT OR IGNORE INTO verdict_history ({','.join(cols)}) VALUES ({placeholders})",
                vals,
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.warning("[{}] record_verdict failed: {}", ticker, e)


def verdicts_needing_outcome(horizon: int) -> list[dict]:
    """Rows whose forward return at `horizon` days is not yet fully filled.

    A row is returned if either its ticker close OR its SPY close for this
    horizon is still NULL, so a row whose ticker leg filled but whose SPY leg
    missed its window keeps getting re-selected until both land.

    Parameters
    ----------
    horizon : int
        One of 7, 30, 90, 365 — selects the `price_<h>d` / `spy_<h>d` columns.

    Returns
    -------
    list of dict
        One dict per row: `id`, `ticker`, `date`, the capture `price` and
        `spy_at_capture`, plus `price_filled` / `spy_filled` (the current column
        values, None if not yet filled) so the caller writes only the NULL leg.
    """
    price_col, spy_col = f"price_{horizon}d", f"spy_{horizon}d"
    if price_col not in _VERDICT_OUTCOME_COLS:
        raise ValueError(f"unknown horizon: {horizon}")
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT id, ticker, date, price, spy_at_capture, {price_col}, {spy_col} "
            f"FROM verdict_history WHERE {price_col} IS NULL OR {spy_col} IS NULL"
        ).fetchall()
    return [
        {"id": r[0], "ticker": r[1], "date": r[2], "price": r[3], "spy_at_capture": r[4],
         "price_filled": r[5], "spy_filled": r[6]}
        for r in rows
    ]


def set_verdict_outcomes(row_id: int, updates: dict) -> None:
    """Fill outcome/benchmark columns on one row.

    Parameters
    ----------
    row_id : int
        Primary key of the verdict-history row to update.
    updates : dict
        Column/value pairs to write; keys outside `_VERDICT_OUTCOME_COLS` are
        ignored, and a no-op update is skipped.
    """
    fields = {k: v for k, v in updates.items() if k in _VERDICT_OUTCOME_COLS}
    if not fields:
        return
    assignments = ",".join(f"{k}=?" for k in fields)
    with _conn() as conn:
        conn.execute(
            f"UPDATE verdict_history SET {assignments} WHERE id=?",
            [*fields.values(), row_id],
        )
        conn.commit()


def all_verdicts() -> list[dict]:
    """Every verdict-history row as a dict (calibration reads this).

    Returns
    -------
    list of dict
        All rows, each keyed by column name.
    """
    with _conn() as conn:
        cur = conn.execute("SELECT * FROM verdict_history")
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]
