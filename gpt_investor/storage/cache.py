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
