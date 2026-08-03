#!/usr/bin/env python
"""Nightly backfill of verdict_history forward returns.

For every horizon (7/30/90/365d), find rows whose outcome column is still NULL
and whose horizon has elapsed, then fill the ticker's close and SPY's close
near (capture_date + horizon) using a ±5d trading-day window. Idempotent —
only NULLs are touched, so re-running never overwrites a filled value.

Schedule via cron/launchd, e.g. nightly at 22:00:
    0 22 * * *  cd /path/to/claude-investor && python scripts/fill_outcomes.py
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yfinance as yf
from loguru import logger

from gpt_investor.storage.cache import verdicts_needing_outcome, set_verdict_outcomes

_HORIZONS = {"price_7d": 7, "price_30d": 30, "price_90d": 90, "price_365d": 365}
_SPY_COL = {"price_7d": "spy_7d", "price_30d": "spy_30d",
            "price_90d": "spy_90d", "price_365d": "spy_365d"}
_WINDOW = 5  # trading-day tolerance either side of the target date

_hist_cache: dict = {}


def _history(ticker: str):
    """Full daily close history for `ticker`, fetched once per run."""
    if ticker in _hist_cache:
        return _hist_cache[ticker]
    try:
        df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False)
    except Exception as e:
        logger.warning("history fetch failed for {}: {}", ticker, e)
        df = None
    _hist_cache[ticker] = df
    return df


def _close_near(ticker: str, target: date) -> float | None:
    """Close on `target`, or the nearest trading day within ±_WINDOW days."""
    df = _history(ticker)
    if df is None or df.empty:
        return None
    idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
    for offset in range(_WINDOW + 1):
        for d in ({target - timedelta(days=offset), target + timedelta(days=offset)}
                  if offset else {target}):
            mask = idx.date == d
            if mask.any():
                return float(df["Close"][mask].iloc[0])
    return None


def main() -> None:
    today = date.today()
    filled = 0
    for price_col, days in _HORIZONS.items():
        rows = verdicts_needing_outcome(price_col)
        for row in rows:
            capture = datetime.fromisoformat(row["date"]).date()
            target = capture + timedelta(days=days)
            if today < target:
                continue  # horizon not elapsed yet
            px = _close_near(row["ticker"], target)
            spy = _close_near("SPY", target)
            updates = {}
            if px is not None:
                updates[price_col] = px
            if spy is not None:
                updates[_SPY_COL[price_col]] = spy
            if updates:
                set_verdict_outcomes(row["id"], updates)
                filled += 1
                logger.info(
                    "filled {} {} ({}d): px={} spy={}",
                    row["ticker"], row["date"], days, updates.get(price_col), updates.get(_SPY_COL[price_col]),
                )
    logger.info("fill_outcomes done — {} cells filled", filled)


if __name__ == "__main__":
    main()
