#!/usr/bin/env python
"""Nightly backfill of verdict_history forward returns.

For every horizon (7/30/90/365d), find rows whose ticker OR SPY outcome is still
NULL and whose horizon has elapsed, then fill each with the capture price scaled
by the SPLIT-ADJUSTED return factor over the window. Using adjusted closes for
both endpoints (and scaling the stored capture price by their ratio) keeps the
return on one consistent basis, so a split between capture and horizon can't
turn a flat stock into a −90% "return". Idempotent — only NULL cells are
touched; the ticker and SPY legs fill independently.

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
_WINDOW = 7  # calendar-day tolerance either side of the target date (covers long closures)

_hist_cache: dict = {}


def _history(ticker: str):
    """Full split/dividend-ADJUSTED daily close history, fetched once per run.

    `auto_adjust=True` back-adjusts the whole series, so both endpoints of a
    return sit on the same basis regardless of splits in between. Cached per
    ticker; a failed fetch caches None.

    Parameters
    ----------
    ticker : str
        Symbol to fetch.

    Returns
    -------
    pandas.DataFrame or None
        The adjusted daily history, or None if the fetch failed.
    """
    if ticker in _hist_cache:
        return _hist_cache[ticker]
    try:
        df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
    except Exception as e:
        logger.warning("history fetch failed for {}: {}", ticker, e)
        df = None
    _hist_cache[ticker] = df
    return df


def _adj_close_near(ticker: str, target: date) -> float | None:
    """Adjusted close on `target`, or the nearest day within ±_WINDOW.

    Parameters
    ----------
    ticker : str
        Symbol to look up.
    target : date
        Desired trading date; nearest available day is searched outward.

    Returns
    -------
    float or None
        The adjusted close, or None if no close falls within the window.
    """
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


def _return_factor(ticker: str, capture: date, target: date) -> float | None:
    """Split-adjusted total-return factor between capture and target dates.

    Computes `adj_target / adj_capture` so a stored capture price can be scaled
    to the horizon on one consistent basis.

    Parameters
    ----------
    ticker : str
        Symbol to look up.
    capture : date
        Verdict capture date (denominator).
    target : date
        Horizon date (numerator).

    Returns
    -------
    float or None
        The return factor, or None if either close is unavailable.
    """
    a = _adj_close_near(ticker, capture)
    b = _adj_close_near(ticker, target)
    if a is None or b is None or a == 0:
        return None
    return b / a


def main() -> None:
    """Backfill every elapsed horizon's NULL ticker/SPY outcome cells.

    Iterates all horizons, selects rows whose window has passed and whose
    outcome is still NULL, scales the capture price by the split-adjusted return
    factor, and writes the result. Idempotent.
    """
    today = date.today()
    filled = 0
    for price_col, days in _HORIZONS.items():
        spy_col = _SPY_COL[price_col]
        rows = verdicts_needing_outcome(price_col, spy_col)
        for row in rows:
            capture = datetime.fromisoformat(row["date"]).date()
            target = capture + timedelta(days=days)
            if today < target:
                continue  # horizon not elapsed yet
            updates = {}
            if row["price_val"] is None and row["price"]:
                f = _return_factor(row["ticker"], capture, target)
                if f is not None:
                    updates[price_col] = row["price"] * f
            if row["spy_val"] is None and row["spy_at_capture"]:
                f = _return_factor("SPY", capture, target)
                if f is not None:
                    updates[spy_col] = row["spy_at_capture"] * f
            if updates:
                set_verdict_outcomes(row["id"], updates)
                filled += 1
                logger.info(
                    "filled {} {} ({}d): {}",
                    row["ticker"], row["date"], days,
                    {k: round(v, 2) for k, v in updates.items()},
                )
    logger.info("fill_outcomes done — {} cells filled", filled)


if __name__ == "__main__":
    main()
