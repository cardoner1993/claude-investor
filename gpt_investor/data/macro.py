"""Deterministic central-bank policy-rate snapshot.

Fetches Fed funds (FRED), ECB MRO (ECB Statistical Data Warehouse) and
PBOC 1Y LPR (ChinaMoney English JSON API). Stance is computed from the
90-day rate delta — no LLM judgment in the numbers.

`get_liquidity_snapshot()` returns a structured dict; `format_liquidity()`
renders the markdown block that replaces the old LLM-paraphrased output.

A 1-paragraph LLM commentary on equity implications can be appended by
calling `commentary_via_llm(snapshot)` from `analysis.py` — the model only
sees the deterministic numbers, can't invent rates.

Setup:
    export FRED_API_KEY=...     # https://fred.stlouisfed.org/docs/api/api_key.html

Missing FRED key → Fed leg reports "n/a" and stance = "unknown".
Network/scrape failure on any leg → that leg reports "n/a", other legs unaffected.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from io import StringIO
from typing import Literal

import requests
from loguru import logger

Stance = Literal["easing", "neutral", "tightening", "unknown"]

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

ECB_BASE = "https://data-api.ecb.europa.eu/service/data"
# Daily MRO fixed rate. Series key documented at
# https://data.ecb.europa.eu/data/data-categories/financial-markets-and-interest-rates/policy-and-exchange-rates
ECB_MRO_KEY = "FM/D.U2.EUR.4F.KR.MRR_FR.LEV"

PBOC_LPR_HISTORY_API = "https://www.chinamoney.com.cn/ags/ms/cm-u-bk-currency/LprHis?lang=EN"
PBOC_LPR_PAGE_URL = "https://www.chinamoney.com.cn/english/bmklpr/"  # human-facing

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

_STANCE_EPS_PP = 0.05  # 5 basis points — below this, treat as flat


def _classify_stance(delta_pp: float | None) -> Stance:
    """Stance from a 90-day rate delta expressed in percentage points."""
    if delta_pp is None:
        return "unknown"
    if delta_pp < -_STANCE_EPS_PP:
        return "easing"
    if delta_pp > _STANCE_EPS_PP:
        return "tightening"
    return "neutral"


def _delta_90d(series: list[tuple[str, float]]) -> float | None:
    """Given desc-sorted [(date_iso, value), ...], return value_latest - value(~90d back)."""
    if not series:
        return None
    latest_date_s, latest_value = series[0]
    try:
        latest_date = datetime.fromisoformat(latest_date_s)
    except ValueError:
        return None
    for date_s, value in series[1:]:
        try:
            d = datetime.fromisoformat(date_s)
        except ValueError:
            continue
        if (latest_date - d).days >= 90:
            return latest_value - value
    return None


# --- FRED -----------------------------------------------------------------

def _fetch_fred_series(series_id: str, limit: int = 180) -> list[tuple[str, float]]:
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        logger.warning("FRED_API_KEY not set; skipping {}", series_id)
        return []
    try:
        r = requests.get(
            FRED_BASE,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
            },
            timeout=10,
        )
        r.raise_for_status()
    except Exception as e:
        logger.warning("FRED {} fetch failed: {}", series_id, e)
        return []
    out: list[tuple[str, float]] = []
    for obs in r.json().get("observations", []):
        v = obs.get("value")
        if v in (None, ".", ""):
            continue
        try:
            out.append((obs["date"], float(v)))
        except (KeyError, ValueError):
            continue
    return out


def get_fed_rate() -> dict:
    """Daily Effective Federal Funds Rate (FRED `DFF`)."""
    series = _fetch_fred_series("DFF", limit=180)
    value = series[0][1] if series else None
    delta = _delta_90d(series)
    return {
        "bank": "Fed",
        "region": "US",
        "rate_pct": value,
        "rate_label": f"{value:.2f}%" if value is not None else "n/a",
        "delta_90d_pp": delta,
        "stance": _classify_stance(delta),
        "as_of": series[0][0] if series else None,
        "source_url": "https://fred.stlouisfed.org/series/DFF",
    }


# --- ECB ------------------------------------------------------------------

def _fetch_ecb_csv(series_key: str) -> list[tuple[str, float]]:
    url = f"{ECB_BASE}/{series_key}?lastNObservations=180&format=csvdata"
    try:
        r = requests.get(url, headers={"Accept": "text/csv"}, timeout=10)
        r.raise_for_status()
    except Exception as e:
        logger.warning("ECB {} fetch failed: {}", series_key, e)
        return []
    rows = list(csv.DictReader(StringIO(r.text)))
    out: list[tuple[str, float]] = []
    for row in rows:
        try:
            out.append((row["TIME_PERIOD"], float(row["OBS_VALUE"])))
        except (KeyError, ValueError):
            continue
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def get_ecb_rate() -> dict:
    """ECB Main Refinancing Operations fixed rate."""
    series = _fetch_ecb_csv(ECB_MRO_KEY)
    value = series[0][1] if series else None
    delta = _delta_90d(series)
    return {
        "bank": "ECB",
        "region": "EU",
        "rate_pct": value,
        "rate_label": f"{value:.2f}%" if value is not None else "n/a",
        "delta_90d_pp": delta,
        "stance": _classify_stance(delta),
        "as_of": series[0][0] if series else None,
        "source_url": "https://data.ecb.europa.eu/data/datasets/FM",
    }


# --- PBOC -----------------------------------------------------------------

def _fetch_pboc_lpr_series() -> list[tuple[str, float]]:
    """Fetch ChinaMoney's English LPR history JSON. Returns desc-sorted
    [(YYYY-MM-DD, 1Y rate), ...]. Empty list on failure.
    """
    try:
        r = requests.get(
            PBOC_LPR_HISTORY_API,
            headers={
                "User-Agent": _UA,
                "Referer": PBOC_LPR_PAGE_URL,
                "Accept": "application/json",
            },
            timeout=10,
        )
        r.raise_for_status()
        records = r.json().get("records", [])
    except Exception as e:
        logger.warning("PBOC LPR fetch failed: {}", e)
        return []

    out: list[tuple[str, float]] = []
    for rec in records:
        date_s = rec.get("showDateCN")  # ISO YYYY-MM-DD
        rate = rec.get("1Y")
        if not date_s or rate in (None, "", "---"):
            continue
        try:
            out.append((date_s, float(rate)))
        except ValueError:
            continue
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def get_pboc_rate() -> dict:
    """PBOC 1-year Loan Prime Rate (LPR) from ChinaMoney's English JSON API."""
    series = _fetch_pboc_lpr_series()
    value = series[0][1] if series else None
    delta = _delta_90d(series)
    return {
        "bank": "PBOC",
        "region": "China",
        "rate_pct": value,
        "rate_label": f"{value:.2f}% LPR" if value is not None else "n/a",
        "delta_90d_pp": delta,
        "stance": _classify_stance(delta),
        "as_of": series[0][0] if series else None,
        "source_url": PBOC_LPR_PAGE_URL,
    }


# --- Snapshot + format ----------------------------------------------------

def snapshot_is_complete(snapshot: dict) -> bool:
    """True iff every leg has a real `rate_pct` (no None / NaN).

    Callers should refuse to disk-cache an incomplete snapshot — otherwise a
    transient outage (missing FRED key, ChinaMoney 5xx, ECB SDW down) freezes
    a bad value into the 6h cache.
    """
    banks = snapshot.get("banks", [])
    if not banks:
        return False
    for leg in banks:
        v = leg.get("rate_pct")
        if v is None:
            return False
        try:
            f = float(v)
        except (TypeError, ValueError):
            return False
        if f != f:  # NaN check
            return False
    return True


def get_liquidity_snapshot() -> dict:
    """Combine all three legs. Any leg may be partial (rate_pct=None)."""
    fed = get_fed_rate()
    ecb = get_ecb_rate()
    pboc = get_pboc_rate()
    snapshot = {
        "as_of": datetime.now(timezone.utc).date().isoformat(),
        "banks": [fed, ecb, pboc],
    }
    logger.info(
        "liquidity snapshot  Fed={}/{} ECB={}/{} PBOC={}/{}",
        fed["rate_label"], fed["stance"],
        ecb["rate_label"], ecb["stance"],
        pboc["rate_label"], pboc["stance"],
    )
    return snapshot


def _delta_str(delta: float | None) -> str:
    if delta is None:
        return ""
    return f" ({delta:+.2f}pp 90d)"


def format_liquidity(snapshot: dict) -> str:
    """Markdown block, same shape callers already expect from the old LLM output."""
    lines = ["**Global Liquidity Snapshot**", ""]
    for leg in snapshot.get("banks", []):
        bank = leg.get("bank", "?")
        region = leg.get("region", "?")
        rate = leg.get("rate_label", "n/a")
        stance = leg.get("stance", "unknown")
        delta_s = _delta_str(leg.get("delta_90d_pp"))
        url = leg.get("source_url", "")
        as_of = leg.get("as_of") or "?"
        lines.append(
            f"**{bank} ({region})**: {rate} — {stance}{delta_s} — as of {as_of} · {url}"
        )
    return "\n".join(lines)
