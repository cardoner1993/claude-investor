"""Higher-signal yfinance data (P4).

Cheap, high-signal additions layered on top of the 5-dimension fundamental
score. All the *scoring/flag* logic is pure (plain dicts/lists in) so it unit
-tests without yfinance; the `fetch_*` helpers are the only yfinance touch
points and each degrades to an empty/None result on failure.

    B1 short interest      — shortPercentOfFloat, >15% flag
    B2 earnings calendar   — days to next earnings, <7d banner
    B3 insider flow        — net 30/90d signed dollar flow
    B4 multi-year trend    — 4y revenue/FCF CAGR + margin slope
    B5 peer comparison     — industry-median fwd P/E / EV-EBITDA (megacap P/B fix)

Each signal is designed to earn a verdict_history column so calibration (P2)
can measure its lift — wiring that column lives with the data layer (#4).
"""

from __future__ import annotations

import math
import statistics
import threading
from datetime import date

import yfinance as yf
from cachetools import TTLCache
from loguru import logger

_HIGH_SHORT = 0.15          # >15% of float short = crowded-short flag
_EARNINGS_SOON = 7          # days to earnings under this = banner

_peer_cache: TTLCache = TTLCache(maxsize=64, ttl=6 * 3600)
_peer_lock = threading.Lock()


def _safe_float(x) -> float | None:
    try:
        v = float(x)
        return v if not (math.isnan(v) or math.isinf(v)) else None
    except (TypeError, ValueError):
        return None


# --- B1 short interest -----------------------------------------------------

def score_short_interest(short_pct: float | None) -> dict:
    """`short_pct` is a fraction (0.18 = 18% of float). Flags crowded shorts."""
    pct = _safe_float(short_pct)
    return {"short_pct": pct, "crowded": pct is not None and pct > _HIGH_SHORT}


# --- B2 earnings calendar --------------------------------------------------

def days_to_earnings(earnings_dates: list, as_of: date) -> int | None:
    """Smallest non-negative day-count to a future earnings date, or None."""
    future = []
    for d in earnings_dates or []:
        if isinstance(d, date):
            delta = (d - as_of).days
            if delta >= 0:
                future.append(delta)
    return min(future) if future else None


def earnings_banner(days: int | None) -> str:
    if days is None or days > _EARNINGS_SOON:
        return ""
    return "EARNINGS TODAY" if days == 0 else f"EARNINGS IN {days}D"


# --- B3 insider flow -------------------------------------------------------

_BUY_WORDS = ("purchase", "buy", "acquisition")
_SELL_WORDS = ("sale", "sell", "disposition")


def _txn_sign(transaction: str) -> int:
    t = (transaction or "").lower()
    if any(w in t for w in _BUY_WORDS):
        return 1
    if any(w in t for w in _SELL_WORDS):
        return -1
    return 0


def net_insider_flow(rows: list[dict], as_of: date, window_days: int) -> float | None:
    """Signed sum of insider transaction dollar value within `window_days`.

    `rows` items: `{value, date, transaction}`. Positive = net buying. Returns
    None when no dated, signed transactions fall in the window.
    """
    total = 0.0
    seen = False
    for r in rows or []:
        d = r.get("date")
        if not isinstance(d, date) or (as_of - d).days > window_days or d > as_of:
            continue
        sign = _txn_sign(r.get("transaction", ""))
        val = _safe_float(r.get("value"))
        if sign == 0 or val is None:
            continue
        total += sign * abs(val)
        seen = True
    return total if seen else None


# --- B4 multi-year trend ---------------------------------------------------

def cagr(first: float | None, last: float | None, years: float) -> float | None:
    """Compound annual growth from `first` to `last` over `years`.

    None if inputs are missing, non-positive (CAGR undefined through zero), or
    years <= 0.
    """
    f, l = _safe_float(first), _safe_float(last)
    if f is None or l is None or f <= 0 or l <= 0 or years <= 0:
        return None
    return round((l / f) ** (1.0 / years) - 1.0, 4)


def margin_slope(margins: list) -> float | None:
    """Least-squares slope of a margin series (oldest→newest), per period.

    Positive = margins expanding. None if fewer than 2 clean points.
    """
    ys = [_safe_float(m) for m in (margins or [])]
    ys = [y for y in ys if y is not None]
    n = len(ys)
    if n < 2:
        return None
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    return round(slope, 4)


# --- B5 peer comparison ----------------------------------------------------

def peer_relative(own: float | None, peer_values: list) -> dict:
    """Compare a ticker's multiple to its peer median.

    Returns `{median, ratio, cheaper}` — ratio = own/median (<1 = cheaper than
    peers). Softens the P/B punishment on asset-light megacaps by judging
    valuation against the sector, not an absolute threshold.
    """
    vals = [v for v in (_safe_float(x) for x in (peer_values or [])) if v is not None and v > 0]
    o = _safe_float(own)
    if not vals or o is None or o <= 0:
        return {"median": None, "ratio": None, "cheaper": None}
    med = statistics.median(vals)
    ratio = round(o / med, 2) if med else None
    return {"median": round(med, 2), "ratio": ratio, "cheaper": ratio is not None and ratio < 1.0}


def fetch_peer_medians(industry_key: str, max_peers: int = 8) -> dict:
    """Median forward P/E and EV/EBITDA across an industry's top companies.

    Cached 6h per industry_key (peers move slowly and many tickers share one
    industry). Best-effort — returns Nones on failure.
    """
    if not industry_key:
        return {"pe_median": None, "ev_median": None, "n": 0}
    with _peer_lock:
        hit = _peer_cache.get(industry_key)
    if hit is not None:
        return hit

    result = {"pe_median": None, "ev_median": None, "n": 0}
    try:
        companies = yf.Industry(industry_key).top_companies
        symbols = [s for s in companies.index if isinstance(s, str)][:max_peers] if companies is not None else []
    except Exception as e:
        logger.warning("peer medians industry {} failed: {}", industry_key, e)
        symbols = []

    pes: list[float] = []
    evs: list[float] = []
    lock = threading.Lock()

    def _one(sym: str) -> None:
        try:
            info = yf.Ticker(sym).info
            pe = _safe_float(info.get("forwardPE") or info.get("trailingPE"))
            ev = _safe_float(info.get("enterpriseToEbitda"))
            with lock:
                if pe and pe > 0:
                    pes.append(pe)
                if ev and ev > 0:
                    evs.append(ev)
        except Exception:
            pass

    threads = [threading.Thread(target=_one, args=(s,), daemon=True) for s in symbols]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=8)

    if pes:
        result["pe_median"] = round(statistics.median(pes), 2)
    if evs:
        result["ev_median"] = round(statistics.median(evs), 2)
    result["n"] = max(len(pes), len(evs))

    with _peer_lock:
        _peer_cache[industry_key] = result
    return result


# --- orchestration + fetch -------------------------------------------------

def fetch_signals(ticker: str, fundamentals: dict | None = None) -> dict:
    """Pull all higher-signal legs for one ticker. Every leg is independently
    guarded — a failure in one degrades that leg to None, never the whole dict.

    `fundamentals` is the scored dict (for the peer-relative comparison + the
    industry key); when omitted, peer comparison is skipped.
    """
    t = yf.Ticker(ticker)
    try:
        info = t.info
    except Exception as e:
        logger.bind(ticker=ticker).warning("signals info fetch failed: {}", e)
        info = {}

    short = score_short_interest(info.get("shortPercentOfFloat"))

    earnings_days = None
    try:
        cal = t.calendar or {}
        ed = cal.get("Earnings Date")
        if ed is not None:
            earnings_days = days_to_earnings(ed if isinstance(ed, list) else [ed], date.today())
    except Exception as e:
        logger.bind(ticker=ticker).debug("signals calendar failed: {}", e)

    insider_30 = insider_90 = None
    try:
        df = t.insider_transactions
        if df is not None and not df.empty:
            rows = [
                {"value": r.get("Value"), "date": _as_date(r.get("Start Date")), "transaction": r.get("Transaction")}
                for r in df.to_dict("records")
            ]
            insider_30 = net_insider_flow(rows, date.today(), 30)
            insider_90 = net_insider_flow(rows, date.today(), 90)
    except Exception as e:
        logger.bind(ticker=ticker).debug("signals insider failed: {}", e)

    rev_cagr = fcf_cagr = op_margin_slope = None
    try:
        fin = t.financials
        if fin is not None and not fin.empty:
            years = fin.shape[1] - 1  # columns are periods, newest first
            rev = _row(fin, "Total Revenue")
            if rev and years > 0:
                rev_cagr = cagr(rev[-1], rev[0], years)
            op_income = _row(fin, "Operating Income")
            if rev and op_income:
                margins = [oi / rv for oi, rv in zip(reversed(op_income), reversed(rev)) if rv]
                op_margin_slope = margin_slope(margins)
        cf = t.cashflow
        if cf is not None and not cf.empty:
            fcf = _row(cf, "Free Cash Flow")
            yrs = cf.shape[1] - 1
            if fcf and yrs > 0:
                fcf_cagr = cagr(fcf[-1], fcf[0], yrs)
    except Exception as e:
        logger.bind(ticker=ticker).debug("signals trend failed: {}", e)

    peers = {"pe_median": None, "ev_median": None, "n": 0, "pe_rel": None}
    if fundamentals is not None:
        industry_key = info.get("industryKey", "")
        pm = fetch_peer_medians(industry_key)
        own_pe = (fundamentals.get("raw", {}) or {}).get("forward_pe") or (fundamentals.get("raw", {}) or {}).get("trailing_pe")
        peers = {**pm, "pe_rel": peer_relative(own_pe, [pm["pe_median"]] if pm["pe_median"] else [])}

    return {
        "short": short,
        "earnings_days": earnings_days,
        "earnings_banner": earnings_banner(earnings_days),
        "insider_30d": insider_30,
        "insider_90d": insider_90,
        "rev_cagr": rev_cagr,
        "fcf_cagr": fcf_cagr,
        "op_margin_slope": op_margin_slope,
        "peers": peers,
    }


def _as_date(x) -> date | None:
    if isinstance(x, date):
        return x
    try:
        return x.date()  # pandas Timestamp
    except AttributeError:
        return None


def _row(df, label: str) -> list[float] | None:
    """Row of a yfinance statement DataFrame as newest→oldest floats, or None."""
    try:
        if label not in df.index:
            return None
        vals = [_safe_float(v) for v in df.loc[label].tolist()]
        return [v for v in vals if v is not None] or None
    except Exception:
        return None


# --- formatting ------------------------------------------------------------

def _fmt_pct(v: float | None) -> str:
    return f"{v:+.1%}" if v is not None else "n/a"


def _fmt_flow(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{'+' if v >= 0 else '-'}${abs(v) / 1e6:.1f}M"


def format_signals(sig: dict) -> str:
    """Markdown block for the final-analysis prompt + dialog."""
    short = sig.get("short", {})
    peers = sig.get("peers", {})
    lines = ["**Higher-signal data**", ""]

    short_pct = short.get("short_pct")
    short_line = f"- Short interest: {short_pct:.1%} of float" if short_pct is not None else "- Short interest: n/a"
    if short.get("crowded"):
        short_line += " ⚠ crowded short (>15%)"
    lines.append(short_line)

    ed = sig.get("earnings_days")
    lines.append(f"- Next earnings: {'in ' + str(ed) + 'd' if ed is not None else 'n/a'}"
                 + (f"  ⚠ {sig['earnings_banner']}" if sig.get("earnings_banner") else ""))

    lines.append(f"- Insider net flow: 30d {_fmt_flow(sig.get('insider_30d'))}, 90d {_fmt_flow(sig.get('insider_90d'))}")
    lines.append(
        f"- Multi-year trend: revenue CAGR {_fmt_pct(sig.get('rev_cagr'))}, "
        f"FCF CAGR {_fmt_pct(sig.get('fcf_cagr'))}, "
        f"op-margin slope {sig.get('op_margin_slope') if sig.get('op_margin_slope') is not None else 'n/a'}"
    )

    pe_med = peers.get("pe_median")
    if pe_med is not None:
        rel = peers.get("pe_rel") or {}
        tag = ""
        if rel.get("ratio") is not None:
            tag = f" — this name at {rel['ratio']}x the median ({'cheaper' if rel.get('cheaper') else 'richer'})"
        lines.append(f"- Peer valuation: industry median fwd P/E {pe_med} (n={peers.get('n')}){tag}")

    return "\n".join(lines)
