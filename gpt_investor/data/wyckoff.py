"""Deterministic Wyckoff-flavoured price/volume timing layer.

Fundamentals say *what to own*; this module says *whether now*. It reads
daily OHLCV from yfinance, computes a handful of price/volume features, maps
them to a Wyckoff phase, and scores the phase as a timing quality on a 0-10
scale — same tier vocabulary as `fundamentals.py` so the card chip is
consistent.

No LLM anywhere. The verdict LLM only ever *reads* the rendered block; it
never invents the chart structure.

The logic is split so it's unit-testable without hitting yfinance:
    - `compute_signals(df)`  — DataFrame in, plain feature dict out
    - `classify_phase(sig)`  — pure function on the feature dict (testable core)
    - `score_wyckoff(sig, phase)` — pure scoring on the feature dict
    - `fetch_ohlcv` / `fetch_ohlcv_batch` — the only yfinance touch points

Phase vocabulary is canonical (PD' discovery ranks on these exact 5 labels):
    accumulation | markup | distribution | markdown | neutral

Score interpretation (mirrors fundamentals tiers):
    >= 8.0  Strong   — clean, confirmed, favourable entry
    >= 6.0  Solid
    >= 4.0  Average
    >= 2.0  Weak
    <  2.0  Avoid     — falling knife / topping
"""

from __future__ import annotations

import math
from typing import Literal

import yfinance as yf
from loguru import logger

Phase = Literal["accumulation", "markup", "distribution", "markdown", "neutral"]

# --- tunables --------------------------------------------------------------

_TREND_EPS = 0.005          # 0.5% — SMA50 within this of SMA200 = flat
_VOL_SURGE_RATIO = 1.5      # 5d avg volume >= 1.5x 50d avg = surge
_CONSOLIDATE_RANGE = 0.12   # 20d (high-low)/close below this = consolidating
_EXPAND_RANGE = 0.25        # ...above this = expanding
_NEAR_HIGH = -0.10          # within 10% of 52w high
_NEAR_LOW = 0.05            # within 5% of 52w low
_OVEREXTENDED = 0.30        # >30% above the 200d SMA
_THIN_HISTORY = 120         # fewer daily bars than this = thin


def _safe_float(x) -> float | None:
    """Coerce a value to float, returning None on failure or NaN.

    Parameters
    ----------
    x : Any
        Value to coerce.

    Returns
    -------
    float | None
        Float value, or None if not coercible or NaN.
    """
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


# --- signal computation ----------------------------------------------------

def compute_signals(df) -> dict:
    """Derive price/volume features from a daily OHLCV DataFrame.

    Tolerates short/empty history by returning Nones; downstream
    `classify_phase` degrades those to `neutral`.

    Parameters
    ----------
    df : pandas.DataFrame | None
        What `fetch_ohlcv` returns — columns Open/High/Low/Close/Volume,
        DatetimeIndex, oldest-first. None/empty yields an all-None feature dict.

    Returns
    -------
    dict
        Feature dict: bar count, SMAs, trend, 52w distances, volume ratio/bias,
        range width, and breakout/new-low flags.
    """
    n = 0 if df is None else len(df)
    sig: dict = {
        "n_bars": n,
        "last_close": None, "sma50": None, "sma200": None,
        "pct_above_200d": None, "trend": None,
        "dist_52w_high": None, "dist_52w_low": None,
        "vol_ratio": None, "vol_surge": False, "up_down_vol_bias": None,
        "range_pct": None, "consolidating": False, "expanding": False,
        "breakout_up": False, "breakout_down": False, "making_new_low": False,
    }
    if n == 0:
        return sig

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    last_close = _safe_float(close.iloc[-1])
    sig["last_close"] = last_close

    if n >= 50:
        sig["sma50"] = _safe_float(close.tail(50).mean())
    if n >= 200:
        sig["sma200"] = _safe_float(close.tail(200).mean())

    if sig["sma50"] is not None and sig["sma200"] is not None and sig["sma200"]:
        gap = (sig["sma50"] - sig["sma200"]) / sig["sma200"]
        sig["trend"] = "up" if gap > _TREND_EPS else "down" if gap < -_TREND_EPS else "flat"
    if sig["sma200"] and last_close is not None:
        sig["pct_above_200d"] = (last_close - sig["sma200"]) / sig["sma200"]

    # 52-week extremes (use whatever window we have, up to ~252 bars)
    hi = _safe_float(high.max())
    lo = _safe_float(low.min())
    if hi and last_close is not None:
        sig["dist_52w_high"] = (last_close / hi) - 1.0   # <= 0, near 0 = at highs
    if lo and last_close is not None:
        sig["dist_52w_low"] = (last_close / lo) - 1.0     # >= 0, near 0 = at lows
    if sig["dist_52w_low"] is not None:
        sig["making_new_low"] = sig["dist_52w_low"] <= _NEAR_LOW

    # volume surge: 5d vs 50d average
    if n >= 50:
        v5 = _safe_float(volume.tail(5).mean())
        v50 = _safe_float(volume.tail(50).mean())
        if v5 is not None and v50:
            sig["vol_ratio"] = v5 / v50
            sig["vol_surge"] = sig["vol_ratio"] >= _VOL_SURGE_RATIO

    # up/down volume bias over the last 20 bars → [-1, +1]
    if n >= 21:
        recent_close = close.tail(21)
        recent_vol = volume.tail(21)
        up_vol = down_vol = 0.0
        prev = None
        for c, v in zip(recent_close, recent_vol):
            cf, vf = _safe_float(c), _safe_float(v)
            if prev is not None and cf is not None and vf is not None:
                if cf > prev:
                    up_vol += vf
                elif cf < prev:
                    down_vol += vf
            prev = cf
        tot = up_vol + down_vol
        if tot > 0:
            sig["up_down_vol_bias"] = (up_vol - down_vol) / tot

    # 20d range width → consolidating / expanding
    if n >= 20 and last_close:
        rng = _safe_float(high.tail(20).max() - low.tail(20).min())
        if rng is not None:
            sig["range_pct"] = rng / last_close
            sig["consolidating"] = sig["range_pct"] < _CONSOLIDATE_RANGE
            sig["expanding"] = sig["range_pct"] > _EXPAND_RANGE

    # breakout: close clears the prior-20d range (excluding today) on a surge
    if n >= 22 and last_close is not None:
        prior_high = _safe_float(high.iloc[-21:-1].max())
        prior_low = _safe_float(low.iloc[-21:-1].min())
        if prior_high is not None and last_close > prior_high and sig["vol_surge"]:
            sig["breakout_up"] = True
        if prior_low is not None and last_close < prior_low and sig["vol_surge"]:
            sig["breakout_down"] = True

    return sig


# --- phase classification (pure, testable core) ----------------------------

def classify_phase(sig: dict) -> Phase:
    """Map a feature dict to a Wyckoff phase. Ordered rules, first match wins.

    Falls back to `neutral` when history is too thin to judge or no rule fires.

    Parameters
    ----------
    sig : dict
        Feature dict from `compute_signals`.

    Returns
    -------
    Phase
        One of accumulation / markup / distribution / markdown / neutral.
    """
    if sig.get("n_bars", 0) < 60 or sig.get("sma200") is None:
        return "neutral"

    trend = sig.get("trend")
    above200 = sig.get("pct_above_200d")
    bias = sig.get("up_down_vol_bias")
    dist_high = sig.get("dist_52w_high")
    consolidating = sig.get("consolidating", False)
    breakout_up = sig.get("breakout_up", False)
    breakout_down = sig.get("breakout_down", False)
    making_new_low = sig.get("making_new_low", False)

    b = bias if bias is not None else 0.0
    near_high = dist_high is not None and dist_high >= _NEAR_HIGH

    # 1) MARKUP — confirmed uptrend: above 200d, rising structure, buyers in control
    if trend == "up" and (above200 is not None and above200 > 0) and (breakout_up or near_high) and b >= 0:
        return "markup"

    # 2) DISTRIBUTION — elevated but rolling over: high/near-high, going sideways,
    #    volume turning negative (supply hitting the bid at the top)
    if (above200 is not None and above200 > 0) and (consolidating or near_high) and b < -0.1:
        return "distribution"
    if (above200 is not None and above200 > 0) and breakout_down:
        return "distribution"

    # 3) MARKDOWN — confirmed downtrend: below 200d, selling pressure, near/making lows
    if trend == "down" and (above200 is not None and above200 < 0) and (b < 0 or making_new_low):
        return "markdown"

    # 4) ACCUMULATION — basing after decline: sideways range, selling abating,
    #    not making fresh lows (the constructive base, distinct from markdown)
    if consolidating and not making_new_low and b >= -0.1:
        return "accumulation"

    return "neutral"


# --- scoring (pure) --------------------------------------------------------

# Timing quality per phase — "how good is *now* as an entry".
_PHASE_BASE = {
    "markup":       8.0,   # trend confirmed, buyers in control
    "accumulation": 7.0,   # early, favourable risk/reward
    "neutral":      5.0,   # no edge either way
    "distribution": 3.0,   # topping — poor entry
    "markdown":     1.0,   # falling knife — avoid
}


def _tier(score: float) -> str:
    """Map a 0-10 timing score to a tier label.

    Parameters
    ----------
    score : float
        Wyckoff timing score.

    Returns
    -------
    str
        One of Strong / Solid / Average / Weak / Avoid.
    """
    if score >= 8.0: return "Strong"
    if score >= 6.0: return "Solid"
    if score >= 4.0: return "Average"
    if score >= 2.0: return "Weak"
    return "Avoid"


def score_wyckoff(sig: dict, phase: Phase | None = None) -> dict:
    """Score a phase as timing quality with confirmation adjustments.

    Pure — takes the feature dict so it can be unit-tested without yfinance.

    Parameters
    ----------
    sig : dict
        Feature dict from `compute_signals` (or a test fixture).
    phase : Phase | None, optional
        Phase to score; defaults to `classify_phase(sig)`.

    Returns
    -------
    dict
        {phase, score (0-10), tier, signals, flags}.
    """
    if phase is None:
        phase = classify_phase(sig)

    score = _PHASE_BASE.get(phase, 5.0)
    above200 = sig.get("pct_above_200d")
    bias = sig.get("up_down_vol_bias")
    thin = sig.get("n_bars", 0) < _THIN_HISTORY
    overextended = above200 is not None and above200 > _OVEREXTENDED

    # confirmation / contradiction adjustments
    if phase in ("markup", "accumulation"):
        if sig.get("breakout_up") and sig.get("vol_surge"):
            score += 1.0                      # volume-confirmed breakout
        if (bias is None or bias <= 0) and not sig.get("vol_surge"):
            score -= 1.0                      # no volume behind the move
    if phase == "markup" and overextended:
        score -= 2.0                          # chasing an extended move
    if phase == "markdown" and sig.get("making_new_low"):
        score -= 1.0

    # thin history can't confirm anything — pull toward neutral, never reward
    if thin:
        score = min(score, 5.0)

    score = round(max(0.0, min(10.0, score)), 2)

    flags: list[str] = []
    if above200 is not None and above200 < 0:
        flags.append("below 200d")
    if (above200 is not None and above200 > 0) and bias is not None and bias < -0.2:
        flags.append("distribution volume")
    if overextended:
        flags.append("overextended")
    if phase in ("markup", "accumulation") and not sig.get("vol_surge") and (bias is None or bias <= 0):
        flags.append("no volume confirmation")
    if sig.get("making_new_low"):
        flags.append("making new lows")
    if thin:
        flags.append("thin history")

    return {
        "phase": phase,
        "score": score,
        "tier": _tier(score),
        "signals": sig,
        "flags": flags,
    }


# --- data fetch ------------------------------------------------------------

def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d"):
    """Fetch single-ticker daily OHLCV from yfinance.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    period : str, optional
        yfinance period string, default "1y".
    interval : str, optional
        yfinance interval string, default "1d".

    Returns
    -------
    pandas.DataFrame | None
        OHLCV frame, or None on failure / empty history.
    """
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    except Exception as e:
        logger.bind(ticker=ticker).warning("wyckoff fetch_ohlcv failed: {}", e)
        return None
    if df is None or df.empty:
        logger.bind(ticker=ticker).warning("wyckoff fetch_ohlcv empty")
        return None
    return df


def fetch_ohlcv_batch(tickers: list[str], period: str = "1y") -> dict:
    """Fetch multi-ticker OHLCV in one download.

    Used by PD' discovery, which scores ~60 names per refresh — a per-ticker
    loop would be 60 calls.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols; deduped, order preserved.
    period : str, optional
        yfinance period string, default "1y".

    Returns
    -------
    dict
        {ticker: DataFrame}; missing/failed tickers are omitted.
    """
    tickers = [t for t in dict.fromkeys(tickers) if t]  # dedupe, keep order
    if not tickers:
        return {}
    try:
        raw = yf.download(
            tickers, period=period, interval="1d",
            auto_adjust=False, group_by="ticker", threads=True, progress=False,
        )
    except Exception as e:
        logger.warning("wyckoff fetch_ohlcv_batch failed: {}", e)
        return {}
    if raw is None or raw.empty:
        return {}

    out: dict = {}
    for t in tickers:
        try:
            sub = raw[t] if len(tickers) > 1 else raw
        except (KeyError, TypeError):
            continue
        sub = sub.dropna(how="all")
        if sub is not None and not sub.empty:
            out[t] = sub
    return out


def score_ticker(ticker: str) -> dict:
    """Fetch, compute, classify, and score one ticker in a single call.

    The card path's convenience wrapper.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    dict
        Output of `score_wyckoff`.
    """
    df = fetch_ohlcv(ticker)
    sig = compute_signals(df)
    return score_wyckoff(sig)


# --- formatting ------------------------------------------------------------

def _fmt_pct(v: float | None) -> str:
    """Format a fraction as a signed percentage, or "n/a" when None.

    Parameters
    ----------
    v : float | None
        Value as a fraction (0.15 = 15%).

    Returns
    -------
    str
        Signed percentage string or "n/a".
    """
    return f"{v:+.1%}" if v is not None else "n/a"


_PHASE_BLURB = {
    "accumulation": "basing after decline — selling abating, constructive",
    "markup":       "confirmed uptrend — buyers in control",
    "distribution": "elevated but rolling over — supply hitting the bid",
    "markdown":     "downtrend — falling, no support yet",
    "neutral":      "no clear structure / thin history",
}


def format_wyckoff(scored: dict) -> str:
    """Render a scored Wyckoff dict as a markdown block.

    Feeds both the verdict prompt and the card dialog.

    Parameters
    ----------
    scored : dict
        Output of `score_wyckoff`.

    Returns
    -------
    str
        Multi-line markdown with the timing score, phase, trend/volume/range
        lines, and any flags.
    """
    sig = scored.get("signals", {})
    phase = scored["phase"]
    lines = [
        f"**Wyckoff timing: {scored['score']}/10 ({scored['tier']}) — phase: {phase}**",
        f"_{_PHASE_BLURB.get(phase, '')}_",
        "",
        f"- Trend: {sig.get('trend') or 'n/a'} "
        f"(price {_fmt_pct(sig.get('pct_above_200d'))} vs 200d SMA)",
        f"- 52w position: {_fmt_pct(sig.get('dist_52w_high'))} from high, "
        f"{_fmt_pct(sig.get('dist_52w_low'))} from low",
        f"- Volume: 5d/50d {('%.2fx' % sig['vol_ratio']) if sig.get('vol_ratio') else 'n/a'}"
        f"{' (surge)' if sig.get('vol_surge') else ''}, "
        f"up/down bias {_fmt_pct(sig.get('up_down_vol_bias'))}",
        f"- Range: {'consolidating' if sig.get('consolidating') else 'expanding' if sig.get('expanding') else 'normal'}"
        f"{', breakout up' if sig.get('breakout_up') else ', breakout down' if sig.get('breakout_down') else ''}",
    ]
    if scored["flags"]:
        lines.append("")
        lines.append("Flags: " + ", ".join(scored["flags"]))
    return "\n".join(lines)
