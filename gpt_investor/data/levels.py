"""Deterministic support/resistance price-levels layer.

Fundamentals say *what to own*, Wyckoff says *whether now* — this module says
*where*: the concrete price barriers a falling/rising stock is likely to stall
or bounce at, so a verdict can read "support at $X, resistance at $Y, currently
$Z". No LLM anywhere; the verdict LLM only ever reads the rendered block.

Levels come from three sources, merged into one weighted list:
    - swing pivots (local highs/lows) clustered by proximity, weighted by touch
      count and the volume traded at each touch
    - moving averages (SMA50 / SMA200) — dynamic levels price respects
    - structural anchors (52-week high/low, nearest round number)

The pure core is testable on synthetic OHLCV without yfinance:
    - `find_pivots(df)`       — swing highs/lows from OHLCV
    - `cluster_levels(...)`   — merge nearby pivots into weighted levels
    - `build_levels(df)`      — full level set (pivots + SMA + anchors)
    - `nearest_levels(...)`   — closest support/resistance to a price
    - `score_levels(...)`     — 0-10 quality of price's position in the range
    - `fetch_ohlcv` reuses `wyckoff.fetch_ohlcv` (shared 1y daily download)

Score interpretation mirrors fundamentals/wyckoff tiers:
    >= 8.0  Strong   — just off strong support, clear room to resistance
    >= 6.0  Solid
    >= 4.0  Average
    >= 2.0  Weak
    <  2.0  Avoid     — pinned under strong resistance, little downside cushion
"""

from __future__ import annotations

import math

from loguru import logger

from gpt_investor.data.wyckoff import fetch_ohlcv, _safe_float

# --- tunables --------------------------------------------------------------

_PIVOT_LOOKBACK = 5         # bars each side that a swing high/low must dominate
_CLUSTER_TOL = 0.02         # pivots within 2% of each other merge into one level
_NEAR_LEVEL = 0.02          # price within 2% of a level counts as "at" it
_STRONG_TOUCHES = 3         # a level touched this many times is "strong"
_MIN_BARS = 60              # fewer daily bars than this = can't build a structure


def _round_number(price: float) -> float:
    """Nearest psychologically significant round number to a price.

    Uses a step scaled to the price's magnitude (10s for >$500, 5s for >$100,
    1s for >$20, else 0.5) — the levels traders actually watch.

    Parameters
    ----------
    price : float
        Reference price.

    Returns
    -------
    float
        Closest round number on the magnitude-appropriate grid.
    """
    if price > 500:
        step = 10.0
    elif price > 100:
        step = 5.0
    elif price > 20:
        step = 1.0
    else:
        step = 0.5
    return round(price / step) * step


# --- pivots (pure, testable core) ------------------------------------------

def find_pivots(df, lookback: int = _PIVOT_LOOKBACK) -> list[dict]:
    """Extract swing highs/lows from a daily OHLCV DataFrame.

    A bar is a swing high if its High is the strict maximum of the window
    ``[i-lookback, i+lookback]``, a swing low if its Low is the strict minimum.
    The first/last ``lookback`` bars can't be centred so they're skipped.

    Parameters
    ----------
    df : pandas.DataFrame | None
        OHLCV frame from `fetch_ohlcv` (Open/High/Low/Close/Volume, oldest
        first). None/short history yields an empty list.
    lookback : int, optional
        Bars each side the pivot must dominate. Default 5.

    Returns
    -------
    list[dict]
        Each pivot ``{price, kind, volume}`` where kind is "high"/"low".
    """
    n = 0 if df is None else len(df)
    if n < 2 * lookback + 1:
        return []

    high = df["High"].astype(float).tolist()
    low = df["Low"].astype(float).tolist()
    volume = df["Volume"].astype(float).tolist()

    pivots: list[dict] = []
    for i in range(lookback, n - lookback):
        window_hi = high[i - lookback : i + lookback + 1]
        window_lo = low[i - lookback : i + lookback + 1]
        vol = _safe_float(volume[i]) or 0.0
        if high[i] == max(window_hi) and window_hi.count(high[i]) == 1:
            pivots.append({"price": high[i], "kind": "high", "volume": vol})
        if low[i] == min(window_lo) and window_lo.count(low[i]) == 1:
            pivots.append({"price": low[i], "kind": "low", "volume": vol})
    return pivots


def cluster_levels(pivots: list[dict], tol: float = _CLUSTER_TOL) -> list[dict]:
    """Merge nearby pivots into weighted price levels.

    Pivots within ``tol`` (fractional) of a running cluster mean collapse into
    one level. Strength combines touch count with the share of total pivot
    volume traded at the level — a level defended on heavy volume outranks one
    grazed a few times on light volume.

    Parameters
    ----------
    pivots : list[dict]
        Output of `find_pivots`.
    tol : float, optional
        Fractional proximity for merging. Default 0.02 (2%).

    Returns
    -------
    list[dict]
        Levels ``{price, touches, volume, strength}`` sorted by price ascending.
        ``strength`` is a 0+ float (roughly touches, volume-boosted).
    """
    if not pivots:
        return []

    total_vol = sum(p["volume"] for p in pivots) or 1.0
    clusters: list[dict] = []
    for p in sorted(pivots, key=lambda x: x["price"]):
        if clusters and abs(p["price"] - clusters[-1]["_mean"]) / clusters[-1]["_mean"] <= tol:
            c = clusters[-1]
            c["_prices"].append(p["price"])
            c["touches"] += 1
            c["volume"] += p["volume"]
            c["_mean"] = sum(c["_prices"]) / len(c["_prices"])
        else:
            clusters.append({"_prices": [p["price"]], "_mean": p["price"],
                             "touches": 1, "volume": p["volume"]})

    levels: list[dict] = []
    for c in clusters:
        vol_share = c["volume"] / total_vol
        levels.append({
            "price": round(c["_mean"], 2),
            "touches": c["touches"],
            "volume": c["volume"],
            "strength": round(c["touches"] * (1.0 + vol_share), 2),
        })
    return levels


def build_levels(df, price: float | None = None) -> list[dict]:
    """Build the full level set: clustered pivots plus dynamic/structural lines.

    Adds SMA50, SMA200, the 52-week high/low, and the nearest round number to
    the clustered swing levels, deduping any that collapse onto an existing
    pivot level (within `_CLUSTER_TOL`) by boosting its strength instead.

    Parameters
    ----------
    df : pandas.DataFrame | None
        OHLCV frame. None/short history yields an empty list.
    price : float | None, optional
        Reference price for the round-number anchor; defaults to last close.

    Returns
    -------
    list[dict]
        Levels ``{price, kind, strength, source}`` sorted by price ascending.
    """
    n = 0 if df is None else len(df)
    if n < _MIN_BARS:
        return []

    close = df["Close"].astype(float)
    last_close = _safe_float(close.iloc[-1])
    if price is None:
        price = last_close
    if price is None:
        return []

    levels = [dict(lv, source="pivot") for lv in cluster_levels(find_pivots(df))]

    anchors: list[dict] = []
    if n >= 50:
        anchors.append({"price": _safe_float(close.tail(50).mean()), "strength": 2.0, "source": "SMA50"})
    if n >= 200:
        anchors.append({"price": _safe_float(close.tail(200).mean()), "strength": 2.5, "source": "SMA200"})
    hi = _safe_float(df["High"].astype(float).max())
    lo = _safe_float(df["Low"].astype(float).min())
    if hi is not None:
        anchors.append({"price": round(hi, 2), "strength": 3.0, "source": "52w high"})
    if lo is not None:
        anchors.append({"price": round(lo, 2), "strength": 3.0, "source": "52w low"})
    anchors.append({"price": round(_round_number(price), 2), "strength": 1.0, "source": "round"})

    for a in anchors:
        if a["price"] is None or a["price"] <= 0:
            continue
        merged = False
        for lv in levels:
            if abs(a["price"] - lv["price"]) / lv["price"] <= _CLUSTER_TOL:
                lv["strength"] = round(lv["strength"] + a["strength"], 2)
                lv["source"] = f"{lv.get('source', 'pivot')}+{a['source']}"
                merged = True
                break
        if not merged:
            levels.append({"price": a["price"], "touches": 1,
                           "strength": a["strength"], "source": a["source"]})

    for lv in levels:
        lv.setdefault("kind", None)
    return sorted(levels, key=lambda x: x["price"])


def nearest_levels(levels: list[dict], price: float) -> dict:
    """Find the closest support (below) and resistance (above) to a price.

    Parameters
    ----------
    levels : list[dict]
        Output of `build_levels`.
    price : float
        Current price.

    Returns
    -------
    dict
        ``{support, resistance, pct_to_support, pct_to_resistance}`` where
        support/resistance are the level dicts (or None if none on that side),
        and the pcts are positive fractional distances (None when absent).
    """
    below = [lv for lv in levels if lv["price"] < price]
    above = [lv for lv in levels if lv["price"] > price]
    support = max(below, key=lambda x: x["price"]) if below else None
    resistance = min(above, key=lambda x: x["price"]) if above else None
    return {
        "support": support,
        "resistance": resistance,
        "pct_to_support": (price - support["price"]) / price if support else None,
        "pct_to_resistance": (resistance["price"] - price) / price if resistance else None,
    }


# --- scoring (pure) --------------------------------------------------------

def _tier(score: float) -> str:
    """Map a 0-10 levels score to a tier label.

    Parameters
    ----------
    score : float
        Levels quality score.

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


def score_levels(df, price: float | None = None) -> dict:
    """Score where price sits in its support/resistance structure, 0-10.

    The read is from a *long* entry's perspective: good is just off strong
    support with clear headroom to resistance (favourable reward-to-risk); bad
    is pinned right under strong resistance with little cushion below. The core
    driver is reward-to-risk = (room up to resistance) / (drop to support).

    Pure — takes the DataFrame so it's unit-testable without yfinance.

    Parameters
    ----------
    df : pandas.DataFrame | None
        OHLCV frame from `fetch_ohlcv`.
    price : float | None, optional
        Price to locate in the structure; defaults to last close.

    Returns
    -------
    dict
        ``{score, tier, support, resistance, pct_to_support, pct_to_resistance,
        reward_to_risk, levels, flags}``. Thin/empty history yields a neutral
        5.0 with a "no clear levels" flag.
    """
    if price is None and df is not None and len(df):
        price = _safe_float(df["Close"].astype(float).iloc[-1])

    levels = build_levels(df, price)
    flags: list[str] = []

    if not levels or price is None:
        return {
            "score": 5.0, "tier": _tier(5.0),
            "support": None, "resistance": None,
            "pct_to_support": None, "pct_to_resistance": None,
            "reward_to_risk": None, "levels": [], "flags": ["no clear levels"],
        }

    near = nearest_levels(levels, price)
    support, resistance = near["support"], near["resistance"]
    pts, ptr = near["pct_to_support"], near["pct_to_resistance"]

    score = 5.0
    reward_to_risk = None
    if pts and ptr and pts > 0:
        reward_to_risk = ptr / pts
        # rr 1.0 is neutral; scale +/- around it, capped so one leg can't swing
        # the whole score. rr 3+ ~ +2.5, rr 0.33 ~ -2.5.
        score += max(-2.5, min(2.5, (reward_to_risk - 1.0) * 1.25))

    strong = _STRONG_TOUCHES
    if resistance and ptr is not None and ptr <= _NEAR_LEVEL and resistance["strength"] >= strong:
        score -= 1.5
        flags.append("pinned under resistance")
    if support and pts is not None and pts <= _NEAR_LEVEL and support["strength"] >= strong:
        score += 1.0
        flags.append("holding at support")
    if resistance is None:
        flags.append("no overhead resistance")   # blue-sky breakout, no cap above
        score += 0.5
    if support is None:
        flags.append("no support below")          # nothing to catch a fall
        score -= 1.0

    score = round(max(0.0, min(10.0, score)), 2)
    return {
        "score": score,
        "tier": _tier(score),
        "support": support,
        "resistance": resistance,
        "pct_to_support": pts,
        "pct_to_resistance": ptr,
        "reward_to_risk": round(reward_to_risk, 2) if reward_to_risk is not None else None,
        "levels": levels,
        "flags": flags,
    }


def score_ticker(ticker: str) -> dict:
    """Fetch OHLCV and score the ticker's level structure in one call.

    The card path's convenience wrapper. Shares `wyckoff.fetch_ohlcv` so the 1y
    daily history is the same download shape both layers use.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    dict
        Output of `score_levels`.
    """
    df = fetch_ohlcv(ticker)
    return score_levels(df)


# --- formatting ------------------------------------------------------------

def _fmt_pct(v: float | None) -> str:
    """Format a fraction as a percentage, or "n/a" when None.

    Parameters
    ----------
    v : float | None
        Value as a fraction (0.05 = 5%).

    Returns
    -------
    str
        Percentage string or "n/a".
    """
    return f"{v:.1%}" if v is not None else "n/a"


def _fmt_level(lv: dict | None) -> str:
    """Render a level dict as "$price (source, strength)".

    Parameters
    ----------
    lv : dict | None
        A level from `build_levels`, or None.

    Returns
    -------
    str
        Human-readable one-liner, or "none" when absent.
    """
    if not lv:
        return "none"
    return f"${lv['price']:.2f} ({lv.get('source', 'pivot')}, str {lv['strength']})"


def format_levels(scored: dict) -> str:
    """Render a scored levels dict as a markdown block.

    Feeds both the verdict prompt and the card dialog.

    Parameters
    ----------
    scored : dict
        Output of `score_levels`.

    Returns
    -------
    str
        Multi-line markdown with the score, nearest support/resistance,
        reward-to-risk, and any flags.
    """
    rr = scored.get("reward_to_risk")
    lines = [
        f"**Price levels: {scored['score']}/10 ({scored['tier']})**",
        "",
        f"- Resistance: {_fmt_level(scored.get('resistance'))} "
        f"({_fmt_pct(scored.get('pct_to_resistance'))} above)",
        f"- Support: {_fmt_level(scored.get('support'))} "
        f"({_fmt_pct(scored.get('pct_to_support'))} below)",
        f"- Reward-to-risk (room up / drop to support): {rr if rr is not None else 'n/a'}",
    ]
    if scored.get("flags"):
        lines.append("")
        lines.append("Flags: " + ", ".join(scored["flags"]))
    return "\n".join(lines)


def chip_label(scored: dict) -> str:
    """One-line card chip: "S $X / R $Y".

    Parameters
    ----------
    scored : dict
        Output of `score_levels`.

    Returns
    -------
    str
        Compact support/resistance summary for the card badge.
    """
    s = scored.get("support")
    r = scored.get("resistance")
    s_txt = f"${s['price']:.0f}" if s else "—"
    r_txt = f"${r['price']:.0f}" if r else "—"
    return f"S {s_txt} / R {r_txt}"
