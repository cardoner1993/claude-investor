"""Market regime indicators — VIX, yield curve, DXY, HY credit, gold.

Deterministic, no LLM. One yfinance `download()` call pulls a 10-day window
across all tickers; 5-day deltas drive a rising/falling direction tag per
indicator. A code-side rule classifies the overall regime label.

Used by `get_final_analysis` (verdict input) and `liquidity_panel` (UI).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Literal

import yfinance as yf
from loguru import logger

# yfinance symbols. Order matters only for log/output stability.
_INDICATORS: dict[str, str] = {
    "vix":  "^VIX",          # CBOE volatility index
    "tnx":  "^TNX",          # 10-year Treasury yield (×10 raw)
    "irx":  "^IRX",          # 3-month Treasury yield (×10 raw — but we read level)
    "dxy":  "DX-Y.NYB",      # US dollar index
    "hyg":  "HYG",           # iShares high-yield corporate bond ETF (credit risk proxy)
    "gold": "GC=F",          # gold futures
}

RegimeLabel = Literal[
    "panic-opportunity",
    "recession-warning",
    "late-cycle-caution",
    "risk-on-bull",
    "mixed",
]


def _direction(delta: float, eps: float = 1e-6) -> Literal["rising", "falling", "flat"]:
    if delta > eps:
        return "rising"
    if delta < -eps:
        return "falling"
    return "flat"


def _safe_float(x) -> float | None:
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


def get_market_regime() -> dict:
    """Fetch indicators + classify regime. One yfinance download call.

    Returns a dict suitable for `format_regime()` and for inclusion in the
    final-analysis prompt. On any fetch failure, returns a partial dict with
    `label = "mixed"` and whatever indicators were obtainable.
    """
    end = datetime.utcnow().date() + timedelta(days=1)
    start = end - timedelta(days=14)  # extra buffer for weekends/holidays
    tickers = list(_INDICATORS.values())

    try:
        df = yf.download(
            tickers,
            start=start.isoformat(),
            end=end.isoformat(),
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:
        logger.warning("market_regime yfinance download failed: {}", e)
        df = None

    indicators: dict[str, dict] = {}
    for key, sym in _INDICATORS.items():
        value: float | None = None
        delta_5d: float | None = None
        if df is not None and not df.empty:
            try:
                series = df[sym]["Close"].dropna() if (sym, "Close") in df.columns else df["Close"].dropna()
            except Exception:
                series = None
            if series is not None and len(series) >= 1:
                value = _safe_float(series.iloc[-1])
                if len(series) >= 6:
                    delta_5d = _safe_float(series.iloc[-1] - series.iloc[-6])
        indicators[key] = {
            "value": value,
            "delta_5d": delta_5d,
            "direction": _direction(delta_5d) if delta_5d is not None else "unknown",
        }

    # ^TNX / ^IRX are quoted ×10 (raw 42.10 = 4.21%). Normalise both to
    # percentage points up-front so all downstream thresholds are in pp.
    tnx_pp = (indicators["tnx"]["value"] / 10) if indicators["tnx"]["value"] is not None else None
    irx_pp = (indicators["irx"]["value"] / 10) if indicators["irx"]["value"] is not None else None
    curve = (tnx_pp - irx_pp) if (tnx_pp is not None and irx_pp is not None) else None

    curve_prev = None
    if df is not None and not df.empty:
        try:
            tnx_series = df["^TNX"]["Close"].dropna() if ("^TNX", "Close") in df.columns else None
            irx_series = df["^IRX"]["Close"].dropna() if ("^IRX", "Close") in df.columns else None
            if tnx_series is not None and irx_series is not None and len(tnx_series) >= 6 and len(irx_series) >= 6:
                curve_prev = _safe_float((tnx_series.iloc[-6] - irx_series.iloc[-6]) / 10)
        except Exception:
            curve_prev = None
    curve_delta = (curve - curve_prev) if (curve is not None and curve_prev is not None) else None

    label = classify_regime(indicators, curve)
    summary = _summary(indicators, curve, label)

    regime = {
        "as_of": datetime.utcnow().date().isoformat(),
        "indicators": indicators,
        "curve": curve,
        "curve_direction": _direction(curve_delta) if curve_delta is not None else "unknown",
        "label": label,
        "summary": summary,
    }
    logger.info(
        "market_regime label={} curve={} vix={} hyg_dir={}",
        label,
        f"{curve:+.2f}" if curve is not None else "?",
        indicators["vix"]["value"],
        indicators["hyg"]["direction"],
    )
    return regime


def classify_regime(indicators: dict[str, dict], curve: float | None) -> RegimeLabel:
    """Deterministic regime label from indicator dict.

    Pure function — exposed for unit testing. Rules ordered from most-extreme
    to most-benign; first match wins. Falls back to "mixed" if any required
    input is missing.
    """
    vix = indicators.get("vix", {}).get("value")
    hyg_dir = indicators.get("hyg", {}).get("direction")

    if vix is None or curve is None:
        return "mixed"

    # Extreme fear during yield-curve inversion → washout / dip-buy setup
    if vix > 35 and curve < 0:
        return "panic-opportunity"

    # Inversion + elevated vol + credit weakening → classic recession signal
    if curve < 0 and vix > 20 and hyg_dir == "falling":
        return "recession-warning"

    # Flat-to-inverted curve + elevated vol → cycle late
    # Flat curve (<50bps) or inversion with elevated vol → cycle late
    if curve < 0.5 and vix > 20:  # curve in percentage points: 0.5pp = 50bps
        return "late-cycle-caution"

    # Calm vol + healthy positive curve (>50bps) + credit rising → benign risk-on
    if vix < 15 and curve > 0.5 and hyg_dir == "rising":
        return "risk-on-bull"

    return "mixed"


def _summary(indicators: dict, curve: float | None, label: RegimeLabel) -> str:
    """One-line human summary. `curve` is in percentage points."""
    vix = indicators.get("vix", {}).get("value")
    parts: list[str] = [f"Regime: {label}."]
    if vix is not None:
        parts.append(f"VIX {vix:.1f}.")
    if curve is not None:
        parts.append(f"10y−3m spread {curve:+.2f}pp.")
    hyg_dir = indicators.get("hyg", {}).get("direction")
    if hyg_dir in {"rising", "falling"}:
        parts.append(f"HY credit {hyg_dir}.")
    return " ".join(parts)


_DIR_ARROW = {"rising": "↑", "falling": "↓", "flat": "→", "unknown": "?"}


def _fmt_value(key: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    # TNX and IRX are quoted in tenths of a percent on yfinance (e.g. 42.1 = 4.21%)
    if key in {"tnx", "irx"}:
        return f"{value / 10:.2f}%"
    if key in {"vix", "hyg", "dxy"}:
        return f"{value:.2f}"
    if key == "gold":
        return f"{value:,.0f}"
    return f"{value:.2f}"


def format_regime(regime: dict) -> str:
    """Markdown block for the liquidity_panel UI and the final-analysis prompt."""
    label = regime.get("label", "unknown")
    indicators = regime.get("indicators", {})
    curve = regime.get("curve")
    curve_dir = regime.get("curve_direction", "unknown")

    lines = [
        f"**Market Regime**: {label}",
        "",
        f"- **VIX**: {_fmt_value('vix', indicators.get('vix', {}).get('value'))} "
        f"({_DIR_ARROW[indicators.get('vix', {}).get('direction', 'unknown')]})",
    ]
    if curve is not None:
        lines.append(f"- **10y−3m spread**: {curve:+.2f}pp ({_DIR_ARROW[curve_dir]})")
    else:
        lines.append("- **10y−3m spread**: n/a")

    for key, label_text in [
        ("hyg", "HY credit (HYG)"),
        ("dxy", "USD index (DXY)"),
        ("gold", "Gold"),
    ]:
        ind = indicators.get(key, {})
        lines.append(
            f"- **{label_text}**: {_fmt_value(key, ind.get('value'))} "
            f"({_DIR_ARROW[ind.get('direction', 'unknown')]})"
        )
    return "\n".join(lines)
