"""Deterministic fundamental scoring.

Pulls raw metrics from yfinance, scores 5 dimensions on a 0-10 scale, and
returns both a structured dict (for UI/comparison) and a markdown block
(for the final-analysis LLM prompt).

The scoring is pure — `score_fundamentals(metrics)` takes a plain dict so
it can be unit-tested without hitting yfinance. The data fetch is split
into `fetch_fundamentals(ticker)`.

Score interpretation:
    >= 8.0  Strong
    >= 6.0  Solid
    >= 4.0  Average
    >= 2.0  Weak
    <  2.0  Avoid

yfinance quirk: `debtToEquity` is reported as a percentage (100 = 1.0x).
We normalise to a ratio inside _score_debt_equity.
"""

import yfinance as yf


# --- per-dimension scoring helpers -----------------------------------------

def _score_pe(pe: float | None) -> float:
    if pe is None or pe <= 0:
        return 0.0
    if pe < 12: return 10.0
    if pe < 18: return 8.0
    if pe < 25: return 6.0
    if pe < 35: return 4.0
    if pe < 50: return 2.0
    return 0.0


def _score_pb(pb: float | None) -> float:
    if pb is None or pb <= 0:
        return 5.0  # neutral when missing
    if pb < 1.5: return 10.0
    if pb < 3.0: return 8.0
    if pb < 5.0: return 6.0
    if pb < 8.0: return 4.0
    return 2.0


def _score_ev_ebitda(ev: float | None) -> float:
    if ev is None or ev <= 0:
        return 5.0
    if ev < 8:  return 10.0
    if ev < 12: return 8.0
    if ev < 18: return 6.0
    if ev < 25: return 4.0
    return 2.0


def _score_growth(g: float | None) -> float:
    if g is None:
        return 5.0
    if g >= 0.25:  return 10.0
    if g >= 0.15:  return 8.0
    if g >= 0.08:  return 6.0
    if g >= 0.0:   return 4.0
    if g >= -0.05: return 2.0
    return 0.0


def _score_roe(roe: float | None) -> float:
    if roe is None:
        return 5.0
    if roe >= 0.25: return 10.0
    if roe >= 0.18: return 8.0
    if roe >= 0.12: return 6.0
    if roe >= 0.05: return 4.0
    if roe >= 0.0:  return 2.0
    return 0.0


def _score_op_margin(m: float | None) -> float:
    if m is None:
        return 5.0
    if m >= 0.25: return 10.0
    if m >= 0.18: return 8.0
    if m >= 0.10: return 6.0
    if m >= 0.05: return 4.0
    if m >= 0.0:  return 2.0
    return 0.0


def _score_fcf_margin(fcf: float | None, revenue: float | None) -> tuple[float, float | None]:
    if not fcf or not revenue or revenue <= 0:
        return 0.0, None
    m = fcf / revenue
    if m >= 0.25: return 10.0, m
    if m >= 0.15: return 8.0, m
    if m >= 0.08: return 6.0, m
    if m >= 0.03: return 4.0, m
    if m >= 0.0:  return 2.0, m
    return 0.0, m


def _score_debt_equity(de_pct: float | None) -> tuple[float, float | None]:
    """yfinance reports D/E as a percentage (100 = 1.0x). Normalise."""
    if de_pct is None:
        return 5.0, None
    de = de_pct / 100.0
    if de < 0.3: return 10.0, de
    if de < 0.6: return 8.0, de
    if de < 1.0: return 6.0, de
    if de < 2.0: return 4.0, de
    return 2.0, de


# --- main scoring entry point ----------------------------------------------

WEIGHTS = {
    "valuation":     0.25,
    "growth":        0.20,
    "profitability": 0.20,
    "cash":          0.20,
    "balance":       0.15,
}


def _tier(score: float) -> str:
    if score >= 8.0: return "Strong"
    if score >= 6.0: return "Solid"
    if score >= 4.0: return "Average"
    if score >= 2.0: return "Weak"
    return "Avoid"


def score_fundamentals(m: dict) -> dict:
    """Pure scoring. `m` is a metrics dict from fetch_fundamentals (or test fixture)."""
    pe_score = _score_pe(m.get("forward_pe") or m.get("trailing_pe"))
    pb_score = _score_pb(m.get("price_to_book"))
    ev_score = _score_ev_ebitda(m.get("ev_ebitda"))
    valuation = round((pe_score * 0.5) + (pb_score * 0.25) + (ev_score * 0.25), 2)

    rev_g_score = _score_growth(m.get("revenue_growth"))
    earn_g_score = _score_growth(m.get("earnings_growth"))
    growth = round((rev_g_score * 0.5) + (earn_g_score * 0.5), 2)

    roe_score = _score_roe(m.get("roe"))
    op_score = _score_op_margin(m.get("operating_margin"))
    profitability = round((roe_score * 0.5) + (op_score * 0.5), 2)

    fcf_score, fcf_margin = _score_fcf_margin(m.get("free_cashflow"), m.get("revenue"))
    cash = fcf_score

    bs_score, de_ratio = _score_debt_equity(m.get("debt_to_equity"))
    balance = bs_score

    composite = round(
        valuation * WEIGHTS["valuation"]
        + growth * WEIGHTS["growth"]
        + profitability * WEIGHTS["profitability"]
        + cash * WEIGHTS["cash"]
        + balance * WEIGHTS["balance"],
        2,
    )

    flags: list[str] = []
    if (m.get("free_cashflow") or 0) < 0:
        flags.append("negative FCF")
    if (m.get("revenue_growth") or 0) < 0:
        flags.append("shrinking revenue")
    if (m.get("earnings_growth") or 0) < 0:
        flags.append("declining earnings")
    if de_ratio is not None and de_ratio > 2.0:
        flags.append("high leverage")
    if (m.get("trailing_eps") or 0) < 0:
        flags.append("earnings loss")
    pe_for_flag = m.get("forward_pe") or m.get("trailing_pe")
    if pe_for_flag is not None and pe_for_flag > 35:
        flags.append("premium valuation")
    if m.get("free_cashflow") is None:
        flags.append("no FCF data")

    return {
        "score": composite,
        "tier": _tier(composite),
        "dimensions": {
            "valuation":     {"score": valuation,     "metrics": {"forward_pe": m.get("forward_pe"), "trailing_pe": m.get("trailing_pe"), "price_to_book": m.get("price_to_book"), "ev_ebitda": m.get("ev_ebitda")}},
            "growth":        {"score": growth,        "metrics": {"revenue_growth": m.get("revenue_growth"), "earnings_growth": m.get("earnings_growth")}},
            "profitability": {"score": profitability, "metrics": {"roe": m.get("roe"), "operating_margin": m.get("operating_margin"), "profit_margin": m.get("profit_margin")}},
            "cash":          {"score": cash,          "metrics": {"free_cashflow": m.get("free_cashflow"), "fcf_margin": fcf_margin, "revenue": m.get("revenue")}},
            "balance":       {"score": balance,       "metrics": {"debt_to_equity_ratio": de_ratio}},
        },
        "flags": flags,
        "raw": m,
    }


# --- data fetch ------------------------------------------------------------

def fetch_fundamentals(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    return {
        "trailing_pe":      info.get("trailingPE"),
        "forward_pe":       info.get("forwardPE"),
        "price_to_book":    info.get("priceToBook"),
        "ev_ebitda":        info.get("enterpriseToEbitda"),
        "revenue_growth":   info.get("revenueGrowth"),
        "earnings_growth":  info.get("earningsGrowth"),
        "roe":              info.get("returnOnEquity"),
        "operating_margin": info.get("operatingMargins"),
        "profit_margin":    info.get("profitMargins"),
        "gross_margin":     info.get("grossMargins"),
        "free_cashflow":    info.get("freeCashflow"),
        "revenue":          info.get("totalRevenue"),
        "debt_to_equity":   info.get("debtToEquity"),  # % form: 100 = 1.0x
        "trailing_eps":     info.get("trailingEps"),
        "market_cap":       info.get("marketCap"),
    }


# --- formatting ------------------------------------------------------------

def _fmt_pct(v: float | None) -> str:
    return f"{v:.1%}" if v is not None else "n/a"


def _fmt_num(v: float | None, fmt: str = ".2f") -> str:
    return f"{v:{fmt}}" if v is not None else "n/a"


def _fmt_billions(v: float | None) -> str:
    return f"${v / 1e9:.1f}B" if v else "n/a"


def format_fundamentals(scored: dict) -> str:
    raw = scored["raw"]
    d = scored["dimensions"]
    lines = [
        f"**Fundamental score: {scored['score']}/10 ({scored['tier']})**",
        "",
        f"- Valuation {d['valuation']['score']}/10 — "
        f"fwd P/E {_fmt_num(raw.get('forward_pe'))}, "
        f"trailing P/E {_fmt_num(raw.get('trailing_pe'))}, "
        f"P/B {_fmt_num(raw.get('price_to_book'))}, "
        f"EV/EBITDA {_fmt_num(raw.get('ev_ebitda'))}",

        f"- Growth {d['growth']['score']}/10 — "
        f"revenue {_fmt_pct(raw.get('revenue_growth'))}, "
        f"earnings {_fmt_pct(raw.get('earnings_growth'))}",

        f"- Profitability {d['profitability']['score']}/10 — "
        f"ROE {_fmt_pct(raw.get('roe'))}, "
        f"op margin {_fmt_pct(raw.get('operating_margin'))}, "
        f"profit margin {_fmt_pct(raw.get('profit_margin'))}",

        f"- Cash {d['cash']['score']}/10 — "
        f"FCF {_fmt_billions(raw.get('free_cashflow'))}, "
        f"FCF margin {_fmt_pct(d['cash']['metrics']['fcf_margin'])}",

        f"- Balance {d['balance']['score']}/10 — "
        f"D/E ratio {_fmt_num(d['balance']['metrics']['debt_to_equity_ratio'])}",
    ]
    if scored["flags"]:
        lines.append("")
        lines.append("Flags: " + ", ".join(scored["flags"]))
    return "\n".join(lines)
