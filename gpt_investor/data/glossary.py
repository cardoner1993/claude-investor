"""Single source of truth for the app's vocabulary.

The UI throws a lot of jargon at the user — Wyckoff phases, fundamental tiers,
regime labels, Brier/log-loss, Kelly, short interest. This module defines every
term once, plain-language, with an authoritative link. Deterministic, no LLM.

Two surfaces consume this: the glossary dialog (grouped list) and the chip
tooltips on the cards/dialog — both read `GLOSSARY` so a definition is written
in exactly one place.

`GLOSSARY[term]` -> `{"definition": str, "url": str}`.
`GROUPS` -> ordered `[(group_title, [term, ...]), ...]` for the dialog layout.
`define(term)` -> the one-line definition (for tooltips), or "" if unknown.
"""

from __future__ import annotations

_INVESTOPEDIA = "https://www.investopedia.com"

GLOSSARY: dict[str, dict[str, str]] = {
    # --- Fundamental tiers ------------------------------------------------
    "Strong": {"definition": "Top fundamental tier — composite score ≥ 8/10.",
               "url": f"{_INVESTOPEDIA}/terms/f/fundamentalanalysis.asp"},
    "Solid": {"definition": "Second fundamental tier — composite score ≥ 6/10.",
              "url": f"{_INVESTOPEDIA}/terms/f/fundamentalanalysis.asp"},
    "Average": {"definition": "Middle fundamental tier — composite score ≥ 4/10.",
                "url": f"{_INVESTOPEDIA}/terms/f/fundamentalanalysis.asp"},
    "Weak": {"definition": "Low fundamental tier — composite score ≥ 2/10.",
             "url": f"{_INVESTOPEDIA}/terms/f/fundamentalanalysis.asp"},
    "Avoid": {"definition": "Bottom fundamental tier — composite score < 2/10.",
              "url": f"{_INVESTOPEDIA}/terms/f/fundamentalanalysis.asp"},
    # --- Fundamental dimensions ------------------------------------------
    "Valuation": {"definition": "How cheap/expensive vs earnings, book, and cash flow (P/E, P/B, EV/EBITDA).",
                  "url": f"{_INVESTOPEDIA}/terms/v/valuation.asp"},
    "Growth": {"definition": "Rate of revenue and earnings expansion.",
               "url": f"{_INVESTOPEDIA}/terms/g/growthrates.asp"},
    "Profitability": {"definition": "Return on equity and operating margin — how efficiently profit is made.",
                      "url": f"{_INVESTOPEDIA}/terms/p/profitabilityratios.asp"},
    "Cash": {"definition": "Free-cash-flow margin — cash generated per dollar of revenue.",
             "url": f"{_INVESTOPEDIA}/terms/f/freecashflow.asp"},
    "Balance": {"definition": "Balance-sheet strength, driven by debt-to-equity leverage.",
                "url": f"{_INVESTOPEDIA}/terms/d/debtequityratio.asp"},
    # --- Wyckoff phases ---------------------------------------------------
    "Accumulation": {"definition": "Basing after a decline — selling abates, smart money builds positions.",
                     "url": f"{_INVESTOPEDIA}/terms/w/wyckoff-method.asp"},
    "Markup": {"definition": "Confirmed uptrend — buyers in control, price rising on demand.",
               "url": f"{_INVESTOPEDIA}/terms/w/wyckoff-method.asp"},
    "Distribution": {"definition": "Elevated but rolling over — supply hits the bid, topping.",
                     "url": f"{_INVESTOPEDIA}/terms/w/wyckoff-method.asp"},
    "Markdown": {"definition": "Confirmed downtrend — falling, no support yet (avoid).",
                 "url": f"{_INVESTOPEDIA}/terms/w/wyckoff-method.asp"},
    "Neutral": {"definition": "No clear price structure, or history too thin to judge.",
                "url": f"{_INVESTOPEDIA}/terms/w/wyckoff-method.asp"},
    # --- Support / resistance --------------------------------------------
    "Support": {"definition": "Price level below where buying tends to halt a fall.",
                "url": f"{_INVESTOPEDIA}/trading/support-and-resistance-basics/"},
    "Resistance": {"definition": "Price level above where selling tends to cap a rise.",
                   "url": f"{_INVESTOPEDIA}/trading/support-and-resistance-basics/"},
    "Reward-to-risk": {"definition": "Room up to resistance divided by drop to support — higher favours a long.",
                       "url": f"{_INVESTOPEDIA}/terms/r/riskrewardratio.asp"},
    # --- Regime labels ----------------------------------------------------
    "risk-on-bull": {"definition": "Market regime: low fear, healthy credit — risk appetite favourable.",
                     "url": f"{_INVESTOPEDIA}/terms/r/risk-on-risk-off.asp"},
    "panic-opportunity": {"definition": "Regime: fear spiking — historically a contrarian buying window.",
                          "url": f"{_INVESTOPEDIA}/terms/r/risk-on-risk-off.asp"},
    "recession-warning": {"definition": "Regime: inverted curve + widening credit — recession risk elevated.",
                          "url": f"{_INVESTOPEDIA}/terms/y/yieldcurve.asp"},
    "late-cycle-caution": {"definition": "Regime: extended bull, thinning breadth — reduce risk.",
                           "url": f"{_INVESTOPEDIA}/terms/m/marketcycle.asp"},
    "mixed": {"definition": "Regime: signals conflict — no clear risk-on/off read.",
              "url": f"{_INVESTOPEDIA}/terms/r/risk-on-risk-off.asp"},
    "VIX": {"definition": "CBOE volatility index — the market's expected 30-day swing ('fear gauge').",
            "url": f"{_INVESTOPEDIA}/terms/v/vix.asp"},
    "Yield curve": {"definition": "10y minus 3m Treasury yield; inverted (negative) warns of recession.",
                    "url": f"{_INVESTOPEDIA}/terms/y/yieldcurve.asp"},
    "HY credit": {"definition": "High-yield (junk) bond spread/HYG direction — a risk-appetite proxy.",
                  "url": f"{_INVESTOPEDIA}/terms/h/high_yield_bond.asp"},
    "DXY": {"definition": "US dollar index; a strong dollar tightens global financial conditions.",
            "url": f"{_INVESTOPEDIA}/terms/u/usdx.asp"},
    # --- Sentiment --------------------------------------------------------
    "VADER": {"definition": "Rule-based sentiment scorer applied to each news headline.",
              "url": "https://github.com/cjhutto/vaderSentiment"},
    "Sentiment score": {"definition": "News sentiment on −1 (bearish) to +1 (bullish), blending VADER and the LLM.",
                        "url": f"{_INVESTOPEDIA}/terms/m/marketsentiment.asp"},
    "Confidence": {"definition": "How much VADER and the LLM agree — low/med/high.",
                   "url": f"{_INVESTOPEDIA}/terms/m/marketsentiment.asp"},
    # --- Higher-signal ----------------------------------------------------
    "Short interest": {"definition": "Shares sold short as a % of float — crowded bets against the stock.",
                       "url": f"{_INVESTOPEDIA}/terms/s/shortinterest.asp"},
    "Insider net flow": {"definition": "Net buying/selling by company officers and directors.",
                         "url": f"{_INVESTOPEDIA}/terms/i/insidertrading.asp"},
    "CAGR": {"definition": "Compound annual growth rate — smoothed multi-year revenue/FCF growth.",
             "url": f"{_INVESTOPEDIA}/terms/c/cagr.asp"},
    "Operating margin": {"definition": "Operating profit as a % of revenue; its slope shows the trend.",
                         "url": f"{_INVESTOPEDIA}/terms/o/operatingmargin.asp"},
    "Peer-median P/E": {"definition": "Median P/E of industry peers — a valuation yardstick.",
                        "url": f"{_INVESTOPEDIA}/terms/p/price-earningsratio.asp"},
    # --- Probabilistic ----------------------------------------------------
    "prob up/flat/down": {"definition": "Model's ~90-day odds of >+2% / ±2% / <−2%; they sum to ~1.",
                          "url": f"{_INVESTOPEDIA}/terms/p/probabilitydistribution.asp"},
    "Brier score": {"definition": "Accuracy of probabilistic calls — lower is better (0 = perfect).",
                    "url": "https://en.wikipedia.org/wiki/Brier_score"},
    "Log-loss": {"definition": "Penalises confident wrong probabilities harshly — lower is better.",
                 "url": "https://en.wikipedia.org/wiki/Cross-entropy"},
    "Fractional Kelly": {"definition": "Bet-sizing fraction of the Kelly-optimal stake to curb volatility.",
                         "url": f"{_INVESTOPEDIA}/articles/trading/04/091504.asp"},
    "Payoff": {"definition": "Reward-to-risk of the trade — expected gain vs expected loss.",
               "url": f"{_INVESTOPEDIA}/terms/r/riskrewardratio.asp"},
    # --- Audit ------------------------------------------------------------
    "Audit": {"definition": "Advisory second opinion — agree / caution / disagree vs similar past outcomes.",
              "url": f"{_INVESTOPEDIA}/terms/b/backtesting.asp"},
    # --- Macro ------------------------------------------------------------
    "Fed": {"definition": "US Federal Reserve policy rate — sets the price of dollar liquidity.",
            "url": "https://www.federalreserve.gov/monetarypolicy.htm"},
    "ECB": {"definition": "European Central Bank policy rate — euro-area monetary stance.",
            "url": "https://www.ecb.europa.eu/mopo/html/index.en.html"},
    "PBOC": {"definition": "People's Bank of China — sets reserve requirements and Chinese liquidity.",
             "url": "http://www.pbc.gov.cn/en/3688229/index.html"},
}

GROUPS: list[tuple[str, list[str]]] = [
    ("Fundamental tiers", ["Strong", "Solid", "Average", "Weak", "Avoid"]),
    ("Fundamental dimensions", ["Valuation", "Growth", "Profitability", "Cash", "Balance"]),
    ("Wyckoff phases", ["Accumulation", "Markup", "Distribution", "Markdown", "Neutral"]),
    ("Price levels", ["Support", "Resistance", "Reward-to-risk"]),
    ("Market regime", ["risk-on-bull", "panic-opportunity", "recession-warning",
                       "late-cycle-caution", "mixed", "VIX", "Yield curve", "HY credit", "DXY"]),
    ("Sentiment", ["VADER", "Sentiment score", "Confidence"]),
    ("Higher-signal", ["Short interest", "Insider net flow", "CAGR",
                       "Operating margin", "Peer-median P/E"]),
    ("Probabilistic", ["prob up/flat/down", "Brier score", "Log-loss",
                       "Fractional Kelly", "Payoff"]),
    ("Audit", ["Audit"]),
    ("Macro", ["Fed", "ECB", "PBOC"]),
]


def define(term: str) -> str:
    """Return the one-line definition for a term, or "" if unknown.

    Parameters
    ----------
    term : str
        Glossary key.

    Returns
    -------
    str
        Plain-language definition, or empty string when the term isn't defined.
    """
    entry = GLOSSARY.get(term)
    return entry["definition"] if entry else ""
