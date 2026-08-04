"""Probabilistic-verdict math (P5).

Instead of a bare "Buy", the model gives odds on what the stock does over the
next quarter — `prob_up / prob_flat / prob_down`. This module turns those odds
into two things:

    1. A **position size** — roughly how much of your money to put in, from the
       Kelly formula (how much to bet given your edge and the reward-to-risk).
    2. **Report-card metrics** (Brier score, log loss) that later grade how good
       those odds actually were, once the real price move is known.

Everything here is pure arithmetic — no yfinance, no LLM. The sizing feeds the
verdict text; the metrics feed `scripts/calibration.py`.
"""

from __future__ import annotations

import math

_CLASSES = ("up", "flat", "down")
_KELLY_FRACTION = 0.5     # half-Kelly — full Kelly is too aggressive in practice
_MAX_SIZE = 0.10          # never suggest more than 10% of book in one name
_EPS = 1e-9


def normalize_probs(up, flat, down) -> dict:
    """Tidy up three raw probabilities so they add up to exactly 1 (100%).

    The model's up/flat/down often don't add to exactly 1 (e.g. 0.55/0.30/0.20
    = 1.05). This clamps any negatives to 0 and rescales the rest to sum to 1,
    so downstream math is well-defined. If all three are 0 (nothing to go on),
    falls back to an even 1/3 each.

    Parameters
    ----------
    up : float
        Raw weight the model gave to the price rising.
    flat : float
        Raw weight for the price staying roughly flat.
    down : float
        Raw weight for the price falling.

    Returns
    -------
    dict
        ``{"up", "flat", "down"}`` of non-negative floats that sum to 1.
    """
    vals = [max(0.0, float(x or 0.0)) for x in (up, flat, down)]
    total = sum(vals)
    if total <= 0:
        return {"up": 1 / 3, "flat": 1 / 3, "down": 1 / 3}
    return dict(zip(_CLASSES, (v / total for v in vals)))


def brier_score(probs: dict, realized: str) -> float | None:
    """Grade a set of probabilities against what actually happened. Lower = better.

    Measures how far the forecast was from reality: squared distance between the
    predicted probabilities and the outcome (the class that happened counts as
    1, the others 0). 0 = perfect (said 100% for what occurred); ~2 = worst
    (said 100% for the wrong thing). Averaged over many verdicts, it tells you
    whether the model's stated odds are trustworthy.

    Parameters
    ----------
    probs : dict
        Predicted distribution with ``up``/``flat``/``down`` weights.
    realized : str
        What actually happened — one of ``up``/``flat``/``down``.

    Returns
    -------
    float | None
        Brier score in [0, 2] (lower better), or None if `realized` isn't a
        valid class.
    """
    if realized not in _CLASSES:
        return None
    p = normalize_probs(probs.get("up"), probs.get("flat"), probs.get("down"))
    return round(sum((p[c] - (1.0 if c == realized else 0.0)) ** 2 for c in _CLASSES), 4)


def log_loss(probs: dict, realized: str) -> float | None:
    """Grade probabilities, punishing confident-but-wrong much harder. Lower = better.

    Looks only at the probability the model gave to the outcome that actually
    happened, and penalises being confidently wrong far more than Brier does —
    saying "5% up" and then it goes up is a big penalty. 0 = said 100% for what
    occurred; grows without bound as that probability approaches 0.

    Parameters
    ----------
    probs : dict
        Predicted distribution with ``up``/``flat``/``down`` weights.
    realized : str
        What actually happened — one of ``up``/``flat``/``down``.

    Returns
    -------
    float | None
        Log loss (>= 0, lower better), or None if `realized` isn't a valid
        class. The realised probability is floored at a tiny value so a 0%
        forecast doesn't blow up to infinity.
    """
    if realized not in _CLASSES:
        return None
    p = normalize_probs(probs.get("up"), probs.get("flat"), probs.get("down"))
    return round(-math.log(max(p[realized], _EPS)), 4)


def realized_class(ret: float | None, flat_band: float = 0.02) -> str | None:
    """Label what actually happened from the forward return: up / flat / down.

    Anything within ±`flat_band` of zero counts as "flat" (didn't really move);
    above is "up", below is "down". This is how a raw return becomes the outcome
    label that `brier_score` / `log_loss` grade against.

    Parameters
    ----------
    ret : float | None
        Forward return as a fraction (0.03 = +3%).
    flat_band : float, optional
        Half-width of the "didn't move" zone around 0; default 0.02 (±2%).

    Returns
    -------
    str | None
        ``up``/``flat``/``down``, or None if `ret` is None.
    """
    if ret is None:
        return None
    if ret > flat_band:
        return "up"
    if ret < -flat_band:
        return "down"
    return "flat"


def kelly_fraction(prob_up: float, prob_down: float, payoff: float, fraction: float = _KELLY_FRACTION) -> float:
    """How much of your money to put in, from the Kelly betting formula.

    Kelly answers "given my odds of winning and the reward-to-risk, what bet
    size grows my money fastest?" — full Kelly is `(p·b − q) / b` where `p` is
    win probability, `q` loss probability, `b` the payoff. Full Kelly is very
    swingy, so we take **half** of it (`fraction`) and never suggest more than
    `_MAX_SIZE` (10%) in one name. No edge (the sum goes negative) → 0, i.e.
    don't put money in.

    Example: 60% up, 20% down, payoff 2.0 → full Kelly 0.5 → half-Kelly 0.25 →
    capped to 0.10.

    Parameters
    ----------
    prob_up : float
        Probability the bet wins (price rises).
    prob_down : float
        Probability the bet loses (price falls).
    payoff : float
        Reward-to-risk odds — gain $`payoff` for every $1 risked (see
        `payoff_from_target`).
    fraction : float, optional
        Fraction of full Kelly to actually use; default 0.5 (half-Kelly).

    Returns
    -------
    float
        Suggested fraction of the portfolio, in [0, 0.10].
    """
    if payoff <= 0:
        return 0.0
    f = (prob_up * payoff - prob_down) / payoff
    return round(max(0.0, min(_MAX_SIZE, f * fraction)), 4)


def payoff_from_target(current_price, price_target, downside: float = 0.15) -> float:
    """Reward-to-risk odds implied by the price target.

    Compares the upside to the target against an assumed downside if the trade
    goes wrong: `upside% / downside%`. Example: price 100, target 130 → +30%
    upside; assuming a 15% downside → payoff 2.0 (you'd make twice what you'd
    lose). Falls back to even odds (1.0) when there's no target or no upside.

    Parameters
    ----------
    current_price : float
        Price now.
    price_target : float
        The verdict's price target.
    downside : float, optional
        Assumed loss fraction if the trade goes against you; default 0.15 (15%).

    Returns
    -------
    float
        Reward-to-risk odds (2.0 = upside twice the downside), or 1.0 when
        inputs are missing or there's no upside.
    """
    if not current_price or current_price <= 0 or not price_target or downside <= 0:
        return 1.0
    upside = (price_target - current_price) / current_price
    if upside <= 0:
        return 1.0
    return round(upside / downside, 3)


def position_advice(kelly: float, held: bool) -> str:
    """Turn a Kelly stake into plain advice, aware of whether you already own it.

    Buckets the size into a human action. If you don't hold it: no edge → skip,
    small stake (<3%) → starter position, bigger → full position. If you already
    hold it: no edge → trim/exit, small → hold, bigger → add.

    Parameters
    ----------
    kelly : float
        Suggested position fraction from `kelly_fraction`.
    held : bool
        Whether you already own the position.

    Returns
    -------
    str
        One of: ``skip``, ``starter position``, ``full position`` (not held);
        ``trim / exit``, ``hold``, ``add`` (held).
    """
    if kelly <= 0:
        return "trim / exit" if held else "skip"
    if held:
        return "hold" if kelly < 0.03 else "add"
    return "starter position" if kelly < 0.03 else "full position"


def render_probability_block(prob_up, prob_flat, prob_down, premortem, current_price, price_target, held: bool = False) -> str:
    """Build the probability / sizing / pre-mortem lines shown under the verdict.

    Produces up to three markdown lines, e.g.::

        **Probabilities**: up 55% / flat 30% / down 15%
        **Suggested size**: 6.2% (½-Kelly, payoff 2.00) — full position
        **Pre-mortem**: AI-capex cycle rolls over and the multiple compresses.

    "Suggested size" reads as: put ~6.2% of the portfolio in, from half-Kelly at
    2.00 reward-to-risk odds; "full position" is the hold-aware read of that
    size.

    Parameters
    ----------
    prob_up, prob_flat, prob_down : float
        The model's raw up/flat/down probabilities (normalised here).
    premortem : str | None
        The "most likely reason this is wrong" note; skipped from output when
        empty.
    current_price : float
        Price now (for the payoff calculation).
    price_target : float
        The verdict's price target (for the payoff calculation).
    held : bool, optional
        Whether you already own the position; default False.

    Returns
    -------
    str
        The markdown block, sections separated by blank lines.
    """
    p = normalize_probs(prob_up, prob_flat, prob_down)
    payoff = payoff_from_target(current_price, price_target)
    kelly = kelly_fraction(p["up"], p["down"], payoff)
    advice = position_advice(kelly, held)
    lines = [
        f"**Probabilities**: up {p['up']:.0%} / flat {p['flat']:.0%} / down {p['down']:.0%}",
        f"**Suggested size**: {kelly:.1%} (½-Kelly, payoff {payoff:.2f}) — {advice}",
    ]
    if premortem:
        lines.append(f"**Pre-mortem**: {premortem}")
    return "\n\n".join(lines)
