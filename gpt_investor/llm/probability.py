"""Probabilistic-verdict math (P5).

Turns the LLM's `prob_up / prob_flat / prob_down` distribution into:
    - calibration metrics (multiclass Brier score, log loss) so we can score
      the model's probabilities against realised outcomes
    - a fractional-Kelly position size + a hold-aware trim/add/skip suggestion

All pure — no yfinance, no LLM. Metrics feed `scripts/calibration.py`; the
sizing feeds the verdict render.
"""

from __future__ import annotations

import math

_CLASSES = ("up", "flat", "down")
_KELLY_FRACTION = 0.5     # half-Kelly — full Kelly is too aggressive in practice
_MAX_SIZE = 0.10          # never suggest more than 10% of book in one name
_EPS = 1e-9


def normalize_probs(up, flat, down) -> dict:
    """Clamp negatives and renormalise to sum 1.

    A degenerate all-zero input falls back to a uniform distribution.

    Parameters
    ----------
    up : float
        Raw probability weight for the up class.
    flat : float
        Raw probability weight for the flat class.
    down : float
        Raw probability weight for the down class.

    Returns
    -------
    dict
        Keys ``up``/``flat``/``down`` with non-negative floats summing to 1.
    """
    vals = [max(0.0, float(x or 0.0)) for x in (up, flat, down)]
    total = sum(vals)
    if total <= 0:
        return {"up": 1 / 3, "flat": 1 / 3, "down": 1 / 3}
    return dict(zip(_CLASSES, (v / total for v in vals)))


def brier_score(probs: dict, realized: str) -> float | None:
    """Multiclass Brier score of the probability vector vs the realised class.

    Squared error against the one-hot realised class; lower is better
    (0 = perfect, 2 = worst).

    Parameters
    ----------
    probs : dict
        Probability distribution with ``up``/``flat``/``down`` weights.
    realized : str
        The realised class; one of ``up``/``flat``/``down``.

    Returns
    -------
    float | None
        Rounded Brier score, or None if `realized` is not a valid class.
    """
    if realized not in _CLASSES:
        return None
    p = normalize_probs(probs.get("up"), probs.get("flat"), probs.get("down"))
    return round(sum((p[c] - (1.0 if c == realized else 0.0)) ** 2 for c in _CLASSES), 4)


def log_loss(probs: dict, realized: str) -> float | None:
    """Negative log-likelihood of the realised class.

    Lower is better. The realised probability is clamped to `_EPS` to avoid a
    log of zero blowing up.

    Parameters
    ----------
    probs : dict
        Probability distribution with ``up``/``flat``/``down`` weights.
    realized : str
        The realised class; one of ``up``/``flat``/``down``.

    Returns
    -------
    float | None
        Rounded log loss, or None if `realized` is not a valid class.
    """
    if realized not in _CLASSES:
        return None
    p = normalize_probs(probs.get("up"), probs.get("flat"), probs.get("down"))
    return round(-math.log(max(p[realized], _EPS)), 4)


def realized_class(ret: float | None, flat_band: float = 0.02) -> str | None:
    """Bucket a forward return into up/flat/down.

    Returns within ±`flat_band` count as flat.

    Parameters
    ----------
    ret : float | None
        Forward return as a fraction (e.g. 0.03 for +3%).
    flat_band : float, optional
        Symmetric dead-band around zero; default 0.02.

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
    """Fractional-Kelly stake for a win/loss bet.

    Full Kelly f* = (p·b − q)/b, scaled by `fraction` (½-Kelly by default) and
    clamped to [0, _MAX_SIZE]. Negative-edge bets return 0.

    Parameters
    ----------
    prob_up : float
        Win probability.
    prob_down : float
        Loss probability.
    payoff : float
        Net odds — win $payoff per $1 risked.
    fraction : float, optional
        Kelly scaling factor; default ½-Kelly (`_KELLY_FRACTION`).

    Returns
    -------
    float
        Rounded stake fraction in [0, _MAX_SIZE].
    """
    if payoff <= 0:
        return 0.0
    f = (prob_up * payoff - prob_down) / payoff
    return round(max(0.0, min(_MAX_SIZE, f * fraction)), 4)


def payoff_from_target(current_price, price_target, downside: float = 0.15) -> float:
    """Net odds implied by the price target: upside% / assumed downside%.

    Falls back to even odds (1.0) when no target or non-positive upside.

    Parameters
    ----------
    current_price : float
        Current price of the asset.
    price_target : float
        Analyst/model price target.
    downside : float, optional
        Assumed downside fraction if the bet loses; default 0.15.

    Returns
    -------
    float
        Rounded net odds, or 1.0 on missing inputs / non-positive upside.
    """
    if not current_price or current_price <= 0 or not price_target or downside <= 0:
        return 1.0
    upside = (price_target - current_price) / current_price
    if upside <= 0:
        return 1.0
    return round(upside / downside, 3)


def position_advice(kelly: float, held: bool) -> str:
    """Hold-aware trim/add/skip suggestion from the Kelly stake.

    Parameters
    ----------
    kelly : float
        Fractional-Kelly stake.
    held : bool
        Whether the position is already held.

    Returns
    -------
    str
        Action suggestion (e.g. ``skip``, ``add``, ``full position``).
    """
    if kelly <= 0:
        return "trim / exit" if held else "skip"
    if held:
        return "hold" if kelly < 0.03 else "add"
    return "starter position" if kelly < 0.03 else "full position"


def render_probability_block(prob_up, prob_flat, prob_down, premortem, current_price, price_target, held: bool = False) -> str:
    """Build the markdown addendum appended to the verdict.

    Renders the probability distribution, fractional-Kelly size + hold-aware
    suggestion, and an optional pre-mortem note.

    Parameters
    ----------
    prob_up : float
        Raw up-class probability weight.
    prob_flat : float
        Raw flat-class probability weight.
    prob_down : float
        Raw down-class probability weight.
    premortem : str | None
        Optional pre-mortem note; omitted from output when falsy.
    current_price : float
        Current price of the asset.
    price_target : float
        Analyst/model price target.
    held : bool, optional
        Whether the position is already held; default False.

    Returns
    -------
    str
        Markdown block, sections joined by blank lines.
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
