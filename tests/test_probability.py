import math

import pytest

from gpt_investor.llm import probability as p


def test_normalize_probs():
    out = p.normalize_probs(0.55, 0.30, 0.20)  # sums to 1.05
    assert math.isclose(sum(out.values()), 1.0)
    assert out["up"] > out["down"]
    # degenerate all-zero → uniform
    assert p.normalize_probs(0, 0, 0) == {"up": 1 / 3, "flat": 1 / 3, "down": 1 / 3}
    # negatives clamped
    assert p.normalize_probs(-1, 1, 0)["up"] == 0.0


def test_brier_score():
    perfect = p.brier_score({"up": 1.0, "flat": 0.0, "down": 0.0}, "up")
    assert perfect == 0.0
    worst = p.brier_score({"up": 0.0, "flat": 0.0, "down": 1.0}, "up")
    assert worst == 2.0
    assert p.brier_score({"up": 0.5, "flat": 0.3, "down": 0.2}, "bogus") is None


def test_log_loss():
    assert p.log_loss({"up": 1.0, "flat": 0.0, "down": 0.0}, "up") == 0.0
    # confident-and-wrong → large loss
    assert p.log_loss({"up": 0.0, "flat": 0.0, "down": 1.0}, "up") > 10
    assert p.log_loss({"up": 0.5, "flat": 0.3, "down": 0.2}, "nope") is None


def test_realized_class():
    assert p.realized_class(0.05) == "up"
    assert p.realized_class(-0.05) == "down"
    assert p.realized_class(0.0) == "flat"
    assert p.realized_class(0.01) == "flat"     # within ±2% band
    assert p.realized_class(None) is None


def test_kelly_fraction():
    # positive edge, favourable payoff → positive stake, capped at 10%
    f = p.kelly_fraction(0.6, 0.2, payoff=2.0)
    assert 0 < f <= 0.10
    # negative edge → no stake
    assert p.kelly_fraction(0.2, 0.6, payoff=1.0) == 0.0
    assert p.kelly_fraction(0.6, 0.2, payoff=0) == 0.0


def test_payoff_from_target():
    assert p.payoff_from_target(100, 130, downside=0.15) == pytest.approx(0.30 / 0.15)
    assert p.payoff_from_target(100, 90) == 1.0     # no upside → even odds
    assert p.payoff_from_target(None, 130) == 1.0


def test_position_advice():
    assert p.position_advice(0.0, held=False) == "skip"
    assert p.position_advice(0.0, held=True) == "trim / exit"
    assert p.position_advice(0.01, held=False) == "starter position"
    assert p.position_advice(0.08, held=False) == "full position"
    assert p.position_advice(0.01, held=True) == "hold"
    assert p.position_advice(0.08, held=True) == "add"


def test_render_probability_block():
    out = p.render_probability_block(0.55, 0.30, 0.15, "capex rolls over", 100, 130)
    assert "Probabilities" in out
    assert "Suggested size" in out
    assert "Pre-mortem" in out
