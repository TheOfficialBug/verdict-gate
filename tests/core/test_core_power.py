"""Power simulation: model identities and sanity of the simulated operating characteristics (small trial counts)."""

import numpy as np
import pytest
from core_fixtures import DATA

from verdict_gate.core.policy import load_policy
from verdict_gate.core.power import (
    Scenario,
    draw_questions,
    draw_repeats,
    fixed_n_pass_power,
    mde,
    simulate_ground_truth,
    simulate_policies,
)
from verdict_gate.core.schemas import Decision

SC = Scenario("test", parent_acc=0.18, discordance=0.14)
STRATA = {"never-changing": 17, "slow-changing": 26, "fast-changing": 17}
FLOOR = {8: 0.25, 16: 0.19, 24: 0.15, 32: 0.13, 40: 0.12, 48: 0.11, 56: 0.10, 60: 0.10}


def _fast(policy, draws=1000):
    return policy.model_copy(update={"posterior": policy.posterior.model_copy(update={"draws": draws})})


@pytest.fixture(scope="module")
def policies():
    out = {n: load_policy(DATA / f"configs/policies/{n}.yaml") for n in ("naive", "fixed_repeat", "moderate")}
    out["moderate"] = _fast(out["moderate"])
    return out


def test_rates_identity_and_shares():
    for churn in (0.0, 0.05):
        sc = Scenario("s", 0.18, 0.14, churn)
        for d in (0.0, 0.05, 0.15):
            rep, reg = sc.rates(d)
            assert rep - reg == pytest.approx(d)
    assert SC.rates(0.0) == pytest.approx((0.07, 0.07))  # null: p_repair = p_regression = δ/2
    assert sum(SC.shares()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        Scenario("bad", parent_acc=0.10, discordance=0.20).shares()


def test_harmful_effects_break_parent_successes():
    for churn in (0.0, 0.05):
        sc = Scenario("s", 0.18, 0.14, churn)
        for d in (-0.05, -0.09, -0.12):
            rep, reg = sc.rates(d)
            assert rep - reg == pytest.approx(d)
    assert SC.fix_break(-0.09) == pytest.approx((0.0, 0.5))  # half of the parent's successes broken
    assert SC.fix_break(-0.18) == pytest.approx((0.0, 1.0))
    assert Scenario("c", 0.18, 0.14, 0.05).fix_break(-1e-12) == pytest.approx(Scenario("c", 0.18, 0.14, 0.05)
                                                                            .fix_break(0.0))  # continuous at 0
    with pytest.raises(ValueError):
        Scenario("c", 0.18, 0.14, 0.05).fix_break(-0.18)  # needs h > 1


def test_harmful_candidate_is_called_harmful_more_often_than_null(policies):
    seq = {"moderate": policies["moderate"]}
    null = simulate_policies(SC, 0.0, seq, FLOOR, 0.14, STRATA, 60, 5, 5, 3)["moderate"]
    bad = simulate_policies(SC, -0.18, seq, FLOOR, 0.14, STRATA, 60, 5, 5, 3)["moderate"]
    assert null.p(Decision.HARMFUL) <= 0.05 < bad.p(Decision.HARMFUL)
    assert bad.questions_used > null.questions_used  # A3: harmful-leaning runs are not stopped by futility


def test_simulated_outcomes_match_the_model():
    rng = np.random.default_rng(0)
    n = 200_000
    pi_p, pi_c = draw_questions(SC, 0.10, n, rng)
    Yp, Yc = draw_repeats(pi_p, 2, rng), draw_repeats(pi_c, 1, rng)[0]
    assert Yp[0].mean() == pytest.approx(0.18, abs=0.005)
    assert np.mean(Yp[0] != Yp[1]) == pytest.approx(0.14, abs=0.005)  # discordance
    rep, reg = SC.rates(0.10)
    assert np.mean((Yp[0] == 0) & (Yc == 1)) == pytest.approx(rep, abs=0.005)
    assert np.mean((Yp[0] == 1) & (Yc == 0)) == pytest.approx(reg, abs=0.005)


def test_null_pass_rate_small_for_moderate_and_large_for_naive(policies):
    stats = simulate_policies(SC, 0.0, policies, FLOOR, 0.14, STRATA, trials=80, rng_seed=1, parent_repeats=5,
                              candidate_repeats=3)
    assert stats["moderate"].p_pass <= 0.10
    assert stats["naive"].p_pass >= 0.25  # keep-if-higher has no noise control
    assert stats["fixed_repeat"].candidate_evals == 180 and stats["naive"].questions_used == 60
    assert 8 <= stats["moderate"].questions_used <= 60


def test_pass_rate_monotone_in_delta(policies):
    p = [simulate_policies(SC, d, policies, FLOOR, 0.14, STRATA, trials=60, rng_seed=2, parent_repeats=5,
                           candidate_repeats=3)["moderate"].p_pass for d in (0.0, 0.15, 0.40)]
    assert p[0] <= p[1] <= p[2] and p[2] >= 0.8


def test_ground_truth_monotone_and_worse_vanishes():
    gt = [simulate_ground_truth(SC, d, 30, 5, 3, trials=80, rng_seed=3, verified_p=0.9, worse_p=0.9, draws=2000)
          for d in (0.0, 0.15, 0.40)]
    assert gt[0]["verified"] <= gt[1]["verified"] <= gt[2]["verified"] and gt[2]["verified"] >= 0.9
    assert gt[2]["worse"] == 0.0


def test_fixed_n_power_grows_with_n(policies):
    power = fixed_n_pass_power(SC, 0.30, [8, 60], policies["moderate"], FLOOR, trials=60, rng_seed=4)
    assert power[8] <= power[60]


def test_simulation_is_seeded(policies):
    run = lambda: simulate_policies(SC, 0.05, {"naive": policies["naive"]}, FLOOR, 0.14, STRATA, 30, 7, 5, 3)
    assert run()["naive"].decisions == run()["naive"].decisions


def test_mde():
    assert mde({0.0: 0.05, 0.05: 0.5, 0.10: 0.85, 0.15: 0.95}, 0.8) == 0.10
    assert mde({0.0: 0.9, 0.05: 0.5}, 0.8) is None  # Δ = 0 never counts as a detectable effect
