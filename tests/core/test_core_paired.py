"""equations: paired difference, transitions (majority of valid repeats, ties), harm, posterior, Dirichlet check."""

import numpy as np
import pytest
from core_fixtures import make_table, outcome, outcomes

from verdict_gate.core.harm import conditional_regression_rate, optimism
from verdict_gate.core.paired import (
    CORRECT,
    INCORRECT,
    TIE,
    build_paired_table,
    delta_hat,
    differences,
    majority_label,
    transitions,
)
from verdict_gate.core.posterior import (
    bayes_bootstrap,
    credible_interval,
    dirichlet_multinomial_fixture,
    prob_gt,
    prob_lt,
)
from verdict_gate.core.schemas import SchemaError

# D_i and Δ̂ --------------------------------------------------------------------------------------------------------


def test_differences_and_delta_hat():
    t = make_table([(0, 1), (1, 1), (1, 0), (0.5, 1), (1 / 3, 2 / 3)])
    assert np.allclose(differences(t), [1, 0, -1, 0.5, 1 / 3])
    assert delta_hat(t) == pytest.approx((1 + 0 - 1 + 0.5 + 1 / 3) / 5)


def test_delta_hat_empty_raises():
    with pytest.raises(ValueError):
        delta_hat(make_table([]))


# transitions ------------------------------------------------------------------------------------------------------


def test_transition_counts_binary():
    t = transitions(make_table(outcomes(repairs=3, regressions=2, preserved=4, unresolved=5)))
    assert (t.repairs, t.regressions, t.preserved, t.unresolved, t.unstable, t.ties) == (3, 2, 4, 5, 0, 0)
    assert t.net_repair == 1 and t.n == 14 and t.parent_correct == 6


@pytest.mark.parametrize("r_plus,r_minus,pres,unres", [(3, 2, 4, 5), (0, 0, 0, 7), (10, 0, 1, 1), (0, 6, 6, 0)])
def test_binary_single_repeat_identity(r_plus, r_minus, pres, unres):
    table = make_table(outcomes(r_plus, r_minus, pres, unres))
    t = transitions(table)
    assert delta_hat(table) == pytest.approx((t.repairs - t.regressions) / table.n)


def test_majority_label():
    assert majority_label(2 / 3) == CORRECT and majority_label(1 / 3) == INCORRECT
    assert majority_label(0.5) == TIE and majority_label(1.0) == CORRECT and majority_label(0.0) == INCORRECT


def test_transitions_use_majority_of_repeats():
    # 3 repeats: parent 1/3 (incorrect), candidate 2/3 (correct) → repair, and both arms are unstable
    t = transitions(make_table([(1 / 3, 2 / 3)], repeats=3))
    assert t.repairs == 1 and t.unstable == 1 and t.unresolved == 0


def test_tie_is_unstable_and_unresolved():
    # 2 repeats, parent 1 of 2 correct (exact tie), candidate 2 of 2
    t = transitions(make_table([(0.5, 1.0), (1.0, 0.5)], repeats=2))
    assert t.ties == 2 and t.unresolved == 2 and t.unstable == 2
    assert t.repairs == t.regressions == t.preserved == 0
    assert t.parent_correct == 1  # the second row's parent is a clear majority-correct


def test_transition_classes_partition_n():
    rows = [(0, 1), (1, 0), (1, 1), (0, 0), (0.5, 1), (2 / 3, 1 / 3), (1 / 3, 1 / 3)]
    t = transitions(make_table(rows, repeats=3))
    assert t.repairs + t.regressions + t.preserved + t.unresolved == t.n


def test_unstable_counts_either_arm():
    table = make_table([(1, 1), (0, 0)])
    table.unstable_parent[0] = True
    table.unstable_candidate[1] = True
    assert transitions(table).unstable == 2


# harm ---------------------------------------------------------------------------------------------------------------


def test_conditional_regression_rate():
    t = transitions(make_table(outcomes(repairs=1, regressions=2, preserved=6, unresolved=1)))
    assert conditional_regression_rate(t) == pytest.approx(2 / 8)


def test_conditional_regression_rate_none_without_parent_correct():
    t = transitions(make_table(outcomes(repairs=3, unresolved=2)))
    assert conditional_regression_rate(t) is None


def test_conditional_regression_rate_zero_is_zero():
    t = transitions(make_table(outcomes(preserved=4)))
    assert conditional_regression_rate(t) == 0.0


def test_optimism():
    assert optimism(0.15, 0.05) == pytest.approx(0.10)
    assert optimism(0.05, 0.08) == pytest.approx(-0.03)


# building a table from task outcomes -------------------------------------------------------------------------------


def test_build_paired_table_means_and_exclusions():
    P, C = "P", "C"
    parent = [outcome(run_id=P, task_id="q1", seed=11, score=1.0), outcome(run_id=P, task_id="q1", seed=22, score=0.0),
              outcome(run_id=P, task_id="q2", seed=11, score=1.0),
              outcome(run_id=P, task_id="q2", seed=22, score=None, valid_execution=False, failure_type="timeout"),
              outcome(run_id=P, task_id="q3", seed=11, score=0.0)]
    cand = [outcome(run_id=C, task_id="q1", seed=11, score=1.0), outcome(run_id=C, task_id="q1", seed=22, score=1.0),
            outcome(run_id=C, task_id="q2", seed=11, score=0.0), outcome(run_id=C, task_id="q2", seed=22, score=1.0),
            outcome(run_id=C, task_id="q3", seed=11, score=None, valid_execution=False, failure_type="grader_crash"),
            outcome(run_id=C, task_id="q3", seed=11, score=1.0, retry_of="q3@11")]
    t = build_paired_table(parent, cand)
    assert t.task_ids == ["q1", "q2", "q3"]
    assert t.parent == [0.5, 1.0, 0.0] and t.candidate == [1.0, 0.0, 1.0]  # q2 seed 22 dropped from both arms
    assert t.parent_repeats == [2, 1, 1] and t.candidate_repeats == [2, 1, 1]
    assert t.unstable_parent == [True, False, False] and t.unstable_candidate == [False, False, False]
    assert t.context["excluded"] == ["q2@seed22"]


def test_build_paired_table_drops_task_without_valid_repeat():
    parent = [outcome(run_id="P", task_id="q1", score=None, valid_execution=False, failure_type="timeout"),
              outcome(run_id="P", task_id="q2", score=1.0)]
    cand = [outcome(run_id="C", task_id="q1", score=1.0), outcome(run_id="C", task_id="q2", score=1.0)]
    t = build_paired_table(parent, cand)
    assert t.task_ids == ["q2"] and t.context["dropped_tasks"] == ["q1"]


def test_build_paired_table_rejects_two_valid_rows_and_mixed_strata():
    with pytest.raises(SchemaError, match="two valid"):
        build_paired_table([outcome(), outcome()], [outcome()])
    with pytest.raises(SchemaError, match="inconsistent strata"):
        build_paired_table([outcome(stratum="a")], [outcome(stratum="b")])


# posterior ----------------------------------------------------------------------------------------------------------


def test_bayes_bootstrap_is_a_weighted_mean():
    D = np.array([1.0, 0.0, -1.0, 1.0, 0.0])
    draws = bayes_bootstrap(D, B=5000, rng_seed=3)
    assert draws.shape == (5000,)
    assert draws.min() >= D.min() and draws.max() <= D.max()
    assert draws.mean() == pytest.approx(D.mean(), abs=0.02)


def test_bayes_bootstrap_reproducible_with_seed():
    D = [1, 0, -1, 1]
    assert np.array_equal(bayes_bootstrap(D, 100, 9001), bayes_bootstrap(D, 100, 9001))
    assert not np.array_equal(bayes_bootstrap(D, 100, 9001), bayes_bootstrap(D, 100, 9002))


def test_bayes_bootstrap_constant_differences():
    assert np.allclose(bayes_bootstrap([0.0] * 10, 100, 1), 0.0)
    with pytest.raises(ValueError):
        bayes_bootstrap([], 10, 1)


def test_prob_gt_lt_strict():
    draws = np.array([-1.0, 0.0, 0.0, 1.0])
    assert prob_gt(draws, 0) == 0.25 and prob_lt(draws, 0) == 0.25
    assert prob_gt(draws, -2) == 1.0 and prob_lt(draws, -1) == 0.0


def test_credible_interval():
    lo, hi = credible_interval(np.linspace(0, 1, 1001))
    assert lo == pytest.approx(0.025) and hi == pytest.approx(0.975)


def test_dirichlet_fixture_mean():
    draws = dirichlet_multinomial_fixture(15, 5, 40, B=20000, rng_seed=1)
    assert draws.mean() == pytest.approx((16 - 6) / 63, abs=0.005)  # E[θ₊ − θ₋] under Dirichlet(16, 6, 41)


@pytest.mark.parametrize("r_plus,r_minus,u,tol", [(15, 5, 40, 0.05), (20, 10, 30, 0.05), (8, 4, 48, 0.05),
                                                   (75, 25, 200, 0.02)])
def test_bayes_bootstrap_agrees_with_dirichlet_fixture(r_plus, r_minus, u, tol):
    """Binary single-repeat data: the two posteriors agree up to Monte Carlo error plus the O(1/n) prior difference.

    Aggregating Dirichlet(1,...,1) question weights gives Dirichlet(R₊, R₋, U) on the three outcome classes; the
    fixture adds a uniform prior, Dirichlet(1+R₊, 1+R₋, 1+U). The gap shrinks as n grows (tolerance 0.02 at n=300).
    """
    D = np.array([1.0] * r_plus + [-1.0] * r_minus + [0.0] * u)
    bb = bayes_bootstrap(D, B=20000, rng_seed=11)
    dm = dirichlet_multinomial_fixture(r_plus, r_minus, u, B=20000, rng_seed=12)
    for t in (0.0, 0.05, 0.1):
        assert prob_gt(bb, t) == pytest.approx(prob_gt(dm, t), abs=tol)
    assert bb.mean() == pytest.approx(dm.mean(), abs=tol / 2)
    assert bb.std() == pytest.approx(dm.std(), abs=tol / 2)
