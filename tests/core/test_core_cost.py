"""cost equations: cost_ratio, questions_per_decision, cost_per_verified, per-question cost arrays."""

import pytest
from core_fixtures import outcome

from verdict_gate.core.cost import (
    cost_per_verified,
    cost_ratio,
    cost_ratio_on,
    mean_cost_per_question,
    per_question_costs,
    questions_per_decision,
    sum_line_items,
)


def test_per_question_cost_divides_by_seeds_and_counts_retries():
    rows = [outcome(task_id="q1", seed=11, cost_usd=0.002), outcome(task_id="q1", seed=22, cost_usd=0.004),
            outcome(task_id="q2", seed=11, score=None, valid_execution=False, failure_type="timeout",
                    cost_usd=0.001),
            outcome(task_id="q2", seed=11, retry_of="q2@seed11", cost_usd=0.003)]
    assert per_question_costs(rows) == pytest.approx({"q1": 0.003, "q2": 0.004})


def test_mean_cost_per_question_on_a_subset():
    costs = {"a": 1.0, "b": 3.0, "c": 5.0}
    assert mean_cost_per_question(costs, ["a", "b"]) == 2.0
    assert mean_cost_per_question(costs, ["zzz"]) is None


def test_cost_ratio_is_candidate_over_parent():
    assert cost_ratio(0.002, 0.003) == pytest.approx(1.5)


@pytest.mark.parametrize("parent,candidate", [(None, 1.0), (1.0, None), (0.0, 1.0)])
def test_cost_ratio_unknown_is_none(parent, candidate):
    assert cost_ratio(parent, candidate) is None


def test_cost_ratio_per_block():
    parent = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    candidate = {"a": 1.0, "b": 1.0, "c": 4.0, "d": 4.0}
    assert cost_ratio_on(parent, candidate, ["a", "b"]) == pytest.approx(1.0)  # first block
    assert cost_ratio_on(parent, candidate, ["a", "b", "c", "d"]) == pytest.approx(2.5)  # after two blocks


def test_questions_per_decision():
    assert questions_per_decision(96, 6) == 16.0
    assert questions_per_decision(0, 0) is None


def test_cost_per_verified():
    assert cost_per_verified(10.0, 20.0, 3) == pytest.approx(10.0)


def test_cost_per_verified_zero_verified_is_none_not_inf_or_zero():
    assert cost_per_verified(10.0, 20.0, 0) is None


def test_sum_line_items():
    assert sum_line_items([{"model": 1.0, "search": 0.5}, {"model": 2.0, "grader": 0.25}]) == {
        "grader": 0.25, "model": 3.0, "search": 0.5}
