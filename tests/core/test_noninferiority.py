import pytest

from verdict_gate.core.noninferiority import INCONCLUSIVE, INFERIOR, NON_INFERIOR, fidelity, noninferiority


def test_equal_arms_are_non_inferior_at_a_reasonable_margin():
    D = [0.0] * 140 + [1.0] * 5 + [-1.0] * 5  # same average, a few flips each way
    r = noninferiority(D, margin=0.05)
    assert r["decision"] == NON_INFERIOR and r["delta_hat"] == 0.0


def test_clear_loss_is_inferior():
    D = [0.0] * 100 + [-1.0] * 40
    assert noninferiority(D, margin=0.05)["decision"] == INFERIOR


def test_small_sample_is_inconclusive():
    D = [0.0] * 6 + [1.0, -1.0, -1.0]
    assert noninferiority(D, margin=0.05)["decision"] == INCONCLUSIVE


def test_margin_must_be_positive():
    with pytest.raises(ValueError):
        noninferiority([0.0], margin=0)


def test_fidelity_counts_flips_a_mean_hides():
    teacher = [1, 1, 1, 1, 0, 0, 0, 0]
    student = [1, 1, 0, 0, 1, 1, 0, 0]  # same accuracy, 50% agreement
    f = fidelity(teacher, student, teacher_cost=4.0, student_cost=1.0)
    assert f["retention"] == 1.0 and f["agreement"] == 0.5
    assert (f["new_errors"], f["fixes"], f["inherited_errors"], f["both_right"]) == (2, 2, 2, 2)
    assert f["cost_ratio"] == 0.25
