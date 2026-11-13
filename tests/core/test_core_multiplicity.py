"""multiplicity: sign-flip p-values (exact and Monte Carlo), Holm, round accounting; ground-truth labels and matrix."""

import itertools
from math import comb

import numpy as np
import pytest

from verdict_gate.core.multiplicity import (
    BASELINE_DECISIONS,
    NOT_VERIFIED,
    VERIFIED,
    WORSE,
    confusion_matrices,
    confusion_matrix,
    ground_truth_label,
    holm,
    holm_adjusted,
    round_accounting,
    sign_flip_pvalue,
)

# ---- sign flip ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("D,expected", [
    ([1, 1, 1, 1, 1], 1 / 32),  # only the observed signs reach mean 1
    ([1, 2, 3], 1 / 8),
    ([1, -1], 3 / 4),  # sums 0, -2, 2, 0 against observed 0
    ([0, 0, 0], 1.0),  # no change: every flip ties the observed mean
    ([-1, -1, -1], 1.0),  # one-sided: a loss is never evidence of gain
])
def test_sign_flip_exact_known_values(D, expected):
    assert sign_flip_pvalue(D) == pytest.approx(expected)


def brute_force(D):
    d = np.asarray(D, float)
    means = [np.mean(np.array(s) * d) for s in itertools.product((1, -1), repeat=len(d))]
    return np.mean([m >= d.mean() - 1e-12 for m in means])


def test_sign_flip_exact_equals_brute_force_on_fractional_differences():
    D = np.random.default_rng(3).choice([-1, -2 / 3, -1 / 3, 0, 1 / 3, 2 / 3, 1], size=12)
    assert sign_flip_pvalue(D) == pytest.approx(brute_force(D))


def test_sign_flip_exact_at_twenty_questions():
    D = [1] * 14 + [-1] * 4 + [0] * 2  # k = 18 nonzero, observed sum 10 → P(2·Bin(18,½) − 18 ≥ 10) = P(Bin ≥ 14)
    expected = sum(comb(18, j) for j in range(14, 19)) / 2**18
    assert sign_flip_pvalue(D) == pytest.approx(expected)


def test_sign_flip_monte_carlo_above_twenty_is_close_to_exact_and_reproducible():
    D = [1] * 15 + [-1] * 7 + [0] * 8  # n = 30 → permutations
    expected = sum(comb(22, j) for j in range(15, 23)) / 2**22  # P(Bin(22,½) ≥ 15)
    p = sign_flip_pvalue(D, n_perm=50_000, rng_seed=8001)
    assert p == pytest.approx(expected, abs=0.005)
    assert p == sign_flip_pvalue(D, n_perm=50_000, rng_seed=8001)
    assert p != sign_flip_pvalue(D, n_perm=50_000, rng_seed=8002)


def test_sign_flip_monte_carlo_never_returns_zero():
    assert sign_flip_pvalue([1] * 30, n_perm=1000, rng_seed=1) == pytest.approx(1 / 1001)


# ---- Holm -----------------------------------------------------------------------------------------------------------


def test_holm_textbook_example():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted 0.005·4 = 0.02 ≤ .05 reject; 0.01·3 = 0.03 reject; 0.03·2 = 0.06 > .05 stop
    assert holm(p, alpha=0.05) == [True, False, False, True]
    assert holm_adjusted(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_adjusted_is_capped_at_one_and_monotone():
    assert holm_adjusted([0.5, 0.6, 0.9]) == pytest.approx([1.0, 1.0, 1.0])
    assert holm([0.04], alpha=0.05) == [True]


# ---- round accounting -------------------------------------------------------------------------------------------------


def card(decision, n):
    return {"decision": decision, "sequential": {"questions_used": n}}


def test_round_accounting():
    cards = [card("PASS_GLOBAL", 16), card("PASS_CONDITIONAL", 60), card("UNSUPPORTED", 8),
             card("HARMFUL", 24), card("UNSUPPORTED", 60)]
    assert round_accounting(cards, generated=6, max_n=60) == {
        "generated": 6, "evaluated": 5, "stopped_early": 3, "admitted_global": 1, "admitted_conditional": 1,
        "unsupported": 2, "harmful": 1, "needs_more_evidence": 0}


def test_round_accounting_rejects_more_evaluated_than_generated():
    with pytest.raises(ValueError):
        round_accounting([card("HARMFUL", 8)] * 2, generated=1)


# ---- ground truth and confusion matrix  -------------------------------------------------------------------------


def test_ground_truth_labels():
    assert ground_truth_label([1] * 10 + [0] * 20) == VERIFIED
    assert ground_truth_label([-1] * 10 + [0] * 20) == WORSE
    assert ground_truth_label([1, -1] * 10) == NOT_VERIFIED


def test_confusion_matrix_cells():
    pairs = [("PASS_GLOBAL", VERIFIED), ("PASS_CONDITIONAL", NOT_VERIFIED), ("PASS_GLOBAL", WORSE),
             ("UNSUPPORTED", VERIFIED), ("HARMFUL", WORSE), ("NEEDS_MORE_EVIDENCE", NOT_VERIFIED)]
    assert confusion_matrix(pairs) == {"supported_admission": 1, "unsupported_admission": 1, "harmful_admission": 1,
                                       "missed_improvement": 1, "correct_reject_defer": 2, "total": 6}


LABELS = {"c1": VERIFIED, "c2": NOT_VERIFIED, "c3": WORSE, "c4": VERIFIED}


def test_confusion_matrices_add_accept_all_and_reject_all_reference_rows():
    decisions = {"moderate": {"c1": "PASS_GLOBAL", "c2": "UNSUPPORTED", "c3": "HARMFUL", "c4": "NEEDS_MORE_EVIDENCE"},
                 "naive": {"c1": "PASS_GLOBAL", "c2": "PASS_GLOBAL", "c3": "UNSUPPORTED", "c4": "PASS_GLOBAL"}}
    m = confusion_matrices(decisions, LABELS)
    assert list(m) == ["moderate", "naive", "accept_all", "reject_all"]
    assert m["accept_all"] == {"supported_admission": 2, "unsupported_admission": 1, "harmful_admission": 1,
                               "missed_improvement": 0, "correct_reject_defer": 0, "total": 4}
    assert m["reject_all"] == {"supported_admission": 0, "unsupported_admission": 0, "harmful_admission": 0,
                               "missed_improvement": 2, "correct_reject_defer": 2, "total": 4}
    assert m["moderate"]["supported_admission"] == 1 and m["moderate"]["missed_improvement"] == 1
    assert {row["total"] for row in m.values()} == {4}  # one denominator for every row
    assert BASELINE_DECISIONS == {"accept_all": "PASS_GLOBAL", "reject_all": "UNSUPPORTED"}
    assert list(confusion_matrices(decisions, LABELS, baselines=False)) == ["moderate", "naive"]


def test_confusion_matrices_need_every_labelled_candidate_and_reserve_the_baseline_names():
    with pytest.raises(ValueError, match="decided"):
        confusion_matrices({"moderate": {"c1": "PASS_GLOBAL"}}, LABELS)
    with pytest.raises(ValueError, match="reserved"):
        confusion_matrices({"accept_all": dict.fromkeys(LABELS, "PASS_GLOBAL")}, LABELS)
