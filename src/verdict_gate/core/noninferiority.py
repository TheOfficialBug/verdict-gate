"""Non-inferiority and fidelity: the acceptance test for distillation (a cheaper student replacing its teacher).

The admission gate asks "is the candidate better?". Distillation asks the opposite question: "is the cheaper version
(a smaller model, a compressed skill, a consolidated memory) NOT WORSE than the original by more than a margin δ?".
An average that "looks the same" is not evidence of that; the posterior of the paired effect is.

    D_i = Y_student,i − Y_teacher,i  (paired per question, mean over seeds), Bayesian bootstrap as in posterior.py.
    NON_INFERIOR  when P(Δ > −δ) ≥ threshold   (the student loses less than δ, with high probability)
    INFERIOR      when P(Δ < −δ) ≥ threshold   (the student loses more than δ)
    INCONCLUSIVE  otherwise                     (collect more questions)

Fidelity (per question, on scores in [0, 1] rounded at 0.5): agreement with the teacher, inherited errors (both
wrong), new errors (teacher right, student wrong: the negative flips a mean hides), fixes, and retention =
student accuracy / teacher accuracy. With costs, the efficiency ratio is student cost / teacher cost.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from verdict_gate.core.posterior import bayes_bootstrap, credible_interval, prob_gt, prob_lt

NON_INFERIOR, INFERIOR, INCONCLUSIVE = "NON_INFERIOR", "INFERIOR", "INCONCLUSIVE"


def noninferiority(D: Sequence[float], margin: float, B: int = 10_000, rng_seed: int = 9001,
                   threshold: float = 0.95) -> dict:
    """Decision on paired differences D (student − teacher) for a non-inferiority margin δ = margin > 0."""
    if margin <= 0:
        raise ValueError("margin must be > 0 (the largest loss you accept, e.g. 0.03 = 3 points)")
    d = np.asarray(D, dtype=float)
    post = bayes_bootstrap(d, B=B, rng_seed=rng_seed)
    p_not_worse, p_worse = prob_gt(post, -margin), prob_lt(post, -margin)
    decision = NON_INFERIOR if p_not_worse >= threshold else INFERIOR if p_worse >= threshold else INCONCLUSIVE
    lo, hi = credible_interval(post)
    return {"decision": decision, "margin": margin, "threshold": threshold, "n": int(d.size),
            "delta_hat": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "p_not_worse_than_margin": round(p_not_worse, 4), "p_worse_than_margin": round(p_worse, 4),
            "draws": B, "rng_seed": rng_seed}


def fidelity(teacher: Sequence[float], student: Sequence[float], teacher_cost: float | None = None,
             student_cost: float | None = None) -> dict:
    """Per-question agreement between a teacher and its student (same questions, same order)."""
    t = np.asarray(teacher, dtype=float) >= 0.5
    s = np.asarray(student, dtype=float) >= 0.5
    if t.shape != s.shape or t.size == 0:
        raise ValueError("teacher and student need the same, non-empty list of questions")
    n = int(t.size)
    out = {"n": n, "agreement": round(float((t == s).mean()), 4),
           "both_right": int((t & s).sum()), "inherited_errors": int((~t & ~s).sum()),
           "new_errors": int((t & ~s).sum()), "fixes": int((~t & s).sum()),
           "negative_flip_rate": round(float((t & ~s).mean()), 4),  # Yan et al. 2021 NFR: old right, new wrong
           "teacher_accuracy": round(float(t.mean()), 4), "student_accuracy": round(float(s.mean()), 4),
           "retention": round(float(s.mean() / t.mean()), 4) if t.any() else None}
    if teacher_cost and student_cost is not None:
        out["cost_ratio"] = round(student_cost / teacher_cost, 4)
    return out
