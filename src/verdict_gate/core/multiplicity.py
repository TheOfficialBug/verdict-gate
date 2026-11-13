"""Multiplicity, round accounting and ground-truth labels.

sign_flip_pvalue: one-sided paired sign-flip permutation test of H0 "Δ = 0, D_i symmetric about 0" against Δ > 0.
    Statistic T = mean(D). p = share of sign vectors s with mean(s·D) ≥ T (the observed vector included).
    n ≤ 20: exact, all 2^n sign vectors. n > 20: n_perm random sign vectors from rng_seed, p = (1 + #{T* ≥ T}) /
    (1 + n_perm).
holm: Holm step-down at level alpha. Used only as an ablation (no family-wise claim is made for the primary policy).

Round accounting: generated / evaluated / stopped early / admitted global / admitted conditional / unsupported /
harmful (+ still undecided). Ground truth: Verified P(Δ>0) ≥ 0.90, Worse P(Δ<0) ≥ 0.90, else Not verified;
the confusion matrix crosses policy decisions with those labels; confusion_matrices() adds the reference rows
accept_all (every candidate PASS_GLOBAL) and reject_all (every candidate UNSUPPORTED).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from verdict_gate.core.posterior import bayes_bootstrap, prob_gt, prob_lt
from verdict_gate.core.schemas import Decision

EXACT_MAX_N = 20
_TOL = 1e-12  # floating-point slack when comparing permuted means with the observed one


def sign_flip_pvalue(D: Sequence[float], n_perm: int = 100_000, rng_seed: int = 0) -> float:
    d = np.asarray(D, dtype=float)
    n = d.size
    if n == 0:
        raise ValueError("sign_flip_pvalue needs at least one paired question")
    observed = d.mean()
    if n <= EXACT_MAX_N:
        hits = 0
        chunk = 1 << 16
        for start in range(0, 1 << n, chunk):
            codes = np.arange(start, min(start + chunk, 1 << n))
            signs = 1 - 2 * ((codes[:, None] >> np.arange(n)) & 1)  # bit j set → flip D_j
            hits += int(np.sum(signs @ d / n >= observed - _TOL))
        return hits / (1 << n)
    rng = np.random.default_rng(rng_seed)
    hits = 0
    chunk = 10_000
    for start in range(0, n_perm, chunk):
        size = min(chunk, n_perm - start)
        signs = rng.choice((-1.0, 1.0), size=(size, n))
        hits += int(np.sum(signs @ d / n >= observed - _TOL))
    return (1 + hits) / (1 + n_perm)


def holm_adjusted(pvalues: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values: p_(i) · (m − i + 1), made monotone, capped at 1 (returned in input order)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return [float(x) for x in adjusted]


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Which hypotheses Holm's step-down procedure rejects at family-wise level alpha (input order)."""
    return [a <= alpha for a in holm_adjusted(pvalues)]


# ---- round accounting -----------------------------------------------------------------------------------------------


def round_accounting(cards: Iterable[dict], generated: int, max_n: int | None = None) -> dict[str, int]:
    """Counts for one round. `generated` = candidates proposed (evaluated ones have a card). A card stopped early if
    its decision used fewer than max_n questions (max_n None → stopped early is not counted)."""
    cards = list(cards)
    if generated < len(cards):
        raise ValueError(f"generated ({generated}) < evaluated ({len(cards)})")
    count = lambda d: sum(c["decision"] == d.value for c in cards)
    return {
        "generated": generated,
        "evaluated": len(cards),
        "stopped_early": sum(c["sequential"]["questions_used"] < max_n for c in cards) if max_n else 0,
        "admitted_global": count(Decision.PASS_GLOBAL),
        "admitted_conditional": count(Decision.PASS_CONDITIONAL),
        "unsupported": count(Decision.UNSUPPORTED),
        "harmful": count(Decision.HARMFUL),
        "needs_more_evidence": count(Decision.NEEDS_MORE_EVIDENCE),
    }


# ---- ground truth and confusion matrix -----------------------------------------------------------------------------

VERIFIED, NOT_VERIFIED, WORSE = "verified", "not_verified", "worse"
ADMITTED = {Decision.PASS_GLOBAL.value, Decision.PASS_CONDITIONAL.value}


def ground_truth_label(D: Sequence[float], draws: int = 10_000, rng_seed: int = 0, threshold: float = 0.90) -> str:
    """Verified if P(Δ>0) ≥ threshold, Worse if P(Δ<0) ≥ threshold, else Not verified (Bayesian bootstrap)."""
    post = bayes_bootstrap(D, B=draws, rng_seed=rng_seed)
    if prob_gt(post, 0.0) >= threshold:
        return VERIFIED
    if prob_lt(post, 0.0) >= threshold:
        return WORSE
    return NOT_VERIFIED


def confusion_matrix(pairs: Iterable[tuple[str, str]]) -> dict[str, int]:
    """pairs = (policy decision, ground-truth label). Cells as in confusion_matrix's keys, plus the total."""
    cells = {"supported_admission": 0, "unsupported_admission": 0, "harmful_admission": 0,
             "missed_improvement": 0, "correct_reject_defer": 0, "total": 0}
    for decision, label in pairs:
        decision = decision.value if isinstance(decision, Decision) else decision
        cells["total"] += 1
        if decision in ADMITTED:
            key = {VERIFIED: "supported_admission", NOT_VERIFIED: "unsupported_admission",
                   WORSE: "harmful_admission"}[label]
        else:
            key = "missed_improvement" if label == VERIFIED else "correct_reject_defer"
        cells[key] += 1
    return cells


# Offline reference policies: the two trivial admission rules, shown beside naive / fixed-repeat /
# moderate / strict in every confusion matrix. They need no evidence and no model calls.
BASELINE_DECISIONS = {"accept_all": Decision.PASS_GLOBAL.value, "reject_all": Decision.UNSUPPORTED.value}


def confusion_matrices(decisions: dict[str, dict[str, str]], labels: dict[str, str],
                       baselines: bool = True) -> dict[str, dict[str, int]]:
    """One confusion matrix per policy, plus the accept_all / reject_all reference rows.

    decisions = {policy: {revision_id: decision}}; labels = {revision_id: ground-truth label}. Every policy must have
    decided exactly the labelled candidates, so all rows share one denominator."""
    out = {}
    for policy, by_revision in decisions.items():
        if policy in BASELINE_DECISIONS:
            raise ValueError(f"policy name {policy!r} is reserved for a reference row")
        if set(by_revision) != set(labels):
            raise ValueError(f"policy {policy!r} decided {sorted(by_revision)}, labelled {sorted(labels)}")
        out[policy] = confusion_matrix((by_revision[r], labels[r]) for r in labels)
    if baselines:
        for name, decision in BASELINE_DECISIONS.items():
            out[name] = confusion_matrix((decision, labels[r]) for r in labels)
    return out
