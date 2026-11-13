"""Power simulation. No model calls, seeded, offline.

Generative model for one question, chosen to reproduce the persistence seen in repeated runs of a search-QA benchmark (most questions
are always wrong, a few always right, the rest flip between repeats):

    parent class     always right (π_p = 1), flaky (π_p = 1/2), always wrong (π_p = 0)
                     with shares w1 = a − δ, f = 2δ, w0 = 1 − w1 − f, where a = parent accuracy and
                     δ = discordance = P(two parent repeats of a question disagree)
    candidate        fixed (π_c = 1) with probability g, broken (π_c = 0) with probability h ("churn"),
                     otherwise unchanged (π_c = π_p).  Net effect Δ = g(1 − a) − h·a, so g = (Δ + h·a) / (1 − a);
                     a harmful candidate (Δ < 0) instead breaks parent successes: h = churn + |Δ|/a.
    repeats          independent Bernoulli(π) draws per repeat.

With one repeat per arm this gives paired binary outcomes with
    p_repair     = P(parent wrong, candidate right) = (1 − g − h)·f/4 + g(1 − a)
    p_regression = P(parent right, candidate wrong) = (1 − g − h)·f/4 + h·a          (p_repair − p_regression = Δ)

Policies are applied with the analysis core itself (decide / run_sequential), so the simulated decisions are the
decisions the frozen policies would make on such data. Ground-truth labels follow multiplicity.py.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from verdict_gate.core.calibrate import floor_at
from verdict_gate.core.decision import Evidence, decide
from verdict_gate.core.policy import (
    FixedRepeatPolicy,
    KeepIfHigherPolicy,
    Policy,
    SequentialBayesPolicy,
    pass_met,
    summarize,
)
from verdict_gate.core.posterior import bayes_bootstrap, prob_gt, prob_lt
from verdict_gate.core.schemas import Decision, PairedTable
from verdict_gate.core.sequential import run_sequential

PASSES = (Decision.PASS_GLOBAL, Decision.PASS_CONDITIONAL)


@dataclass(frozen=True)
class Scenario:
    name: str
    parent_acc: float  # a
    discordance: float  # δ
    churn: float = 0.0  # h

    def shares(self) -> tuple[float, float, float]:
        """(always wrong, flaky, always right)."""
        w1, f = self.parent_acc - self.discordance, 2 * self.discordance
        w0 = 1.0 - w1 - f
        if min(w0, f, w1) < -1e-12:
            raise ValueError(f"scenario {self.name}: accuracy {self.parent_acc} and discordance {self.discordance} "
                             "are incompatible (need discordance ≤ accuracy and accuracy + discordance ≤ 1)")
        return max(w0, 0.0), f, max(w1, 0.0)

    def fix_break(self, delta: float) -> tuple[float, float]:
        """(g, h) for net effect Δ. Δ ≥ 0: extra fixes on top of the churn. Δ < 0 (harmful candidate): extra breaks
        of parent successes, h = churn + |Δ|/a, with the churn's own fixes g = churn·a/(1 − a) kept (continuous at 0)."""
        a, h = self.parent_acc, self.churn
        if delta < 0:
            h = h - delta / a
            g = self.churn * a / (1.0 - a)
        else:
            g = (delta + h * a) / (1.0 - a)
        if g < 0 or g + h > 1:
            raise ValueError(f"scenario {self.name}: Δ={delta} not reachable with churn {h}")
        return g, h

    def rates(self, delta: float) -> tuple[float, float]:
        """(p_repair, p_regression) for one repeat per arm."""
        g, h = self.fix_break(delta)
        _, f, _ = self.shares()
        same = (1 - g - h) * f / 4
        return same + g * (1 - self.parent_acc), same + h * self.parent_acc


def draw_questions(sc: Scenario, delta: float, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Per-question correctness probabilities (π_parent, π_candidate)."""
    w0, f, w1 = sc.shares()
    pi_p = rng.choice(np.array([0.0, 0.5, 1.0]), size=n, p=np.array([w0, f, w1]) / (w0 + f + w1))
    g, h = sc.fix_break(delta)
    u = rng.random(n)
    pi_c = np.where(u < g, 1.0, np.where(u < g + h, 0.0, pi_p))
    return pi_p, pi_c


def draw_repeats(pi: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k × n binary outcomes."""
    return (rng.random((k, pi.size)) < pi).astype(float)


def paired_table(parent: np.ndarray, candidate: np.ndarray, strata: Sequence[str | None]) -> PairedTable:
    """Table from k_p × n and k_c × n outcome matrices (per-question means over repeats)."""
    n = parent.shape[1]
    return PairedTable(
        task_ids=[f"q{i:03d}" for i in range(n)], strata=list(strata),
        parent=parent.mean(axis=0).tolist(), candidate=candidate.mean(axis=0).tolist(),
        parent_repeats=[parent.shape[0]] * n, candidate_repeats=[candidate.shape[0]] * n,
        unstable_parent=(parent.min(axis=0) != parent.max(axis=0)).tolist(),
        unstable_candidate=(candidate.min(axis=0) != candidate.max(axis=0)).tolist(),
        model="simulated", context={"simulated": True},
    )


def strata_labels(counts: dict[str, int]) -> list[str]:
    return [level for level, k in counts.items() for _ in range(k)]


@dataclass
class PolicyStats:
    trials: int
    decisions: Counter
    questions_used: float  # mean paired questions looked at
    candidate_evals: float  # mean candidate executions (questions × candidate repeats)
    stops: Counter = field(default_factory=Counter)  # rule that ended the run -> count (sequential policies only)

    def p(self, *decisions: Decision) -> float:
        return sum(self.decisions[d] for d in decisions) / self.trials

    @property
    def p_pass(self) -> float:
        return self.p(*PASSES)


STOP_RULES = (("P(Δ<−F)", "harm"), ("cost ratio", "cost"), ("pass rule met overall", "pass"),
              ("stratum harmed", "conditional"), ("pass rule not met overall", "conditional"),
              ("futility", "futility"), ("maximum n", "max_n"), ("curtailed", "curtailed"), ("undecided", "undecided"))


def stop_rule(reason: str) -> str:
    """Which decision rule ended a sequential run, from the reason text written by decision.decide."""
    return next((rule for prefix, rule in STOP_RULES if reason.startswith(prefix)), "other")


def simulate_policies(sc: Scenario, delta: float, policies: dict[str, Policy], floor_by_n: dict[int, float],
                      legacy_floor: float, strata_counts: dict[str, int], trials: int, rng_seed: int,
                      parent_repeats: int, candidate_repeats: int) -> dict[str, PolicyStats]:
    """Apply every policy to the same simulated selection sets (one fresh set of questions per trial).

    keep_if_higher and sequential_bayes see one repeat per arm (the first); fixed_repeat sees per-question means
    over `parent_repeats` parent repeats and the policy's own number of candidate seeds.
    """
    rng = np.random.default_rng(rng_seed)
    strata = strata_labels(strata_counts)
    n = len(strata)
    stats = {name: PolicyStats(trials, Counter(), 0.0, 0.0) for name in policies}
    for _ in range(trials):
        pi_p, pi_c = draw_questions(sc, delta, n, rng)
        Yp, Yc = draw_repeats(pi_p, parent_repeats, rng), draw_repeats(pi_c, candidate_repeats, rng)
        single = paired_table(Yp[:1], Yc[:1], strata)
        for name, policy in policies.items():
            if isinstance(policy, SequentialBayesPolicy):
                res = run_sequential(single, policy, floor_by_n, invalid_rate=0.0, pairing_ok=True)
                decision, used, evals = res.final.decision, res.questions_used, res.questions_used
                stats[name].stops[stop_rule(res.final.reason)] += 1
            elif isinstance(policy, KeepIfHigherPolicy):
                decision = decide(Evidence(single, 0.0, True), policy).decision
                used, evals = n, n
            elif isinstance(policy, FixedRepeatPolicy):
                k = len(policy.seeds)
                table = paired_table(Yp, Yc[:k], strata)
                decision = decide(Evidence(table, 0.0, True, legacy_floor=legacy_floor), policy).decision
                used, evals = n, n * k
            else:
                raise TypeError(type(policy).__name__)
            s = stats[name]
            s.decisions[decision] += 1
            s.questions_used += used / trials
            s.candidate_evals += evals / trials
    return stats


def simulate_ground_truth(sc: Scenario, delta: float, n: int, parent_repeats: int, candidate_repeats: int,
                          trials: int, rng_seed: int, verified_p: float, worse_p: float,
                          draws: int) -> dict[str, float]:
    """P(Verified) and P(Worse) for a confirmation set of n questions (labels as in multiplicity.py)."""
    rng = np.random.default_rng(rng_seed)
    verified = worse = 0
    for t in range(trials):
        pi_p, pi_c = draw_questions(sc, delta, n, rng)
        D = draw_repeats(pi_c, candidate_repeats, rng).mean(axis=0) - draw_repeats(pi_p, parent_repeats, rng).mean(axis=0)
        post = bayes_bootstrap(D, B=draws, rng_seed=rng_seed + t)
        verified += prob_gt(post, 0.0) >= verified_p
        worse += prob_lt(post, 0.0) >= worse_p
    return {"verified": verified / trials, "worse": worse / trials}


def fixed_n_pass_power(sc: Scenario, delta: float, ns: Sequence[int], policy: SequentialBayesPolicy,
                       floor_by_n: dict[int, float], trials: int, rng_seed: int) -> dict[int, float]:
    """P(the policy's pass rule holds) when it is checked once, at a fixed n, on one repeat per arm."""
    rng = np.random.default_rng(rng_seed)
    hits = dict.fromkeys(ns, 0)
    for _ in range(trials):
        pi_p, pi_c = draw_questions(sc, delta, max(ns), rng)
        D = draw_repeats(pi_c, 1, rng)[0] - draw_repeats(pi_p, 1, rng)[0]
        for n in ns:
            hits[n] += pass_met(summarize(D[:n], floor_at(floor_by_n, n), policy.posterior), policy.pass_rule)
    return {n: hits[n] / trials for n in ns}


def mde(curve: dict[float, float], target: float) -> float | None:
    """Smallest Δ > 0 on the grid whose probability reaches `target`; None if no grid point does."""
    hits = [d for d, p in sorted(curve.items()) if d > 0 and p >= target]
    return hits[0] if hits else None
