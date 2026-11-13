"""Admission decision.

sequential_bayes, evaluated on the n questions seen so far with floor F = F_noise(n) (nearest frozen n at or below).
Steps are checked in this order and the first match wins:

  1. validity       invalid rate > limit or pairing broken → raise EvidenceInvalid (halts the run; not a decision)
  2. HARMFUL        P(Δ<−F) ≥ harm.p_gain_lt_neg_floor  (2a)   or   cost_ratio > harm.max_cost_ratio and pass rule
                    not met (2b). 2a uses the pass rule's floor F; with no floor F = 0.
                    With harm.min_n set (optional fallback), 2a is checked only when n ≥ harm.min_n.
  3. PASS_GLOBAL    pass rule met, no harmed stratum, cost_ratio ≤ harm.max_cost_ratio
  4. PASS_CONDITIONAL  (pass met but a stratum harmed (4a) or pass not met (4b)) and ≥ 1 supported stratum
  5. UNSUPPORTED    futility (5a) or n = max_n (5b) or curtailment: even all-repair remaining questions fail (5c)
  6. NEEDS_MORE_EVIDENCE otherwise

With futility.hold_while_negative, 5a and 5c are skipped while Δ̂ < 0, so a run leaning
harmful continues block by block until HARMFUL fires, another rule decides, or n = max_n (5b, UNSUPPORTED).

keep_if_higher returns PASS_GLOBAL or UNSUPPORTED; fixed_repeat returns PASS_GLOBAL, UNSUPPORTED or HARMFUL.
A cost ratio of None (costs not available) makes the cost conditions of steps 2 and 3 inactive; the reason says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from verdict_gate.core.calibrate import floor_at
from verdict_gate.core.paired import delta_hat, differences
from verdict_gate.core.policy import (
    FixedRepeatPolicy,
    KeepIfHigherPolicy,
    Policy,
    SequentialBayesPolicy,
    Summary,
    futility_met,
    pass_met,
    summarize,
)
from verdict_gate.core.schemas import Decision, PairedTable
from verdict_gate.core.scope import HARMFUL, SUPPORTED, StratumEffect, stratum_effects


class EvidenceInvalid(Exception):
    """Stop rule: the evidence cannot support any decision (too many invalid executions, or broken pairing)."""


@dataclass
class Evidence:
    table: PairedTable  # the questions seen so far
    invalid_rate: float
    pairing_ok: bool
    cost_ratio: float | None = None
    floor_by_n: dict[int, float] | None = None  # paired F_noise(n) (sequential_bayes)
    legacy_floor: float | None = None  # F_legacy (fixed_repeat)
    available_n: int | None = None  # questions that exist for this comparison; max_n = min(policy.max_n, this)


@dataclass
class DecisionResult:
    decision: Decision
    reason: str
    n: int
    delta_hat: float
    summary: Summary | None = None
    strata: list[StratumEffect] = field(default_factory=list)
    activation_conditions: list[dict] = field(default_factory=list)
    inhibition_conditions: list[dict] = field(default_factory=list)
    parent_fallback: list[dict] = field(default_factory=list)
    held_out_replication: str = "not_applicable"


def decide(evidence: Evidence, policy: Policy) -> DecisionResult:
    _check_validity(evidence, policy)
    if evidence.table.n == 0:
        raise EvidenceInvalid("no valid paired questions")
    if isinstance(policy, SequentialBayesPolicy):
        return _decide_sequential(evidence, policy)
    if isinstance(policy, KeepIfHigherPolicy):
        return _decide_keep_if_higher(evidence.table)
    if isinstance(policy, FixedRepeatPolicy):
        return _decide_fixed_repeat(evidence, policy)
    raise TypeError(f"unknown policy type {type(policy).__name__}")


def _check_validity(evidence: Evidence, policy: Policy) -> None:  # step 1
    if not evidence.pairing_ok:
        raise EvidenceInvalid("pairing broken")
    if evidence.invalid_rate > policy.max_invalid_rate:
        raise EvidenceInvalid(f"invalid rate {evidence.invalid_rate:.3f} > limit {policy.max_invalid_rate}")


def _condition(strata_field: str, level: str) -> dict:
    return {"field": strata_field, "level": level}


def _decide_sequential(ev: Evidence, policy: SequentialBayesPolicy) -> DecisionResult:
    if ev.floor_by_n is None:
        raise ValueError("sequential_bayes needs the paired floor F_noise(n) (Evidence.floor_by_n)")
    table, n = ev.table, ev.table.n
    max_n = min(policy.max_n, ev.available_n) if ev.available_n is not None else policy.max_n
    D = differences(table)
    s = summarize(D, floor_at(ev.floor_by_n, n), policy.posterior)
    passed = pass_met(s, policy.pass_rule)
    cost_known = ev.cost_ratio is not None
    cost_too_high = cost_known and ev.cost_ratio > policy.harm.max_cost_ratio
    cost_note = "" if cost_known else "; cost ratio unknown, cost conditions not applied"
    stats = (f"n={n}, Δ̂={s.delta_hat:+.3f}, F={s.floor:.3f}, P(Δ>0)={s.p_gain_gt_0:.3f}, "
             f"P(Δ>F)={s.p_gain_gt_floor:.3f}, P(Δ<0)={s.p_gain_lt_0:.3f}, P(Δ<−F)={s.p_gain_lt_neg_floor:.3f}")
    result = lambda decision, why, **kw: DecisionResult(decision, f"{why} ({stats}){cost_note}", n, s.delta_hat,
                                                        s, **kw)

    effects = stratum_effects(table, policy.strata, policy, ev.floor_by_n)
    harmed = [e.level for e in effects if e.label == HARMFUL]
    supported = [e.level for e in effects if e.label == SUPPORTED]

    # 1b. evidence gate (v4 option): too few questions or too few discordant ones for any decision yet
    gate = policy.evidence_gate
    discordant = int(np.count_nonzero(D))
    enough_disc = gate is None or discordant >= gate.min_discordant
    if gate is not None and n < max_n and n < gate.min_n:
        return result(Decision.NEEDS_MORE_EVIDENCE, f"evidence gate: n={n} < min_n={gate.min_n}", strata=effects)
    if gate is not None:
        stats += f", discordant={discordant}"
    passed = passed and enough_disc

    # 2. HARMFUL
    harm_checked = (policy.harm.min_n is None or n >= policy.harm.min_n) and enough_disc
    if harm_checked and s.p_gain_lt_neg_floor >= policy.harm.p_gain_lt_neg_floor:
        return result(Decision.HARMFUL, f"P(Δ<−F) ≥ {policy.harm.p_gain_lt_neg_floor}", strata=effects)
    if cost_too_high and not passed:
        return result(Decision.HARMFUL, f"cost ratio {ev.cost_ratio:.2f} > {policy.harm.max_cost_ratio} "
                                        "and the pass rule is not met", strata=effects)

    # 3. PASS_GLOBAL
    if passed and not harmed and not cost_too_high:
        return result(Decision.PASS_GLOBAL, "pass rule met overall, no harmed stratum, cost ratio within limit",
                      strata=effects, held_out_replication="pending")

    # 4. PASS_CONDITIONAL
    if (harmed or not passed) and supported:
        f = policy.strata.field
        why = "stratum harmed: " + ", ".join(harmed) if passed else "pass rule not met overall"
        return result(Decision.PASS_CONDITIONAL, f"{why}; supported strata: {', '.join(supported)}",
                      strata=effects, held_out_replication="pending",
                      activation_conditions=[_condition(f, lv) for lv in supported],
                      inhibition_conditions=[_condition(f, lv) for lv in harmed],
                      parent_fallback=[_condition(f, lv) for lv in policy.strata.levels
                                       if lv not in supported and lv not in harmed]
                      + [{"field": f, "level": None}])

    # 5. UNSUPPORTED (with hold_while_negative, futility and curtailment hold while Δ̂ < 0)
    hold = policy.futility.hold_while_negative and s.delta_hat < 0
    if not hold and futility_met(s, policy.futility):
        return result(Decision.UNSUPPORTED, "futility rule met", strata=effects)
    if n >= max_n:
        return result(Decision.UNSUPPORTED, f"maximum n={max_n} reached without a pass", strata=effects)
    if not hold and policy.curtailment and not _can_still_pass(D, max_n, ev.floor_by_n, policy):
        return result(Decision.UNSUPPORTED, f"curtailed: even {max_n - n} more repairs could not meet the pass rule",
                      strata=effects)

    # 6. NEEDS_MORE_EVIDENCE
    if hold:
        return result(Decision.NEEDS_MORE_EVIDENCE, "undecided; Δ̂ < 0, futility and curtailment held (A3)",
                      strata=effects)
    return result(Decision.NEEDS_MORE_EVIDENCE, "undecided; another block can change the decision", strata=effects)


def _can_still_pass(D: np.ndarray, max_n: int, floor_by_n: dict[int, float], policy: SequentialBayesPolicy) -> bool:
    """Would the pass rule hold at max_n if every remaining question were a repair (D = +1)?"""
    best = np.concatenate([D, np.ones(max_n - len(D))])
    return pass_met(summarize(best, floor_at(floor_by_n, max_n), policy.posterior), policy.pass_rule)


def _decide_keep_if_higher(table: PairedTable) -> DecisionResult:
    acc_p, acc_c = float(np.mean(table.parent)), float(np.mean(table.candidate))
    why = f"candidate accuracy {acc_c:.3f} vs parent {acc_p:.3f} (n={table.n})"
    decision = Decision.PASS_GLOBAL if acc_c > acc_p else Decision.UNSUPPORTED
    return DecisionResult(decision, why, table.n, acc_c - acc_p,
                          held_out_replication="pending" if decision == Decision.PASS_GLOBAL else "not_applicable")


def _decide_fixed_repeat(ev: Evidence, policy: FixedRepeatPolicy) -> DecisionResult:
    if ev.legacy_floor is None:
        raise ValueError("fixed_repeat needs the legacy floor F_legacy (Evidence.legacy_floor)")
    d, F = delta_hat(ev.table), ev.legacy_floor
    why = f"Δ̂={d:+.3f}, F_legacy={F:.3f}, n={ev.table.n}, seeds={policy.seeds}"
    if d > 0 and d > F:
        return DecisionResult(Decision.PASS_GLOBAL, f"Δ̂ > 0 and Δ̂ > F_legacy ({why})", ev.table.n, d,
                              held_out_replication="pending")
    if d < -F:
        return DecisionResult(Decision.HARMFUL, f"Δ̂ < −F_legacy ({why})", ev.table.n, d)
    return DecisionResult(Decision.UNSUPPORTED, f"Δ̂ within the legacy floor ({why})", ev.table.n, d)
