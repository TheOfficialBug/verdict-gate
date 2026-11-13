"""Task scope: per-stratum effects and conditional routes.

Only the field named in the strata config is used (e.g. `freshness` for a search-QA benchmark). Descriptive tags such as
`question_types` or `topic` never reach this module. Labels use the policy's own thresholds:

    harmful      n_s ≥ harm_min_n    and P(Δ_s < −F(n_s)) ≥ harm_p_lt_neg_floor   (F = 0 without a floor)
    supported    n_s ≥ support_min_n and the policy's pass rule holds on the stratum's questions (floor F(n_s))
    inconclusive otherwise

label() is that rule for any set of paired questions; the evidence card also uses it for whole domains
(scope.domains_tested), so a domain and a stratum are labelled by identical thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

from verdict_gate.core.calibrate import floor_at
from verdict_gate.core.paired import differences, transitions
from verdict_gate.core.policy import SequentialBayesPolicy, StrataConfig, Summary, pass_met, summarize
from verdict_gate.core.schemas import PairedTable, SchemaError

SUPPORTED, INCONCLUSIVE, HARMFUL = "supported", "inconclusive", "harmful"


@dataclass(frozen=True)
class StratumEffect:
    field: str
    level: str
    n: int
    delta_hat: float | None
    floor: float | None
    p_gain_gt_0: float | None
    p_gain_gt_floor: float | None
    p_gain_lt_0: float | None
    p_gain_lt_neg_floor: float | None
    repairs: int
    regressions: int
    label: str


def label(n: int, s: Summary, policy: SequentialBayesPolicy) -> str:
    """harmful / supported / inconclusive for n paired questions with posterior summary s (floor = s.floor)."""
    sub = policy.subgroup
    if n >= sub.harm_min_n and s.p_gain_lt_neg_floor >= sub.harm_p_lt_neg_floor:
        return HARMFUL
    if n >= sub.support_min_n and pass_met(s, policy.pass_rule):
        return SUPPORTED
    return INCONCLUSIVE


def check_strata(table: PairedTable, strata: StrataConfig) -> None:
    """Every stratum value in the table must be a level of the configured field (None = unstratified)."""
    unknown = {s for s in table.strata if s is not None and s not in strata.levels}
    if unknown:
        raise SchemaError(f"strata {sorted(unknown)} are not levels of {strata.field!r} in the strata config")


def stratum_effects(table: PairedTable, strata: StrataConfig, policy: SequentialBayesPolicy,
                    floor_by_n: dict[int, float]) -> list[StratumEffect]:
    """One StratumEffect per configured level, in config order (levels without questions are inconclusive)."""
    check_strata(table, strata)
    effects = []
    for level in strata.levels:
        idx = [i for i, s in enumerate(table.strata) if s == level]
        if not idx:
            effects.append(StratumEffect(strata.field, level, 0, None, None, None, None, None, None, 0, 0,
                                         INCONCLUSIVE))
            continue
        part = table.subset(idx)
        s = summarize(differences(part), floor_at(floor_by_n, part.n), policy.posterior)
        t = transitions(part)
        effects.append(StratumEffect(strata.field, level, part.n, s.delta_hat, s.floor, s.p_gain_gt_0,
                                     s.p_gain_gt_floor, s.p_gain_lt_0, s.p_gain_lt_neg_floor, t.repairs, t.regressions,
                                     label(part.n, s, policy)))
    return effects


def route(table: PairedTable, routing_cfg: dict, strata: StrataConfig) -> list[float]:
    """Conditional-route outcome per question: candidate in the routed strata, parent everywhere else.

    routing_cfg = {"field": <strata field>, "candidate_strata": [levels]}. Uses only pre-execution features and the
    two global runs already in the table, so no model calls.
    """
    if routing_cfg["field"] != strata.field:
        raise SchemaError(f"routing field {routing_cfg['field']!r} is not the strata field {strata.field!r}")
    chosen = set(routing_cfg["candidate_strata"])
    if not chosen <= set(strata.levels):
        raise SchemaError(f"routing levels {sorted(chosen - set(strata.levels))} are not in the strata config")
    check_strata(table, strata)
    return [c if s in chosen else p for s, p, c in zip(table.strata, table.parent, table.candidate)]
