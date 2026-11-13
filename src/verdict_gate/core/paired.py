"""Paired effect and transitions.

    D_i        = Y_cand,i − Y_parent,i
    Δ̂          = (1/n) Σ D_i
    R₊         = #{parent incorrect → candidate correct}          (repairs)
    R₋         = #{parent correct → candidate incorrect}          (regressions)
    Preserved  = #{correct → correct};  Unresolved = #{incorrect → incorrect}
    Unstable   = #{either arm tagged unstable}
    NetRepair  = R₊ − R₋
    Binary single-repeat identity: Δ̂ = (R₊ − R₋)/n

Correct/incorrect per question = majority of its valid repeats (mean above / below one half). An exact tie has no
majority: it is tagged unstable and counted as unresolved, so R₊ + R₋ + Preserved + Unresolved = n always holds.
Any question whose repeats disagree is also tagged unstable; Unstable is a tag count that overlaps the four
transition classes, not a fifth class.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from verdict_gate.core.schemas import PairedTable, SchemaError, TaskOutcome

CORRECT, INCORRECT, TIE = "correct", "incorrect", "tie"


def differences(table: PairedTable) -> np.ndarray:
    """D_i = Y_cand,i − Y_parent,i."""
    return np.asarray(table.candidate, dtype=float) - np.asarray(table.parent, dtype=float)


def delta_hat(table: PairedTable) -> float:
    """Δ̂ = mean(D_i)."""
    if table.n == 0:
        raise ValueError("Δ̂ needs at least one paired question")
    return float(np.mean(differences(table)))


def majority_label(mean: float) -> str:
    """More than half of the valid repeats correct → correct; fewer than half → incorrect; exactly half → tie."""
    if mean > 0.5:
        return CORRECT
    if mean < 0.5:
        return INCORRECT
    return TIE


@dataclass(frozen=True)
class Transitions:
    n: int
    repairs: int  # R₊
    regressions: int  # R₋
    preserved: int
    unresolved: int  # includes questions with a majority tie in either arm
    unstable: int  # either arm tagged unstable (overlaps the four classes above)
    ties: int  # questions with a majority tie in either arm (already counted in unresolved)
    parent_correct: int  # denominator of the conditional regression rate

    @property
    def net_repair(self) -> int:
        return self.repairs - self.regressions


def transitions(table: PairedTable) -> Transitions:
    repairs = regressions = preserved = unresolved = unstable = ties = parent_correct = 0
    for p, c, up, uc in zip(table.parent, table.candidate, table.unstable_parent, table.unstable_candidate):
        lp, lc = majority_label(p), majority_label(c)
        unstable += up or uc or lp == TIE or lc == TIE
        parent_correct += lp == CORRECT
        if lp == TIE or lc == TIE:
            ties += 1
            unresolved += 1
        elif lp == INCORRECT and lc == CORRECT:
            repairs += 1
        elif lp == CORRECT and lc == INCORRECT:
            regressions += 1
        elif lp == CORRECT:
            preserved += 1
        else:
            unresolved += 1
    return Transitions(table.n, repairs, regressions, preserved, unresolved, unstable, ties, parent_correct)


# ---- building a table from task outcomes ----------------------------------------------------------------------


def _valid_scores(outcomes: Iterable[TaskOutcome]) -> tuple[dict, set, dict]:
    """(task, seed) -> score of its valid execution; the (task, seed) keys with no valid execution; task -> strata."""
    scores: dict[tuple[str, int | None], float] = {}
    seen: set[tuple[str, int | None]] = set()
    strata: dict[str, set] = {}
    for o in outcomes:
        key = (o.task_id, o.seed)
        seen.add(key)
        strata.setdefault(o.task_id, set()).add(o.stratum)
        if o.valid_execution:
            if key in scores:
                raise SchemaError(f"two valid executions of task {o.task_id} with seed {o.seed}")
            scores[key] = float(o.score)
    return scores, seen - set(scores), strata


def build_paired_table(parent_outcomes: Iterable[TaskOutcome], candidate_outcomes: Iterable[TaskOutcome],
                       context: dict | None = None) -> PairedTable:
    """Per-question means over valid repeats.

    A (task, seed) with no valid execution after its retry is excluded from both arms for that seed. A task left
    with no valid repeat in either arm is dropped. Exclusions are listed in context["excluded"].
    """
    parent_outcomes, candidate_outcomes = list(parent_outcomes), list(candidate_outcomes)
    models = {o.model for o in parent_outcomes + candidate_outcomes}
    if len(models) != 1:
        raise SchemaError(f"a paired table needs one model, got {sorted(models)}")
    p_scores, p_invalid, p_strata = _valid_scores(parent_outcomes)
    c_scores, c_invalid, c_strata = _valid_scores(candidate_outcomes)
    excluded = p_invalid | c_invalid

    rows = {name: [] for name in ("task_ids", "strata", "parent", "candidate", "parent_repeats",
                                  "candidate_repeats", "unstable_parent", "unstable_candidate")}
    dropped = []
    for task in sorted(set(p_strata) | set(c_strata)):
        levels = p_strata.get(task, set()) | c_strata.get(task, set())
        if len(levels) != 1:
            raise SchemaError(f"task {task} has inconsistent strata {sorted(map(str, levels))}")
        ys_p = [y for (t, s), y in sorted(p_scores.items(), key=str) if t == task and (t, s) not in excluded]
        ys_c = [y for (t, s), y in sorted(c_scores.items(), key=str) if t == task and (t, s) not in excluded]
        if not ys_p or not ys_c:
            dropped.append(task)
            continue
        rows["task_ids"].append(task)
        rows["strata"].append(next(iter(levels)))
        rows["parent"].append(float(np.mean(ys_p)))
        rows["candidate"].append(float(np.mean(ys_c)))
        rows["parent_repeats"].append(len(ys_p))
        rows["candidate_repeats"].append(len(ys_c))
        rows["unstable_parent"].append(len(set(ys_p)) > 1)
        rows["unstable_candidate"].append(len(set(ys_c)) > 1)
    ctx = dict(context or {})
    ctx["excluded"] = [f"{t}@seed{s}" for t, s in sorted(excluded, key=str)]
    ctx["dropped_tasks"] = dropped
    return PairedTable(model=models.pop(), context=ctx, **rows)
