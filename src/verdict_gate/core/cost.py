"""Cost of a decision.

    cost_ratio              = mean cost/question (candidate) / mean cost/question (parent)
    questions_per_decision  = total paired questions evaluated / candidate decisions
    cost_per_verified       = total evolution + evaluation cost / verified admitted revisions   (0 verified → None)

Cost per question (per arm) = every execution of that task in that arm, invalid ones and retries included, divided by
the number of seeds the task was run with. Keeping one number per task lets the cost ratio be computed on any subset
of questions, e.g. the questions seen after each sequential block.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from verdict_gate.core.schemas import TaskOutcome


def per_question_costs(outcomes: Iterable[TaskOutcome]) -> dict[str, float]:
    """task_id -> USD per question: total cost of all its executions / number of distinct seeds."""
    total: dict[str, float] = {}
    seeds: dict[str, set] = {}
    for o in outcomes:
        total[o.task_id] = total.get(o.task_id, 0.0) + o.cost_usd
        seeds.setdefault(o.task_id, set()).add(o.seed)
    return {t: total[t] / len(seeds[t]) for t in total}


def mean_cost_per_question(costs: dict[str, float], task_ids: Sequence[str]) -> float | None:
    """Mean USD per question over task_ids (None if no question has a cost)."""
    values = [costs[t] for t in task_ids if t in costs]
    return float(np.mean(values)) if values else None


def cost_ratio(parent_usd_per_q: float | None, candidate_usd_per_q: float | None) -> float | None:
    """Candidate / parent mean cost per question. None when either is unknown or the parent cost is 0."""
    if parent_usd_per_q is None or candidate_usd_per_q is None or parent_usd_per_q <= 0:
        return None
    return candidate_usd_per_q / parent_usd_per_q


def cost_ratio_on(parent_costs: dict[str, float], candidate_costs: dict[str, float],
                  task_ids: Sequence[str]) -> float | None:
    """Cost ratio on a subset of questions (e.g. the questions seen after a block)."""
    return cost_ratio(mean_cost_per_question(parent_costs, task_ids), mean_cost_per_question(candidate_costs, task_ids))


def questions_per_decision(total_questions: int, decisions: int) -> float | None:
    """Total paired questions evaluated / number of candidate decisions (None if there were no decisions)."""
    if decisions <= 0:
        return None
    return total_questions / decisions


def cost_per_verified(evolution_usd: float, evaluation_usd: float, verified_admitted: int) -> float | None:
    """(Evolution + evaluation cost) / verified admitted revisions. None (not 0, not inf) when none was verified."""
    if verified_admitted <= 0:
        return None
    return (evolution_usd + evaluation_usd) / verified_admitted


def sum_line_items(items: Iterable[dict[str, float]]) -> dict[str, float]:
    """Add up cost line items ({"model": ..., "search": ...}) across runs."""
    out: dict[str, float] = {}
    for d in items:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + float(v)
    return dict(sorted(out.items()))
