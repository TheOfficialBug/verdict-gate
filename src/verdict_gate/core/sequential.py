"""Sequential evaluation in blocks.

Questions are visited in a pre-registered random order (order_seed). After each block of block_size questions,
decide() runs on all questions seen so far; evaluation stops at the first decision other than NEEDS_MORE_EVIDENCE.
With futility.hold_while_negative a run whose running Δ̂ < 0 is not stopped by futility or
curtailment, so it continues until HARMFUL, another rule decides, or max_n. The last block may be shorter (max n = 60 with blocks of 8: 7 blocks of 8 and one of 4).

`pool` (optional) is the pre-registered question pool, e.g. all 60 ids of a 60-question split. A live run that stops early stores only
the questions it executed, so replaying it must visit the pool's order (block_order(pool)) rather than an order drawn
over the stored ids, and must keep max_n and curtailment at the pool's size. Blocks then advance over pool positions;
a presented question with no valid pair (invalid after its retry) is skipped inside its block. Without a pool the
pool is the table's own ids, which is the behaviour of a complete stored run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from verdict_gate.core.decision import DecisionResult, Evidence, decide
from verdict_gate.core.policy import SequentialBayesPolicy
from verdict_gate.core.schemas import Decision, PairedTable


def block_order(task_ids: list[str], order_seed: int) -> list[str]:
    """Pre-registered visiting order: a seeded permutation of the sorted task IDs."""
    ids = sorted(task_ids)
    perm = np.random.default_rng(order_seed).permutation(len(ids))
    return [ids[i] for i in perm]


@dataclass
class SequentialResult:
    final: DecisionResult
    blocks_evaluated: int
    questions_used: int
    stopping_reason: str
    history: list[tuple[int, Decision]] = field(default_factory=list)  # (n after the block, decision)
    seen_ids: list[str] = field(default_factory=list)  # task ids of the final decision's table, in visiting order


def run_sequential(table: PairedTable, policy: SequentialBayesPolicy, floor_by_n: dict[int, float],
                   invalid_rate: float, pairing_ok: bool, cost_ratio: float | None = None,
                   cost_ratio_at: Callable[[list[str]], float | None] | None = None,
                   early_stop: bool = True, pool: list[str] | None = None) -> SequentialResult:
    """Replay the governed policy block by block over a full paired table (no model calls).

    cost_ratio_at (optional): maps the task IDs seen so far to the cost ratio on those questions, so the ratio is
    recomputed after every block; it overrides the fixed cost_ratio. early_stop=False (ablation) keeps evaluating
    blocks until max_n and returns the decision at max_n. pool: the pre-registered question pool (module docstring);
    questions_used then counts the pool questions presented.
    """
    position = {t: i for i, t in enumerate(table.task_ids)}
    if pool is None:
        pool = table.task_ids
    elif set(table.task_ids) - set(pool):
        raise ValueError(f"{len(set(table.task_ids) - set(pool))} table task(s) are not in the pool")
    visit = block_order(pool, policy.order_seed)
    max_n = min(policy.max_n, len(pool))
    history: list[tuple[int, Decision]] = []
    n = 0
    while True:
        n = min(n + policy.block_size, max_n)
        seen_ids = [t for t in visit[:n] if t in position]
        seen = table.subset([position[t] for t in seen_ids])
        ratio = cost_ratio_at(seen.task_ids) if cost_ratio_at is not None else cost_ratio
        evidence = Evidence(table=seen, invalid_rate=invalid_rate, pairing_ok=pairing_ok,
                            cost_ratio=ratio, floor_by_n=floor_by_n, available_n=len(pool))
        result = decide(evidence, policy)
        history.append((n, result.decision))
        if (early_stop and result.decision != Decision.NEEDS_MORE_EVIDENCE) or n >= max_n:
            return SequentialResult(result, len(history), n, result.reason, history, seen_ids)
