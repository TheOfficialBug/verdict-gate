"""Offline replay, live/replay agreement and ablations. No network, no model.

Stored arm file (one arm of a comparison: one revision, one or more seeds), JSON:

    {"revision_text_hash": "<sha256 of the revision text>",
     "manifests": [RunManifest, ...],          # one per run (seed)
     "outcomes":  [TaskOutcome, ...],          # every execution, invalid ones and retries included
     "pool":      [task_id, ...]}              # optional: the pre-registered question pool (sequential.py)

A candidate arm's manifests name its parent through parent_id; pair_arms() matches it to the arm whose revision_id
equals that parent_id. Which seeds each policy kind uses:

    keep_if_higher    both arms at policy.seed
    fixed_repeat      candidate at policy.seeds, parent at every seed it has (per-question means)
    sequential_bayes  candidate at every seed it has, parent at the candidate's seeds (seed 11 paired with seed 11)

evaluate() builds the paired table, checks pairing and the invalid rate, runs the policy (sequential: block by block
with the cost ratio recomputed on the questions seen) and returns the evidence card. replay() does that for every
comparison under one order seed; check_agreement() compares a live decision sequence with the replay; run_ablations()
runs the ablation variants from configs/ablations/*.yaml and returns one table. next_batch() drives a sequential
comparison one block at a time (stop, or which tasks to run next).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from verdict_gate.core.calibrate import load_fnoise
from verdict_gate.core.cost import cost_ratio_on, mean_cost_per_question, per_question_costs, sum_line_items
from verdict_gate.core.decision import DecisionResult, Evidence, EvidenceInvalid, decide
from verdict_gate.core.evidence_card import CardContext, build_card, file_hash
from verdict_gate.core.multiplicity import BASELINE_DECISIONS, holm, holm_adjusted, sign_flip_pvalue
from verdict_gate.core.paired import build_paired_table, differences, transitions
from verdict_gate.core.policy import (
    FixedRepeatPolicy,
    KeepIfHigherPolicy,
    Policy,
    SequentialBayesPolicy,
    _resolve,
    load_policy,
)
from verdict_gate.core.posterior import dirichlet_multinomial_fixture, prob_gt, prob_lt
from verdict_gate.core.schemas import Decision, PairedTable, RunManifest, SchemaError, TaskOutcome, check_pairing
from verdict_gate.core.sequential import block_order, run_sequential

FloorMode = Literal["policy", "zero", "legacy"]


# ---- stored arms ----------------------------------------------------------------------------------------------------


@dataclass
class Arm:
    manifests: list[RunManifest]
    outcomes: list[TaskOutcome]
    revision_text_hash: str | None = None
    path: str | None = None
    pool: list[str] | None = None  # pre-registered question pool; sequential replay visits block_order(pool)

    @property
    def revision_id(self) -> str:
        ids = {m.revision_id for m in self.manifests}
        if len(ids) != 1:
            raise SchemaError(f"arm {self.path}: runs of different revisions {sorted(ids)}")
        return ids.pop()

    @property
    def parent_id(self) -> str | None:
        return self.manifests[0].parent_id

    def at_seeds(self, seeds) -> Arm:
        seeds = set(seeds)
        return Arm([m for m in self.manifests if m.seed in seeds], [o for o in self.outcomes if o.seed in seeds],
                   self.revision_text_hash, self.path, self.pool)


def load_arm(path: str | Path) -> Arm:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Arm([RunManifest(**m) for m in raw["manifests"]], [TaskOutcome(**o) for o in raw["outcomes"]],
               raw.get("revision_text_hash"), str(path), raw.get("pool"))


def pair_arms(arms: list[Arm]) -> list[tuple[Arm, Arm]]:
    """(parent, candidate) for every arm whose parent_id names another arm, in input order."""
    by_id = {a.revision_id: a for a in arms}
    return [(by_id[a.parent_id], a) for a in arms if a.parent_id in by_id]


# ---- floors ---------------------------------------------------------------------------------------------------------


@dataclass
class Floors:
    paired: dict[int, float] = field(default_factory=dict)
    legacy: float | None = None
    file_hash: str | None = None
    source_run_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> Floors:
        rec = load_fnoise(path)
        return cls(rec["paired_fnoise"], rec.get("legacy_fnoise"), file_hash(path), list(rec.get("source_run_ids", [])))


NO_FLOOR = Floors()


# ---- one comparison -------------------------------------------------------------------------------------------------


@dataclass
class Prepared:
    parent: Arm
    candidate: Arm
    table: PairedTable
    invalid_rate: float
    parent_costs: dict[str, float]
    candidate_costs: dict[str, float]


def _seeds(arm: Arm) -> set:
    return {m.seed for m in arm.manifests}


def select_seeds(parent: Arm, candidate: Arm, policy: Policy) -> tuple[Arm, Arm]:
    if isinstance(policy, KeepIfHigherPolicy):
        return parent.at_seeds([policy.seed]), candidate.at_seeds([policy.seed])
    if isinstance(policy, FixedRepeatPolicy):
        return parent, candidate.at_seeds(policy.seeds)
    return parent.at_seeds(_seeds(candidate)), candidate


def invalid_rate(parent: Arm, candidate: Arm) -> float:
    """Share of (arm, task, seed) executions with no valid result after retries, over both arms."""
    keys, valid = set(), set()
    for arm, outcomes in (("p", parent.outcomes), ("c", candidate.outcomes)):
        for o in outcomes:
            keys.add((arm, o.task_id, o.seed))
            if o.valid_execution:
                valid.add((arm, o.task_id, o.seed))
    return (len(keys) - len(valid)) / len(keys) if keys else 0.0


def prepare(parent: Arm, candidate: Arm, policy: Policy) -> Prepared:
    parent, candidate = select_seeds(parent, candidate, policy)
    if not parent.manifests or not candidate.manifests:
        raise EvidenceInvalid(f"{candidate.revision_id}: no runs at the seeds policy {policy.name} uses")
    try:
        check_pairing(parent.manifests, candidate.manifests, parent.outcomes, candidate.outcomes)
    except SchemaError as e:
        raise EvidenceInvalid(str(e)) from None
    table = build_paired_table(parent.outcomes, candidate.outcomes)
    return Prepared(parent, candidate, table, invalid_rate(parent, candidate),
                    per_question_costs(parent.outcomes), per_question_costs(candidate.outcomes))


def _floor_by_n(policy: SequentialBayesPolicy, floors: Floors, mode: FloorMode) -> tuple[dict[int, float], str]:
    if mode == "zero":
        return {1: 0.0}, "none"
    if mode == "legacy" or policy.floor.source == "legacy_fnoise":
        if floors.legacy is None:
            raise ValueError("legacy floor requested but the fnoise file has no legacy_fnoise")
        return {1: floors.legacy}, "legacy_fnoise"
    if not floors.paired:
        raise ValueError("paired floor requested but no paired_fnoise values were loaded")
    return floors.paired, "paired_fnoise"


def _sorted_seeds(seeds) -> list:
    return sorted(seeds, key=lambda s: (s is None, s or 0))


def evaluate(parent: Arm, candidate: Arm, policy: Policy, floors: Floors = NO_FLOOR, *, floor_mode: FloorMode = "policy",
             early_stop: bool = True, git_commit: str = "unknown", created_at: str | None = None) -> dict:
    """Decide one candidate against its parent from stored artifacts and return its evidence card."""
    if candidate.revision_text_hash is None:
        raise SchemaError(f"arm {candidate.path}: revision_text_hash is missing")
    prep = prepare(parent, candidate, policy)
    table = prep.table
    if table.n == 0:
        raise EvidenceInvalid(f"{candidate.revision_id}: no valid paired questions")
    floor_value, floor_type, used = None, "none", table
    fnoise_hash, fnoise_runs = None, []

    if isinstance(policy, SequentialBayesPolicy):
        floor_by_n, floor_type = _floor_by_n(policy, floors, floor_mode)
        res = run_sequential(table, policy, floor_by_n, prep.invalid_rate, True,
                             cost_ratio_at=lambda ids: cost_ratio_on(prep.parent_costs, prep.candidate_costs, ids),
                             early_stop=early_stop, pool=candidate.pool)
        result = res.final
        position = {t: i for i, t in enumerate(table.task_ids)}
        used = table.subset([position[t] for t in res.seen_ids])
        sequential = {"blocks_evaluated": res.blocks_evaluated, "questions_used": res.questions_used,
                      "stopping_reason": res.stopping_reason}
    else:
        if isinstance(policy, FixedRepeatPolicy):
            if floors.legacy is None:
                raise ValueError("fixed_repeat needs legacy_fnoise in the fnoise file")
            floor_value, floor_type = floors.legacy, "legacy_fnoise"
        ratio = cost_ratio_on(prep.parent_costs, prep.candidate_costs, table.task_ids)
        result = decide(Evidence(table=table, invalid_rate=prep.invalid_rate, pairing_ok=True, cost_ratio=ratio,
                                 legacy_floor=floor_value), policy)
        sequential = {"blocks_evaluated": 1, "questions_used": table.n,
                      "stopping_reason": f"fixed design: all {table.n} paired questions evaluated"}
    if floor_type != "none":
        fnoise_hash, fnoise_runs = floors.file_hash, floors.source_run_ids

    p_q = mean_cost_per_question(prep.parent_costs, used.task_ids)
    c_q = mean_cost_per_question(prep.candidate_costs, used.task_ids)
    cm = prep.candidate.manifests[0]
    ctx = CardContext(
        revision_id=prep.candidate.revision_id, parent_id=prep.parent.revision_id,
        revision_type=cm.revision_type.value, revision_text_hash=candidate.revision_text_hash,
        evaluation_context={"benchmark": cm.benchmark, "split": cm.split, "model": cm.model, "provider": cm.provider,
                            "seeds": _sorted_seeds(_seeds(prep.candidate)),
                            "retrieval_cache_version": cm.retrieval_cache_version, "max_tokens": cm.max_tokens},
        git_commit=git_commit,
        prereg_tag=next((m.prereg_tag for m in prep.candidate.manifests if m.prereg_tag), None),
        created_at=created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        source_run_ids=[m.run_id for m in prep.parent.manifests + prep.candidate.manifests],
        preliminary=len(prep.candidate.manifests) == 1,
        floor_type=floor_type, fnoise_file_hash=fnoise_hash, fnoise_source_run_ids=fnoise_runs,
        costs={"parent_usd_per_q": p_q, "candidate_usd_per_q": c_q,
               "cost_ratio": cost_ratio_on(prep.parent_costs, prep.candidate_costs, used.task_ids),
               "line_items": {"parent": sum_line_items(m.cost_line_items for m in prep.parent.manifests),
                              "candidate": sum_line_items(m.cost_line_items for m in prep.candidate.manifests)}},
        sequential=sequential, floor_value=floor_value,
    )
    return build_card(result, policy, used, ctx)


def with_order_seed(policy: Policy, order_seed: int | None) -> Policy:
    if order_seed is None or not isinstance(policy, SequentialBayesPolicy):
        return policy
    return policy.model_copy(update={"order_seed": order_seed})


def replay(policy: Policy, arms: list[Arm], order_seed: int | None, floors_by_model: dict[str, Floors] | None = None,
           **kwargs) -> list[dict]:
    """Evidence cards for every (parent, candidate) pair in `arms`, recomputed from the stored artifacts."""
    policy = with_order_seed(policy, order_seed)
    floors_by_model = floors_by_model or {}
    return [evaluate(p, c, policy, floors_by_model.get(c.manifests[0].model, NO_FLOOR), **kwargs)
            for p, c in pair_arms(arms)]


# ---- one block at a time (live runs, next_batch) --------------------------------------------------------------------


def floor_by_n(policy: SequentialBayesPolicy, floors: Floors) -> dict[int, float]:
    return _floor_by_n(policy, floors, "policy")[0]


def decide_block(parent: Arm, candidate: Arm, presented: list[str], policy: SequentialBayesPolicy, floors: Floors,
                 pool_size: int) -> DecisionResult:
    """The decision on the questions presented so far (rows in visiting order), from the executed outcomes."""
    try:
        check_pairing(parent.manifests, candidate.manifests, parent.outcomes, candidate.outcomes)
    except SchemaError as e:
        raise EvidenceInvalid(str(e)) from None
    table = build_paired_table(parent.outcomes, candidate.outcomes)
    position = {t: i for i, t in enumerate(table.task_ids)}
    table = table.subset([position[t] for t in presented if t in position])
    ratio = cost_ratio_on(per_question_costs(parent.outcomes), per_question_costs(candidate.outcomes), table.task_ids)
    return decide(Evidence(table=table, invalid_rate=invalid_rate(parent, candidate), pairing_ok=True,
                           cost_ratio=ratio, floor_by_n=floor_by_n(policy, floors), available_n=pool_size), policy)


@dataclass
class NextBatch:
    decision: str  # decision on the questions run so far (NEEDS_MORE_EVIDENCE before the first block)
    reason: str
    questions_done: int  # tasks (a prefix of the visiting order) the decision is based on
    task_ids: list[str]  # run these next, in both arms; [] = stop, the decision is final

    @property
    def stop(self) -> bool:
        return not self.task_ids


def next_batch(policy: SequentialBayesPolicy, pool: list[str], parent: Arm | None = None,
               candidate: Arm | None = None, floors: Floors = NO_FLOOR) -> NextBatch:
    """Run a sequential comparison in batches: stop, or which tasks to run next.

    The policy visits the pool in a fixed, pre-registered order (block_order(pool, policy.order_seed)), block_size
    tasks at a time, up to max_n. Run the returned task_ids in both arms (same seed), append the rows to your results
    file, re-import, and call again. Without arms (nothing run yet) it returns the first block. The decision after each
    complete block is the same one `verdict compare` gives on the final files when the arms carry the pool.
    """
    if not isinstance(policy, SequentialBayesPolicy):
        raise TypeError("batches are for a sequential_bayes policy (e.g. moderate or strict)")
    visit = block_order(pool, policy.order_seed)
    max_n = min(policy.max_n, len(visit))
    done = set()
    if parent is not None and candidate is not None:
        parent, candidate = select_seeds(parent, candidate, policy)
        done = {o.task_id for o in parent.outcomes} & {o.task_id for o in candidate.outcomes}
    n = 0
    while n < max_n and visit[n] in done:
        n += 1
    stray = sorted(done - set(visit[:n]))
    if stray:
        raise EvidenceInvalid(f"{len(stray)} task(s) were run out of the visiting order (e.g. {stray[0]}); "
                              "run the tasks next_batch returns, in both arms")
    block_end = min((n // policy.block_size + 1) * policy.block_size, max_n)
    if n == 0:
        return NextBatch(Decision.NEEDS_MORE_EVIDENCE.value, "nothing run yet", 0, visit[:block_end])
    # Decide after every complete block, as the sequential policy does, so extra tasks run past a final decision
    # never change it. The parent may hold more tasks (e.g. a full calibration run): pair on the prefix only.
    ends = list(range(policy.block_size, n + 1, policy.block_size)) + ([n] if n == max_n and n % policy.block_size
                                                                          else [])
    for end in ends:
        seen = set(visit[:end])
        p, c = (Arm(a.manifests, [o for o in a.outcomes if o.task_id in seen], a.revision_text_hash, a.path, a.pool)
                for a in (parent, candidate))
        result = decide_block(p, c, visit[:end], policy, floors, len(visit))
        if result.decision != Decision.NEEDS_MORE_EVIDENCE or end >= max_n:
            return NextBatch(result.decision.value, result.reason, end, [])
    if n % policy.block_size:
        return NextBatch(Decision.NEEDS_MORE_EVIDENCE.value, "current block incomplete", n, visit[n:block_end])
    return NextBatch(result.decision.value, result.reason, n, visit[n:block_end])


AGREEMENT_FIELDS = (("decision",), ("sequential", "questions_used"), ("sequential", "blocks_evaluated"))


def check_agreement(live_cards: list[dict], replay_cards: list[dict]) -> list[str]:
    """Stop rule 7: the live decision sequence must equal replay on the same artifacts. Returns the mismatches
    (decision, stopping point, blocks) per revision; [] means they agree."""
    def get(card, path):
        for key in path:
            card = card[key]
        return card

    live = {c["revision_id"]: c for c in live_cards}
    rep = {c["revision_id"]: c for c in replay_cards}
    problems = [f"{r}: in live but not in replay" for r in live if r not in rep]
    problems += [f"{r}: in replay but not in live" for r in rep if r not in live]
    for r in [r for r in live if r in rep]:
        for path in AGREEMENT_FIELDS:
            a, b = get(live[r], path), get(rep[r], path)
            if a != b:
                problems.append(f"{r}: {'.'.join(path)} live={a} replay={b}")
    return problems


# ---- ablations -------------------------------------------------------------------------------------------------------


class OrderSeeds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: int
    count: int


class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    # accept_all / reject_all: offline reference rows (multiplicity.BASELINE_DECISIONS); no evidence is used.
    # regression_bounded: SkillBoost gate (arXiv:2607.26643 Eq. 3, eps as in EvoSkill v2 App. 2; reports/evoskill_replay.md)
    method: Literal["sequential", "holm_sign_flip", "dirichlet_multinomial", "accept_all", "reject_all",
                    "regression_bounded"] = "sequential"
    eps: float = 0.05  # regression_bounded: PASS_GLOBAL iff R+ − R− > 0 and R− / n < eps
    policy: str | None = None  # another policy file (relative to the repo root); default = base_policy
    overrides: dict = {}  # field overrides on the policy, e.g. {block_size: 4}
    floor: FloorMode = "policy"
    early_stop: bool = True
    order_seeds: OrderSeeds | list[int] | None = None  # default: the policy's own order_seed
    alpha: float = 0.05  # holm_sign_flip
    n_perm: int = 100_000  # holm_sign_flip (only when n > 20)
    rng_seed: int = 0  # holm_sign_flip / dirichlet_multinomial
    draws: int = 10_000  # dirichlet_multinomial
    p_pass: float = 0.95  # dirichlet_multinomial: P(Δ>0) needed to admit
    p_harm: float = 0.95  # dirichlet_multinomial: P(Δ<0) that makes it HARMFUL


class AblationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    version: int
    base_policy: str
    variants: list[Variant]


TABLE_COLUMNS = ("variant", "order_seed", "revision_id", "decision", "questions_used", "blocks_evaluated",
                 "delta_hat", "p_gain_gt_0", "p_value", "p_holm", "reason")


def _variant_policy(v: Variant, base: Policy, root: Path) -> Policy:
    policy = load_policy(_resolve(root, v.policy), root=root) if v.policy else base
    if v.overrides:
        policy = type(policy)(**{**policy.model_dump(by_alias=True), **v.overrides})
    return policy


def _order_seeds(v: Variant, policy: Policy) -> list[int | None]:
    if isinstance(v.order_seeds, OrderSeeds):
        return list(range(v.order_seeds.start, v.order_seeds.start + v.order_seeds.count))
    if isinstance(v.order_seeds, list):
        return list(v.order_seeds)
    return [getattr(policy, "order_seed", None)]


def _full_design(prep: Prepared, policy: Policy) -> PairedTable:
    """The first max_n questions in the policy's visiting order (the whole design, no early stop)."""
    table = prep.table
    if not isinstance(policy, SequentialBayesPolicy):
        return table
    position = {t: i for i, t in enumerate(table.task_ids)}
    order = block_order(table.task_ids, policy.order_seed)[:min(policy.max_n, table.n)]
    return table.subset([position[t] for t in order])


def _row(variant, order_seed, revision_id, decision, n, blocks, delta, p_gt0=None, p=None, p_holm=None, reason=""):
    return {"variant": variant, "order_seed": order_seed, "revision_id": revision_id, "decision": decision,
            "questions_used": n, "blocks_evaluated": blocks, "delta_hat": round(delta, 6),
            "p_gain_gt_0": None if p_gt0 is None else round(p_gt0, 6), "p_value": None if p is None else round(p, 6),
            "p_holm": None if p_holm is None else round(p_holm, 6), "reason": reason}


def run_ablations(cfg: AblationConfig, arms: list[Arm], floors_by_model: dict[str, Floors], root: str | Path) -> list[dict]:
    """One row per variant × order seed × candidate. All comparisons in `arms` form one round (for Holm)."""
    root = Path(root)
    base = load_policy(_resolve(root, cfg.base_policy), root=root)
    pairs = pair_arms(arms)
    rows: list[dict] = []
    for v in cfg.variants:
        policy = _variant_policy(v, base, root)
        for seed in _order_seeds(v, policy):
            pol = with_order_seed(policy, seed)
            if v.method == "sequential":
                for p, c in pairs:
                    card = evaluate(p, c, pol, floors_by_model.get(c.manifests[0].model, NO_FLOOR),
                                    floor_mode=v.floor, early_stop=v.early_stop, created_at="replay")
                    rows.append(_row(v.name, seed, card["revision_id"], card["decision"],
                                     card["sequential"]["questions_used"], card["sequential"]["blocks_evaluated"],
                                     card["effect"]["delta_hat"], card["effect"]["p_gain_gt_0"],
                                     reason=card["decision_reason"]))
                continue
            tables = [(c.revision_id, _full_design(prepare(p, c, pol), pol)) for p, c in pairs]
            if v.method in BASELINE_DECISIONS:  # reference row: same decision for every candidate, 0 questions used
                style = " (EvoSkill v2-style, no admission control)" if v.method == "accept_all" else ""
                for rid, t in tables:
                    rows.append(_row(v.name, seed, rid, BASELINE_DECISIONS[v.method], 0, 0,
                                     float(differences(t).mean()),
                                     reason=f"reference policy {v.method}{style}: every candidate "
                                            f"{BASELINE_DECISIONS[v.method]} (Δ̂ shown on the full design)"))
            elif v.method == "regression_bounded":  # SkillBoost: gain > 0 and regressions / n < eps; never HARMFUL
                for rid, t in tables:
                    tr = transitions(t)
                    ok = tr.repairs - tr.regressions > 0 and tr.regressions / t.n < v.eps
                    decision = Decision.PASS_GLOBAL if ok else Decision.UNSUPPORTED
                    rows.append(_row(v.name, seed, rid, decision.value, t.n, 1, float(differences(t).mean()),
                                     reason=f"R+={tr.repairs} R-={tr.regressions} n={t.n} eps={v.eps}"))
            elif v.method == "holm_sign_flip":
                pvals = [sign_flip_pvalue(differences(t), n_perm=v.n_perm, rng_seed=v.rng_seed) for _, t in tables]
                adjusted, rejected = holm_adjusted(pvals), holm(pvals, v.alpha)
                for (rid, t), p, a, rej in zip(tables, pvals, adjusted, rejected):
                    decision = Decision.PASS_GLOBAL if rej else Decision.UNSUPPORTED
                    rows.append(_row(v.name, seed, rid, decision.value, t.n, 1, float(differences(t).mean()),
                                     p=p, p_holm=a, reason=f"Holm across {len(tables)} candidates, alpha={v.alpha}"))
            else:  # dirichlet_multinomial: transitions, no floor
                for rid, t in tables:
                    tr = transitions(t)
                    post = dirichlet_multinomial_fixture(tr.repairs, tr.regressions,
                                                         t.n - tr.repairs - tr.regressions, B=v.draws,
                                                         rng_seed=v.rng_seed)
                    p_gt, p_lt = prob_gt(post, 0.0), prob_lt(post, 0.0)
                    decision = (Decision.PASS_GLOBAL if p_gt >= v.p_pass else
                                Decision.HARMFUL if p_lt >= v.p_harm else Decision.UNSUPPORTED)
                    rows.append(_row(v.name, seed, rid, decision.value, t.n, 1, float(differences(t).mean()),
                                     p_gt0=p_gt, reason=f"Dirichlet(1+R+, 1+R-, 1+U); P(Δ<0)={p_lt:.3f}"))
    return rows


def load_ablation_config(path: str | Path) -> AblationConfig:
    return AblationConfig(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def write_table(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
