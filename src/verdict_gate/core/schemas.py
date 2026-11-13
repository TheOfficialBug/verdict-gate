"""Data schemas and validation rules.

Validation rules, each with a test in tests/core/test_schemas.py:
  1. score is None iff valid_execution is False                       (TaskOutcome)
  2. failure_type is present iff valid_execution is False             (TaskOutcome)
  3. pairing keys identical between arms                              (check_pairing)
  4. task_subset_hash matches the declared split file                 (check_task_subset_hash)
  5. no task appears in two splits                                    (check_splits_disjoint)
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from datetime import datetime
from enum import Enum
from itertools import combinations
from pathlib import Path

from pydantic import BaseModel, model_validator


class Decision(str, Enum):
    PASS_GLOBAL = "PASS_GLOBAL"
    PASS_CONDITIONAL = "PASS_CONDITIONAL"
    NEEDS_MORE_EVIDENCE = "NEEDS_MORE_EVIDENCE"
    UNSUPPORTED = "UNSUPPORTED"
    HARMFUL = "HARMFUL"


class RevisionType(str, Enum):
    SKILL = "skill"
    PROMPT = "prompt"
    MODEL = "model"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    MEMORY_RULE = "memory_rule"
    ROUTER = "router"
    AGENT_GRAPH = "agent_graph"
    DONE_CONDITION_CHECKER = "done_condition_checker"


class SchemaError(ValueError):
    """A validation rule of this module is broken (pairing, split hash, split overlap)."""


class TaskOutcome(BaseModel):
    """One execution of one task (one repeat). Retries are separate rows that point at the original via retry_of."""

    run_id: str
    task_id: str
    stratum: str | None
    model: str
    provider: str
    seed: int | None
    revision_id: str
    score: float | None  # None iff valid_execution is False
    valid_execution: bool
    failure_type: str | None  # required iff not valid
    retry_of: str | None
    tokens_in: int
    tokens_out: int
    tool_calls: int
    latency_s: float
    cost_usd: float

    @model_validator(mode="after")
    def _validity_rules(self) -> TaskOutcome:
        if self.valid_execution and self.score is None:
            raise ValueError("a valid execution must have a score")
        if not self.valid_execution and self.score is not None:
            raise ValueError("an invalid execution must not have a score (invalid is not wrong)")
        if self.valid_execution and self.failure_type is not None:
            raise ValueError("failure_type is only allowed on invalid executions")
        if not self.valid_execution and not self.failure_type:
            raise ValueError("an invalid execution needs a failure_type")
        return self


class RunManifest(BaseModel):
    run_id: str
    timestamp: datetime
    git_commit: str
    prereg_tag: str | None
    benchmark: str
    benchmark_version: str
    split: str
    task_subset_hash: str
    model: str
    openrouter_model_id: str
    provider: str
    seed: int | None
    temperature: float
    max_tokens: int
    revision_id: str
    parent_id: str | None
    revision_type: RevisionType
    policy_version: str | None
    retrieval_cache_version: str | None
    environment_version: str
    estimated_cost_usd: float
    actual_cost_usd: float
    cost_line_items: dict[str, float]  # e.g. {"model": ..., "search": ..., "grader": ..., "user_sim": ...}
    wall_time_s: float
    invalid_execution_count: int
    preliminary: bool  # True for single-run numbers
    checks: dict[str, bool]
    closed_hash: str | None


class PairedTable(BaseModel):
    """One row per task: parent and candidate per-question means over valid repeats."""

    task_ids: list[str]
    strata: list[str | None]
    parent: list[float]
    candidate: list[float]
    parent_repeats: list[int]
    candidate_repeats: list[int]
    unstable_parent: list[bool]
    unstable_candidate: list[bool]
    model: str
    context: dict

    @model_validator(mode="after")
    def _aligned(self) -> PairedTable:
        n = len(self.task_ids)
        columns = (self.strata, self.parent, self.candidate, self.parent_repeats, self.candidate_repeats,
                   self.unstable_parent, self.unstable_candidate)
        if any(len(col) != n for col in columns):
            raise ValueError("every PairedTable column must have one entry per task")
        if len(set(self.task_ids)) != n:
            raise ValueError("task_ids must be unique (one row per task)")
        if any(r < 1 for r in self.parent_repeats + self.candidate_repeats):
            raise ValueError("every row needs at least one valid repeat in each arm")
        if any(not 0.0 <= y <= 1.0 for y in self.parent + self.candidate):
            raise ValueError("per-question means must lie in [0, 1]")
        return self

    @property
    def n(self) -> int:
        return len(self.task_ids)

    def subset(self, indices: Sequence[int]) -> PairedTable:
        """Rows at `indices`, in that order (used for blocks and strata)."""
        pick = lambda col: [col[i] for i in indices]
        return PairedTable(
            task_ids=pick(self.task_ids), strata=pick(self.strata), parent=pick(self.parent),
            candidate=pick(self.candidate), parent_repeats=pick(self.parent_repeats),
            candidate_repeats=pick(self.candidate_repeats), unstable_parent=pick(self.unstable_parent),
            unstable_candidate=pick(self.unstable_candidate), model=self.model, context=dict(self.context),
        )


# ---- pairing (rule 3) -------------------------------------------------------------------------------------------

# Keys every run of both arms must share. The seed is checked separately: a candidate repeat must have a parent
# repeat with the same seed (fixed-repeat pairs candidate seeds {11,22,33} with parent seeds {11,...,55}).
SHARED_PAIRING_KEYS = ("model", "provider", "retrieval_cache_version", "max_tokens")


def check_pairing(parent_runs: Sequence[RunManifest], candidate_runs: Sequence[RunManifest],
                  parent_outcomes: Iterable[TaskOutcome] = (), candidate_outcomes: Iterable[TaskOutcome] = ()) -> None:
    """Raise SchemaError if the two arms are not paired on (task_id, model, provider, seed, cache version, max_tokens).

    Runs: the shared keys must be identical across every run of both arms, and every candidate seed must also be a
    parent seed. Outcomes (optional): each row must match its run's model/provider/seed, and for every candidate
    seed both arms must cover the same task IDs.
    """
    if not parent_runs or not candidate_runs:
        raise SchemaError("pairing needs at least one run per arm")
    runs = list(parent_runs) + list(candidate_runs)
    for key in SHARED_PAIRING_KEYS:
        values = {getattr(r, key) for r in runs}
        if len(values) != 1:
            raise SchemaError(f"pairing broken: {key} differs between runs ({sorted(map(str, values))})")
    parent_seeds = {r.seed for r in parent_runs}
    missing = {r.seed for r in candidate_runs} - parent_seeds
    if missing:
        raise SchemaError(f"pairing broken: candidate seeds {sorted(map(str, missing))} have no parent run")

    by_run = {r.run_id: r for r in runs}
    tasks: dict[tuple[str, int | None], set[str]] = {}
    for arm, outcomes in (("parent", parent_outcomes), ("candidate", candidate_outcomes)):
        for o in outcomes:
            run = by_run.get(o.run_id)
            if run is None:
                raise SchemaError(f"outcome {o.task_id} names run {o.run_id}, which is not in either arm")
            for key in ("model", "provider", "seed"):
                if getattr(o, key) != getattr(run, key):
                    raise SchemaError(f"pairing broken: outcome {o.task_id} {key} differs from run {o.run_id}")
            tasks.setdefault((arm, o.seed), set()).add(o.task_id)
    if tasks:
        for seed in {r.seed for r in candidate_runs}:
            if tasks.get(("parent", seed), set()) != tasks.get(("candidate", seed), set()):
                raise SchemaError(f"pairing broken: task IDs differ between arms for seed {seed}")


# ---- splits (rules 4 and 5) -------------------------------------------------------------------------------------


def task_subset_hash(task_ids: Iterable[str]) -> str:
    """SHA-256 of the sorted IDs, one per line with a trailing newline: the bytes of a configs/splits/*.txt file."""
    text = "\n".join(sorted(task_ids)) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_split(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").split()


def check_task_subset_hash(manifest: RunManifest, split_file: str | Path, task_ids: Iterable[str] | None = None) -> None:
    """The run's declared hash must equal the split file's hash (and, if given, the hash of the run's task IDs)."""
    file_hash = hashlib.sha256(Path(split_file).read_bytes()).hexdigest()
    if manifest.task_subset_hash != file_hash:
        raise SchemaError(f"run {manifest.run_id}: task_subset_hash does not match split file {split_file}")
    if task_ids is not None and task_subset_hash(task_ids) != file_hash:
        raise SchemaError(f"run {manifest.run_id}: its task IDs are not the tasks of split file {split_file}")


def check_splits_disjoint(splits: dict[str, Iterable[str]]) -> None:
    """No task may appear in two splits."""
    sets = {name: set(ids) for name, ids in splits.items()}
    for a, b in combinations(sorted(sets), 2):
        overlap = sets[a] & sets[b]
        if overlap:
            raise SchemaError(f"splits {a} and {b} share {len(overlap)} task(s), e.g. {sorted(overlap)[0]}")
