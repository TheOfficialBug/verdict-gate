"""Hand-built fixtures for the analysis-core tests. Everything is synthetic; nothing touches runs/ or the network."""

from datetime import datetime
from pathlib import Path

from verdict_gate.core.schemas import PairedTable, RunManifest, TaskOutcome

ROOT = Path(__file__).resolve().parents[2]  # repository root
DATA = ROOT / "src" / "verdict_gate" / "_data"  # bundled policies, schema, strata, example data
FIXTURES = ROOT / "tests" / "fixtures"
NEVER, SLOW, FAST = "never-changing", "slow-changing", "fast-changing"


def make_table(pairs, strata=None, repeats=1, model="m") -> PairedTable:
    """pairs = [(parent_mean, candidate_mean), ...]; a mean strictly between 0 and 1 means the repeats disagree."""
    n = len(pairs)
    strata = strata if isinstance(strata, list) else [strata] * n
    return PairedTable(
        task_ids=[f"t{i:03d}" for i in range(n)], strata=strata,
        parent=[float(p) for p, _ in pairs], candidate=[float(c) for _, c in pairs],
        parent_repeats=[repeats] * n, candidate_repeats=[repeats] * n,
        unstable_parent=[0 < p < 1 for p, _ in pairs], unstable_candidate=[0 < c < 1 for _, c in pairs],
        model=model, context={},
    )


def outcomes(repairs=0, regressions=0, preserved=0, unresolved=0):
    """Binary single-repeat (parent, candidate) pairs: D = +1, −1, 0 (both right), 0 (both wrong)."""
    return [(0, 1)] * repairs + [(1, 0)] * regressions + [(1, 1)] * preserved + [(0, 0)] * unresolved


def stratified(*groups):
    """groups = (level, pairs), ...  →  one table with strata in group order."""
    pairs, strata = [], []
    for level, group in groups:
        pairs += group
        strata += [level] * len(group)
    return make_table(pairs, strata)


def outcome(**kw) -> TaskOutcome:
    base = dict(run_id="R1", task_id="q1", stratum="never-changing", model="m", provider="p", seed=11,
                revision_id="s0", score=1.0, valid_execution=True, failure_type=None, retry_of=None,
                tokens_in=10, tokens_out=5, tool_calls=1, latency_s=1.0, cost_usd=0.001)
    return TaskOutcome(**{**base, **kw})


def manifest(**kw) -> RunManifest:
    base = dict(run_id="R1", timestamp=datetime(2026, 11, 2), git_commit="abc", prereg_tag=None, benchmark="sealqa",
                benchmark_version="v1", split="V60", task_subset_hash="x", model="m", openrouter_model_id="m/id",
                provider="p", seed=11, temperature=0.0, max_tokens=4096, revision_id="s0", parent_id=None,
                revision_type="skill", policy_version=None, retrieval_cache_version="c1", environment_version="e1",
                estimated_cost_usd=0.1, actual_cost_usd=0.1, cost_line_items={"model": 0.1}, wall_time_s=10.0,
                invalid_execution_count=0, preliminary=True, checks={}, closed_hash=None)
    return RunManifest(**{**base, **kw})
