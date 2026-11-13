"""Writes the synthetic replay fixture in this folder (run once; the outputs are committed and never edited).

    uv run python tests/fixtures/replay/make_fixtures.py

24 questions, model "m", seeds 11/22/33, parent s0 and four candidates:
    c_good    repairs most of the parent's failures              (cost 1.2x)
    c_bad     breaks every one of the parent's successes         (cost 1.0x; harm beyond F(8))
    c_null    same outcomes as the parent                        (cost 1.0x)
    c_costly  same outcomes as the parent                        (cost 3.0x)
    c_mixed   6 repairs and 4 regressions                       (cost 1.0x; needs more than one block)
Parent task q05 seed 33 fails once (invalid) and succeeds on retry. live_cards.json is the decision sequence of a
simulated live run (tests/core/test_core_replay.py::simulate_live), stored so replay can be checked against it.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (11, 22, 33)
LEVELS = ("never-changing", "slow-changing", "fast-changing")
N = 24


def parent_score(i, seed):
    if i == 1 and seed == 22:
        return 1.0  # one unstable question
    return 1.0 if i % 2 == 0 else 0.0


SCORES = {
    "s0": parent_score,
    "c_good": lambda i, seed: 0.0 if i == 3 else 1.0,
    "c_bad": lambda i, seed: 0.0,
    "c_null": parent_score,
    "c_costly": parent_score,
    "c_mixed": lambda i, seed: 1.0 if i % 4 == 1 else 0.0 if i in (0, 4, 12, 16) else parent_score(i, seed),
}
COST = {"s0": 0.001, "c_good": 0.0012, "c_bad": 0.001, "c_null": 0.001, "c_costly": 0.003, "c_mixed": 0.001}


def manifest(rev, seed):
    return {"run_id": f"FX_{rev}_s{seed}", "timestamp": "2026-11-03T00:00:00", "git_commit": "fixture",
            "prereg_tag": None, "benchmark": "sealqa", "benchmark_version": "fixture", "split": "V24",
            "task_subset_hash": "fixture", "model": "m", "openrouter_model_id": "fixture/m", "provider": "p",
            "seed": seed, "temperature": 0.0, "max_tokens": 4096, "revision_id": rev,
            "parent_id": None if rev == "s0" else "s0", "revision_type": "skill", "policy_version": None,
            "retrieval_cache_version": "c1", "environment_version": "e1", "estimated_cost_usd": COST[rev] * N,
            "actual_cost_usd": COST[rev] * N, "cost_line_items": {"model": COST[rev] * N * 0.8,
                                                                   "search": COST[rev] * N * 0.2},
            "wall_time_s": 60.0, "invalid_execution_count": int(rev == "s0" and seed == 33), "preliminary": True,
            "checks": {}, "closed_hash": None}


def outcome(rev, i, seed, score, valid=True, retry_of=None):
    return {"run_id": f"FX_{rev}_s{seed}", "task_id": f"q{i:02d}", "stratum": LEVELS[i % 3], "model": "m",
            "provider": "p", "seed": seed, "revision_id": rev, "score": score if valid else None,
            "valid_execution": valid, "failure_type": None if valid else "timeout", "retry_of": retry_of,
            "tokens_in": 100, "tokens_out": 20, "tool_calls": 1, "latency_s": 1.0, "cost_usd": COST[rev]}


def arm(rev):
    outcomes = []
    for seed in SEEDS:
        for i in range(N):
            if rev == "s0" and i == 5 and seed == 33:
                outcomes.append(outcome(rev, i, seed, None, valid=False))
                outcomes.append(outcome(rev, i, seed, SCORES[rev](i, seed), retry_of="q05@seed33"))
            else:
                outcomes.append(outcome(rev, i, seed, SCORES[rev](i, seed)))
    return {"revision_text_hash": f"sha256-of-{rev}-text", "manifests": [manifest(rev, s) for s in SEEDS],
            "outcomes": outcomes}


def write_json(path, data):
    """One manifest / outcome per line, so the files stay small and diffable."""
    rows = lambda key: ",\n  ".join(json.dumps(x) for x in data[key])
    text = (f'{{"revision_text_hash": {json.dumps(data["revision_text_hash"])},\n'
            f' "manifests": [\n  {rows("manifests")}],\n "outcomes": [\n  {rows("outcomes")}]}}\n')
    path.write_text(text, encoding="utf-8", newline="\n")


def main():
    for rev in SCORES:
        write_json(HERE / f"{rev}.json", arm(rev))
    (HERE / "fnoise_m.yaml").write_text(
        "model: m\npaired_fnoise: {8: 0.10, 16: 0.07, 24: 0.06}\nlegacy_fnoise: 0.05\nsensitivity: 0.06\n"
        "source_run_ids: [FX_E1_s11, FX_E1_s22, FX_E1_s33]\nrng_seed: 1\ndraws: 1000\nquantile: 0.95\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
