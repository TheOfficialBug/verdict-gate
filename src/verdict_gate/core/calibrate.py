"""Noise calibration.

    paired_fnoise: for every unordered seed pair (s, s'): d_i = Y_s,i − Y_s',i and −d_i, pooled over pairs
                   (pairing="symmetric": the floor does not depend on seed numbering).
                   For each block size n: F(n) = q-quantile of |mean of n bootstrap-sampled d|.
                   pairing="ordered" keeps the original one-sided d (s < s' only); sensitivity analyses only.
    legacy_fnoise: G_ss' = |S_s − S_s'| over whole-run scores; F_legacy = q-quantile of G.
    sensitivity:   mean(G) + 2·sd(G)   (sample sd).
    flip_rate:     per question, the fraction of seed pairs whose outcomes differ.

The frozen output is configs/fnoise/<model-slug>.yaml (write_fnoise; slug = policy.model_slug); floor_at() reads F at the nearest frozen n at or
below the current n.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
import yaml


def _seed_matrix(repeats_by_seed: dict[int, Sequence[float]]) -> tuple[list[int], np.ndarray]:
    seeds = sorted(repeats_by_seed)
    if len(seeds) < 2:
        raise ValueError("calibration needs at least two seeds")
    lengths = {len(repeats_by_seed[s]) for s in seeds}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("every seed needs the same, non-empty list of questions")
    return seeds, np.array([repeats_by_seed[s] for s in seeds], dtype=float)


Pairing = Literal["symmetric", "ordered"]
PAIRINGS: tuple[Pairing, ...] = ("symmetric", "ordered")


def pooled_pair_differences(repeats_by_seed: dict[int, Sequence[float]],
                            pairing: Pairing = "ordered") -> np.ndarray:
    """d_i = Y_s,i − Y_s',i for every unordered seed pair (s < s'), concatenated ("ordered").

    "symmetric" appends −d (both orders of every pair), the convention paired_fnoise uses.
    """
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, not {pairing!r}")
    _, Y = _seed_matrix(repeats_by_seed)
    d = np.concatenate([Y[a] - Y[b] for a, b in combinations(range(len(Y)), 2)])
    return np.concatenate([d, -d]) if pairing == "symmetric" else d


def paired_fnoise(repeats_by_seed: dict[int, Sequence[float]], block_sizes: Sequence[int], B: int = 10_000,
                  q: float = 0.95, rng_seed: int = 0, pairing: Pairing = "symmetric") -> dict[int, float]:
    """F_noise(n) for each block size n. pairing="ordered" (spec-literal d, s < s') is for sensitivity rows only."""
    d = pooled_pair_differences(repeats_by_seed, pairing)
    if pairing == "symmetric":  # sorted: the draws depend only on the multiset of d, so relabelling seeds is exact
        d = np.sort(d)
    rng = np.random.default_rng(rng_seed)
    floors = {}
    for n in sorted(block_sizes):
        sample_means = d[rng.integers(0, d.size, size=(B, n))].mean(axis=1)
        floors[int(n)] = float(np.quantile(np.abs(sample_means), q))
    return floors


def pairwise_run_gaps(run_scores: Sequence[float]) -> np.ndarray:
    """G_ss' = |S_s − S_s'| over unordered pairs of whole-run scores."""
    if len(run_scores) < 2:
        raise ValueError("need at least two run scores")
    return np.array([abs(a - b) for a, b in combinations(run_scores, 2)], dtype=float)


def legacy_fnoise(run_scores: Sequence[float], q: float = 0.95) -> float:
    return float(np.quantile(pairwise_run_gaps(run_scores), q))


def sensitivity(run_scores: Sequence[float]) -> float:
    G = pairwise_run_gaps(run_scores)
    sd = float(np.std(G, ddof=1)) if G.size > 1 else 0.0
    return float(np.mean(G)) + 2.0 * sd


def flip_rate(repeats_by_seed: dict[int, Sequence[float]]) -> list[float]:
    _, Y = _seed_matrix(repeats_by_seed)
    pairs = list(combinations(range(len(Y)), 2))
    disagree = sum((Y[a] != Y[b]).astype(float) for a, b in pairs)
    return [float(x) for x in disagree / len(pairs)]


def repeats_from_outcomes(outcomes) -> tuple[list[str], dict[int, list[float]], list[str], list[str]]:
    """Repeat runs of one revision (TaskOutcome rows, several run_ids) -> calibration input.

    Returns (run_ids, {run index: scores over the common questions}, common task ids, excluded task ids). A question
    is used only if every run has a valid execution of it (an invalid execution is not a wrong answer)."""
    by_run: dict[str, dict[str, float]] = {}
    all_tasks: set[str] = set()
    for o in outcomes:
        all_tasks.add(o.task_id)
        scores = by_run.setdefault(o.run_id, {})
        if o.valid_execution:
            if o.task_id in scores:
                raise ValueError(f"run {o.run_id}: two valid executions of task {o.task_id}")
            scores[o.task_id] = float(o.score)
    run_ids = list(by_run)
    if len(run_ids) < 2:
        raise ValueError(f"calibration needs at least two repeat runs, got {run_ids}")
    common = sorted(set.intersection(*[set(s) for s in by_run.values()]))
    if not common:
        raise ValueError("no question has a valid execution in every run")
    return run_ids, {i: [by_run[r][t] for t in common] for i, r in enumerate(run_ids)}, common, \
        sorted(all_tasks - set(common))


# ---- frozen floor files -------------------------------------------------------------------------------------------


def write_fnoise(path: str | Path, model: str, paired: dict[int, float], legacy: float | None,
                 sensitivity_value: float | None, source_run_ids: Sequence[str], rng_seed: int, B: int,
                 q: float, pairing: Pairing = "symmetric", extra: dict | None = None) -> dict:
    """Write the frozen floor file. `extra` (provenance: run hashes, flip rates, ...) is appended after the fields
    the readers use; it may not overwrite them."""
    record = {
        "model": model,
        "paired_fnoise": {int(n): float(f) for n, f in sorted(paired.items())},
        "legacy_fnoise": legacy,
        "sensitivity": sensitivity_value,
        "source_run_ids": list(source_run_ids),
        "rng_seed": rng_seed,
        "draws": B,
        "quantile": q,
        "pairing": pairing,
    }
    clash = sorted(set(extra or {}) & set(record))
    if clash:
        raise ValueError(f"extra fields would overwrite {clash}")
    record.update(extra or {})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF on every OS: the file is hashed (freeze manifest, evidence cards) and committed with eol=lf
    path.write_text(yaml.safe_dump(record, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
    return record


def load_fnoise(path: str | Path) -> dict:
    record = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    record["paired_fnoise"] = {int(n): float(f) for n, f in (record.get("paired_fnoise") or {}).items()}
    return record


def floor_at(floor_by_n: dict[int, float], n: int) -> float:
    """F at the nearest frozen n at or below n. Below the smallest frozen n, the smallest n's (largest) floor."""
    if not floor_by_n:
        raise ValueError("no frozen floor values")
    at_or_below = [k for k in floor_by_n if k <= n]
    return float(floor_by_n[max(at_or_below) if at_or_below else min(floor_by_n)])
