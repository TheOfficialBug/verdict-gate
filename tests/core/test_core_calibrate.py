"""calibration: paired F_noise(n), legacy floor, sensitivity, flip rate, frozen floor files."""

import math
from itertools import combinations

import numpy as np
import pytest

from verdict_gate.core.calibrate import (
    flip_rate,
    floor_at,
    legacy_fnoise,
    load_fnoise,
    paired_fnoise,
    pairwise_run_gaps,
    pooled_pair_differences,
    sensitivity,
    write_fnoise,
)

BLOCKS = [8, 16, 24, 32, 40, 48, 56, 60]


def test_identical_repeats_give_zero_floor():
    y = [1, 0, 1, 1, 0, 0, 1, 0] * 5
    floors = paired_fnoise({11: y, 22: y, 33: y}, BLOCKS, B=2000, rng_seed=1)
    assert sorted(floors) == BLOCKS and all(f == 0.0 for f in floors.values())


def test_pooled_differences_over_unordered_seed_pairs():
    d = pooled_pair_differences({33: [1, 1], 11: [0, 1], 22: [1, 0]})
    # pairs in sorted seed order: (11,22), (11,33), (22,33); d = Y_s − Y_s'
    assert d.tolist() == [-1, 1, -1, 0, 0, -1]
    sym = pooled_pair_differences({33: [1, 1], 11: [0, 1], 22: [1, 0]}, pairing="symmetric")
    assert sym.tolist() == [-1, 1, -1, 0, 0, -1, 1, -1, 1, 0, 0, 1]
    with pytest.raises(ValueError, match="pairing"):
        pooled_pair_differences({11: [1], 22: [0]}, pairing="both")


def test_pooled_floor_is_invariant_to_seed_labels():
    """Amendment B: with drift between runs (seed 33 much better), the ordered floor depends on which seed is
    numbered first; the default (symmetric) floor does not."""
    rng = np.random.default_rng(3)
    a, b, c = (rng.random(60) < p for p in (0.15, 0.12, 0.40))
    reps = {11: a.tolist(), 22: b.tolist(), 33: c.tolist()}
    relabelled = {11: c.tolist(), 22: b.tolist(), 33: a.tolist()}  # swap the labels of seeds 11 and 33
    kw = dict(block_sizes=BLOCKS, B=4000, rng_seed=2)
    assert paired_fnoise(reps, **kw) == paired_fnoise(relabelled, **kw)
    assert paired_fnoise(reps, **kw) == paired_fnoise(reps, pairing="symmetric", **kw)
    assert paired_fnoise(reps, pairing="ordered", **kw) != paired_fnoise(relabelled, pairing="ordered", **kw)
    d = pooled_pair_differences(reps, pairing="symmetric")
    assert sorted(d.tolist()) == sorted(pooled_pair_differences(relabelled, pairing="symmetric").tolist())
    assert d.mean() == 0.0


@pytest.mark.parametrize("flip", [0.1, 0.3])
def test_floor_matches_analytic_value(flip):
    """Two seeds whose outcomes differ on a share `flip` of 400 questions, half in each direction.

    Then d ∈ {−1, 0, +1} with P(±1) = flip/2, Var(d) = flip, and the 95th percentile of |mean of n draws| is close to
    1.96·sqrt(flip/n) (normal approximation; checked at the larger block sizes where it holds).
    """
    n_q = 400
    k = int(flip * n_q)
    a = np.zeros(n_q)
    b = np.zeros(n_q)
    a[: k // 2] = 1  # a right, b wrong
    b[k // 2: k] = 1  # b right, a wrong
    floors = paired_fnoise({11: a.tolist(), 22: b.tolist()}, [40, 60], B=20000, rng_seed=5)
    for n in (40, 60):
        assert floors[n] == pytest.approx(1.96 * math.sqrt(flip / n), rel=0.12)


def test_floor_shrinks_with_n():
    rng = np.random.default_rng(0)
    reps = {s: rng.integers(0, 2, 90).tolist() for s in (11, 22, 33, 44, 55)}
    floors = paired_fnoise(reps, BLOCKS, B=5000, rng_seed=9)
    assert floors[8] > floors[24] > floors[60] > 0


def test_paired_fnoise_reproducible():
    reps = {11: [1, 0, 1, 0], 22: [0, 0, 1, 1]}
    assert paired_fnoise(reps, [8], 1000, rng_seed=4) == paired_fnoise(reps, [8], 1000, rng_seed=4)


def test_calibration_input_checks():
    with pytest.raises(ValueError, match="two seeds"):
        paired_fnoise({11: [1, 0]}, [8])
    with pytest.raises(ValueError, match="same"):
        paired_fnoise({11: [1, 0], 22: [1]}, [8])


def test_legacy_fnoise_and_sensitivity():
    scores = [0.50, 0.55, 0.45, 0.60]
    G = [abs(a - b) for a, b in combinations(scores, 2)]
    assert pairwise_run_gaps(scores).tolist() == pytest.approx(G)
    assert legacy_fnoise(scores) == pytest.approx(float(np.quantile(G, 0.95)))
    assert legacy_fnoise(scores, q=0.5) == pytest.approx(float(np.median(G)))
    assert sensitivity(scores) == pytest.approx(np.mean(G) + 2 * np.std(G, ddof=1))


def test_legacy_identical_runs():
    assert legacy_fnoise([0.4, 0.4, 0.4]) == 0.0 and sensitivity([0.4, 0.4, 0.4]) == 0.0


def test_sensitivity_two_runs_has_zero_sd():
    assert sensitivity([0.4, 0.5]) == pytest.approx(0.1)


def test_flip_rate():
    rates = flip_rate({11: [1, 1, 0, 1], 22: [1, 0, 0, 0], 33: [1, 1, 1, 0]})
    # q0: no disagreement; q1: (11,22),(22,33) of 3 pairs; q2: (11,33),(22,33); q3: (11,22),(11,33)
    assert rates == pytest.approx([0.0, 2 / 3, 2 / 3, 2 / 3])


def test_floor_at_nearest_frozen_n_at_or_below():
    floors = {8: 0.30, 16: 0.20, 24: 0.15, 60: 0.08}
    assert floor_at(floors, 8) == 0.30
    assert floor_at(floors, 20) == 0.20
    assert floor_at(floors, 59) == 0.15
    assert floor_at(floors, 60) == 0.08
    assert floor_at(floors, 4) == 0.30  # below the smallest frozen n: the smallest n's floor
    with pytest.raises(ValueError):
        floor_at({}, 8)


def test_fnoise_file_round_trip(tmp_path):
    path = tmp_path / "fnoise" / "qwen.yaml"
    write_fnoise(path, "qwen", {16: 0.2, 8: 0.3}, legacy=0.05, sensitivity_value=0.07,
                 source_run_ids=["R21", "R22"], rng_seed=9001, B=10000, q=0.95)
    rec = load_fnoise(path)
    assert rec["paired_fnoise"] == {8: 0.3, 16: 0.2}
    assert rec["source_run_ids"] == ["R21", "R22"] and rec["rng_seed"] == 9001 and rec["legacy_fnoise"] == 0.05
    assert rec["pairing"] == "symmetric"
