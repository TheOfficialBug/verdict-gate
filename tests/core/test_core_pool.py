"""The optional question pool of a stored arm (sequential.py, replay.py): the pre-registered order for live runs that
stop early. A pool equal to the stored ids changes nothing; a pool larger than the stored ids keeps max_n and the
visiting order of the full design."""

import dataclasses

import pytest
from core_fixtures import ROOT

from verdict_gate.core.replay import Floors, load_arm, replay
from verdict_gate.core.sequential import block_order

FIX = ROOT / "tests" / "fixtures" / "replay"
REVS = ("s0", "c_good", "c_bad", "c_null", "c_costly", "c_mixed")


@pytest.fixture(scope="module")
def arms():
    return [load_arm(FIX / f"{r}.json") for r in REVS]


@pytest.fixture(scope="module")
def floors():
    return {"m": Floors.from_file(FIX / "fnoise_m.yaml")}


def test_a_pool_equal_to_the_stored_ids_changes_nothing(arms, floors, moderate):
    pooled = [dataclasses.replace(a, pool=sorted({o.task_id for o in a.outcomes})) for a in arms]
    assert replay(moderate, pooled, None, floors, created_at="t") == replay(moderate, arms, None, floors, created_at="t")


def test_a_truncated_arm_replays_in_the_pool_order(arms, floors, moderate):
    """Keep only the first block of the pool's order: with the pool, replay decides on exactly those 8 rows."""
    pool = sorted({o.task_id for o in arms[0].outcomes})
    first = set(block_order(pool, moderate.order_seed)[:8])
    cut = [dataclasses.replace(a, outcomes=[o for o in a.outcomes if o.task_id in first], pool=pool) for a in arms]
    cards = {c["revision_id"]: c for c in replay(moderate, cut, None, floors, created_at="t")}
    good = cards["c_good"]
    assert good["sequential"]["questions_used"] == 8 and good["effect"]["n"] == 8
    with pytest.raises(ValueError, match="not in the pool"):
        replay(moderate, [dataclasses.replace(a, pool=pool[:4]) for a in arms], None, floors, created_at="t")
