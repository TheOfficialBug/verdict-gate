"""validation rules: one test (or more) per rule."""

import pytest
from core_fixtures import make_table, manifest, outcome
from pydantic import ValidationError

from verdict_gate.core.schemas import (
    Decision,
    PairedTable,
    RevisionType,
    SchemaError,
    check_pairing,
    check_splits_disjoint,
    check_task_subset_hash,
    task_subset_hash,
)


def test_enums_have_exact_names():
    assert [d.value for d in Decision] == ["PASS_GLOBAL", "PASS_CONDITIONAL", "NEEDS_MORE_EVIDENCE",
                                           "UNSUPPORTED", "HARMFUL"]
    assert RevisionType("done_condition_checker") is RevisionType.DONE_CONDITION_CHECKER
    assert len(RevisionType) == 9


# rule 1: score is None iff invalid ---------------------------------------------------------------------------------

def test_valid_outcome_needs_score():
    with pytest.raises(ValidationError, match="must have a score"):
        outcome(score=None)


def test_invalid_outcome_must_not_have_score():
    with pytest.raises(ValidationError, match="must not have a score"):
        outcome(valid_execution=False, score=0.0, failure_type="timeout")


def test_invalid_outcome_ok():
    o = outcome(valid_execution=False, score=None, failure_type="provider_error")
    assert o.score is None and not o.valid_execution


# rule 2: failure_type present iff invalid ------------------------------------------------------------------------

def test_invalid_outcome_needs_failure_type():
    with pytest.raises(ValidationError, match="needs a failure_type"):
        outcome(valid_execution=False, score=None, failure_type=None)


def test_valid_outcome_must_not_have_failure_type():
    with pytest.raises(ValidationError, match="only allowed on invalid"):
        outcome(failure_type="timeout")


def test_wrong_answer_is_valid():
    assert outcome(score=0.0).valid_execution


# rule 3: pairing keys identical between arms ---------------------------------------------------------------------

def test_pairing_ok():
    check_pairing([manifest(run_id="P")], [manifest(run_id="C", revision_id="r1")],
                  [outcome(run_id="P")], [outcome(run_id="C", revision_id="r1")])


@pytest.mark.parametrize("key,value", [("model", "other"), ("provider", "other"),
                                       ("retrieval_cache_version", "c2"), ("max_tokens", 2048)])
def test_pairing_broken_by_each_shared_key(key, value):
    with pytest.raises(SchemaError, match=key):
        check_pairing([manifest(run_id="P")], [manifest(run_id="C", **{key: value})])


def test_pairing_broken_by_candidate_seed_without_parent():
    with pytest.raises(SchemaError, match="seeds"):
        check_pairing([manifest(run_id="P", seed=11)], [manifest(run_id="C", seed=22)])


def test_pairing_allows_more_parent_seeds():  # fixed-repeat: candidate {11} against parent {11, 22}
    check_pairing([manifest(run_id="P1", seed=11), manifest(run_id="P2", seed=22)], [manifest(run_id="C", seed=11)])


def test_pairing_broken_by_task_ids():
    with pytest.raises(SchemaError, match="task IDs differ"):
        check_pairing([manifest(run_id="P")], [manifest(run_id="C")],
                      [outcome(run_id="P", task_id="q1")], [outcome(run_id="C", task_id="q2")])


def test_pairing_broken_by_outcome_not_matching_its_run():
    with pytest.raises(SchemaError, match="seed differs"):
        check_pairing([manifest(run_id="P")], [manifest(run_id="C")],
                      [outcome(run_id="P")], [outcome(run_id="C", seed=22)])


# rule 4: task_subset_hash matches the declared split file -------------------------------------------------------

def test_task_subset_hash_is_order_free():
    assert task_subset_hash(["b", "a"]) == task_subset_hash(["a", "b"])


def _split(tmp_path, name, ids):
    path = tmp_path / name
    path.write_bytes(("\n".join(ids) + "\n").encode("utf-8"))  # LF on every OS (the hash is of the bytes)
    return path


def test_check_task_subset_hash(tmp_path):
    ids = [f"q{i:03d}" for i in range(60)]
    s60, s30 = _split(tmp_path, "a.txt", ids), _split(tmp_path, "b.txt", [f"r{i:03d}" for i in range(30)])
    good = manifest(task_subset_hash=task_subset_hash(ids))
    check_task_subset_hash(good, s60, ids)
    with pytest.raises(SchemaError, match="does not match split file"):
        check_task_subset_hash(good, s30)
    with pytest.raises(SchemaError, match="not the tasks"):
        check_task_subset_hash(good, s60, ids[:-1])


# rule 5: no task appears in two splits ---------------------------------------------------------------------------

def test_splits_are_disjoint():
    check_splits_disjoint({"dev": ["a", "b"], "held": ["c", "d"], "final": ["e"]})
    with pytest.raises(SchemaError):
        check_splits_disjoint({"dev": ["a", "b"], "held": ["b", "c"]})


def test_overlapping_splits_rejected():
    with pytest.raises(SchemaError, match="share 1 task"):
        check_splits_disjoint({"V60": ["a", "b"], "V30": ["b", "c"]})


# PairedTable shape -----------------------------------------------------------------------------------------------

def test_paired_table_columns_must_align():
    t = make_table([(0, 1), (1, 1)])
    with pytest.raises(ValidationError, match="one entry per task"):
        PairedTable(**{**t.model_dump(), "parent": [0.0]})


def test_paired_table_task_ids_unique():
    t = make_table([(0, 1), (1, 1)])
    with pytest.raises(ValidationError, match="unique"):
        PairedTable(**{**t.model_dump(), "task_ids": ["a", "a"]})


def test_paired_table_values_in_unit_interval():
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        make_table([(0, 2)])


def test_paired_table_subset_keeps_order():
    t = make_table([(0, 1), (1, 1), (1, 0)], strata=["a", "b", "c"])
    s = t.subset([2, 0])
    assert s.task_ids == ["t002", "t000"] and s.strata == ["c", "a"] and s.candidate == [0.0, 1.0]
