"""0.2.0 bring-your-own-task flow: calibrate, import, compare by policy name, next_batch, example (examples/byot)."""

import json

import pytest
from core_fixtures import DATA, ROOT
from typer.testing import CliRunner

from verdict_gate import next_batch
from verdict_gate.adapters.json_csv import import_runs
from verdict_gate.cli import app
from verdict_gate.core.calibrate import load_fnoise, repeats_from_outcomes
from verdict_gate.core.decision import EvidenceInvalid
from verdict_gate.core.policy import load_policy
from verdict_gate.core.replay import Arm, Floors, evaluate, load_arm
from verdict_gate.core.sequential import block_order

BYOT = ROOT / "examples" / "byot"
RESULTS = str(BYOT / "results.csv")
FIX = ROOT / "tests" / "fixtures" / "replay"
MODERATE = load_policy(DATA / "configs" / "policies" / "moderate.yaml")
runner = CliRunner()


def test_repeats_use_only_tasks_valid_in_every_run():
    parent = [o for r in import_runs(RESULTS) if r.manifest.revision_id == "parent" for o in r.outcomes]
    run_ids, reps, common, excluded = repeats_from_outcomes(parent)
    assert run_ids == ["parent_s11", "parent_s22", "parent_s33"]
    assert excluded == ["T005"] and len(common) == 59  # T005 timed out at seed 33 and was not retried
    assert all(len(v) == 59 for v in reps.values())  # T017 timed out at seed 22 but its retry is valid
    with pytest.raises(ValueError, match="two repeat runs"):
        repeats_from_outcomes([o for o in parent if o.run_id == "parent_s11"])


def test_calibrate_writes_a_floor_file_compare_can_read(tmp_path):
    out = tmp_path / "floor.yaml"
    r = runner.invoke(
        app,
        [
            "calibrate",
            RESULTS,
            "--revision",
            "parent",
            "--out",
            str(out),
            "--block-sizes",
            "8,16,24",
            "--draws",
            "2000",
            "--seed",
            "1",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "flip rate" in r.output and "floor written" in r.output
    rec = load_fnoise(out)
    assert rec["model"] == "my-support-model" and sorted(rec["paired_fnoise"]) == [8, 16, 24]
    assert rec["paired_fnoise"][8] >= rec["paired_fnoise"][24] > 0
    assert rec["rng_seed"] == 1 and rec["draws"] == 2000 and rec["quantile"] == 0.95
    assert rec["source_run_ids"] == ["parent_s11", "parent_s22", "parent_s33"]
    assert Floors.from_file(out).paired == rec["paired_fnoise"]


def test_calibrate_needs_runs_of_one_model_and_two_repeats(tmp_path):
    out = str(tmp_path / "f.yaml")
    assert runner.invoke(app, ["calibrate", RESULTS, "--out", out]).exit_code == 2  # no runs named
    assert runner.invoke(app, ["calibrate", RESULTS, "--run", "parent_s11", "--out", out]).exit_code == 2
    assert runner.invoke(app, ["calibrate", RESULTS, "--revision", "nope", "--out", out]).exit_code == 2


def _import(tmp_path, *extra):
    p, c = tmp_path / "parent.json", tmp_path / "candidate.json"
    r1 = runner.invoke(app, ["import", RESULTS, "--revision", "parent", "--max-tokens", "256", "--out", str(p)])
    r2 = runner.invoke(
        app,
        [
            "import",
            RESULTS,
            "--revision",
            "cand-1",
            "--parent-id",
            "parent",
            "--max-tokens",
            "256",
            "--revision-file",
            str(BYOT / "candidate_skill.md"),
            "--out",
            str(c),
            *extra,
        ],
    )
    assert r1.exit_code == 0 and r2.exit_code == 0, r1.output + r2.output
    return p, c


def test_import_then_compare_by_policy_name(tmp_path):
    p, c = _import(tmp_path)
    arm = load_arm(c)
    assert arm.parent_id == "parent" and len(arm.revision_text_hash) == 64 and arm.manifests[0].max_tokens == 256
    assert {m.seed for m in load_arm(p).manifests} == {11, 22, 33}
    floor = tmp_path / "floor.yaml"
    assert (
        runner.invoke(
            app, ["calibrate", RESULTS, "--revision", "parent", "--out", str(floor), "--draws", "2000"]
        ).exit_code
        == 0
    )
    naive = runner.invoke(
        app,
        [
            "compare",
            "--baseline",
            str(p),
            "--candidate",
            str(c),
            "--policy",
            "naive",
            "--out",
            str(tmp_path / "naive.json"),
        ],
    )
    assert naive.exit_code == 0 and "PASS_GLOBAL" in naive.output  # keep-if-higher keeps the lucky run
    card_path = tmp_path / "card.json"
    mod = runner.invoke(
        app,
        [
            "compare",
            "--baseline",
            str(p),
            "--candidate",
            str(c),
            "--policy",
            "moderate",
            "--fnoise",
            str(floor),
            "--out",
            str(card_path),
        ],
    )
    assert mod.exit_code == 30, mod.output  # UNSUPPORTED: the gain is inside the parent's own noise
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["revision_id"] == "cand-1" and card["parent_id"] == "parent"
    assert runner.invoke(app, ["card", "render", str(card_path)]).exit_code == 0
    rep = runner.invoke(
        app,
        [
            "replay",
            "--policy",
            "moderate",
            "--runs",
            str(p),
            "--runs",
            str(c),
            "--fnoise",
            str(floor),
            "--out",
            str(tmp_path / "replay.json"),
        ],
    )
    assert rep.exit_code == 0 and "cand-1: UNSUPPORTED" in rep.output


def test_import_and_compare_refusals(tmp_path):
    no_hash = runner.invoke(
        app,
        [
            "import",
            RESULTS,
            "--revision",
            "cand-1",
            "--parent-id",
            "parent",
            "--max-tokens",
            "256",
            "--out",
            str(tmp_path / "c.json"),
        ],
    )
    assert no_hash.exit_code == 2 and not (tmp_path / "c.json").exists()
    mixed = runner.invoke(
        app,
        [
            "import",
            RESULTS,
            "--run",
            "parent_s11",
            "--run",
            "cand1_s11",
            "--max-tokens",
            "1",
            "--out",
            str(tmp_path / "m.json"),
        ],
    )
    assert mixed.exit_code == 2
    p, c = _import(tmp_path)
    unknown_policy = runner.invoke(
        app,
        [
            "compare",
            "--baseline",
            str(p),
            "--candidate",
            str(c),
            "--policy",
            "nope",
            "--out",
            str(tmp_path / "card.json"),
        ],
    )
    assert unknown_policy.exit_code == 2
    no_floor = runner.invoke(
        app,
        [
            "compare",
            "--baseline",
            str(p),
            "--candidate",
            str(c),
            "--policy",
            "moderate",
            "--out",
            str(tmp_path / "card.json"),
        ],
    )
    assert no_floor.exit_code == 2 and "verdict-gate calibrate" in no_floor.output


def _ran(arm: Arm, tasks: set) -> Arm:
    return Arm(arm.manifests, [o for o in arm.outcomes if o.task_id in tasks], arm.revision_text_hash, arm.path)


@pytest.mark.parametrize("candidate", ["c_good", "c_null", "c_bad", "c_mixed"])
def test_next_batch_loop_reaches_the_replay_decision(candidate):
    """Running only the tasks next_batch asks for ends at the decision and stopping point of the full replay."""
    parent, cand = load_arm(FIX / "s0.json"), load_arm(FIX / f"{candidate}.json")
    floors = Floors.from_file(FIX / "fnoise_m.yaml")
    pool = sorted({o.task_id for o in cand.outcomes})
    card = evaluate(parent, cand, MODERATE, floors)
    ran: set = set()
    nb = next_batch(MODERATE, pool)
    assert nb.task_ids == block_order(pool, MODERATE.order_seed)[: MODERATE.block_size] and not nb.stop
    while not nb.stop:
        ran |= set(nb.task_ids)
        nb = next_batch(MODERATE, pool, _ran(parent, ran), _ran(cand, ran), floors)
    assert nb.decision == card["decision"]
    assert nb.questions_done == card["sequential"]["questions_used"]
    assert len(ran) == nb.questions_done  # no task beyond the stopping point was run


def test_next_batch_incomplete_block_and_out_of_order_tasks():
    parent, cand = load_arm(FIX / "s0.json"), load_arm(FIX / "c_good.json")
    floors = Floors.from_file(FIX / "fnoise_m.yaml")
    pool = sorted({o.task_id for o in cand.outcomes})
    visit = block_order(pool, MODERATE.order_seed)
    nb = next_batch(MODERATE, pool, _ran(parent, set(visit[:5])), _ran(cand, set(visit[:5])), floors)
    assert nb.task_ids == visit[5:8] and nb.questions_done == 5
    with pytest.raises(EvidenceInvalid, match="out of the visiting order"):
        next_batch(MODERATE, pool, _ran(parent, {visit[-1]}), _ran(cand, {visit[-1]}), floors)
    with pytest.raises(TypeError):
        next_batch(load_policy(DATA / "configs" / "policies" / "naive.yaml"), pool)


def test_next_batch_cli_and_example(tmp_path):
    r = runner.invoke(app, ["example", "--out", str(tmp_path)])
    assert r.exit_code == 0 and (tmp_path / "results.csv").read_bytes() == (BYOT / "results.csv").read_bytes()
    first = runner.invoke(app, ["next-batch", "--policy", "moderate", "--pool", str(tmp_path / "tasks.txt")])
    assert first.exit_code == 0 and "run next 8 task(s)" in first.output
