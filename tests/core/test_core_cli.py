"""CLI: compare (exit code = decision), replay (cards, accounting, live check, ablations), card render."""

import json

import pytest
from core_fixtures import DATA, FIXTURES, ROOT
from typer.testing import CliRunner

from verdict_gate.cli import app

FIX = ROOT / "tests" / "fixtures" / "replay"
POLICY = str(DATA / "configs/policies/moderate.yaml")
FNOISE = str(FIX / "fnoise_m.yaml")
runner = CliRunner()


def compare(candidate, out, policy=POLICY, baseline="s0"):
    return runner.invoke(app, ["compare", "--baseline", str(FIX / f"{baseline}.json"),
                               "--candidate", str(FIX / f"{candidate}.json"), "--policy", policy,
                               "--out", str(out), "--fnoise", FNOISE])


@pytest.mark.parametrize("candidate,code", [("c_good", 0), ("c_null", 30), ("c_bad", 40)])
def test_compare_exit_code_is_the_decision(candidate, code, tmp_path):
    out = tmp_path / "card.json"
    r = compare(candidate, out)
    assert r.exit_code == code, r.output
    assert json.loads(out.read_text(encoding="utf-8"))["revision_id"] == candidate


def test_compare_invalid_evidence_exits_2(tmp_path):
    bad = json.loads((FIX / "c_good.json").read_text(encoding="utf-8"))
    for m in bad["manifests"]:
        m["max_tokens"] = 1  # pairing key differs from the parent
    path = tmp_path / "c_bad_pairing.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    r = runner.invoke(app, ["compare", "--baseline", str(FIX / "s0.json"), "--candidate", str(path),
                            "--policy", POLICY, "--out", str(tmp_path / "card.json"), "--fnoise", FNOISE])
    assert r.exit_code == 2 and not (tmp_path / "card.json").exists()


def test_card_render(tmp_path):
    compare("c_good", tmp_path / "card.json")
    r = runner.invoke(app, ["card", "render", str(tmp_path / "card.json")])
    assert r.exit_code == 0 and "## Decision: **PASS_GLOBAL**" in r.output


def test_card_render_rejects_an_invalid_card(tmp_path):
    path = tmp_path / "card.json"
    path.write_text(json.dumps({"decision": "PASS_GLOBAL"}), encoding="utf-8")
    assert runner.invoke(app, ["card", "render", str(path)]).exit_code == 2


def replay_args(out, *extra):
    runs = [a for r in ("s0", "c_good", "c_bad", "c_null", "c_costly", "c_mixed") for a in ("--runs", str(FIX / f"{r}.json"))]
    return ["replay", "--policy", POLICY, *runs, "--order-seed", "7001", "--fnoise", FNOISE, "--out", str(out), *extra]


def test_replay_writes_cards_and_accounting_and_matches_live(tmp_path):
    out = tmp_path / "replay.json"
    r = runner.invoke(app, replay_args(out, "--live", str(FIX / "live_cards.json")))
    assert r.exit_code == 0, r.output
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["cards"]) == 5 and data["order_seed"] == 7001
    assert data["accounting"]["evaluated"] == 5 and data["accounting"]["harmful"] == 2


def test_replay_disagreeing_with_live_exits_1(tmp_path):
    live = json.loads((FIX / "live_cards.json").read_text(encoding="utf-8"))
    live[0]["decision"] = "HARMFUL"
    path = tmp_path / "live.json"
    path.write_text(json.dumps(live), encoding="utf-8")
    r = runner.invoke(app, replay_args(tmp_path / "replay.json", "--live", str(path)))
    assert r.exit_code == 1


def test_replay_with_ablations_writes_the_table(tmp_path):
    table = tmp_path / "e3a.csv"
    r = runner.invoke(app, replay_args(tmp_path / "replay.json", "--ablations",
                                       str(FIXTURES / "ablation_variants.yaml"), "--table", str(table)))
    assert r.exit_code == 0, r.output
    assert table.read_text(encoding="utf-8").startswith("variant,order_seed,revision_id,decision")
