"""replay: replay-equals-live on a stored fixture, agreement check, order seeds, invalid evidence, ablation runner."""

import csv
import json

import pytest
from core_fixtures import DATA, FIXTURES, ROOT, manifest, outcome

from verdict_gate.core.cost import cost_ratio_on, per_question_costs
from verdict_gate.core.decision import Evidence, EvidenceInvalid, decide
from verdict_gate.core.paired import build_paired_table
from verdict_gate.core.replay import (
    TABLE_COLUMNS,
    AblationConfig,
    Arm,
    Floors,
    Variant,
    check_agreement,
    evaluate,
    invalid_rate,
    load_ablation_config,
    load_arm,
    pair_arms,
    replay,
    run_ablations,
    select_seeds,
    write_table,
)
from verdict_gate.core.schemas import Decision
from verdict_gate.core.sequential import block_order

FIX = ROOT / "tests" / "fixtures" / "replay"
REVS = ("s0", "c_good", "c_bad", "c_null", "c_costly", "c_mixed")


@pytest.fixture(scope="module")
def arms():
    return [load_arm(FIX / f"{r}.json") for r in REVS]


@pytest.fixture(scope="module")
def floors():
    return {"m": Floors.from_file(FIX / "fnoise_m.yaml")}


def simulate_live(parent, candidate, policy, floors):
    """What a live runner does: execute one block at a time, then decide on only the outcomes executed so far.

    Written independently of run_sequential: the table is rebuilt from raw outcomes after every block (rows in
    visiting order) and the cost ratio is recomputed from the executed outcomes.
    """
    parent, candidate = select_seeds(parent, candidate, policy)
    tasks = sorted({o.task_id for o in candidate.outcomes})
    order = block_order(tasks, policy.order_seed)
    max_n = min(policy.max_n, len(tasks))
    n, blocks = 0, 0
    while True:
        n, blocks = min(n + policy.block_size, max_n), blocks + 1
        done = set(order[:n])
        p_out = [o for o in parent.outcomes if o.task_id in done]
        c_out = [o for o in candidate.outcomes if o.task_id in done]
        table = build_paired_table(p_out, c_out)
        position = {t: i for i, t in enumerate(table.task_ids)}
        table = table.subset([position[t] for t in order[:n]])
        ratio = cost_ratio_on(per_question_costs(p_out), per_question_costs(c_out), table.task_ids)
        rate = invalid_rate(Arm(parent.manifests, p_out), Arm(candidate.manifests, c_out))
        result = decide(Evidence(table=table, invalid_rate=rate, pairing_ok=True, cost_ratio=ratio,
                                 floor_by_n=floors.paired, available_n=len(tasks)), policy)
        if result.decision != Decision.NEEDS_MORE_EVIDENCE or n >= max_n:
            return {"revision_id": candidate.revision_id, "decision": result.decision.value,
                    "sequential": {"questions_used": n, "blocks_evaluated": blocks}}


def test_fixture_pairs_every_candidate_with_s0(arms):
    assert [(p.revision_id, c.revision_id) for p, c in pair_arms(arms)] == [("s0", r) for r in REVS[1:]]


def test_replay_equals_simulated_live_run(arms, floors, moderate):
    live = [simulate_live(p, c, moderate, floors["m"]) for p, c in pair_arms(arms)]
    assert check_agreement(live, replay(moderate, arms, None, floors, created_at="t")) == []


def test_replay_equals_stored_live_sequence(arms, floors, moderate):
    stored = json.loads((FIX / "live_cards.json").read_text(encoding="utf-8"))
    assert check_agreement(stored, replay(moderate, arms, None, floors, created_at="t")) == []
    # the stored sequence covers more than one block and every decision kind the fixture produces
    assert {c["decision"] for c in stored} == {"PASS_GLOBAL", "HARMFUL", "UNSUPPORTED"}
    assert max(c["sequential"]["blocks_evaluated"] for c in stored) > 1


def test_agreement_reports_each_kind_of_mismatch(arms, floors, moderate):
    cards = replay(moderate, arms, None, floors, created_at="t")
    live = json.loads(json.dumps(cards))
    live[0]["decision"] = "UNSUPPORTED"
    live[1]["sequential"]["questions_used"] += 8
    del live[2]
    problems = check_agreement(live, cards)
    assert any("c_good: decision" in p for p in problems)
    assert any("c_bad: sequential.questions_used" in p for p in problems)
    assert any("c_null: in replay but not in live" in p for p in problems)


def test_replay_is_deterministic_and_order_seed_is_recorded(arms, floors, moderate):
    a = replay(moderate, arms, 7001, floors, created_at="t")
    assert a == replay(moderate, arms, None, floors, created_at="t")  # 7001 is the policy's own seed
    b = replay(moderate, arms, 7005, floors, created_at="t")
    assert a[0]["policy"]["config_hash"] != b[0]["policy"]["config_hash"]


def test_expected_fixture_decisions(arms, floors, moderate, naive, fixed_repeat):
    got = lambda policy: {c["revision_id"]: c["decision"] for c in replay(policy, arms, None, floors, created_at="t")}
    assert got(moderate) == {"c_good": "PASS_GLOBAL", "c_bad": "HARMFUL", "c_null": "UNSUPPORTED",
                             "c_costly": "HARMFUL", "c_mixed": "UNSUPPORTED"}
    assert got(naive)["c_mixed"] == "PASS_GLOBAL"  # keep-if-higher admits what the governed policy rejects
    assert got(fixed_repeat)["c_costly"] == "UNSUPPORTED"


def test_cost_ratio_on_the_card_uses_the_questions_seen(arms, floors, moderate):
    costly = {c["revision_id"]: c for c in replay(moderate, arms, None, floors, created_at="t")}["c_costly"]
    assert "cost ratio" in costly["decision_reason"]
    # parent q05 carries one extra (invalid) execution, so the ratio depends on whether q05 was seen
    assert costly["costs"]["cost_ratio"] == pytest.approx(costly["costs"]["candidate_usd_per_q"]
                                                          / costly["costs"]["parent_usd_per_q"])


def test_broken_pairing_is_invalid_evidence(arms, floors, moderate):
    parent, candidate = arms[0], arms[1]
    other = load_arm(FIX / "c_good.json")
    for m in other.manifests:
        m.provider = "other"
    for o in other.outcomes:
        o.provider = "other"
    with pytest.raises(EvidenceInvalid, match="provider"):
        evaluate(parent, other, moderate, floors["m"])
    assert evaluate(parent, candidate, moderate, floors["m"])["decision"] == "PASS_GLOBAL"


def test_too_many_invalid_executions_is_invalid_evidence(arms, floors, moderate):
    bad = load_arm(FIX / "c_good.json")
    for o in bad.outcomes[:10]:
        o.valid_execution, o.score, o.failure_type = False, None, "timeout"
    with pytest.raises(EvidenceInvalid, match="invalid rate"):
        evaluate(arms[0], bad, moderate, floors["m"])


def test_ablation_runner_writes_one_table(arms, floors, tmp_path):
    cfg = load_ablation_config(FIXTURES / "ablation_variants.yaml")
    rows = run_ablations(cfg, arms, floors, DATA)
    n_cand = len(REVS) - 1
    assert len(rows) == (len(cfg.variants) - 1 + 20) * n_cand  # orderings: 20 seeds
    by = {(r["variant"], r["revision_id"]): r for r in rows if r["variant"] != "orderings"}
    assert by[("no_early_stop", "c_good")]["questions_used"] == 24
    assert by[("block_4", "c_good")]["questions_used"] == 4
    assert by[("moderate", "c_mixed")]["blocks_evaluated"] == 3
    assert by[("holm_sign_flip", "c_good")]["decision"] == "PASS_GLOBAL"
    assert by[("holm_sign_flip", "c_mixed")]["p_holm"] >= by[("holm_sign_flip", "c_mixed")]["p_value"]
    assert by[("dirichlet_multinomial", "c_bad")]["decision"] == "HARMFUL"
    assert by[("no_floor", "c_good")]["decision"] == "PASS_GLOBAL"
    assert sorted({r["order_seed"] for r in rows if r["variant"] == "orderings"}) == list(range(7002, 7022))
    for name, decision in (("accept_all", "PASS_GLOBAL"), ("reject_all", "UNSUPPORTED")):  # reference rows
        ref = [r for r in rows if r["variant"] == name]
        assert len(ref) == n_cand and {r["decision"] for r in ref} == {decision}
        assert {r["questions_used"] for r in ref} == {0}
    assert by[("accept_all", "c_good")]["delta_hat"] == by[("no_early_stop", "c_good")]["delta_hat"]
    assert {v for v, _ in by} >= {"naive", "fixed_repeat", "moderate", "strict", "accept_all", "reject_all"}
    assert by[("naive", "c_good")]["decision"] == "PASS_GLOBAL"
    out = tmp_path / "e3a.csv"
    write_table(rows, out)
    with out.open(encoding="utf-8") as f:
        read = list(csv.DictReader(f))
    assert tuple(read[0]) == TABLE_COLUMNS and len(read) == len(rows)


# ---- SkillBoost gate and EvoSkill v2-style accept-all ------------------------------


def gate_arms(repairs, regressions, n=60):
    """Parent s0 and candidate c at seed 11 on n questions: R+ repairs, R- regressions, the rest preserved."""
    pairs = [(0, 1)] * repairs + [(1, 0)] * regressions + [(1, 1)] * (n - repairs - regressions)
    out = []
    for rev, parent_id, col in (("s0", None, 0), ("c", "s0", 1)):
        run = f"R_{rev}"
        out.append(Arm([manifest(run_id=run, revision_id=rev, parent_id=parent_id)],
                       [outcome(run_id=run, task_id=f"q{i:02d}", revision_id=rev, score=float(p[col]))
                        for i, p in enumerate(pairs)], revision_text_hash="h"))
    return out


GATE_CFG = AblationConfig(name="gate", version=1, base_policy="configs/policies/moderate.yaml", variants=[
    Variant(name="skillboost_gate", method="regression_bounded", policy="configs/policies/naive.yaml"),
    Variant(name="accept_all", method="accept_all")])


@pytest.mark.parametrize("repairs,regressions,decision", [(4, 2, "PASS_GLOBAL"), (4, 3, "UNSUPPORTED"),
                                                          (3, 3, "UNSUPPORTED"), (0, 0, "UNSUPPORTED")])
def test_skillboost_gate(repairs, regressions, decision):
    rows = {r["variant"]: r for r in run_ablations(GATE_CFG, gate_arms(repairs, regressions), {}, DATA)}
    gate = rows["skillboost_gate"]
    assert gate["decision"] == decision and gate["questions_used"] == 60
    assert gate["reason"] == f"R+={repairs} R-={regressions} n=60 eps=0.05"
    assert rows["accept_all"]["decision"] == "PASS_GLOBAL"  # even with no gain
    assert "EvoSkill v2-style" in rows["accept_all"]["reason"]


def test_skillboost_gate_eps_is_a_config_field():
    variant = GATE_CFG.variants[0].model_copy(update={"eps": 0.10})
    cfg = GATE_CFG.model_copy(update={"variants": [variant]})
    assert run_ablations(cfg, gate_arms(4, 3), {}, DATA)[0]["decision"] == "PASS_GLOBAL"


def test_skillboost_3seed_variant_uses_the_fixed_repeat_arms(arms, floors):
    rows = run_ablations(load_ablation_config(FIXTURES / "ablation_variants.yaml"), arms, floors, DATA)
    by = {(r["variant"], r["revision_id"]): r for r in rows}
    assert by[("skillboost_gate", "c_good")]["questions_used"] == by[("naive", "c_good")]["questions_used"]
    three = by[("skillboost_gate_3seed", "c_good")]
    assert three["questions_used"] == by[("fixed_repeat", "c_good")]["questions_used"]
    assert three["reason"].startswith("R+=")
    assert by[("skillboost_gate", "c_bad")]["decision"] == "UNSUPPORTED"
