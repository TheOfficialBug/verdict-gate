"""Scope (stratum effects, routes), sequential block evaluation and the policy/strata config files."""

import pytest
import yaml
from core_fixtures import DATA, FAST, NEVER, SLOW, make_table, outcomes, stratified

from verdict_gate.core.policy import load_policy, load_strata
from verdict_gate.core.schemas import Decision, SchemaError
from verdict_gate.core.scope import route, stratum_effects
from verdict_gate.core.sequential import block_order, run_sequential

FLOOR = {n: 0.05 for n in (8, 16, 24, 32, 40, 48, 56, 60)}


# config files ----------------------------------------------------------------------------------------------------

def test_strata_config_is_freshness_only():
    strata = load_strata(DATA / "configs/strata/sealqa.yaml")
    assert strata.field == "freshness"
    assert strata.levels == [NEVER, SLOW, FAST]


def test_policy_files_match_spec():
    m = load_policy(DATA / "configs/policies/moderate.yaml")
    assert (m.kind, m.block_size, m.max_n, m.order_seed) == ("sequential_bayes", 8, 60, 7001)
    assert (m.posterior.draws, m.posterior.rng_seed, m.floor.source) == (10000, 9001, "paired_fnoise")
    assert (m.pass_rule.p_gain_gt_0, m.pass_rule.p_gain_gt_floor) == (0.95, 0.50)
    assert (m.futility.p_gain_gt_0_max, m.futility.p_gain_gt_floor_max) == (0.05, 0.10)
    assert (m.harm.p_gain_lt_neg_floor, m.harm.max_cost_ratio) == (0.95, 2.0)
    assert (m.subgroup.harm_min_n, m.subgroup.harm_p_lt_neg_floor, m.subgroup.support_min_n) == (8, 0.90, 16)
    assert m.curtailment and m.max_invalid_rate == 0.05
    s = load_policy(DATA / "configs/policies/strict.yaml")
    assert (s.pass_rule.p_gain_gt_0, s.pass_rule.p_gain_gt_floor) == (None, 0.95)
    assert s.model_dump(exclude={"name", "pass_rule"}) == m.model_dump(exclude={"name", "pass_rule"})
    n = load_policy(DATA / "configs/policies/naive.yaml")
    assert (n.kind, n.seed) == ("keep_if_higher", 11)
    f = load_policy(DATA / "configs/policies/fixed_repeat.yaml")
    assert (f.kind, f.seeds, f.floor.source) == ("fixed_repeat", [11, 22, 33], "legacy_fnoise")


def test_policy_rejects_unknown_keys(tmp_path):
    raw = yaml.safe_load((DATA / "configs/policies/naive.yaml").read_text(encoding="utf-8"))
    raw["threshold"] = 0.5
    (tmp_path / "configs" / "policies").mkdir(parents=True)
    (tmp_path / "configs" / "validity.yaml").write_text("max_invalid_rate: 0.05\n", encoding="utf-8")
    path = tmp_path / "configs" / "policies" / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_policy(path)


# stratum effects --------------------------------------------------------------------------------------------------

def test_stratum_effects_labels(moderate):
    table = stratified((NEVER, outcomes(repairs=12, preserved=4)),
                       (SLOW, outcomes(repairs=4, regressions=6, preserved=6)),
                       (FAST, outcomes(regressions=3, preserved=5)))
    effects = {e.level: e for e in stratum_effects(table, moderate.strata, moderate, FLOOR)}
    assert effects[NEVER].label == "supported" and effects[NEVER].n == 16 and effects[NEVER].repairs == 12
    assert effects[SLOW].label == "inconclusive" and effects[SLOW].regressions == 6
    assert effects[FAST].label == "harmful" and effects[FAST].p_gain_lt_0 == 1.0
    assert effects[FAST].p_gain_lt_neg_floor >= 0.90
    assert effects[NEVER].delta_hat == pytest.approx(12 / 16) and effects[NEVER].floor == 0.05


def test_stratum_harm_uses_the_floor_at_n_s(moderate):
    # harmful needs P(Δ_s < −F(n_s)) ≥ 0.90; F(n_s) is read at the stratum's own n (8 here)
    table = stratified((NEVER, outcomes(repairs=12, preserved=4)), (FAST, outcomes(regressions=1, preserved=7)))
    no_floor = {e.level: e for e in stratum_effects(table, moderate.strata, moderate, {8: 0.0})}
    floored = {e.level: e for e in stratum_effects(table, moderate.strata, moderate, {8: 0.25, 16: 0.0})}
    assert no_floor[FAST].label == "harmful"
    assert floored[FAST].floor == 0.25 and floored[FAST].label == "inconclusive"
    assert floored[NEVER].floor == 0.0


def test_stratum_without_questions_is_inconclusive(moderate):
    effects = stratum_effects(make_table(outcomes(repairs=16), NEVER), moderate.strata, moderate, FLOOR)
    assert [(e.level, e.n, e.label, e.p_gain_gt_0) for e in effects[1:]] == [
        (SLOW, 0, "inconclusive", None), (FAST, 0, "inconclusive", None)]


def test_unstratified_questions_are_ignored_by_scope(moderate):
    table = make_table(outcomes(repairs=8), [None] * 8)
    assert all(e.n == 0 for e in stratum_effects(table, moderate.strata, moderate, FLOOR))


def test_descriptive_tags_cannot_be_used_as_strata(moderate):
    # a question_types tag (D11: descriptive only) is not a level of the configured field
    table = make_table(outcomes(repairs=8), "advanced reasoning")
    with pytest.raises(SchemaError, match="not levels of 'freshness'"):
        stratum_effects(table, moderate.strata, moderate, FLOOR)


def test_route_uses_candidate_only_in_routed_strata(moderate):
    table = make_table([(0, 1), (1, 0), (0, 1)], [NEVER, FAST, None])
    routed = route(table, {"field": "freshness", "candidate_strata": [NEVER]}, moderate.strata)
    assert routed == [1.0, 1.0, 0.0]


def test_route_rejects_other_fields_and_levels(moderate):
    table = make_table([(0, 1)], NEVER)
    with pytest.raises(SchemaError, match="routing field"):
        route(table, {"field": "question_types", "candidate_strata": [NEVER]}, moderate.strata)
    with pytest.raises(SchemaError, match="routing levels"):
        route(table, {"field": "freshness", "candidate_strata": ["Sports"]}, moderate.strata)


# sequential blocks ------------------------------------------------------------------------------------------------

def test_block_order_is_a_seeded_permutation():
    ids = [f"q{i}" for i in range(60)]
    order = block_order(ids, 7001)
    assert sorted(order) == sorted(ids) and order != sorted(ids)
    assert order == block_order(list(reversed(ids)), 7001)  # input order does not matter
    assert order != block_order(ids, 7002)


def test_run_sequential_stops_at_first_decision(moderate):
    table = make_table(outcomes(repairs=60), NEVER)
    res = run_sequential(table, moderate, FLOOR, invalid_rate=0.0, pairing_ok=True, cost_ratio=1.0)
    assert res.final.decision == Decision.PASS_GLOBAL
    assert (res.blocks_evaluated, res.questions_used) == (1, 8)
    assert res.history == [(8, Decision.PASS_GLOBAL)]


def test_run_sequential_null_revision_stops_for_futility(moderate):
    table = make_table(outcomes(preserved=30, unresolved=30), NEVER)
    res = run_sequential(table, moderate, FLOOR, invalid_rate=0.0, pairing_ok=True)
    assert res.final.decision == Decision.UNSUPPORTED and "futility" in res.stopping_reason
    assert res.questions_used == 8


def test_run_sequential_last_block_is_partial(moderate):
    # 20 questions with D ∈ {+1/3, −1/3, 0}: no prefix is conclusive, curtailment is off, so the blocks are
    # 8, 16 and a partial block of 4, and the decision at n = max_n = 20 is UNSUPPORTED
    pairs = [(1 / 3, 2 / 3), (2 / 3, 1 / 3), (1 / 3, 1 / 3), (1 / 3, 2 / 3)] * 5
    table = make_table(pairs, NEVER, repeats=3)
    policy = moderate.model_copy(update={"curtailment": False})
    res = run_sequential(table, policy, {8: 0.0}, invalid_rate=0.0, pairing_ok=True)
    assert [n for n, _ in res.history] == [8, 16, 20]
    assert res.final.decision == Decision.UNSUPPORTED and "maximum n=20" in res.stopping_reason


POWER_FLOOR = {8: 0.25, 16: 0.19, 24: 0.17, 32: 0.13, 40: 0.13, 48: 0.10, 56: 0.09, 60: 0.10}


def _by_visit(policy, kinds):
    """Table whose i-th visited question has outcome kinds[i] (D = +1 'rep', −1 'reg', 0 'unr')."""
    pair = {"rep": (0, 1), "reg": (1, 0), "unr": (0, 0)}
    ids = [f"t{i:03d}" for i in range(len(kinds))]
    slot = {t: i for i, t in enumerate(block_order(ids, policy.order_seed))}
    return make_table([pair[kinds[slot[t]]] for t in ids], NEVER)


def _no_hold(policy):
    return policy.model_copy(update={"futility": policy.futility.model_copy(update={"hold_while_negative": False})})


def test_a3_harmful_candidate_continues_to_harmful(moderate, strict):
    # 3 regressions per block of 8 (Δ̂ = −0.375): block 1 has P(Δ>0) = 0 but P(Δ<−F(8)) < 0.95
    kinds = (["reg"] * 3 + ["unr"] * 5) * 7 + ["reg", "unr", "unr", "unr"]
    for policy in (moderate, strict):
        table = _by_visit(policy, kinds)
        old = run_sequential(table, _no_hold(policy), POWER_FLOOR, invalid_rate=0.0, pairing_ok=True)
        assert old.final.decision == Decision.UNSUPPORTED and old.questions_used == 8
        assert "futility" in old.stopping_reason
        new = run_sequential(table, policy, POWER_FLOOR, invalid_rate=0.0, pairing_ok=True)
        assert new.final.decision == Decision.HARMFUL and new.questions_used > 8
        assert all(d == Decision.NEEDS_MORE_EVIDENCE for _, d in new.history[:-1])


def test_a3_negative_within_noise_reaches_max_n_unsupported(moderate):
    # running Δ̂ = −1/n at every block boundary: never beyond −F(n), never ≥ 0 → held to max_n, then UNSUPPORTED
    kinds = ["reg"] + ["unr"] * 7 + (["reg", "rep"] + ["unr"] * 6) * 6 + ["reg", "rep", "unr", "unr"]
    table = _by_visit(moderate, kinds)
    res = run_sequential(table, moderate, POWER_FLOOR, invalid_rate=0.0, pairing_ok=True)
    assert [n for n, _ in res.history] == [8, 16, 24, 32, 40, 48, 56, 60]
    assert res.final.decision == Decision.UNSUPPORTED and "maximum n=60" in res.stopping_reason
    assert res.final.delta_hat < 0
    old = run_sequential(table, _no_hold(moderate), POWER_FLOOR, invalid_rate=0.0, pairing_ok=True)
    assert old.questions_used == 8 and "futility" in old.stopping_reason


def test_run_sequential_raises_on_invalid_evidence(moderate):
    from verdict_gate.core.decision import EvidenceInvalid
    with pytest.raises(EvidenceInvalid):
        run_sequential(make_table(outcomes(repairs=8), NEVER), moderate, FLOOR, invalid_rate=0.2, pairing_ok=True)
