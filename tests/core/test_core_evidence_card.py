"""evidence card: schema-valid cards from fixtures, the null rules, and the Markdown report."""

import copy
import json

import pytest
from core_fixtures import DATA, FAST, NEVER, ROOT, SLOW, make_table, outcomes, stratified

from verdict_gate.core.decision import Evidence, decide
from verdict_gate.core.evidence_card import (
    CARD_VERSION,
    DOMAIN,
    CardContext,
    add_domain_result,
    build_card,
    load_schema,
    policy_hash,
    render_report,
    validate_card,
)
from verdict_gate.core.replay import Floors, load_arm, replay
from verdict_gate.core.scope import HARMFUL, SUPPORTED

FIX = ROOT / "tests" / "fixtures" / "replay"
REVS = ("s0", "c_good", "c_bad", "c_null", "c_costly", "c_mixed")


@pytest.fixture(scope="module")
def arms():
    return [load_arm(FIX / f"{r}.json") for r in REVS]


@pytest.fixture(scope="module")
def floors():
    return {"m": Floors.from_file(FIX / "fnoise_m.yaml")}


def context(**kw):
    base = dict(revision_id="c1", parent_id="s0", revision_type="skill", revision_text_hash="h",
                evaluation_context={"benchmark": "sealqa", "split": "V60", "model": "m", "provider": "p",
                                    "seeds": [11], "retrieval_cache_version": "c1", "max_tokens": 4096},
                git_commit="abc", prereg_tag=None, created_at="2026-11-03T00:00:00+00:00", source_run_ids=["R1", "R2"],
                preliminary=True, floor_type="paired_fnoise", fnoise_file_hash="f", fnoise_source_run_ids=["E1"],
                costs={"parent_usd_per_q": 0.001, "candidate_usd_per_q": 0.001, "cost_ratio": 1.0,
                       "line_items": {"parent": {"model": 0.06}, "candidate": {"model": 0.06}}},
                sequential={"blocks_evaluated": 5, "questions_used": 39, "stopping_reason": "decided"})
    return CardContext(**{**base, **kw})


def conditional_card(moderate):
    table = stratified((NEVER, outcomes(repairs=12, preserved=4)),
                       (SLOW, outcomes(repairs=4, regressions=6, preserved=6)), (FAST, outcomes(regressions=7)))
    result = decide(Evidence(table=table, invalid_rate=0.0, pairing_ok=True, cost_ratio=1.0,
                             floor_by_n={8: 0.05}), moderate)
    return build_card(result, moderate, table, context())


def test_schema_file_is_a_valid_json_schema():
    import jsonschema

    jsonschema.Draft202012Validator.check_schema(load_schema())


@pytest.mark.parametrize("policy_name", ["moderate", "strict", "naive", "fixed_repeat"])
def test_cards_from_fixture_are_schema_valid(policy_name, arms, floors):
    from verdict_gate.core.policy import load_policy

    cards = replay(load_policy(DATA / f"configs/policies/{policy_name}.yaml"), arms, None, floors, created_at="t")
    assert len(cards) == 5
    for card in cards:
        assert validate_card(card) == [], card["revision_id"]
        assert card["card_version"] == CARD_VERSION
        json.dumps(card)  # serialisable


def test_conditional_card_is_valid_and_records_the_fallback(moderate):
    card = conditional_card(moderate)
    assert card["decision"] == "PASS_CONDITIONAL"
    assert validate_card(card) == []
    assert card["activation_conditions"] == [{"field": "freshness", "level": NEVER}]
    assert {"field": "freshness", "level": None} in card["parent_fallback"]
    assert card["held_out_replication"]["status"] == "pending"
    assert card["scope"]["supported"] == [NEVER] and len(card["scope"]["strata"]) == 3
    assert card["policy"]["config_hash"] == policy_hash(moderate)


def test_conditional_card_without_fallback_is_rejected(moderate):
    card = conditional_card(moderate)
    card["parent_fallback"] = []
    assert any("parent_fallback" in e for e in validate_card(card))


@pytest.mark.parametrize("field", list(load_schema()["required"]))
def test_every_field_is_required(field, moderate):
    card = conditional_card(moderate)
    del card[field]
    assert validate_card(card) != []


@pytest.mark.parametrize("path", [("decision",), ("revision_id",), ("effect", "p_gain_gt_0"), ("effect", "n"),
                                  ("transitions", "repairs"), ("sequential", "stopping_reason"),
                                  ("held_out_replication", "status"), ("policy", "config_hash")])
def test_null_is_rejected_where_the_spec_does_not_allow_it(path, moderate):
    card = conditional_card(moderate)
    target = card
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = None
    assert validate_card(card) != []


@pytest.mark.parametrize("path", [("prereg_tag",), ("models",), ("effect", "p_gain_gt_floor"),
                                  ("transitions", "conditional_regression_rate"), ("costs", "cost_ratio"),
                                  ("noise_calibration", "floor_value_at_n"), ("noise_calibration", "fnoise_file_hash")])
def test_null_is_allowed_where_noted(path, moderate):
    card = conditional_card(moderate)
    target = card
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = None
    assert validate_card(card) == []


def test_bad_values_are_rejected(moderate):
    good = conditional_card(moderate)
    for path, value in [(("decision",), "ACCEPT"), (("held_out_replication", "status"), "done"),
                        (("effect", "p_gain_gt_0"), 1.5), (("revision_type",), "lora"), (("extra_field",), 1)]:
        card = copy.deepcopy(good)
        card[path[0]] = value if len(path) == 1 else {**card[path[0]], path[1]: value}
        assert validate_card(card) != [], path


def test_naive_card_has_no_floor(arms, floors, naive):
    card = replay(naive, arms, None, floors, created_at="t")[0]
    assert card["noise_calibration"]["floor_type"] == "none"
    assert card["noise_calibration"]["floor_value_at_n"] is None
    assert card["effect"]["p_gain_gt_floor"] is None
    assert card["effect"]["rng_seed"] == 9001  # reporting posterior


def test_fixed_repeat_card_uses_the_legacy_floor(arms, floors, fixed_repeat):
    card = replay(fixed_repeat, arms, None, floors, created_at="t")[0]
    assert card["noise_calibration"]["floor_type"] == "legacy_fnoise"
    assert card["noise_calibration"]["floor_value_at_n"] == 0.05
    assert card["noise_calibration"]["source_run_ids"] == ["FX_E1_s11", "FX_E1_s22", "FX_E1_s33"]


def test_report_puts_the_decision_first_then_the_sections_in_order(moderate):
    text = render_report(conditional_card(moderate))
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings[0].startswith("## Decision: **PASS_CONDITIONAL**")
    assert [h.split(":")[0] for h in headings[1:]] == ["## Effect", "## Transitions", "## Scope", "## Cost",
                                                       "## Stopping", "## Caveats"]
    assert ("Parent fallback: freshness = slow-changing, freshness = fast-changing, freshness = (missing)"
            in text)
    assert "Preliminary" in text


# ---- domain scope (card 1.1) ----------------------------------------------------------------------------------------

FRAMES = "frames-single-label"


def global_card(moderate, **kw):
    """PASS_GLOBAL on 24 never-changing questions (12 repairs, no regression): the primary domain is supported."""
    table = stratified((NEVER, outcomes(repairs=12, preserved=12)))
    result = decide(Evidence(table=table, invalid_rate=0.0, pairing_ok=True, cost_ratio=1.0,
                             floor_by_n={8: 0.05}), moderate)
    assert result.decision.value == "PASS_GLOBAL"
    return build_card(result, moderate, table, context(**kw))


def add_frames(card, policy, pairs):
    return add_domain_result(card, FRAMES, make_table(pairs), policy, benchmark="frames", split="held",
                             seeds=[11, 22], run_ids=["R200", "R201"])


def test_primary_domain_entry(moderate):
    card = global_card(moderate)
    [primary] = card["scope"]["domains_tested"]
    assert primary["domain"] == "sealqa-v60" and primary["role"] == "primary" and primary["label"] == SUPPORTED
    assert primary["n"] == 24 and primary["seeds"] == [11] and primary["run_ids"] == ["R1", "R2"]
    assert primary["effect"]["delta_hat"] == card["effect"]["delta_hat"] == 0.5
    assert primary["effect"]["floor"] == 0.05
    assert card["activation_conditions"] == [{"field": DOMAIN, "level": "sealqa-v60"}]
    assert card["parent_fallback"] == [{"field": DOMAIN, "level": None}]
    assert validate_card(card) == []


def test_supported_and_harmful_domain_activate_only_in_the_supported_one(moderate):
    card = add_frames(global_card(moderate), moderate, outcomes(regressions=12, preserved=4))
    labels = {d["domain"]: d["label"] for d in card["scope"]["domains_tested"]}
    assert labels == {"sealqa-v60": SUPPORTED, FRAMES: HARMFUL}
    assert card["activation_conditions"] == [{"field": DOMAIN, "level": "sealqa-v60"}]
    assert {"field": DOMAIN, "level": FRAMES} in card["parent_fallback"]
    assert {"field": DOMAIN, "level": None} in card["parent_fallback"]
    assert card["scope"]["domains_tested"][1]["role"] == "transfer"
    assert validate_card(card) == []
    text = render_report(card)
    assert f"| {FRAMES} | transfer | frames / held | 16 | 11, 22 |" in text and "| harmful | R200, R201 |" in text
    assert "Parent fallback: domain = frames-single-label, domain = (any untested domain)" in text


def test_untested_domain_is_in_parent_fallback(moderate):
    card = global_card(moderate, untested_domains=["tau3-banking"])
    assert {"field": DOMAIN, "level": "tau3-banking"} in card["parent_fallback"]
    assert {"field": DOMAIN, "level": None} in card["parent_fallback"]
    assert all(c["level"] != "tau3-banking" for c in card["activation_conditions"])
    assert validate_card(card) == []
    add_frames(card, moderate, outcomes(repairs=12, preserved=4))  # a tested transfer domain stays; tau3 untested
    assert {"field": DOMAIN, "level": "tau3-banking"} in card["parent_fallback"]
    assert {"field": DOMAIN, "level": FRAMES} in card["activation_conditions"]


@pytest.mark.parametrize("level,message", [("tau3-banking", "untested"), (FRAMES, "harmful, not supported")])
def test_activation_in_an_untested_or_non_supported_domain_fails_validation(moderate, level, message):
    card = add_frames(global_card(moderate), moderate, outcomes(regressions=12, preserved=4))
    card["activation_conditions"].append({"field": DOMAIN, "level": level})
    assert any(message in e for e in validate_card(card))


def test_inconclusive_transfer_domain_is_not_activated(moderate):
    card = add_frames(global_card(moderate), moderate, outcomes(repairs=1, regressions=1, preserved=4))
    assert card["scope"]["domains_tested"][1]["label"] == "inconclusive"
    assert {"field": DOMAIN, "level": FRAMES} in card["parent_fallback"]
    bad = copy.deepcopy(card)
    bad["parent_fallback"].remove({"field": DOMAIN, "level": FRAMES})
    assert any("not activated and not listed" in e for e in validate_card(bad))


def test_parent_fallback_must_hold_the_untested_domain_catch_all(moderate):
    card = global_card(moderate)
    card["parent_fallback"] = [{"field": DOMAIN, "level": "sealqa-v60"}]
    card["activation_conditions"] = []
    assert any(e.startswith("parent_fallback") for e in validate_card(card))


def test_rejected_card_activates_nowhere(arms, floors, moderate):
    card = next(c for c in replay(moderate, arms, None, floors, created_at="t") if c["decision"] == "HARMFUL")
    assert card["activation_conditions"] == []
    assert {"field": DOMAIN, "level": card["scope"]["domains_tested"][0]["domain"]} in card["parent_fallback"]
    card["activation_conditions"] = [{"field": DOMAIN, "level": card["scope"]["domains_tested"][0]["domain"]}]
    assert validate_card(card) != []


def test_conditional_card_activates_strata_and_supported_transfer_domains(moderate):
    card = add_frames(conditional_card(moderate), moderate, outcomes(repairs=12, preserved=4))
    assert card["activation_conditions"] == [{"field": "freshness", "level": NEVER},
                                             {"field": DOMAIN, "level": FRAMES}]
    assert all(c["level"] != "sealqa-v60" for c in card["parent_fallback"])  # covered by the strata conditions
    assert validate_card(card) == []
    card["activation_conditions"][0] = {"field": "freshness", "level": FAST}
    assert any("not a supported stratum" in e for e in validate_card(card))


def test_naive_card_labels_domains_with_moderate_thresholds(arms, floors, naive):
    for card in replay(naive, arms, None, floors, created_at="t"):
        assert validate_card(card) == []
        [primary] = card["scope"]["domains_tested"]
        on = card["activation_conditions"] == [{"field": DOMAIN, "level": primary["domain"]}]
        assert on == (card["decision"] == "PASS_GLOBAL" and primary["label"] == SUPPORTED)


def test_early_governed_pass_global_still_activates_its_domain(arms, floors, moderate):
    card = next(c for c in replay(moderate, arms, None, floors, created_at="t") if c["revision_id"] == "c_good")
    primary = card["scope"]["domains_tested"][0]
    assert card["decision"] == "PASS_GLOBAL" and primary["n"] < moderate.subgroup.support_min_n
    assert primary["label"] == SUPPORTED
    assert card["activation_conditions"] == [{"field": DOMAIN, "level": primary["domain"]}]


def test_add_domain_result_checks_policy_and_duplicates(moderate, strict):
    card = global_card(moderate)
    with pytest.raises(ValueError, match="not the policy"):
        add_frames(card, strict, outcomes(repairs=4))
    add_frames(card, moderate, outcomes(repairs=4))
    with pytest.raises(ValueError, match="already"):
        add_frames(card, moderate, outcomes(repairs=4))


def test_report_renders_every_fixture_card(arms, floors, moderate, naive):
    for policy in (moderate, naive):
        for card in replay(policy, arms, None, floors, created_at="t"):
            assert render_report(card).startswith(f"# Evidence card: {card['revision_id']} vs s0")
