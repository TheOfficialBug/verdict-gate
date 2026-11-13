"""decision order: every branch and every sub-condition of steps 1, 2, 3, 4 and 5, plus naive and fixed-repeat.

Fixtures are hand-built binary single-repeat tables (core_fixtures.outcomes); the posterior values they produce under
the moderate policy's posterior seed are noted next to each fixture.
"""

import pytest
from core_fixtures import FAST, NEVER, SLOW, make_table, outcomes, stratified

from verdict_gate.core.decision import Evidence, EvidenceInvalid, decide
from verdict_gate.core.schemas import Decision

BLOCKS = [8, 16, 24, 32, 40, 48, 56, 60]


def floor(value):
    return {n: value for n in BLOCKS}


def ev(table, F=0.05, cost_ratio=1.0, invalid_rate=0.0, pairing_ok=True, available_n=None, legacy_floor=None):
    return Evidence(table=table, invalid_rate=invalid_rate, pairing_ok=pairing_ok, cost_ratio=cost_ratio,
                    floor_by_n=floor(F), legacy_floor=legacy_floor, available_n=available_n)


def single(**counts):
    return make_table(outcomes(**counts), NEVER)


# step 1: validity precondition -------------------------------------------------------------------------------------

def test_step1_invalid_rate_above_limit_raises(moderate):
    with pytest.raises(EvidenceInvalid, match="invalid rate"):
        decide(ev(single(repairs=8), invalid_rate=0.06), moderate)


def test_step1_invalid_rate_at_limit_is_allowed(moderate):
    assert decide(ev(single(repairs=8), invalid_rate=0.05), moderate).decision == Decision.PASS_GLOBAL


def test_step1_broken_pairing_raises(moderate):
    with pytest.raises(EvidenceInvalid, match="pairing"):
        decide(ev(single(repairs=8), pairing_ok=False), moderate)


def test_step1_applies_to_every_policy_kind(naive, fixed_repeat):
    for policy in (naive, fixed_repeat):
        with pytest.raises(EvidenceInvalid):
            decide(ev(single(repairs=8), invalid_rate=0.5, legacy_floor=0.0), policy)


def test_step1_no_questions_raises(moderate):
    with pytest.raises(EvidenceInvalid, match="no valid"):
        decide(ev(make_table([])), moderate)


# step 2: HARMFUL ----------------------------------------------------------------------------------------------------

def test_step2a_posterior_harm(moderate):
    r = decide(ev(single(regressions=10, preserved=6)), moderate)  # P(Δ<−0.05) = 1
    assert r.decision == Decision.HARMFUL and "P(Δ<−F) ≥ 0.95" in r.reason
    assert r.summary.p_gain_lt_neg_floor == 1.0


def test_step2a_harm_is_judged_against_minus_floor(moderate, strict):
    # a degenerate first block (one regression, no repair, n = 8) has P(Δ<0) = 1, but with a floor
    # F(8) > 0 it is not evidence of harm beyond noise. P(Δ<−F) = P(Beta(1,7) weight > F) = (1 − F)^7.
    table = single(regressions=1, preserved=7)
    for F in (0.05, 0.25):
        for policy in (moderate, strict):
            r = decide(ev(table, F=F), policy)
            assert r.summary.p_gain_lt_0 == 1.0
            assert r.summary.p_gain_lt_neg_floor == pytest.approx((1 - F) ** 7, abs=0.02)
            assert r.decision != Decision.HARMFUL


def test_step2a_harm_beyond_the_floor_is_harmful(moderate):
    # Δ̂ = −0.5 at n = 16 with F = 0.25: P(Δ < −0.25) ≥ 0.95
    r = decide(ev(single(regressions=8, preserved=8), F=0.25), moderate)
    assert r.summary.p_gain_lt_neg_floor >= 0.95 and r.decision == Decision.HARMFUL


def test_step2a_without_a_floor_is_unchanged(moderate):
    # F = 0 (no floor, e.g. replay floor_mode="zero"): P(Δ<−F) = P(Δ<0), so the pre-amendment rule applies
    table = single(regressions=1, preserved=7)
    r = decide(ev(table, F=0.0), moderate)
    assert r.summary.p_gain_lt_neg_floor == r.summary.p_gain_lt_0 == 1.0
    assert r.decision == Decision.HARMFUL
    for counts in (dict(regressions=10, preserved=6), dict(repairs=1, regressions=3, preserved=12),
                   dict(repairs=3, regressions=1, unresolved=4), dict(repairs=12, preserved=4)):
        s = decide(ev(single(**counts), F=0.0), moderate).summary
        assert s.p_gain_lt_neg_floor == s.p_gain_lt_0


def test_step2a_wins_over_a_supported_stratum(moderate):
    table = stratified((NEVER, outcomes(repairs=16)), (FAST, outcomes(regressions=40)))
    assert decide(ev(table, available_n=60), moderate).decision == Decision.HARMFUL


def test_step2b_cost_ratio_without_pass(moderate):
    r = decide(ev(single(preserved=8, unresolved=8), cost_ratio=3.0), moderate)  # no gain, P(Δ<0) = 0
    assert r.decision == Decision.HARMFUL and "cost ratio" in r.reason


def test_step2b_cost_ratio_with_pass_is_not_harmful(moderate):
    # pass met but cost too high: not HARMFUL (2b needs "pass not met"), not PASS_GLOBAL (cost), not step 4
    # (no harmed stratum, pass met); undecided below max_n, UNSUPPORTED at max_n
    table = single(repairs=12, preserved=4)
    assert decide(ev(table, cost_ratio=3.0), moderate).decision == Decision.NEEDS_MORE_EVIDENCE
    assert decide(ev(table, cost_ratio=3.0, available_n=16), moderate).decision == Decision.UNSUPPORTED


# step 3: PASS_GLOBAL -----------------------------------------------------------------------------------------------

def test_step3_pass_global(moderate):
    r = decide(ev(single(repairs=12, preserved=4)), moderate)
    assert r.decision == Decision.PASS_GLOBAL and r.held_out_replication == "pending"
    assert r.summary.p_gain_gt_0 >= 0.95 and r.summary.p_gain_gt_floor >= 0.5


def test_step3_cost_ratio_at_limit_passes(moderate):
    assert decide(ev(single(repairs=12, preserved=4), cost_ratio=2.0), moderate).decision == Decision.PASS_GLOBAL


def test_step3_unknown_cost_ratio_passes_and_says_so(moderate):
    r = decide(ev(single(repairs=12, preserved=4), cost_ratio=None), moderate)
    assert r.decision == Decision.PASS_GLOBAL and "cost ratio unknown" in r.reason


def test_step3_stratum_within_the_floor_does_not_block(moderate):
    # fast-changing: 1 regression in 8, P(Δ_s<0) = 1 but P(Δ_s<−F) with F = 0.25 is (0.75)^7 ≈ 0.13 < 0.90
    table = stratified((NEVER, outcomes(repairs=20, preserved=4)), (FAST, outcomes(regressions=1, preserved=7)))
    r = decide(ev(table, F=0.25), moderate)
    fast = next(e for e in r.strata if e.level == FAST)
    assert fast.p_gain_lt_0 == 1.0 and fast.label != "harmful"
    assert r.decision == Decision.PASS_GLOBAL
    assert decide(ev(table, F=0.0), moderate).decision == Decision.PASS_CONDITIONAL  # no floor: harmed, as before


def test_step3_small_harmed_looking_stratum_does_not_block(moderate):
    # fast-changing: 3 regressions in 7 questions, P(Δ_s<0) = 1 but n_s = 7 < harm_min_n = 8
    table = stratified((NEVER, outcomes(repairs=20, preserved=4)), (FAST, outcomes(regressions=3, preserved=4)))
    assert decide(ev(table), moderate).decision == Decision.PASS_GLOBAL


def test_strict_needs_more_than_moderate(moderate, strict):
    table = single(repairs=10, regressions=2, preserved=20)  # P(Δ>0)≈0.99, P(Δ>0.15)≈0.86
    assert decide(ev(table, F=0.15), moderate).decision == Decision.PASS_GLOBAL
    assert decide(ev(table, F=0.15), strict).decision == Decision.NEEDS_MORE_EVIDENCE


# step 4: PASS_CONDITIONAL ------------------------------------------------------------------------------------------

def test_step4a_pass_overall_but_a_stratum_harmed(moderate):
    # fast-changing: n_s = 8 ≥ harm_min_n, P(Δ_s<−0.05) ≈ 1 → inhibited; never-changing: n_s = 24, supported
    table = stratified((NEVER, outcomes(repairs=20, preserved=4)), (FAST, outcomes(regressions=3, preserved=5)))
    r = decide(ev(table), moderate)
    assert r.decision == Decision.PASS_CONDITIONAL and "stratum harmed" in r.reason
    assert r.activation_conditions == [{"field": "freshness", "level": NEVER}]
    assert r.inhibition_conditions == [{"field": "freshness", "level": FAST}]
    assert r.parent_fallback == [{"field": "freshness", "level": SLOW}, {"field": "freshness", "level": None}]
    assert r.held_out_replication == "pending"


def test_step4b_pass_not_met_overall_but_a_stratum_supported(moderate):
    # overall 16 repairs vs 13 regressions in 39: P(Δ>0)≈0.7, no pass, no harm, no futility
    table = stratified((NEVER, outcomes(repairs=12, preserved=4)),
                       (SLOW, outcomes(repairs=4, regressions=6, preserved=6)),
                       (FAST, outcomes(regressions=7)))  # n_s = 7 < harm_min_n: not inhibited
    r = decide(ev(table), moderate)
    assert r.decision == Decision.PASS_CONDITIONAL and "pass rule not met overall" in r.reason
    assert r.activation_conditions == [{"field": "freshness", "level": NEVER}]
    assert r.inhibition_conditions == []
    assert {c["level"] for c in r.parent_fallback} == {SLOW, FAST, None}


def test_step4_support_needs_min_n(moderate):
    # same as 4b but never-changing has 15 questions < support_min_n = 16: no supported stratum
    table = stratified((NEVER, outcomes(repairs=11, preserved=4)),
                       (SLOW, outcomes(repairs=4, regressions=6, preserved=6)),
                       (FAST, outcomes(regressions=7)))
    r = decide(ev(table), moderate)
    assert r.decision == Decision.NEEDS_MORE_EVIDENCE
    assert [e.label for e in r.strata] == ["inconclusive"] * 3


def test_step4_harmed_stratum_without_support_is_not_conditional(moderate):
    table = stratified((NEVER, outcomes(repairs=10, preserved=4)), (FAST, outcomes(regressions=3, preserved=5)))
    assert decide(ev(table), moderate).decision != Decision.PASS_CONDITIONAL


# step 5: UNSUPPORTED -----------------------------------------------------------------------------------------------

def test_step5a_futility_by_p_gain_gt_0(moderate):
    r = decide(ev(single(preserved=8, unresolved=8)), moderate)  # P(Δ>0) = 0
    assert r.decision == Decision.UNSUPPORTED and "futility" in r.reason


def test_step5a_futility_by_floor_only(moderate):
    r = decide(ev(single(repairs=2, regressions=1, preserved=13), F=0.3), moderate)  # P(Δ>0)≈0.75, P(Δ>F)≈0.02
    assert r.summary.p_gain_gt_0 > 0.05 and r.summary.p_gain_gt_floor <= 0.10
    assert r.decision == Decision.UNSUPPORTED and "futility" in r.reason


def test_step5b_max_n_reached(moderate):
    table = single(repairs=4, regressions=2, preserved=10)  # P(Δ>0)≈0.81: no pass, no futility
    r = decide(ev(table, F=0.01, available_n=16), moderate)
    assert r.decision == Decision.UNSUPPORTED and "maximum n=16" in r.reason
    assert decide(ev(table, F=0.01), moderate).decision == Decision.NEEDS_MORE_EVIDENCE  # 44 more could pass


def test_step5c_curtailment(moderate):
    # n = 56 of 60: even 4 more repairs give 14 vs 8 in 60, P(Δ>0)≈0.90 < 0.95
    table = single(repairs=10, regressions=8, preserved=38)
    r = decide(ev(table, F=0.001), moderate)
    assert r.decision == Decision.UNSUPPORTED and "curtailed" in r.reason
    no_curtail = moderate.model_copy(update={"curtailment": False})
    assert decide(ev(table, F=0.001), no_curtail).decision == Decision.NEEDS_MORE_EVIDENCE


# futility and curtailment hold while Δ̂ < 0 ----------------------------------------------------------

def _no_hold(policy):
    return policy.model_copy(update={"futility": policy.futility.model_copy(update={"hold_while_negative": False})})


def test_a3_policies_hold_while_negative(moderate, strict):
    assert moderate.futility.hold_while_negative and strict.futility.hold_while_negative
    assert moderate.version == strict.version == 3


def test_a3_harmful_leaning_is_not_stopped_by_futility(moderate, strict):
    # Δ̂ = −0.25 at n = 8, F = 0.25: P(Δ>0) = 0 (futility), P(Δ<−F) ≈ 0.55 (not yet HARMFUL)
    table = single(regressions=2, unresolved=6)
    for policy in (moderate, strict):
        r = decide(ev(table, F=0.25), policy)
        assert r.summary.p_gain_gt_0 <= 0.05 and r.summary.p_gain_lt_neg_floor < 0.95
        assert r.decision == Decision.NEEDS_MORE_EVIDENCE and "held (A3)" in r.reason
        old = decide(ev(table, F=0.25), _no_hold(policy))
        assert old.decision == Decision.UNSUPPORTED and "futility" in old.reason


def test_a3_harmful_leaning_is_not_stopped_by_curtailment(moderate):
    # n = 56 of 60, Δ̂ = −1/56, P(Δ>0) ≈ 0.4 (no futility); 4 more repairs cannot pass → previously curtailed
    table = single(repairs=9, regressions=10, preserved=37)
    r = decide(ev(table, F=0.001), moderate)
    assert r.delta_hat < 0 and r.summary.p_gain_gt_0 > 0.05 and r.summary.p_gain_gt_floor > 0.10
    assert r.decision == Decision.NEEDS_MORE_EVIDENCE and "held (A3)" in r.reason
    old = decide(ev(table, F=0.001), _no_hold(moderate))
    assert old.decision == Decision.UNSUPPORTED and "curtailed" in old.reason


def test_a3_negative_at_max_n_is_unsupported(moderate):
    # the hold never extends a run beyond max_n
    r = decide(ev(single(repairs=9, regressions=10, preserved=41), F=0.001), moderate)
    assert r.n == 60 and r.delta_hat < 0
    assert r.decision == Decision.UNSUPPORTED and "maximum n=60" in r.reason


@pytest.mark.parametrize("counts", [dict(repairs=2, regressions=1, preserved=13),   # Δ̂ = +1/16 > 0
                                    dict(repairs=1, regressions=1, preserved=14)])  # Δ̂ = 0
def test_a3_null_leaning_still_stops_by_floor_futility(moderate, strict, counts):
    # Δ̂ ≥ 0, small, F = 0.3: P(Δ>F) ≤ 0.10 → UNSUPPORTED exactly as without A3
    for policy in (moderate, strict):
        r = decide(ev(single(**counts), F=0.3), policy)
        assert r.delta_hat >= 0 and r.summary.p_gain_gt_floor <= 0.10
        assert r.decision == Decision.UNSUPPORTED and "futility" in r.reason
        old = decide(ev(single(**counts), F=0.3), _no_hold(policy))
        assert (old.decision, old.reason) == (r.decision, r.reason)


def test_a3_null_leaning_curtailment_unchanged(moderate):
    # Δ̂ = +2/56 ≥ 0 at n = 56: curtailment applies as before
    table = single(repairs=10, regressions=8, preserved=38)
    r = decide(ev(table, F=0.001), moderate)
    assert r.delta_hat >= 0 and r.decision == Decision.UNSUPPORTED and "curtailed" in r.reason


def test_a3_fallback_harm_min_n(moderate):
    # pre-set A3 fallback: harm.min_n = 24 → the posterior harm rule cannot fire before n = 24
    gated = moderate.model_copy(update={"harm": moderate.harm.model_copy(update={"min_n": 24})})
    early = single(regressions=10, preserved=6)  # n = 16, P(Δ<−0.05) = 1
    assert decide(ev(early), moderate).decision == Decision.HARMFUL
    assert decide(ev(early), gated).decision == Decision.NEEDS_MORE_EVIDENCE
    assert decide(ev(single(regressions=15, preserved=9)), gated).decision == Decision.HARMFUL  # n = 24


# step 6: NEEDS_MORE_EVIDENCE ---------------------------------------------------------------------------------------

def test_step6_needs_more_evidence(moderate):
    r = decide(ev(single(repairs=3, regressions=1, unresolved=4)), moderate)
    assert r.decision == Decision.NEEDS_MORE_EVIDENCE and r.n == 8


def test_floor_uses_nearest_frozen_n_at_or_below(moderate):
    table = single(repairs=6, preserved=14)  # n = 20 → floor of n = 16
    r = decide(Evidence(table, 0.0, True, 1.0, floor_by_n={8: 0.9, 16: 0.2, 24: 0.0}), moderate)
    assert r.summary.floor == 0.2


def test_sequential_needs_paired_floor(moderate):
    with pytest.raises(ValueError, match="F_noise"):
        decide(Evidence(single(repairs=8), 0.0, True), moderate)


# naive keep-if-higher ----------------------------------------------------------------------------------------------

def test_naive_higher_passes(naive):
    r = decide(ev(single(repairs=1, preserved=7)), naive)
    assert r.decision == Decision.PASS_GLOBAL and r.delta_hat == pytest.approx(1 / 8)


@pytest.mark.parametrize("counts", [dict(repairs=1, regressions=1, preserved=6), dict(regressions=2, preserved=6)])
def test_naive_equal_or_lower_is_unsupported(naive, counts):
    assert decide(ev(single(**counts)), naive).decision == Decision.UNSUPPORTED


def test_naive_never_returns_harmful(naive):
    assert decide(ev(single(regressions=8)), naive).decision == Decision.UNSUPPORTED


# fixed-repeat -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("pairs,expected", [
    ([(0, 1)] * 2 + [(1, 1)] * 8, Decision.PASS_GLOBAL),          # Δ̂ = 0.2 > F = 0.1
    ([(0, 1)] * 1 + [(1, 1)] * 9, Decision.UNSUPPORTED),          # Δ̂ = 0.1, not > F
    ([(0, 1 / 3)] * 3 + [(1, 1)] * 7, Decision.UNSUPPORTED),      # 0 < Δ̂ = 0.1 ≤ F
    ([(1, 0)] * 1 + [(1, 1)] * 9, Decision.UNSUPPORTED),          # Δ̂ = −0.1, not < −F
    ([(1, 0)] * 2 + [(1, 1)] * 8, Decision.HARMFUL),              # Δ̂ = −0.2 < −F
])
def test_fixed_repeat_branches(fixed_repeat, pairs, expected):
    r = decide(ev(make_table(pairs, NEVER, repeats=3), legacy_floor=0.1), fixed_repeat)
    assert r.decision == expected


def test_fixed_repeat_zero_floor_still_needs_positive_gain(fixed_repeat):
    table = make_table([(1, 1)] * 10, NEVER, repeats=3)
    assert decide(ev(table, legacy_floor=0.0), fixed_repeat).decision == Decision.UNSUPPORTED


def test_fixed_repeat_needs_legacy_floor(fixed_repeat):
    with pytest.raises(ValueError, match="F_legacy"):
        decide(ev(single(repairs=8)), fixed_repeat)
