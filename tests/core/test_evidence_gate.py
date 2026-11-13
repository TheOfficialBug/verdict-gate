"""Evidence gate (moderate_v4): no decision on too few questions or too few discordant pairs; v3 unchanged."""

from pathlib import Path

from core_fixtures import make_table, outcomes

from verdict_gate.core.decision import Evidence, decide
from verdict_gate.core.policy import load_policy
from verdict_gate.core.schemas import Decision
from verdict_gate.core.sequential import run_sequential

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "src" / "verdict_gate" / "_data"
V3 = load_policy(DATA / "configs/policies/moderate.yaml", root=ROOT)
V4 = load_policy(DATA / "configs/policies/moderate_v4.yaml", root=ROOT)
ZERO = {1: 0.0}  # a near-deterministic agent: F_noise(n) = 0


def ev(pairs, n_avail=300):
    return Evidence(table=make_table(pairs), invalid_rate=0.0, pairing_ok=True, cost_ratio=1.0, floor_by_n=ZERO,
                    available_n=n_avail)


def test_v3_decides_on_one_discordant_pair_with_zero_floor():
    # 1 repair + 7 ties in the first block: v3 passes (the E22 failure mode); v4 waits
    first_block = outcomes(repairs=1, preserved=7)
    assert decide(ev(first_block), V3).decision == Decision.PASS_GLOBAL
    assert decide(ev(first_block), V4).decision == Decision.NEEDS_MORE_EVIDENCE


def test_v4_holds_harm_and_futility_before_min_n():
    assert decide(ev(outcomes(regressions=1, preserved=7)), V4).decision == Decision.NEEDS_MORE_EVIDENCE
    assert decide(ev(outcomes(preserved=8)), V4).decision == Decision.NEEDS_MORE_EVIDENCE
    assert decide(ev(outcomes(preserved=8)), V3).decision == Decision.UNSUPPORTED  # v3 futility at n = 8


def test_v4_needs_min_discordant_to_pass():
    few = outcomes(repairs=5, preserved=27)  # n = 32, 5 discordant < 6
    assert decide(ev(few), V4).decision != Decision.PASS_GLOBAL
    enough = outcomes(repairs=6, preserved=26)  # n = 32, 6 discordant
    assert decide(ev(enough), V4).decision == Decision.PASS_GLOBAL


def test_v4_harm_needs_min_discordant():
    harm = outcomes(regressions=6, preserved=26)
    assert decide(ev(harm), V4).decision == Decision.HARMFUL
    few = outcomes(regressions=3, preserved=29)
    assert decide(ev(few), V4).decision != Decision.HARMFUL


def test_v4_at_max_n_without_discordant_is_unsupported():
    pairs = outcomes(repairs=4, preserved=116)  # n = max_n = 120, only 4 discordant
    assert decide(ev(pairs, n_avail=120), V4).decision == Decision.UNSUPPORTED


def test_sequential_v4_uses_at_least_min_n():
    table = make_table(outcomes(repairs=30, preserved=270))
    res = run_sequential(table, V4, ZERO, 0.0, True, cost_ratio=1.0)
    assert res.questions_used >= 32
    res3 = run_sequential(table, V3, ZERO, 0.0, True, cost_ratio=1.0)
    assert res3.questions_used <= res.questions_used
