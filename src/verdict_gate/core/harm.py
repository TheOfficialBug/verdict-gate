"""Harm-side quantities.

    ConditionalRegressionRate = R₋ / #{parent correct}      (no parent correct → None, not 0)
    Optimism_j                = initial_gain_j − repeated_gain_j
"""

from __future__ import annotations

from verdict_gate.core.paired import Transitions


def conditional_regression_rate(t: Transitions) -> float | None:
    """Share of the parent's correct questions that the candidate breaks. None when the parent got none right."""
    if t.parent_correct == 0:
        return None
    return t.regressions / t.parent_correct


def optimism(initial_gain: float, repeated_gain: float) -> float:
    """Selection optimism of one admitted revision j: the gain it was admitted on minus the gain on retest."""
    return initial_gain - repeated_gain
