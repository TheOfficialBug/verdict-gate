"""Posterior over the paired effect.

Primary: Bayesian bootstrap over questions. w ~ Dirichlet(1,...,1) over the n paired questions, Δ* = Σ w_i D_i.
Check fixture (tests only): Dirichlet(1,1,1) posterior on (repair, regression, unchanged) proportions, Δ = θ₊ − θ₋.
The RNG seed is always passed in, so every number can be reproduced and written to the evidence card.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bayes_bootstrap(D: Sequence[float], B: int = 10_000, rng_seed: int = 0) -> np.ndarray:
    """B posterior draws of the mean paired difference."""
    d = np.asarray(D, dtype=float)
    if d.size == 0:
        raise ValueError("bayes_bootstrap needs at least one paired question")
    rng = np.random.default_rng(rng_seed)
    weights = rng.dirichlet(np.ones(d.size), size=B)  # B x n, each row sums to 1
    return weights @ d


def prob_gt(draws: np.ndarray, t: float) -> float:
    """P(Δ > t): share of draws strictly above t."""
    return float(np.mean(np.asarray(draws) > t))


def prob_lt(draws: np.ndarray, t: float) -> float:
    """P(Δ < t): share of draws strictly below t."""
    return float(np.mean(np.asarray(draws) < t))


def credible_interval(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Equal-tailed interval of the draws (for reporting only, never for deciding)."""
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(np.asarray(draws), [tail, 1.0 - tail])
    return float(lo), float(hi)


def dirichlet_multinomial_fixture(R_plus: int, R_minus: int, U: int, B: int = 10_000, rng_seed: int = 0) -> np.ndarray:
    """Test-only cross-check for binary single-repeat data: θ ~ Dirichlet(1+R₊, 1+R₋, 1+U), Δ = θ₊ − θ₋."""
    rng = np.random.default_rng(rng_seed)
    theta = rng.dirichlet([1 + R_plus, 1 + R_minus, 1 + U], size=B)
    return theta[:, 0] - theta[:, 1]
