"""Policy and strata configs, and the threshold rules they define.

Every threshold lives in configs/policies/*.yaml, configs/strata/*.yaml or configs/validity.yaml; this module only
reads them. Three policy kinds share one Decision enum:

    sequential_bayes   evidence-governed (moderate, strict): blocks, posterior, floor, harm, subgroups
    keep_if_higher     naive: admit if candidate accuracy > parent accuracy (one seed)
    fixed_repeat       fixed seeds, paired per-question means, legacy floor
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from verdict_gate.core.posterior import bayes_bootstrap, credible_interval, prob_gt, prob_lt


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class StrataConfig(_Cfg):
    """The only grouping field conditional decisions may use, and its pre-registered levels."""

    name: str
    version: int
    field: str
    levels: list[str]


class PosteriorCfg(_Cfg):
    method: Literal["bayes_bootstrap"]
    draws: int
    rng_seed: int


class FloorCfg(_Cfg):
    source: Literal["paired_fnoise", "legacy_fnoise"]
    file: str | None = None  # may contain {model_slug} (or the raw {model}); see floor_file


class PassCfg(_Cfg):
    """Every condition present must hold. Moderate: both; strict: p_gain_gt_floor only."""

    p_gain_gt_0: float | None = None
    p_gain_gt_floor: float | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> PassCfg:
        if self.p_gain_gt_0 is None and self.p_gain_gt_floor is None:
            raise ValueError("a pass rule needs at least one condition")
        return self


class FutilityCfg(_Cfg):
    """Either condition present is enough. hold_while_negative: while the running estimate
    Δ̂ < 0 neither futility nor curtailment may stop the run, so harm evidence can accumulate until HARMFUL fires or
    max_n is reached; with Δ̂ ≥ 0 both behave as before."""

    p_gain_gt_0_max: float | None = None
    p_gain_gt_floor_max: float | None = None
    hold_while_negative: bool = False


class HarmCfg(_Cfg):
    """HARMFUL if P(Δ < −F) ≥ p_gain_lt_neg_floor (F = the pass rule's floor F_noise(n); 0 without a floor),
    or cost ratio > max_cost_ratio and the pass rule is not met. min_n (optional, a
    pre-set fallback): rule 2a is checked only at block boundaries with n ≥ min_n."""

    p_gain_lt_neg_floor: float
    max_cost_ratio: float
    min_n: int | None = None  # optional fallback: the posterior harm rule (2a) may fire only once n ≥ min_n


class SubgroupCfg(_Cfg):
    """A stratum is harmful if n_s ≥ harm_min_n and P(Δ_s < −F(n_s)) ≥ harm_p_lt_neg_floor."""

    strata_file: str
    harm_min_n: int
    harm_p_lt_neg_floor: float
    support_min_n: int


class _PolicyBase(_Cfg):
    name: str
    version: int
    max_invalid_rate: float  # from configs/validity.yaml, injected by load_policy


class EvidenceGateCfg(_Cfg):
    """Minimum evidence before a sequential decision (v4, 2026-11-13; unreleased). With a near-deterministic agent
    the run-to-run floor F_noise(n) is 0 and the Bayesian bootstrap gives P = 1.00 from a single discordant pair, so
    v3 could decide on the first block of 8. min_n: no terminal decision (pass, futility, harm, cost harm,
    curtailment) before n ≥ min_n, unless n = max_n. min_discordant: PASS and the posterior harm rule (2a) also need at
    least this many discordant questions (D ≠ 0); at max_n without them the decision is UNSUPPORTED."""

    min_n: int = 0
    min_discordant: int = 0


class SequentialBayesPolicy(_PolicyBase):
    kind: Literal["sequential_bayes"]
    block_size: int
    max_n: int
    order_seed: int
    posterior: PosteriorCfg
    floor: FloorCfg
    pass_rule: PassCfg = Field(alias="pass")
    futility: FutilityCfg
    harm: HarmCfg
    subgroup: SubgroupCfg
    curtailment: bool
    strata: StrataConfig  # loaded from subgroup.strata_file
    evidence_gate: EvidenceGateCfg | None = None  # v4 option; None = v3 behaviour


class KeepIfHigherPolicy(_PolicyBase):
    kind: Literal["keep_if_higher"]
    seed: int


class FixedRepeatPolicy(_PolicyBase):
    kind: Literal["fixed_repeat"]
    seeds: list[int]
    floor: FloorCfg


Policy = SequentialBayesPolicy | KeepIfHigherPolicy | FixedRepeatPolicy
_KINDS = {"sequential_bayes": SequentialBayesPolicy, "keep_if_higher": KeepIfHigherPolicy,
          "fixed_repeat": FixedRepeatPolicy}


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _repo_root(path: Path) -> Path:
    """The directory holding the `configs/` folder the file lives in (config paths are relative to it)."""
    for parent in path.resolve().parents:
        if parent.name == "configs":
            return parent.parent
    return Path.cwd()


def load_strata(path: str | Path) -> StrataConfig:
    return StrataConfig(**_read_yaml(Path(path)))


BUNDLED = Path(__file__).resolve().parents[1] / "_data"


def _resolve(root: Path, rel: str) -> Path:
    """A config path relative to the policy's root, falling back to the copy bundled with the package (so a user's
    own policy file anywhere on disk can name the bundled validity limit and strata)."""
    candidate = root / rel
    return candidate if candidate.is_file() else BUNDLED / rel


def load_policy(path: str | Path, root: str | Path | None = None,
                validity_file: str = "configs/validity.yaml") -> Policy:
    """Read a policy YAML plus the validity limit and (for sequential_bayes) the strata file it names."""
    path = Path(path)
    root = Path(root) if root is not None else _repo_root(path)
    raw = _read_yaml(path)
    raw["max_invalid_rate"] = _read_yaml(_resolve(root, validity_file))["max_invalid_rate"]
    kind = raw.get("kind")
    if kind not in _KINDS:
        raise ValueError(f"{path}: unknown policy kind {kind!r}")
    if kind == "sequential_bayes":
        raw["strata"] = load_strata(_resolve(root, raw["subgroup"]["strata_file"]))
    return _KINDS[kind](**raw)


def model_slug(model: str) -> str:
    """File-name form of an OpenRouter model id: 'qwen/qwen3.5-9b' -> 'qwen--qwen3.5-9b' (one flat file, no subfolder).

    The floor files configs/fnoise/<slug>.yaml use it; every reader resolves the path through floor_file."""
    if not model or model.strip() != model:
        raise ValueError(f"not a model id: {model!r}")
    slug = model.replace("/", "--")
    if any(c in slug for c in r'\:*?"<>|') or slug in (".", ".."):
        raise ValueError(f"model id {model!r} cannot be a file name")
    return slug


def floor_file(policy: SequentialBayesPolicy | FixedRepeatPolicy, model: str) -> str | None:
    """The policy's floor file for this model. The template may use {model_slug} (configs/fnoise/<slug>.yaml) or the
    raw {model} (kept for old configs; a model id with '/' then names a subfolder)."""
    if not policy.floor.file:
        return None
    return policy.floor.file.format(model=model, model_slug=model_slug(model))


# ---- posterior summary and the rules applied to it ---------------------------------------------------------------


@dataclass(frozen=True)
class Summary:
    n: int
    delta_hat: float
    floor: float
    p_gain_gt_0: float
    p_gain_gt_floor: float
    p_gain_lt_0: float
    p_gain_lt_neg_floor: float  # P(Δ < −floor): the harm statistic; equals p_gain_lt_0 when floor = 0
    ci95: tuple[float, float]


def summarize(D: np.ndarray, floor: float, posterior: PosteriorCfg) -> Summary:
    draws = bayes_bootstrap(D, B=posterior.draws, rng_seed=posterior.rng_seed)
    return Summary(n=int(len(D)), delta_hat=float(np.mean(D)), floor=floor, p_gain_gt_0=prob_gt(draws, 0.0),
                   p_gain_gt_floor=prob_gt(draws, floor), p_gain_lt_0=prob_lt(draws, 0.0),
                   p_gain_lt_neg_floor=prob_lt(draws, -floor), ci95=credible_interval(draws))


def pass_met(s: Summary, rule: PassCfg) -> bool:
    if rule.p_gain_gt_0 is not None and s.p_gain_gt_0 < rule.p_gain_gt_0:
        return False
    if rule.p_gain_gt_floor is not None and s.p_gain_gt_floor < rule.p_gain_gt_floor:
        return False
    return True


def futility_met(s: Summary, rule: FutilityCfg) -> bool:
    by_zero = rule.p_gain_gt_0_max is not None and s.p_gain_gt_0 <= rule.p_gain_gt_0_max
    by_floor = rule.p_gain_gt_floor_max is not None and s.p_gain_gt_floor <= rule.p_gain_gt_floor_max
    return by_zero or by_floor
