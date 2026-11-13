"""Evidence card and human report.

build_card() turns a DecisionResult plus its context into the JSON card defined by schemas/evidence_card.schema.json;
validate_card() checks a card against that schema; render_report() writes the one-page Markdown report (decision
first, then effect, transitions, scope, cost, stopping reason, caveats).

Every field is required. null is allowed only for: prereg_tag (no tag before prereg-v1), retrieval_cache_version and
seeds entries (live retrieval / providers without seed support), noise_calibration.floor_value_at_n and
fnoise_file_hash (policies without a floor), effect.p_gain_gt_floor (no floor), transitions.conditional_regression_rate
(parent never correct), per-stratum statistics of a stratum with no questions, costs.* per-question values and ratio
(costs unknown), models (filled by a cross-model transfer check; null before), and a tested domain's effect floor and floor
probabilities (no floor in that domain).

Policies without a posterior of their own (keep_if_higher, fixed_repeat) still get the effect posterior on the card,
computed for reporting only with REPORT_POSTERIOR; it never changes their decision.

Domain scope (card schema 1.1). scope.domains_tested lists every domain the revision was measured on, each
with its effect and a label from scope.label() (the stratum thresholds: the
policy's own for sequential_bayes; for keep_if_higher / fixed_repeat, which have no subgroup thresholds, those of
configs/policies/moderate.yaml, for reporting only). The first entry is the card's own evaluation domain (role
"primary", named <benchmark>-<split>, e.g. "sealqa-v60"); add_domain_result() appends transfer
domains (e.g. a second QA benchmark).
One exception: a sequential_bayes PASS_GLOBAL labels its primary domain supported even when an early stop left
n < support_min_n, because the decision already applied the same pass rule to the whole domain (otherwise a governed
admission could activate nowhere). A keep_if_higher / fixed_repeat PASS_GLOBAL gets no such exception: if moderate's
thresholds do not support it, the card activates nowhere and the report says so.
Activation rule: a condition {"field": "domain", "level": d} activates the revision in a whole domain d, allowed only
for a tested domain labelled supported; stratum conditions ({"field": <strata field>, ...}, PASS_CONDITIONAL) apply
inside the primary domain and only to supported strata. Only admitted decisions activate anything. Every tested
domain that is not activated is listed in parent_fallback, and so are untested domains: by name when known
(CardContext.untested_domains) and always through the catch-all {"field": "domain", "level": null}.
validate_card() rejects a card that breaks this rule.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jsonschema

from verdict_gate.core.calibrate import floor_at
from verdict_gate.core.harm import conditional_regression_rate
from verdict_gate.core.paired import differences, transitions
from verdict_gate.core.policy import Policy, PosteriorCfg, SequentialBayesPolicy, Summary, load_policy, summarize
from verdict_gate.core.schemas import Decision, PairedTable
from verdict_gate.core.scope import HARMFUL, INCONCLUSIVE, SUPPORTED, label

CARD_VERSION = "verdict.evidence_card/1.1"
# Bundled data (card schema, policies, validity limit, strata, example data) ships inside the package.
ROOT = Path(__file__).resolve().parents[1] / "_data"
SCHEMA_PATH = ROOT / "schemas" / "evidence_card.schema.json"
REPORT_POSTERIOR = PosteriorCfg(method="bayes_bootstrap", draws=10_000, rng_seed=9001)
LABEL_POLICY_PATH = ROOT / "configs" / "policies" / "moderate.yaml"  # domain labels for policies without strata
DOMAIN = "domain"  # condition field for a whole domain; level None = any untested domain
ADMITTED = (Decision.PASS_GLOBAL.value, Decision.PASS_CONDITIONAL.value)


def policy_hash(policy: Policy) -> str:
    """SHA-256 of the policy config as canonical JSON (every resolved field, including strata and validity limit)."""
    text = json.dumps(policy.model_dump(mode="json", by_alias=True), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class CardContext:
    """Everything on the card that the decision itself does not carry."""

    revision_id: str
    parent_id: str
    revision_type: str
    revision_text_hash: str
    evaluation_context: dict  # benchmark, split, model, provider, seeds, retrieval_cache_version, max_tokens
    git_commit: str
    prereg_tag: str | None
    created_at: str  # ISO 8601
    source_run_ids: list[str]
    preliminary: bool
    floor_type: str  # paired_fnoise | legacy_fnoise | none
    fnoise_file_hash: str | None
    fnoise_source_run_ids: list[str]
    costs: dict  # parent_usd_per_q, candidate_usd_per_q, cost_ratio, line_items
    sequential: dict  # blocks_evaluated, questions_used, stopping_reason
    floor_value: float | None = None  # floor for policies that do not report it in their summary (fixed_repeat)
    models: dict | None = None  # cross-model transfer: {supported, inconclusive, harmed}
    held_out_run_ids: list[str] = field(default_factory=list)
    domain: str | None = None  # primary domain name; default <benchmark>-<split>, lower case (e.g. sealqa-v60)
    untested_domains: list[str] = field(default_factory=list)  # named in parent_fallback (beside the catch-all)


def domain_name(benchmark: str, split: str) -> str:
    return f"{benchmark}-{split}".lower()


@functools.cache
def _report_label_policy() -> SequentialBayesPolicy:
    return load_policy(LABEL_POLICY_PATH)


def label_policy(policy: Policy) -> SequentialBayesPolicy:
    """The policy whose stratum thresholds label domains: its own, else moderate's (reporting only)."""
    return policy if isinstance(policy, SequentialBayesPolicy) else _report_label_policy()


def _posterior(policy: Policy) -> PosteriorCfg:
    return policy.posterior if isinstance(policy, SequentialBayesPolicy) else REPORT_POSTERIOR


def _effect_fields(s: Summary, posterior: PosteriorCfg, floor: float | None) -> dict:
    return {
        "n": s.n, "delta_hat": s.delta_hat, "posterior_method": posterior.method, "rng_seed": posterior.rng_seed,
        "p_gain_gt_0": s.p_gain_gt_0, "p_gain_gt_floor": s.p_gain_gt_floor if floor is not None else None,
        "p_gain_lt_0": s.p_gain_lt_0, "p_gain_lt_neg_floor": s.p_gain_lt_neg_floor if floor is not None else None,
        "ci95": [s.ci95[0], s.ci95[1]],
    }


def _effect(result, policy: Policy, table: PairedTable, floor_value: float | None) -> tuple[dict, float | None, Summary]:
    if result.summary is not None:  # sequential_bayes: the posterior the decision used
        s, posterior = result.summary, policy.posterior
        floor = s.floor
    else:
        posterior, floor = REPORT_POSTERIOR, floor_value
        s = summarize(differences(table), floor if floor is not None else 0.0, posterior)
    return _effect_fields(s, posterior, floor), floor, s


def _domain_entry(domain: str, role: str, benchmark: str, split: str, seeds, run_ids, s: Summary,
                  floor: float | None, policy: Policy) -> dict:
    effect = {k: v for k, v in _effect_fields(s, _posterior(policy), floor).items() if k != "n"}
    return {"domain": domain, "role": role, "benchmark": benchmark, "split": split, "n": s.n, "seeds": list(seeds),
            "effect": {**effect, "floor": floor}, "label": label(s.n, s, label_policy(policy)),
            "run_ids": list(run_ids)}


def _cond(level: str | None) -> dict:
    return {"field": DOMAIN, "level": level}


def _apply_domain_scope(card: dict) -> None:
    """Rebuild the domain conditions from scope.domains_tested (stratum conditions are kept as decided).

    PASS_GLOBAL activates every supported domain; PASS_CONDITIONAL activates its supported strata inside the primary
    domain plus every supported transfer domain; other decisions activate nothing. Every tested domain not activated,
    every named untested domain and the catch-all go to parent_fallback."""
    domains = card["scope"]["domains_tested"]
    primary = domains[0]["domain"]
    tested = [d["domain"] for d in domains]
    supported = [d["domain"] for d in domains if d["label"] == SUPPORTED]
    strata_on = [c for c in card["activation_conditions"] if c["field"] != DOMAIN]
    on = {Decision.PASS_GLOBAL.value: supported,
          Decision.PASS_CONDITIONAL.value: [d for d in supported if d != primary]}.get(card["decision"], [])
    untested = [c["level"] for c in card["parent_fallback"]
                if c["field"] == DOMAIN and c["level"] is not None and c["level"] not in tested]
    off = [d for d in tested if d not in on and not (d == primary and strata_on)] + untested
    card["activation_conditions"] = strata_on + [_cond(d) for d in on]
    card["parent_fallback"] = ([c for c in card["parent_fallback"] if c["field"] != DOMAIN]
                               + [_cond(d) for d in off] + [_cond(None)])


def add_domain_result(card: dict, domain: str, table: PairedTable, policy: Policy, *, benchmark: str, split: str,
                      seeds, run_ids, floor_by_n: dict[int, float] | None = None) -> dict:
    """Write one more tested domain (transfer, e.g. "frames-single-label") into the card, in place.

    `table` = the frozen revision vs its parent on that domain; the label uses the same thresholds as the card's
    strata (label_policy). floor_by_n = the domain's frozen F_noise(n); None = no floor there (F = 0). The activation
    and parent-fallback conditions are rebuilt. Returns the card."""
    if policy_hash(policy) != card["policy"]["config_hash"]:
        raise ValueError(f"policy {policy.name} is not the policy of card {card['revision_id']}")
    if domain in {d["domain"] for d in card["scope"]["domains_tested"]}:
        raise ValueError(f"domain {domain!r} is already on card {card['revision_id']}")
    if table.n == 0:
        raise ValueError(f"domain {domain!r}: no paired questions")
    floor = floor_at(floor_by_n, table.n) if floor_by_n else None
    s = summarize(differences(table), floor if floor is not None else 0.0, _posterior(policy))
    card["scope"]["domains_tested"].append(_domain_entry(domain, "transfer", benchmark, split, seeds, run_ids, s,
                                                         floor, policy))
    _apply_domain_scope(card)
    return card


def build_card(result, policy: Policy, table: PairedTable, ctx: CardContext) -> dict:
    """Card for one decision. `table` = the paired questions the decision used (for sequential: the ones seen)."""
    effect, floor, summary = _effect(result, policy, table, ctx.floor_value)
    ec = ctx.evaluation_context
    primary = _domain_entry(ctx.domain or domain_name(ec["benchmark"], ec["split"]), "primary", ec["benchmark"],
                            ec["split"], ec["seeds"], ctx.source_run_ids, summary, floor, policy)
    if isinstance(policy, SequentialBayesPolicy) and result.decision == Decision.PASS_GLOBAL:
        primary["label"] = SUPPORTED  # same pass rule on the whole domain; early stop may leave n < support_min_n
    t = transitions(table)
    strata = [asdict(e) for e in result.strata]
    card = {
        "card_version": CARD_VERSION,
        "created_at": ctx.created_at,
        "git_commit": ctx.git_commit,
        "prereg_tag": ctx.prereg_tag,
        "revision_id": ctx.revision_id,
        "parent_id": ctx.parent_id,
        "revision_type": ctx.revision_type,
        "revision_text_hash": ctx.revision_text_hash,
        "evaluation_context": dict(ctx.evaluation_context),
        "policy": {"name": policy.name, "version": policy.version, "config_hash": policy_hash(policy)},
        "noise_calibration": {"floor_type": ctx.floor_type, "floor_value_at_n": floor,
                              "fnoise_file_hash": ctx.fnoise_file_hash,
                              "source_run_ids": list(ctx.fnoise_source_run_ids)},
        "effect": effect,
        "transitions": {"repairs": t.repairs, "regressions": t.regressions, "preserved": t.preserved,
                        "unresolved": t.unresolved, "unstable": t.unstable,
                        "conditional_regression_rate": conditional_regression_rate(t)},
        "scope": {"strata": strata,
                  "supported": [e["level"] for e in strata if e["label"] == SUPPORTED],
                  "inconclusive": [e["level"] for e in strata if e["label"] == INCONCLUSIVE],
                  "harmed": [e["level"] for e in strata if e["label"] == HARMFUL],
                  "domains_tested": [primary]},
        "models": ctx.models,
        "costs": dict(ctx.costs),
        "sequential": dict(ctx.sequential),
        "decision": result.decision.value,
        "decision_reason": result.reason,
        "activation_conditions": list(result.activation_conditions),
        "inhibition_conditions": list(result.inhibition_conditions),
        "parent_fallback": list(result.parent_fallback) + [_cond(d) for d in ctx.untested_domains],
        "held_out_replication": {"status": result.held_out_replication, "run_ids": list(ctx.held_out_run_ids)},
        "source_run_ids": list(ctx.source_run_ids),
        "preliminary": ctx.preliminary,
    }
    _apply_domain_scope(card)
    return card


# ---- validation -----------------------------------------------------------------------------------------------------


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_card(card: dict) -> list[str]:
    """Schema errors, then activation-rule errors, as readable strings ([] = valid)."""
    validator = jsonschema.Draft202012Validator(load_schema())
    errors = sorted(validator.iter_errors(card), key=lambda e: list(e.absolute_path))
    if errors:
        return [f"{'/'.join(map(str, e.absolute_path)) or '<card>'}: {e.message}" for e in errors]
    return scope_errors(card)


def scope_errors(card: dict) -> list[str]:
    """The activation rule (module docstring) on a schema-valid card."""
    domains = card["scope"]["domains_tested"]
    labels = {d["domain"]: d["label"] for d in domains}
    errors = [] if len(labels) == len(domains) else ["scope/domains_tested: a domain is listed twice"]
    if domains[0]["role"] != "primary" or any(d["role"] != "transfer" for d in domains[1:]):
        errors.append("scope/domains_tested: the first entry must be the primary domain, the others transfer")
    strata_fields = {st["field"] for st in card["scope"]["strata"]}
    domains_on, strata_on = set(), False
    for c in card["activation_conditions"]:
        if c["field"] == DOMAIN:
            if c["level"] not in labels:
                errors.append(f"activation_conditions: domain {c['level']!r} is untested")
            elif labels[c["level"]] != SUPPORTED:
                errors.append(f"activation_conditions: domain {c['level']!r} is {labels[c['level']]}, not supported")
            domains_on.add(c["level"])
        elif c["field"] in strata_fields and c["level"] in card["scope"]["supported"]:
            strata_on = True
        else:
            errors.append(f"activation_conditions: {c['field']} = {c['level']!r} is not a supported stratum")
    if card["activation_conditions"] and card["decision"] not in ADMITTED:
        errors.append(f"activation_conditions: a {card['decision']} revision activates nowhere")
    fallback = {c["level"] for c in card["parent_fallback"] if c["field"] == DOMAIN}
    for d, lab in labels.items():
        activated = d in domains_on or (d == domains[0]["domain"] and strata_on)
        if not activated and d not in fallback:
            errors.append(f"parent_fallback: domain {d!r} ({lab}) is not activated and not listed")
    return errors


# ---- human report ---------------------------------------------------------------------------------------------------


def _num(x, fmt="{:.3f}") -> str:
    return "n/a" if x is None else fmt.format(x)


def _level(c: dict) -> str:
    if c["level"] is not None:
        return c["level"]
    return "(any untested domain)" if c["field"] == DOMAIN else "(missing)"


def _conditions(items: list[dict]) -> str:
    if not items:
        return "none"
    return ", ".join(f"{c['field']} = {_level(c)}" for c in items)


def render_report(card: dict) -> str:
    """One-page Markdown report: decision first, then effect, transitions, scope, cost, stopping reason, caveats."""
    e, t, c, s = card["effect"], card["transitions"], card["costs"], card["sequential"]
    nc, ctx, pol = card["noise_calibration"], card["evaluation_context"], card["policy"]
    lines = [
        f"# Evidence card: {card['revision_id']} vs {card['parent_id']}",
        "",
        f"## Decision: **{card['decision']}**",
        "",
        card["decision_reason"],
        "",
    ]
    lines += [f"- Activate for: {_conditions(card['activation_conditions'])}"]
    if card["decision"] == Decision.PASS_CONDITIONAL.value:
        lines += [f"- Inhibit for: {_conditions(card['inhibition_conditions'])}"]
    lines += [f"- Parent fallback: {_conditions(card['parent_fallback'])}", ""]
    lines += [
        f"Policy `{pol['name']}` v{pol['version']} (config {pol['config_hash'][:12]}); "
        f"held-out replication: {card['held_out_replication']['status']}.",
        "",
        "## Effect",
        "",
        f"- n = {e['n']} paired questions; Δ̂ = {e['delta_hat']:+.3f}; "
        f"95% interval [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]",
        f"- P(Δ > 0) = {e['p_gain_gt_0']:.3f}; P(Δ > floor) = {_num(e['p_gain_gt_floor'])}; "
        f"P(Δ < 0) = {e['p_gain_lt_0']:.3f}; P(Δ < −floor) = {_num(e.get('p_gain_lt_neg_floor'))}",
        f"- Floor: {nc['floor_type']} = {_num(nc['floor_value_at_n'])}; posterior {e['posterior_method']}, "
        f"rng seed {e['rng_seed']}",
        "",
        "## Transitions",
        "",
        "| repairs | regressions | net | preserved | unresolved | unstable | cond. regression rate |",
        "|---|---|---|---|---|---|---|",
        f"| {t['repairs']} | {t['regressions']} | {t['repairs'] - t['regressions']:+d} | {t['preserved']} | "
        f"{t['unresolved']} | {t['unstable']} | {_num(t['conditional_regression_rate'])} |",
        "",
        "## Scope",
        "",
        "| domain | role | benchmark / split | n | seeds | Δ̂ | 95% interval | P(Δ>0) | P(Δ<0) | label | runs |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in card["scope"]["domains_tested"]:
        de = d["effect"]
        lines.append(f"| {d['domain']} | {d['role']} | {d['benchmark']} / {d['split']} | {d['n']} | "
                     f"{', '.join(map(str, d['seeds']))} | {de['delta_hat']:+.3f} | "
                     f"[{de['ci95'][0]:+.3f}, {de['ci95'][1]:+.3f}] | {de['p_gain_gt_0']:.3f} | "
                     f"{de['p_gain_lt_0']:.3f} | {d['label']} | {', '.join(d['run_ids'])} |")
    lines.append("")
    if card["scope"]["strata"]:
        lines += ["| stratum | n | Δ̂ | P(Δ>0) | P(Δ<0) | R₊ | R₋ | label |", "|---|---|---|---|---|---|---|---|"]
        for st in card["scope"]["strata"]:
            lines.append(f"| {st['level']} | {st['n']} | {_num(st['delta_hat'], '{:+.3f}')} | "
                         f"{_num(st['p_gain_gt_0'])} | {_num(st['p_gain_lt_0'])} | {st['repairs']} | "
                         f"{st['regressions']} | {st['label']} |")
    else:
        lines.append("No per-stratum analysis for this policy.")
    if card["models"] is not None:
        m = card["models"]
        lines += ["", f"Models: supported {m['supported']}; inconclusive {m['inconclusive']}; harmed {m['harmed']}"]
    lines += [
        "",
        "## Cost",
        "",
        f"- Parent ${_num(c['parent_usd_per_q'], '{:.5f}')}/question; candidate "
        f"${_num(c['candidate_usd_per_q'], '{:.5f}')}/question; ratio {_num(c['cost_ratio'], '{:.2f}')}",
    ]
    for arm, items in c["line_items"].items():
        lines.append(f"- {arm} line items: " + (", ".join(f"{k} ${v:.4f}" for k, v in items.items()) or "none"))
    lines += [
        "",
        "## Stopping",
        "",
        f"{s['blocks_evaluated']} block(s), {s['questions_used']} question(s): {s['stopping_reason']}",
        "",
        "## Caveats",
        "",
    ]
    caveats = []
    if card["preliminary"]:
        caveats.append("Preliminary: single-run numbers.")
    if card["held_out_replication"]["status"] == "pending":
        caveats.append("Admission is not yet replicated on held-out data.")
    if card["decision"] in ADMITTED and not card["activation_conditions"]:
        caveats.append("Admitted, but no tested domain or stratum is labelled supported: the parent is used "
                       "everywhere.")
    if c["cost_ratio"] is None:
        caveats.append("Cost ratio unknown: cost conditions were not applied.")
    if t["unstable"]:
        caveats.append(f"{t['unstable']} question(s) unstable across repeats.")
    if card["prereg_tag"] is None:
        caveats.append("No pre-registration tag.")
    caveats.append(f"Model {ctx.get('model')} on {ctx.get('benchmark')} {ctx.get('split')}; "
                   f"runs: {', '.join(card['source_run_ids'])}.")
    lines += [f"- {x}" for x in caveats]
    return "\n".join(lines) + "\n"
