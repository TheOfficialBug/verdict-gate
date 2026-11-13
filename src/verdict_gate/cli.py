"""Command-line interface: `verdict-gate` (short alias `vgate`).

Commands: calibrate (noise floor from parent repeats), import (long-format results -> arm file), compare (one
decision; the exit code is the decision), next-batch (sequential runs in batches), replay (recompute decisions from
stored files), card render (Markdown report), validate (check a file before you rely on it), policy (list or show the
bundled policies), example (write the example data). No command makes a network call.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from verdict_gate import __version__
from verdict_gate.adapters.json_csv import import_runs
from verdict_gate.core.calibrate import (
    flip_rate,
    legacy_fnoise,
    paired_fnoise,
    repeats_from_outcomes,
    sensitivity,
    write_fnoise,
)
from verdict_gate.core.decision import EvidenceInvalid
from verdict_gate.core.evidence_card import ROOT as DATA_ROOT
from verdict_gate.core.evidence_card import file_hash, render_report, validate_card
from verdict_gate.core.multiplicity import round_accounting
from verdict_gate.core.policy import KeepIfHigherPolicy, _repo_root, _resolve, floor_file, load_policy
from verdict_gate.core.replay import (
    NO_FLOOR,
    Floors,
    check_agreement,
    evaluate,
    load_ablation_config,
    load_arm,
    next_batch,
    replay,
    run_ablations,
    with_order_seed,
    write_table,
)
from verdict_gate.core.schemas import SchemaError, read_split

app = typer.Typer(help="VERDICT (research preview): evidence-governed keep/reject decisions for changes to LLM "
                       "agents. Docs: https://github.com/TheOfficialBug/verdict-gate", no_args_is_help=True)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"verdict-gate {__version__}")
        raise typer.Exit()


@app.callback()
def main(version: bool = typer.Option(False, "--version", callback=_version, is_eager=True,
                                      help="print the version and exit")) -> None:
    """VERDICT command-line interface."""
    # Decision reasons carry Δ, ≥ and P(·); a Windows console's legacy code page (cp1252) cannot encode them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(errors="replace")


# ---- compare, replay, card render ----------------------------------------------------------------------

EXIT_INVALID_EVIDENCE = 2
EXIT_REPLAY_DISAGREES = 1  # live decisions differ from the replay
EXIT_BY_DECISION = {"PASS_GLOBAL": 0, "PASS_CONDITIONAL": 10, "NEEDS_MORE_EVIDENCE": 20, "UNSUPPORTED": 30,
                    "HARMFUL": 40}
INVALID_ERRORS = (EvidenceInvalid, SchemaError, ValidationError, KeyError, ValueError)

card_app = typer.Typer(help="Evidence cards: render a card as a Markdown report.", no_args_is_help=True)
app.add_typer(card_app, name="card")


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _floors(policy_path: Path, policy, arms: list, fnoise: Path | None) -> dict:
    """Floors per model: --fnoise for every model, else the policy's floor file (configs/fnoise/{model_slug}.yaml)."""
    if isinstance(policy, KeepIfHigherPolicy):
        return {}
    floors = {}
    for model in {a.manifests[0].model for a in arms}:
        path = fnoise or _repo_root(policy_path) / floor_file(policy, model)
        if not Path(path).is_file():
            raise ValueError(f"no noise-floor file for model {model!r} ({path}); make one with `verdict-gate calibrate` "
                             "and pass it as --fnoise")
        floors[model] = Floors.from_file(path)
    return floors


def _policy_path(policy: str) -> Path:
    """A policy YAML path, or the name of a bundled policy: naive, moderate, strict, fixed_repeat."""
    path = Path(policy)
    if path.is_file():
        return path
    bundled = DATA_ROOT / "configs" / "policies" / f"{policy}.yaml"
    if bundled.is_file():
        return bundled
    names = sorted(p.stem for p in (DATA_ROOT / "configs" / "policies").glob("*.yaml"))
    raise typer.BadParameter(f"{policy!r} is neither a policy file nor a bundled policy ({', '.join(names)})")


PolicyOpt = typer.Option(..., help="bundled policy name (naive, moderate, strict, fixed_repeat) or a policy YAML")


def _invalid(e) -> typer.Exit:
    typer.echo(f"INVALID EVIDENCE: {e}", err=True)
    return typer.Exit(EXIT_INVALID_EVIDENCE)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@app.command("compare")
def compare(
    baseline: Path = typer.Option(..., help="parent arm file (JSON: manifests + outcomes)"),
    candidate: Path = typer.Option(..., help="candidate arm file (JSON: manifests + outcomes + revision_text_hash)"),
    policy: str = PolicyOpt,
    out: Path = typer.Option(..., help="evidence card JSON to write"),
    fnoise: Path | None = typer.Option(None, help="noise-floor file (verdict-gate calibrate); default: the policy's "
                                                  "configs/fnoise/{model_slug}.yaml"),
) -> None:
    """Decide one candidate against its parent. The exit code is the decision (0 PASS_GLOBAL, 10 PASS_CONDITIONAL,
    20 NEEDS_MORE_EVIDENCE, 30 UNSUPPORTED, 40 HARMFUL); 2 = invalid evidence.

    Arm files come from `verdict-gate import` (long-format CSV/JSONL results)."""
    policy = _policy_path(policy)
    try:
        pol = load_policy(policy)
        parent_arm, cand_arm = load_arm(baseline), load_arm(candidate)
        floors = _floors(policy, pol, [cand_arm], fnoise)
        card = evaluate(parent_arm, cand_arm, pol, floors.get(cand_arm.manifests[0].model, NO_FLOOR),
                        git_commit=_git_commit())
    except INVALID_ERRORS as e:
        raise _invalid(e) from None
    errors = validate_card(card)
    if errors:
        raise _invalid("card fails the schema: " + "; ".join(errors))
    _write_json(out, card)
    typer.echo(f"{card['decision']}: {card['decision_reason']}")
    raise typer.Exit(EXIT_BY_DECISION[card["decision"]])


@app.command("replay")
def replay_cmd(
    policy: str = PolicyOpt,
    runs: list[Path] = typer.Option(..., help="arm files (repeat --runs); candidates pair with their parent_id"),
    out: Path = typer.Option(..., help="replay JSON to write (cards + round accounting)"),
    order_seed: int | None = typer.Option(None, help="block order seed; default: the policy's order_seed"),
    fnoise: Path | None = typer.Option(None, help="frozen floor file; default: the policy's per-model file"),
    live: Path | None = typer.Option(None, help="live cards JSON (list); exit 1 if the replay disagrees"),
    ablations: Path | None = typer.Option(None, help="ablation variants YAML (policy variants to replay side by side)"),
    table: Path | None = typer.Option(None, help="ablation table CSV (default: <out>.ablations.csv)"),
) -> None:
    """Recompute decisions from stored arm files (no network). Exit 0; 2 invalid evidence; 1 live != replay."""
    policy = _policy_path(policy)
    try:
        pol = load_policy(policy)
        arms = [load_arm(p) for p in runs]
        cards = replay(pol, arms, order_seed, _floors(policy, pol, arms, fnoise), git_commit=_git_commit())
        rows = None
        if ablations is not None:
            cfg = load_ablation_config(ablations)
            base = load_policy(_resolve(_repo_root(ablations), cfg.base_policy))
            rows = run_ablations(cfg, arms, _floors(policy, base, arms, fnoise), _repo_root(ablations))
    except INVALID_ERRORS as e:
        raise _invalid(e) from None
    for c in cards:
        errors = validate_card(c)
        if errors:
            raise _invalid(f"card {c['revision_id']} fails the schema: " + "; ".join(errors))
    _write_json(out, {"policy": pol.name, "order_seed": getattr(with_order_seed(pol, order_seed), "order_seed", None),
                      "cards": cards,
                      "accounting": round_accounting(cards, generated=len(cards), max_n=getattr(pol, "max_n", None))})
    for c in cards:
        typer.echo(f"{c['revision_id']}: {c['decision']} (n={c['sequential']['questions_used']})")
    if rows is not None:
        table = table or out.with_suffix(".ablations.csv")
        write_table(rows, table)
        typer.echo(f"{len(rows)} ablation row(s) written to {table}")
    if live is not None:
        problems = check_agreement(json.loads(live.read_text(encoding="utf-8")), cards)
        for p in problems:
            typer.echo(f"DISAGREES: {p}", err=True)
        if problems:
            raise typer.Exit(EXIT_REPLAY_DISAGREES)
        typer.echo("live decisions equal replay")


@card_app.command("render")
def card_render(
    card: Path = typer.Argument(..., help="evidence card JSON"),
    out: Path | None = typer.Option(None, help="write the Markdown report here instead of stdout"),
) -> None:
    """One-page Markdown report of a card (validated against the schema first; exit 2 if invalid)."""
    data = json.loads(card.read_text(encoding="utf-8"))
    errors = validate_card(data)
    if errors:
        raise _invalid("card fails the schema: " + "; ".join(errors))
    text = render_report(data)
    if out is None:
        typer.echo(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


# ---- your own results: calibrate, import, next-batch, example ------------------------------------------------

BLOCK_SIZES = "8,16,24,32,40,48,56,60"  # the block sizes of the shipped configs/fnoise/*.yaml
EXAMPLE_FILES = ("results.csv", "candidate_skill.md", "tasks.txt")  # examples/byot, bundled in the wheel


def _select_runs(results: Path, run: list[str], revision: str | None, benchmark: str = "UNKNOWN") -> list:
    """The runs of a long-format results file named by --run, or every run of --revision."""
    if not run and revision is None:
        raise typer.BadParameter("name the runs with --run (repeat) or --revision")
    try:
        runs = import_runs(results, benchmark=benchmark)
    except (ValueError, ValidationError) as e:
        raise _invalid(e) from None
    chosen = [r for r in runs if r.manifest.run_id in run or r.manifest.revision_id == revision]
    missing = sorted(set(run) - {r.manifest.run_id for r in runs})
    if missing or not chosen:
        available = ", ".join(f"{r.manifest.run_id} ({r.manifest.revision_id})" for r in runs)
        raise _invalid(f"no such run(s) {missing or revision!r} in {results}; runs (revision): {available}")
    return chosen


@app.command("calibrate")
def calibrate_cmd(
    results: Path = typer.Argument(..., help="long-format results (.csv, .jsonl, .json; see examples/README.md)"),
    out: Path = typer.Option(..., help="noise-floor YAML to write (pass it to compare/replay as --fnoise)"),
    run: list[str] = typer.Option([], help="a repeat run of the PARENT (repeat; >= 2)"),
    revision: str | None = typer.Option(None, help="use every run whose revision_id is this (instead of --run)"),
    block_sizes: str = typer.Option(BLOCK_SIZES, help="comma-separated block sizes n for F_noise(n)"),
    quantile: float = typer.Option(0.95, help="quantile of |mean of n paired differences|"),
    draws: int = typer.Option(10_000, help="bootstrap draws per block size"),
    seed: int = typer.Option(9001, help="bootstrap RNG seed (recorded in the file)"),
) -> None:
    """Noise floor F_noise(n) from >= 2 repeat runs of the unchanged parent on the same tasks.

    The floor is how far two runs of the SAME agent drift apart on n tasks; a candidate must beat it to be admitted.
    Only tasks with a valid execution in every run are used."""
    runs = _select_runs(results, run, revision)
    models = {r.manifest.model for r in runs}
    if len(models) != 1:
        raise _invalid(f"the runs use different models {sorted(models)}; calibrate one model at a time")
    sizes = sorted({int(x) for x in block_sizes.split(",") if x.strip()})
    try:
        run_ids, reps, common, excluded = repeats_from_outcomes(o for r in runs for o in r.outcomes)
    except ValueError as e:
        raise _invalid(e) from None
    paired = paired_fnoise(reps, sizes, B=draws, q=quantile, rng_seed=seed)
    means = [sum(v) / len(v) for v in reps.values()]
    flips = flip_rate(reps)
    flipping = sum(f > 0 for f in flips) / len(flips)
    write_fnoise(out, models.pop(), paired, legacy_fnoise(means, quantile), sensitivity(means), run_ids, seed, draws,
                 quantile, extra={"created_by": "verdict-gate calibrate", "source_file": results.name,
                                  "block_sizes": sizes, "n_questions": len(common), "excluded_questions": excluded,
                                  "run_means": {r: round(m, 4) for r, m in zip(run_ids, means)},
                                  "flip": {"mean_pair_discordance": round(sum(flips) / len(flips), 4),
                                           "share_questions_flipping": round(flipping, 4)}})
    typer.echo(f"{len(run_ids)} runs x {len(common)} tasks valid in every run ({len(excluded)} excluded)")
    typer.echo("run means: " + ", ".join(f"{r}={m:.3f}" for r, m in zip(run_ids, means)))
    typer.echo(f"flip rate: {flipping:.1%} of tasks change outcome between runs; mean pair discordance "
               f"{sum(flips) / len(flips):.3f}")
    typer.echo(f"{'n':>4}  F_noise(n) (q={quantile})")
    for n, f in paired.items():
        typer.echo(f"{n:>4}  {f:.4f}")
    typer.echo(f"floor written to {out}")


@app.command("import")
def import_cmd(
    results: Path = typer.Argument(..., help="long-format results (.csv, .jsonl, .json; see examples/README.md)"),
    out: Path = typer.Option(..., help="arm file (JSON) to write, for compare / replay / next-batch"),
    max_tokens: int = typer.Option(..., min=1, help="output token limit the agent ran with (a pairing key: parent "
                                                    "and candidate must match)"),
    run: list[str] = typer.Option([], help="run_id to include (repeat; one run per seed)"),
    revision: str | None = typer.Option(None, help="include every run whose revision_id is this (instead of --run)"),
    parent_id: str | None = typer.Option(None, help="candidate arms: the parent's revision_id (replay pairs on it)"),
    revision_file: Path | None = typer.Option(None, help="candidate arms: the prompt/skill text; its sha256 goes on "
                                                         "the card"),
    revision_hash: str | None = typer.Option(None, help="candidate arms: the revision text hash, if you have it"),
    pool: Path | None = typer.Option(None, help="all task ids of the evaluation (one per line), for batched runs"),
    benchmark: str = typer.Option("UNKNOWN", help="name of your task set, shown on the card"),
) -> None:
    """One arm file (one revision, one run per seed) from a long-format results file."""
    if parent_id is not None and revision_file is None and revision_hash is None:
        raise typer.BadParameter("a candidate arm (--parent-id) needs --revision-file or --revision-hash")
    runs = _select_runs(results, run, revision, benchmark)
    revs = {r.manifest.revision_id for r in runs}
    if len(revs) != 1:
        raise _invalid(f"an arm is one revision; the runs have revision_id {sorted(revs)}")
    manifests = [r.manifest.model_copy(update={"max_tokens": max_tokens, "parent_id": parent_id}) for r in runs]
    arm = {"revision_text_hash": file_hash(revision_file) if revision_file else revision_hash,
           "manifests": [m.model_dump() for m in manifests],
           "outcomes": [o.model_dump() for r in runs for o in r.outcomes]}
    if pool is not None:
        arm["pool"] = read_split(pool)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(arm, indent=1, default=str) + "\n", encoding="utf-8")
    seeds = ", ".join(str(m.seed) for m in manifests)
    typer.echo(f"{revs.pop()}: {len(runs)} run(s) (seeds {seeds}), {len(arm['outcomes'])} rows -> {out}")


@app.command("next-batch")
def next_batch_cmd(
    policy: str = PolicyOpt,
    pool: Path | None = typer.Option(None, help="all task ids (one per line); default: the candidate arm's pool"),
    baseline: Path | None = typer.Option(None, help="parent arm file so far (omit before the first batch)"),
    candidate: Path | None = typer.Option(None, help="candidate arm file so far (omit before the first batch)"),
    fnoise: Path | None = typer.Option(None, help="noise-floor file (verdict-gate calibrate)"),
) -> None:
    """Batched runs: stop, or which tasks to run next (blocks of the policy's block_size, up to max_n)."""
    policy = _policy_path(policy)
    try:
        pol = load_policy(policy)
        arms = [load_arm(p) for p in (baseline, candidate)] if baseline and candidate else []
        ids = read_split(pool) if pool else (arms[1].pool if arms else None)
        if not ids:
            raise ValueError("give --pool (or arm files that carry one)")
        floors = _floors(policy, pol, arms[1:], fnoise).popitem()[1] if arms else NO_FLOOR
        nb = next_batch(pol, ids, *arms, floors=floors)
    except (*INVALID_ERRORS, TypeError) as e:
        raise _invalid(e) from None
    typer.echo(f"after {nb.questions_done} task(s): {nb.decision}: {nb.reason}")
    if nb.stop:
        typer.echo("STOP: the decision is final")
        return
    typer.echo(f"run next {len(nb.task_ids)} task(s) in both arms:")
    typer.echo("\n".join(nb.task_ids))


@app.command("example")
def example_cmd(out: Path = typer.Option(Path("verdict_example"), help="folder to write the example files to")) -> None:
    """Write the bring-your-own-task example data (60 support tickets, examples/byot) to a folder."""
    src = DATA_ROOT / "examples" / "byot"
    out.mkdir(parents=True, exist_ok=True)
    for name in EXAMPLE_FILES:
        shutil.copyfile(src / name, out / name)
        typer.echo(f"wrote {out / name}")


# ---- validate, policy -------------------------------------------------------------------------------------------------


def _kind(data) -> str:
    if isinstance(data, dict) and "card_version" in data:
        return "card"
    if isinstance(data, dict) and "manifests" in data and "outcomes" in data:
        return "arm"
    return "unknown"


@app.command("validate")
def validate_cmd(path: Path = typer.Argument(..., help="results file (.csv/.jsonl/.json), arm file, floor file "
                                                       "(.yaml) or evidence card")) -> None:
    """Check a file before relying on it. Exit 0 = valid; 2 = invalid (the reason is printed)."""
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            floors = Floors.from_file(path)
            if not floors.paired:
                raise ValueError("no paired_fnoise values in the floor file")
            typer.echo(f"floor file OK: F_noise(n) for n = {sorted(floors.paired)}")
            return
        data = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else None
        kind = _kind(data)
        if kind == "card":
            errors = validate_card(data)
            if errors:
                raise ValueError("card fails the schema: " + "; ".join(errors))
            typer.echo(f"evidence card OK: {data['revision_id']} -> {data['decision']}")
        elif kind == "arm":
            arm = load_arm(path)
            seeds = sorted({m.seed for m in arm.manifests}, key=lambda s: (s is None, s or 0))
            invalid = sum(not o.valid_execution for o in arm.outcomes)
            typer.echo(f"arm file OK: revision {arm.revision_id}, {len(arm.manifests)} run(s) (seeds {seeds}), "
                       f"{len(arm.outcomes)} rows, {invalid} invalid")
        else:
            runs = import_runs(path)
            for r in runs:
                bad = sum(not o.valid_execution for o in r.outcomes)
                unknown = f"; unknown fields: {', '.join(r.unknown_fields)}" if r.unknown_fields else ""
                typer.echo(f"run {r.manifest.run_id} (revision {r.manifest.revision_id}): {len(r.outcomes)} rows, "
                           f"{bad} invalid{unknown}")
            typer.echo(f"results file OK: {len(runs)} run(s)")
    except (*INVALID_ERRORS, json.JSONDecodeError, OSError) as e:
        raise _invalid(e) from None


@app.command("policy")
def policy_cmd(name: str | None = typer.Argument(None, help="bundled policy name or a policy YAML; omit to list")) -> None:
    """List the bundled policies, or print one policy as it will be applied (with its validity limit and strata)."""
    if name is None:
        for p in sorted((DATA_ROOT / "configs" / "policies").glob("*.yaml")):
            pol = load_policy(p)
            typer.echo(f"{p.stem:14s} {pol.__class__.__name__.replace('Policy', ''):16s} version {pol.version}")
        return
    path = _policy_path(name)
    pol = load_policy(path)
    typer.echo(f"# {path.name} (sha256 {file_hash(path)[:16]})")
    typer.echo(yaml.safe_dump(json.loads(pol.model_dump_json(by_alias=True)), sort_keys=False).rstrip())
