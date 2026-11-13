"""Generic long-format import for external users. Format: examples/README.md.

One row per task execution, in CSV or JSON Lines (one object per line) or a JSON list of objects:

    required   run_id, task_id, score, valid
    optional   seed, stratum, model, provider, revision_id, failure_type, retry_of,
               tokens_in, tokens_out, tool_calls, latency_s, cost_usd

`valid` is true/false (also 1/0, yes/no). A valid row needs a score; an invalid row needs a failure_type and its
score is ignored (an invalid execution is not a wrong answer). Optional columns left out or empty become the
UNKNOWN sentinels of verdict.adapters. Rows may hold several runs; each run must name one model, provider, seed and
revision. Run-level metadata the rows cannot carry (temperature, max_tokens, costs, ...) may be passed as `run_info`.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

from verdict_gate.adapters.common import (
    UNKNOWN,
    UNKNOWN_FLOAT,
    UNKNOWN_INT,
    UNKNOWN_TIME,
    AdaptedRun,
    identify_split,
    read_jsonl,
)
from verdict_gate.core.schemas import RevisionType, RunManifest, TaskOutcome, task_subset_hash

REQUIRED = ("run_id", "task_id", "score", "valid")
OPTIONAL_STR = ("stratum", "model", "provider", "revision_id", "failure_type", "retry_of")
OPTIONAL_INT = ("seed", "tokens_in", "tokens_out", "tool_calls")
OPTIONAL_FLOAT = ("latency_s", "cost_usd")
COLUMNS = REQUIRED + OPTIONAL_STR + OPTIONAL_INT + OPTIONAL_FLOAT
_TRUE, _FALSE = {"true", "1", "yes", "y", "t"}, {"false", "0", "no", "n", "f"}


def _empty(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def _bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise ValueError(f"valid must be true/false, got {v!r}")


def read_rows(path: str | Path) -> list[dict]:
    """Rows of a .csv, .jsonl or .json (list of objects) file."""
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".json":
        return list(json.loads(path.read_text(encoding="utf-8")))
    raise ValueError(f"{path}: expected .csv, .jsonl or .json")


def row_to_outcome(row: dict) -> TaskOutcome:
    missing = [c for c in REQUIRED if c not in row or (c != "score" and _empty(row[c]))]
    if missing:
        raise ValueError(f"row {row!r} lacks required column(s) {missing}")
    unknown_cols = sorted(set(row) - set(COLUMNS))
    if unknown_cols:
        raise ValueError(f"unknown column(s) {unknown_cols}; known: {list(COLUMNS)}")
    valid = _bool(row["valid"])
    get_str = lambda c: UNKNOWN if _empty(row.get(c)) else str(row[c])
    get_int = lambda c: UNKNOWN_INT if _empty(row.get(c)) else int(row[c])
    get_float = lambda c: UNKNOWN_FLOAT if _empty(row.get(c)) else float(row[c])
    return TaskOutcome(
        run_id=str(row["run_id"]), task_id=str(row["task_id"]),
        stratum=None if _empty(row.get("stratum")) else str(row["stratum"]),
        model=get_str("model"), provider=get_str("provider"),
        seed=None if _empty(row.get("seed")) else int(row["seed"]), revision_id=get_str("revision_id"),
        score=(None if _empty(row["score"]) else float(row["score"])) if valid else None,
        valid_execution=valid, failure_type=None if _empty(row.get("failure_type")) else str(row["failure_type"]),
        retry_of=None if _empty(row.get("retry_of")) else str(row["retry_of"]),
        tokens_in=get_int("tokens_in"), tokens_out=get_int("tokens_out"), tool_calls=get_int("tool_calls"),
        latency_s=get_float("latency_s"), cost_usd=get_float("cost_usd"),
    )


def outcome_to_row(o: TaskOutcome) -> dict:
    """Inverse of row_to_outcome (UNKNOWN sentinels become empty cells)."""
    out = {}
    for c in COLUMNS:
        v = o.valid_execution if c == "valid" else getattr(o, c)
        if v is None or v == UNKNOWN or (c in OPTIONAL_INT and c != "seed" and v == UNKNOWN_INT) or \
                (isinstance(v, float) and v != v):
            v = ""
        out[c] = v
    return out


def write_csv(path: str | Path, outcomes: Iterable[TaskOutcome]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(COLUMNS))
        w.writeheader()
        for o in outcomes:
            w.writerow(outcome_to_row(o))


def import_runs(path: str | Path, benchmark: str = UNKNOWN, splits_dir: str | Path | None = None,
                run_info: dict[str, dict] | None = None) -> list[AdaptedRun]:
    """One AdaptedRun per run_id, in first-seen order. run_info[run_id] may set any RunManifest field."""
    by_run: dict[str, list[TaskOutcome]] = {}
    for row in read_rows(path):
        o = row_to_outcome(row)
        by_run.setdefault(o.run_id, []).append(o)
    return [_manifest_for(run_id, outs, benchmark, splits_dir, (run_info or {}).get(run_id, {}))
            for run_id, outs in by_run.items()]


def _manifest_for(run_id: str, outs: list[TaskOutcome], benchmark: str, splits_dir, info: dict) -> AdaptedRun:
    def one(field):
        values = {getattr(o, field) for o in outs}
        if len(values) != 1:
            raise ValueError(f"run {run_id}: {field} differs between rows ({sorted(map(str, values))})")
        return values.pop()

    ids = sorted({o.task_id for o in outs})
    known_costs = [o.cost_usd for o in outs if o.cost_usd == o.cost_usd]
    fields = dict(
        run_id=run_id, timestamp=UNKNOWN_TIME, git_commit=UNKNOWN, prereg_tag=None, benchmark=benchmark,
        benchmark_version=UNKNOWN, split=identify_split(ids, splits_dir), task_subset_hash=task_subset_hash(ids),
        model=one("model"), openrouter_model_id=UNKNOWN, provider=one("provider"), seed=one("seed"),
        temperature=UNKNOWN_FLOAT, max_tokens=UNKNOWN_INT, revision_id=one("revision_id"), parent_id=None,
        revision_type=RevisionType.SKILL, policy_version=None, retrieval_cache_version=None,
        environment_version=UNKNOWN, estimated_cost_usd=UNKNOWN_FLOAT,
        actual_cost_usd=sum(known_costs) if len(known_costs) == len(outs) else UNKNOWN_FLOAT,
        cost_line_items={}, wall_time_s=UNKNOWN_FLOAT,
        invalid_execution_count=sum(not o.valid_execution for o in outs), preliminary=True, checks={},
        closed_hash=None,
    )
    fields.update(info)
    manifest = RunManifest(**fields)
    unknown = [k for k, v in manifest.model_dump().items()
               if k not in info and (v == UNKNOWN or v == UNKNOWN_INT or (isinstance(v, float) and v != v)
                                     or (k == "seed" and v is None) or (k == "timestamp" and v == UNKNOWN_TIME))]
    return AdaptedRun(manifest, outs, sorted(unknown), {"file_rows": len(outs)})
