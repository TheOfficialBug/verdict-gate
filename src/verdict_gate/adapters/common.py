"""Shared pieces of the adapters: the UNKNOWN sentinels and the AdaptedRun container."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from verdict_gate.core.schemas import RunManifest, TaskOutcome, task_subset_hash

UNKNOWN = "UNKNOWN"  # str fields
UNKNOWN_INT = -1  # int fields (token and tool-call counts)
UNKNOWN_FLOAT = float("nan")  # float fields (latency, costs, wall time): sums over them stay NaN, never a fake 0
UNKNOWN_TIME = datetime(1970, 1, 1, tzinfo=UTC)  # RunManifest.timestamp


def read_jsonl(path: str | Path) -> list[dict]:
    """One JSON object per line. Splits on '\\n' only: trace text may hold U+2028 and other characters that
    str.splitlines() would treat as line breaks inside a record."""
    import json

    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def is_unknown(value) -> bool:
    if isinstance(value, float):
        return math.isnan(value)
    return value is None or value == UNKNOWN or value == UNKNOWN_INT or value == UNKNOWN_TIME


@dataclass
class AdaptedRun:
    """One run in the analysis schema, plus everything recorded that the schema has no field for.

    unknown_fields: manifest fields (and "outcome.<field>" for per-task fields) the artifacts do not record.
    source:         run-level recorded values kept verbatim (e.g. the harness's own accuracy), so nothing is lost.
    task_extras:    task_id -> recorded per-task values the schema has no field for (category, steps, raw score...).
    """

    manifest: RunManifest
    outcomes: list[TaskOutcome]
    unknown_fields: list[str] = field(default_factory=list)
    source: dict = field(default_factory=dict)
    task_extras: dict[str, dict] = field(default_factory=dict)


def identify_split(task_ids: list[str], splits_dir: str | Path | None) -> str:
    """Name of the split file under splits_dir whose ids are exactly these ids, else UNKNOWN.

    Matching is by content (the hash of the sorted ids), never by a file or run name. *.txt files hold one id per
    line; *.json files need an "ids" list.
    """
    if splits_dir is None or not Path(splits_dir).is_dir():
        return UNKNOWN
    import json

    want = task_subset_hash(task_ids)
    for path in sorted(Path(splits_dir).iterdir()):
        if path.suffix == ".txt":
            ids = path.read_text(encoding="utf-8").split()
        elif path.suffix == ".json":
            ids = json.loads(path.read_text(encoding="utf-8")).get("ids")
            if not isinstance(ids, list):
                continue
        else:
            continue
        if task_subset_hash(ids) == want:
            return path.stem
    return UNKNOWN
