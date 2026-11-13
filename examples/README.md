# examples

## `long_format.csv`: the generic import format (`verdict_gate.adapters.json_csv`)

One row per task execution. CSV, JSON Lines (`.jsonl`, one object per line) or a JSON list of objects (`.json`)
with the same keys.

| Column | Required | Type | Meaning |
|---|---|---|---|
| `run_id` | yes | text | One run = one arm (parent or candidate) × one seed. Rows of several runs may share a file |
| `task_id` | yes | text | Question / task id; pairing is on this id |
| `score` | yes (may be empty when `valid` is false) | number in [0, 1] | Graded outcome of a valid execution |
| `valid` | yes | true/false (1/0, yes/no) | false = infrastructure failure (timeout, provider error, grader crash...). An invalid execution is **not** a wrong answer: its score is ignored |
| `failure_type` | when `valid` is false | text | Why the execution is invalid, e.g. `timeout`, `BadRequestError` |
| `retry_of` | no | text | For a retry row: `<run_id>/<task_id>` of the invalid execution it repeats (same task_id) |
| `seed` | no | integer | Generation seed; empty = not recorded |
| `stratum` | no | text | Level of the pre-registered strata field (e.g. SealQA `freshness`); empty = unstratified |
| `model`, `provider`, `revision_id` | no | text | One value per run |
| `tokens_in`, `tokens_out`, `tool_calls` | no | integer | Per-execution counts |
| `latency_s`, `cost_usd` | no | number | Per-execution wall time and cost |

Missing or empty optional cells are never guessed: text becomes `UNKNOWN`, integers `-1`, numbers `NaN`, and the
manifest field is listed in `AdaptedRun.unknown_fields`. A run's `actual_cost_usd` is the sum of `cost_usd` only
when every row has one. Unknown columns are rejected, so a typo cannot silently drop data.

```python
from verdict_gate.adapters.json_csv import import_runs
from verdict_gate.core.paired import build_paired_table

runs = {r.manifest.run_id: r for r in import_runs("examples/long_format.csv", benchmark="my-bench",
                                                   run_info={"parent_s11": {"temperature": 0.0}})}
table = build_paired_table(runs["parent_s11"].outcomes, runs["candidate_s11"].outcomes)
```

Run-level settings that rows cannot carry (temperature, max_tokens, git commit, costs by line item...) go in
`run_info={run_id: {RunManifest field: value}}`. `write_csv(path, outcomes)` writes the same format back.

From the command line, `verdict-gate import FILE --revision R --max-tokens N --out arm.json` turns the runs of one revision
into the arm file `verdict-gate compare` reads, and `verdict-gate calibrate FILE --revision PARENT --out floor.yaml` turns the
parent's repeat runs into its noise floor. [`byot/`](byot/README.md) walks through the whole flow on 60 synthetic
support tickets.
