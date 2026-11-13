# User guide

This guide covers everything needed to use VERDICT on your own agent: the input format, the workflow, batched
evaluation, CI integration and how to read an evidence card. The statistics are in [methodology.md](methodology.md).

- [Concepts in one minute](#concepts-in-one-minute)
- [Input format](#input-format)
- [Workflow](#workflow)
- [Batched (sequential) evaluation](#batched-sequential-evaluation)
- [Evidence cards](#evidence-cards)
- [Policies](#policies)
- [CI](#ci)
- [Python API](#python-api)
- [Troubleshooting](#troubleshooting)

## Concepts in one minute

| Term | Meaning |
|---|---|
| **Parent** | The current agent (the version you would keep if the change is rejected) |
| **Candidate** | The changed agent: new prompt, skill, memory rule, model, tool, ... |
| **Arm** | All runs of one revision (parent or candidate), one run per seed |
| **Paired** | Parent and candidate are compared task by task on the same tasks and settings |
| **Noise floor F_noise(n)** | How far two runs of the *unchanged* parent drift apart on n tasks (95th percentile) |
| **Repair / regression** | A task the candidate fixes / breaks relative to the parent |
| **Evidence card** | The decision with everything needed to audit and replay it |

## Input format

One row per execution of one task, in CSV, JSON Lines (`.jsonl`) or a JSON list (`.json`). Rows of several runs may
share one file.

| Column | Required | Type | Meaning |
|---|---|---|---|
| `run_id` | yes | text | One run = one revision × one seed |
| `task_id` | yes | text | Stable task identifier; pairing is on this id |
| `score` | yes (empty when `valid` is false) | number in [0, 1] | Graded outcome of a valid execution |
| `valid` | yes | true/false (1/0, yes/no) | `false` = infrastructure failure (timeout, provider error, grader crash). **Not a wrong answer**: its score is ignored |
| `failure_type` | when `valid` is false | text | Why, e.g. `timeout` |
| `retry_of` | no | text | `<run_id>/<task_id>` of the invalid execution this row retries |
| `seed` | no | integer | Generation seed |
| `stratum` | no | text | A subgroup label fixed **before** looking at results |
| `revision_id`, `model`, `provider` | no | text | One value per run |
| `tokens_in`, `tokens_out`, `tool_calls` | no | integer | Per execution |
| `latency_s`, `cost_usd` | no | number | Per execution; costs enable the cost rule |

Missing optional values are never guessed: text becomes `UNKNOWN`, integers `-1`, numbers `NaN`, and the field is
listed as unknown. Unknown columns are rejected so a typo cannot silently drop data. A sample file is
[examples/long_format.csv](../examples/long_format.csv).

```csv
run_id,task_id,score,valid,seed,revision_id,model,provider,failure_type,cost_usd
parent-s11,ticket-001,1,true,11,parent,my-model,my-provider,,0.0012
cand-s11,ticket-001,1,true,11,cand-1,my-model,my-provider,,0.0013
cand-s11,ticket-002,,false,11,cand-1,my-model,my-provider,timeout,0.0004
```

Check a file at any time with `verdict-gate validate results.csv`.

## Workflow

**1. Freeze the evaluation design first.** Before running the candidate, write down the task list, the metric, the
model and settings, the policy and the maximum budget. Changing any of them after seeing results invalidates the
decision.

**2. Calibrate the parent** (≥ 2 repeat runs on the same tasks; 3 to 5 is better):

```bash
verdict-gate calibrate results.csv --revision parent --out floor.yaml
```

It prints the run means, the share of tasks whose outcome flips between runs, and F_noise(n) per block size.

**3. Import both arms:**

```bash
verdict-gate import results.csv --revision parent --max-tokens 512 --benchmark support-v1 --out parent.json
verdict-gate import results.csv --revision cand-1 --parent-id parent --revision-file candidate_prompt.md \
    --max-tokens 512 --benchmark support-v1 --out candidate.json
```

`--max-tokens` is a pairing key: parent and candidate must have run with the same output limit. `--revision-file`
(or `--revision-hash`) records exactly which change was evaluated.

**4. Compare:**

```bash
verdict-gate compare --baseline parent.json --candidate candidate.json --policy moderate \
    --fnoise floor.yaml --out card.json
```

**5. Read the card** (`verdict-gate card render card.json`) before acting on the decision.

**6. Replay** to confirm that the stored files reproduce the decision:

```bash
verdict-gate replay --policy moderate --runs parent.json --runs candidate.json --fnoise floor.yaml --out replay.json
```

## Batched (sequential) evaluation

Running every task in both arms is often unnecessary. `next-batch` returns either "stop" with the decision, or the
next tasks (in a fixed, seeded order) to run in **both** arms:

```bash
verdict-gate next-batch --policy moderate --pool tasks.txt --fnoise floor.yaml            # before the first batch
# ... run the listed tasks in both arms, append the rows, re-import both arms ...
verdict-gate next-batch --policy moderate --pool tasks.txt \
    --baseline parent.json --candidate candidate.json --fnoise floor.yaml               # after each batch
```

The decision equals what `compare` gives on the same tasks. From Python: `verdict_gate.next_batch(policy, pool,
parent_arm, candidate_arm, floors=...)`.

## Evidence cards

A card is JSON validated against a bundled JSON Schema. It records: the revision and its parent, the content hash of
the change, the evaluation context (benchmark, model, provider, seeds, token limit), the policy and its configuration
hash, the noise floor and where it came from, the effect estimate with its uncertainty, repairs and regressions,
per-subgroup results, costs, the stopping reason, and the conditions under which the parent should still be used.
`card render` turns it into a one-page Markdown report.

## Policies

`verdict-gate policy` lists the bundled policies; `verdict-gate policy moderate` prints one as it will be applied.

| Policy | When to use |
|---|---|
| `moderate` | Default. Sequential, paired, against the noise floor |
| `strict` | Higher bar beyond the floor |
| `moderate_v4` | Near-deterministic agents (flip rate close to 0): adds a minimum-evidence gate |
| `fixed_repeat` | When you always run a fixed design |
| `naive` | Only as a comparison baseline |

A custom policy is a YAML file (copy a bundled one). It may name `configs/validity.yaml` and a strata file relative to
its own folder; if they are not there, the bundled copies are used.

## CI

The exit code is the decision:

| Exit | Meaning |
|---:|---|
| 0 | `PASS_GLOBAL` |
| 10 | `PASS_CONDITIONAL` |
| 20 | `NEEDS_MORE_EVIDENCE` |
| 30 | `UNSUPPORTED` |
| 40 | `HARMFUL` |
| 2 | invalid evidence |

Most CI systems fail on any non-zero exit. To map decisions yourself:

```bash
set +e
verdict-gate compare --baseline parent.json --candidate candidate.json --policy moderate \
    --fnoise floor.yaml --out card.json
code=$?
set -e
case "$code" in
  0)  echo "keep the candidate" ;;
  10) echo "keep only in the supported subgroups" ;;
  20) echo "run another batch"; exit 1 ;;
  30|40) echo "keep the parent"; exit 1 ;;
  *)  echo "invalid evidence"; exit 2 ;;
esac
```

Treat a pass as evidence to review, not as automatic release authority. Keep the card with the release record.

## Python API

```python
from verdict_gate import next_batch
from verdict_gate.core.policy import load_policy
from verdict_gate.core.replay import Floors, evaluate, load_arm

policy = load_policy("my_policy.yaml")            # or the bundled: verdict_gate/_data/configs/policies/moderate.yaml
card = evaluate(load_arm("parent.json"), load_arm("candidate.json"), policy, Floors.from_file("floor.yaml"))
print(card["decision"], card["decision_reason"])
```

## Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `INVALID EVIDENCE: no noise-floor file` | `--fnoise` missing | Run `calibrate` on the parent's repeats and pass the file |
| `pairing` errors | Parent and candidate differ in model, provider, token limit or task set | Re-run with matched settings |
| `invalid rate ... > limit` | Too many failed executions | Fix the infrastructure, retry, re-import |
| Decision after very few tasks with `P(Δ>0)=1.000` | Near-deterministic agent, floor 0 | Use `moderate_v4` ([why](methodology.md#evidence-gate)) |
