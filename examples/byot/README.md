# Bring your own task: support-ticket routing

A worked example of VERDICT on results that did not come from the thesis benchmarks. Everything here is synthetic
and offline: no model call, no API key.

**The setup.** A support agent routes each of 60 customer tickets to one queue (`account`, `shipping`, `billing`,
`technical`, `escalation`). The score is exact match against the known right queue: 1 or 0. The current prompt (the
*parent*) was run three times with seeds 11, 22 and 33; a revised prompt (the *candidate*,
[`candidate_skill.md`](candidate_skill.md),
which adds billing and escalation rules) was run once, with seed 11. Three executions failed for infrastructure
reasons (two timeouts, one rate limit); two of them were retried.

The data is drawn by [`make_data.py`](make_data.py)
with a fixed seed. By construction the candidate is **no better** than the parent overall: its billing rule helps
billing tickets and hurts mixed-intent tickets by the same amount. Its one run still happens to score 5 tickets
higher (0.767 vs 0.683), which is what a single lucky run looks like.

| File | What it is |
|---|---|
| `results.csv` | 242 rows in the long format ([examples/README.md](../README.md)): one row per execution, `valid=false` + `failure_type` for infrastructure failures, `retry_of` for retries |
| `candidate_skill.md` | The candidate prompt; its sha256 goes on the evidence card |
| `tasks.txt` | The 60 ticket ids (the question pool, for batched runs) |
| `make_data.py` | Regenerates `results.csv` and `tasks.txt` (repo checkout only) |

## The flow

From `pip install verdict-gate`, `verdict-gate example --out byot && cd byot` writes these files to a folder. In a repo
checkout, run the same commands from `examples/byot/` with `uv run verdict-gate ...`. All output below is real.

**1. Calibrate the noise floor** from the parent's repeat runs: how far two runs of the *same* prompt drift apart on
n tickets.

```console
$ verdict-gate calibrate results.csv --revision parent --out floor.yaml
3 runs x 59 tasks valid in every run (1 excluded)
run means: parent_s11=0.678, parent_s22=0.763, parent_s33=0.712
flip rate: 42.4% of tasks change outcome between runs; mean pair discordance 0.282
   n  F_noise(n) (q=0.95)
   8  0.3750
  16  0.2500
  24  0.2083
  32  0.1875
  40  0.1750
  48  0.1458
  56  0.1429
  60  0.1333
floor written to floor.yaml
```

The unchanged parent already moves between 0.678 and 0.763, and 42% of tickets flip between runs. T005 is excluded:
its seed-33 execution timed out and was not retried, and an invalid execution is never counted as a wrong answer.
Options: `--block-sizes 8,16,...` (default: the shipped floors' sizes), `--quantile` (0.95), `--draws` (10000),
`--seed` (9001), or `--run ID` (repeat) instead of `--revision`.

**2. Import each arm** (one revision, one run per seed) from the long-format file. A candidate names its parent and
its prompt text. `--max-tokens` is your agent's output token limit, a pairing key that must match between arms;
`--benchmark NAME` labels the card.

```console
$ verdict-gate import results.csv --revision parent --max-tokens 256 --out parent.json
parent: 3 run(s) (seeds 11, 22, 33), 181 rows -> parent.json
$ verdict-gate import results.csv --revision cand-1 --parent-id parent --revision-file candidate_skill.md --max-tokens 256 --out candidate.json
cand-1: 1 run(s) (seeds 11), 61 rows -> candidate.json
```

**3. Compare.** `--policy` takes a bundled policy name (`naive` = keep-if-higher, `moderate`, `strict`,
`fixed_repeat`) or a policy YAML. The exit code is the decision (0 PASS_GLOBAL, 10 PASS_CONDITIONAL,
20 NEEDS_MORE_EVIDENCE, 30 UNSUPPORTED, 40 HARMFUL; 2 = invalid evidence).

```console
$ verdict-gate compare --baseline parent.json --candidate candidate.json --policy naive --out naive_card.json
PASS_GLOBAL: candidate accuracy 0.767 vs parent 0.683 (n=60)
$ verdict-gate compare --baseline parent.json --candidate candidate.json --policy moderate --fnoise floor.yaml --out card.json
UNSUPPORTED: futility rule met (n=8, Δ̂=+0.125, F=0.375, P(Δ>0)=1.000, P(Δ>F)=0.040, P(Δ<0)=0.000, P(Δ<−F)=0.000)
```

Keep-if-higher keeps the lucky run. The moderate policy visits the tickets in a fixed random order, 8 at a time, and
stops after the first 8: at that size the parent's own noise is ±0.375, so the chance that the gain clears the
floor is 4%, under the futility limit. It needed 8 paired tickets instead of 60.

**4. Render the evidence card** (`card.json` is validated against the card schema first).

```console
$ verdict-gate card render card.json
# Evidence card: cand-1 vs parent

## Decision: **UNSUPPORTED**

futility rule met (n=8, Δ̂=+0.125, F=0.375, P(Δ>0)=1.000, P(Δ>F)=0.040, P(Δ<0)=0.000, P(Δ<−F)=0.000)

- Activate for: none
- Parent fallback: domain = unknown-unknown, domain = (any untested domain)
...
```

**5. Replay** recomputes every decision from the stored arm files (candidates pair with their parent through
`--parent-id`), e.g. for an audit or to check a live run (`--live`).

```console
$ verdict-gate replay --policy moderate --runs parent.json --runs candidate.json --fnoise floor.yaml --out replay.json
cand-1: UNSUPPORTED (n=8)
```

## Running in batches

You do not have to run all 60 tickets on the candidate first. `verdict-gate next-batch` (Python: `verdict_gate.next_batch`)
says which tickets to run next, in both arms with the same seed, or to stop:

```console
$ verdict-gate next-batch --policy moderate --pool tasks.txt
after 0 task(s): NEEDS_MORE_EVIDENCE: nothing run yet
run next 8 task(s) in both arms:
T038
T060
T029
T017
T011
T023
T057
T019
```

Run those, append the rows to `results.csv`, re-import, and ask again with the arms:

```console
$ verdict-gate next-batch --policy moderate --pool tasks.txt --baseline parent.json --candidate candidate.json --fnoise floor.yaml
after 8 task(s): UNSUPPORTED: futility rule met (n=8, Δ̂=+0.125, F=0.375, P(Δ>0)=1.000, P(Δ>F)=0.040, P(Δ<0)=0.000, P(Δ<−F)=0.000)
STOP: the decision is final
```

The order is fixed by the pool and the policy's `order_seed`, so the batched decision equals the one `compare` gives
on the same tickets. Tickets run outside that order are refused. In Python:

```python
from pathlib import Path

from verdict-gate import next_batch
from verdict_gate.core.evidence_card import ROOT  # bundled configs (or your repo checkout)
from verdict_gate.core.policy import load_policy
from verdict_gate.core.replay import Floors, load_arm

policy = load_policy(ROOT / "configs" / "policies" / "moderate.yaml")
pool = Path("tasks.txt").read_text().split()
batch = next_batch(policy, pool, load_arm("parent.json"), load_arm("candidate.json"), Floors.from_file("floor.yaml"))
print(batch.stop, batch.decision, batch.questions_done, batch.task_ids)
# True UNSUPPORTED 8 []
```

## Notes for your own data

- **Score.** VERDICT does not run your agent or grade its answers; your system does. See "What counts as a score"
  in the [README](../../README.md#what-counts-as-a-score).
- **Repeats.** Calibrate on at least 2 (better 3 to 5) runs of the unchanged parent, on the same tasks, with the
  same model and settings you will compare under. Recalibrate when the model, the task set or the grader changes.
- **Strata.** The bundled policies condition on the thesis' SealQA `freshness` strata. Leave `stratum` empty for your
  own tasks (the card then lists those strata as empty). For your own strata, copy a policy and a strata file into a
  `configs/` folder of your own and point the policy's `subgroup.strata_file` at it.
- **Invalid executions** are excluded, never scored 0. More than 5% invalid (after retries) is refused as invalid
  evidence (`configs/validity.yaml`).
