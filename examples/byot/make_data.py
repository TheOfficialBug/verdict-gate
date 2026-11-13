"""Synthetic "bring your own task" results: 60 customer-support tickets, exact-match intent routing (score 0/1).

    uv run python examples/byot/make_data.py      # rewrites examples/byot/results.csv and tasks.txt

Nothing here calls a model: the outcomes are drawn with a fixed seed so the example is reproducible. The parent
prompt ran three times (seeds 11, 22, 33); the candidate prompt (candidate_skill.md, which adds refund and billing
rules) ran once (seed 11). By construction the candidate is no better than the parent overall, yet its one run
happens to score higher: the case keep-if-higher falls for. Like a real agent, the parent's own repeats disagree on
the mixed-intent tickets. A few executions failed for infrastructure reasons (valid=false + failure_type); most were
retried (retry_of).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL, PROVIDER = "my-support-model", "my-provider"
COLUMNS = [
    "run_id",
    "task_id",
    "seed",
    "score",
    "valid",
    "model",
    "provider",
    "revision_id",
    "failure_type",
    "retry_of",
    "tokens_in",
    "tokens_out",
    "latency_s",
    "cost_usd",
]
USD_PER_TOKEN = 0.2e-6
SEED = 2033  # picked among 2020-2044 so the single candidate run shows a lucky gain (+5 tickets)

# Ticket kinds: probability that the parent / the candidate routes the ticket to the right queue. The candidate's
# billing rule helps billing tickets and hurts mixed-intent ones by the same amount: its true net effect is zero.
KINDS = {  # kind: (count, p_parent, p_candidate)
    "easy": (30, 0.95, 0.95),  # password resets, shipping status: both prompts get these right
    "billing": (12, 0.40, 0.65),  # refunds and double charges: +0.25 with the billing rule
    "ambiguous": (15, 0.50, 0.30),  # mixed intents: the billing rule now grabs some of them (-0.20)
    "escalation": (3, 0.90, 0.90),  # legal threats: unchanged
}
# Infrastructure failures: (run_id, task_id, failure_type, retried?)
FAILURES = [
    ("parent_s22", "T017", "timeout", True),
    ("parent_s33", "T005", "timeout", False),
    ("cand1_s11", "T042", "rate_limit", True),
]


def main() -> None:
    rng = random.Random(SEED)
    tickets = []
    for kind, (count, p_parent, p_cand) in KINDS.items():
        tickets += [(kind, p_parent, p_cand)] * count
    rng.shuffle(tickets)
    task_ids = [f"T{i:03d}" for i in range(1, len(tickets) + 1)]
    runs = [
        ("parent_s11", "parent", 11, 1),
        ("parent_s22", "parent", 22, 1),
        ("parent_s33", "parent", 33, 1),
        ("cand1_s11", "cand-1", 11, 2),
    ]  # (run_id, revision_id, seed, index of the p column)
    failures = {(r, t): (f, retried) for r, t, f, retried in FAILURES}
    rows = []
    for run_id, revision, seed, col in runs:
        prompt_tokens = 900 if revision == "parent" else 1150  # the candidate prompt is longer
        for task_id, ticket in zip(task_ids, tickets):
            tokens_in = prompt_tokens + rng.randint(150, 600)
            tokens_out = rng.randint(20, 80)
            row = {
                "run_id": run_id,
                "task_id": task_id,
                "seed": seed,
                "model": MODEL,
                "provider": PROVIDER,
                "revision_id": revision,
                "failure_type": "",
                "retry_of": "",
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_s": round(rng.uniform(0.8, 3.5), 2),
                "cost_usd": round((tokens_in + tokens_out) * USD_PER_TOKEN, 6),
            }
            score = int(rng.random() < ticket[col])
            if (run_id, task_id) in failures:
                failure, retried = failures[(run_id, task_id)]
                rows.append(
                    {
                        **row,
                        "score": "",
                        "valid": "false",
                        "failure_type": failure,
                        "tokens_out": "",
                        "latency_s": 60.0 if failure == "timeout" else 0.3,
                    }
                )
                if not retried:
                    continue
                row["retry_of"] = f"{run_id}/{task_id}"
            rows.append({**row, "score": score, "valid": "true"})
    with (HERE / "results.csv").open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (HERE / "tasks.txt").write_text("\n".join(task_ids) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} rows ({len(task_ids)} tickets, {len(runs)} runs) to examples/byot/results.csv")


if __name__ == "__main__":
    main()
