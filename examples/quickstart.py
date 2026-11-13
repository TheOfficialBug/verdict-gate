"""Offline quickstart: decide three candidate skills against their parent, under keep-if-higher and under VERDICT's
moderate policy, and write one evidence card. No network, no API key, no model call.

    uv run python examples/quickstart.py [--out DIR]

Inputs are committed synthetic arm files (tests/fixtures/replay/: 24 questions, seeds 11/22/33, a parent s0 and
candidates that repair, break, or do not change it) and their noise-floor file. The same decisions come from the CLI:

    uv run verdict-gate compare --baseline tests/fixtures/replay/s0.json --candidate tests/fixtures/replay/c_good.json \
        --policy configs/policies/moderate.yaml --fnoise tests/fixtures/replay/fnoise_m.yaml --out card.json
    uv run verdict-gate card render card.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from verdict_gate.adapters.json_csv import import_runs
from verdict_gate.core.evidence_card import render_report, validate_card
from verdict_gate.core.paired import build_paired_table, differences
from verdict_gate.core.policy import load_policy
from verdict_gate.core.replay import Floors, evaluate, load_arm

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "replay"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="folder for card.json and card.md (default: a new temp folder)")
    a = ap.parse_args(argv)
    out = a.out or Path(tempfile.mkdtemp(prefix="verdict_quickstart_"))

    # 1. Your own results in the long CSV format (examples/README.md) -> a paired table.
    runs = {r.manifest.run_id: r for r in import_runs(ROOT / "examples" / "long_format.csv", benchmark="my-bench")}
    table = build_paired_table(runs["parent_s11"].outcomes, runs["candidate_s11"].outcomes)
    print(f"long_format.csv: {table.n} paired questions, mean candidate - parent = {differences(table).mean():+.3f}")

    # 2. Decisions from stored arms: keep-if-higher (naive) vs the evidence-governed moderate policy.
    parent = load_arm(FIX / "s0.json")
    floors = Floors.from_file(FIX / "fnoise_m.yaml")
    policies = {name: load_policy(ROOT / "configs" / "policies" / f"{name}.yaml") for name in ("naive", "moderate")}
    cards = {}
    for cand in ("c_good", "c_null", "c_bad"):
        arm = load_arm(FIX / f"{cand}.json")
        row = []
        for name, pol in policies.items():
            card = evaluate(parent, arm, pol, floors, git_commit="example", created_at="2026-01-01T00:00:00+00:00")
            assert validate_card(card) == [], "every card must satisfy schemas/evidence_card.schema.json"
            cards[(cand, name)] = card
            row.append(f"{name} {card['decision']}")
        print(f"{cand:7s} " + " | ".join(row))

    # 3. One evidence card as JSON (the schema'd record) and as a one-page Markdown report.
    out.mkdir(parents=True, exist_ok=True)
    card = cards[("c_good", "moderate")]
    (out / "card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    (out / "card.md").write_text(render_report(card), encoding="utf-8")
    print(f"wrote {out / 'card.json'} and {out / 'card.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
