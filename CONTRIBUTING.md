# Contributing

Thanks for helping. VERDICT is a research preview, so the most valuable contributions are evidence: reproductions,
failure cases, and validation of the statistics.

## Set up

```bash
git clone https://github.com/TheOfficialBug/verdict-gate.git
cd verdict-gate
uv sync                 # package + dev tools (pytest, ruff)
uv run pytest -q        # all tests are offline (network access is blocked in tests)
uv run ruff check src tests examples
```

## Ground rules

- **Decisions must stay reproducible.** Any change to how a decision is computed needs a test, and must not change the
  decision of an existing bundled policy on the fixtures unless the change is intended and documented.
- **Bundled policy files are versioned artifacts.** Do not edit `naive`, `moderate`, `strict` or `fixed_repeat` in
  place (their hashes are recorded on existing evidence cards); add a new versioned policy instead.
- **Claims need denominators.** Documentation that reports a result states the number of candidates, tasks, seeds,
  models, the grader and whether the analysis was pre-registered or post hoc.
- **No network, no keys** in the package or its tests.
- Keep the core (`verdict_gate.core`) free of I/O adapters and the CLI (a test checks this).

## Pull requests

1. Open an issue first for anything beyond a small fix.
2. Add or update tests; run `pytest` and `ruff`.
3. Update `CHANGELOG.md` under "Unreleased".
4. Statistical changes: describe the operating characteristics you expect and how you checked them.

## Reporting a wrong decision

Open an issue with the policy, the floor file, both arm files (or a minimal synthetic version) and the card. If the
data is private, a synthetic table that reproduces the behaviour is enough.
