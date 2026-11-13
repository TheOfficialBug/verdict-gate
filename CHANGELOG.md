# Changelog

All notable changes to `verdict-gate`. Versions follow [semantic versioning](https://semver.org); while the version
is 0.x, minor versions may change the public API.

## Unreleased (0.3.0)

Not yet on PyPI. Source of truth moved to [TheOfficialBug/verdict-gate](https://github.com/TheOfficialBug/verdict-gate).

### Changed (breaking)
- Import package renamed `verdict` → `verdict_gate`; command renamed `verdict` → `verdict-gate` (alias `vgate`).
  An unrelated PyPI distribution is also called `verdict`; the two overwrote each other. See
  [docs/migration.md](docs/migration.md).
- Development status is now **Alpha / research preview**; the README and docs state what is and is not validated.
- Removed the thesis-only `budget` and `freeze` commands and the benchmark-specific adapters and runners from the
  package (they stay in the research repository).

### Added
- `moderate_v4` policy and the optional `evidence_gate` (`min_n`, `min_discordant`) for sequential policies: no
  decision on too few tasks or too few discordant pairs. Fixes decisions on 8 tasks when the agent is
  near-deterministic and the noise floor is 0.
- `verdict-gate validate FILE`: check a results file, arm file, floor file or evidence card.
- `verdict-gate policy [NAME]`: list the bundled policies or print one as applied.
- `--version`.
- A user's own policy file may name `configs/validity.yaml` and a strata file that do not exist next to it; the
  bundled copies are used.
- Bundled data (schema, policies, strata, example) now lives only inside the package.
- CI on Linux, macOS and Windows (Python 3.12 and 3.13, newest and lowest dependency versions), a test of the built
  wheel, and a release workflow with PyPI Trusted Publishing, build provenance and attestations.
- Tested dependency ranges with upper bounds; `py.typed`.
- Documentation: user guide, methodology (with known limitations and roadmap), evidence with denominators.

## 0.2.0 (2026-11-12)

`verdict calibrate`, `verdict import`, `verdict next-batch` / `verdict.next_batch()`, `verdict example` (bring your
own tasks); bundled policy names; a missing noise-floor file is invalid evidence (exit 2).

## 0.1.0 (2026-11-10)

Initial release: paired evidence, noise floors, sequential Bayesian policy and baselines, evidence cards (JSON schema
and Markdown report), offline replay.
