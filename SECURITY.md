# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/TheOfficialBug/verdict-gate/security/advisories/new). Do not open
a public issue. You can expect an acknowledgement within 7 days.

## Supported versions

Only the latest release receives fixes while the project is in 0.x.

## Threat model

VERDICT reads files you give it and writes decision records. It makes no network calls and executes no code from its
inputs. Relevant risks:

| Risk | What VERDICT does | What you should do |
|---|---|---|
| Tampered results or arm files | Validates schemas and pairing; records hashes of the policy and the revision text | Keep results in access-controlled storage; record their hashes with the card |
| Tampered policy | Records the policy's configuration hash on every card | Version policies; review policy changes like code |
| Compromised or noisy grader | Nothing: the decision is only as good as the score | Prefer deterministic checks or reference-based grading |
| Poisoned or unrepresentative tasks | Nothing | Sample tasks from the deployment population; keep a held-out set |
| Revision substitution | The card records the SHA-256 of `--revision-file` | Compare the hash with the deployed artifact |
| Sensitive data in results | Stores only what you put in the results file | Do not put secrets or personal data in task ids or extra fields |

## Releases

Releases are built and published by GitHub Actions with PyPI Trusted Publishing (no long-lived tokens), with build
provenance and PyPI attestations. Workflow actions are pinned to full commit SHAs.
