# Releasing

Releases are published only by the `Release` workflow, from a version tag. No API token is used.

## One-time setup

1. On PyPI, open the `verdict-gate` project → *Publishing* → *Add a trusted publisher* (GitHub):
   owner `TheOfficialBug`, repository `verdict-gate`, workflow `release.yml`, environment `pypi`.
2. On GitHub, *Settings → Environments*: create `pypi` and require a reviewer.
3. *Settings → Branches*: protect `main` (pull requests, passing CI).
4. *Settings → Code security*: enable private vulnerability reporting and Dependabot alerts.

## Each release

1. Update `version` in `pyproject.toml` and `__version__` in `src/verdict_gate/__init__.py`.
2. Move the "Unreleased" notes in `CHANGELOG.md` under the new version and date.
3. Merge to `main` with CI green.
4. Tag and push: `git tag -s v0.3.0 -m "v0.3.0" && git push origin v0.3.0`.
5. Approve the `pypi` environment when the workflow asks.
6. Check that PyPI shows provenance for every file and that the GitHub attestation matches the tag's commit:
   `gh attestation verify dist/verdict_gate-0.3.0-py3-none-any.whl --repo TheOfficialBug/verdict-gate`.
