# Migrating from 0.2 to 0.3

0.3 renames the package to avoid a clash with an unrelated PyPI distribution called `verdict` (installing both
overwrote the same top-level module).

| 0.2 | 0.3 |
|---|---|
| `import verdict` | `import verdict_gate` |
| `from verdict.core.replay import ...` | `from verdict_gate.core.replay import ...` |
| `verdict compare ...` | `verdict-gate compare ...` (or `vgate compare ...`) |
| `verdict budget ...`, `verdict freeze ...` | Removed: thesis-specific tooling, kept in the research repository |

Everything else is unchanged: the input format, arm files, floor files, evidence-card schema (`1.1`) and the bundled
`naive`, `moderate`, `strict` and `fixed_repeat` policies (byte-identical, so their configuration hashes on existing
cards still match). New: `moderate_v4`, `verdict-gate validate`, `verdict-gate policy`, `--version`.

Search-and-replace for code:

```bash
grep -rl "verdict\." your_project | xargs sed -i 's/\bfrom verdict\./from verdict_gate./; s/\bimport verdict\b/import verdict_gate/'
```

`pip uninstall verdict-gate` then `pip install verdict-gate` removes the old `verdict` module.
