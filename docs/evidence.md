# Evidence

Results from the thesis experiments that built and tested VERDICT (Kunal Suresh, Iowa State University, 2026;
thesis forthcoming). Each row gives the denominator, the analysis status and the main caveat. **Pre-registered** = the
analysis was fixed before the data; **post hoc** = run after seeing results, reported as such; **exploratory** = a
pilot, not a confirmatory claim.

Agents: Qwen3.5-9B (most experiments) and Granite-4.2-8B, temperature 0, via OpenRouter with a pinned provider.

## Noise in unchanged agents

| Result | Denominator | Status |
|---|---|---|
| 26.7% of answers flip between identical runs | SealQA, Qwen3.5-9B, repeated runs of the same agent | Pre-registered |
| Near zero (0.0–0.2%) on single-call classification and tool-selection tasks | DDXPlus, BANKING77, CLINC150, ToolBench, 3 repeats | Measured |

Noise depends strongly on the task. Search agents are noisy; single-call classifiers are nearly deterministic.

## Admission decisions

| Result | Denominator | Status | Caveat |
|---|---|---|---|
| Keep-if-higher kept 5 candidates, 3 of them worse than the parent; VERDICT kept 1, none worse | 6 candidate skills, fresh SealQA questions | Pre-registered | Small number of candidates |
| Same decisions as a fixed 3-seed design with 26 instead of 180 evaluations ($1.22 vs $9.28) | SealQA candidates | Pre-registered | One benchmark |
| In a self-improvement loop VERDICT kept 4 revisions instead of 17, at $10.39 vs $39.55 | 3 evolution seeds | Pre-registered | — |
| Final held-out accuracy of that loop: non-inferior at a 3-point margin on FreshQA (Δ = +0.004, 95% CI [−0.007, +0.015], 146 questions); inconclusive on H48 (Δ = −0.014, CI [−0.042, +0.012], 48 questions) | Governed vs naive final agents | Post hoc | The pre-registered accuracy-gain test was not met |

## Across all stored comparisons (decision benchmark)

34 candidate-vs-parent comparisons with ≥ 2 seeds per arm (search agents and memory tasks). Ground truth: the paired
effect on all seeds and all held questions (12 verified gains, 20 no verified change, 2 verified worse). Each rule
was replayed under 50 task orders. Expected counts:

| Rule | Wrong decisions | Kept a non-gain | Kept a harmful change | Missed a real gain | Tasks used |
|---|---:|---:|---:|---:|---:|
| Keep everything | 22.0 | 20.0 | 2.0 | 0.0 | 0 |
| Keep-if-higher, one seed, all held tasks | 6.0 | 5.0 | 0.0 | 1.0 | 167 |
| Keep-if-higher, one seed, 60 tasks | 8.3 | 6.2 | 0.0 | 2.1 | 58 |
| Fixed repeats, 60 tasks | 7.3 | 6.3 | 0.0 | 1.0 | 58 |
| Paired bootstrap test, 60 tasks | 7.6 | 0.3 | 0.0 | 7.3 | 58 |
| VERDICT `moderate` | 10.7 | 0.6 | 0.0 | 10.1 | 18 |
| VERDICT `moderate_v4` | 8.6 | 0.3 | 0.0 | 8.4 | 56 |

Status: post hoc. Caveat: the rules decide on subsets of the same data that defines the ground truth, which favours
rules that see more of it (keep-if-higher on all held tasks). Reading: VERDICT is the most cautious rule. It rarely
keeps a change that did not help, and it misses most small real gains, because it requires a gain to clear the
run-to-run noise floor.

## Memory of past tasks (exploratory pilots)

Qwen3.5-9B, 300 held tasks × 3 seeds per dataset, deterministic graders (label or API-name match).

| Memory | DDXPlus | ToolBench | BANKING77 | CLINC150 |
|---|---:|---:|---:|---:|
| None | 68.1% | 83.3% | 72.7% | 66.7% |
| 16 most similar past tasks the agent got right | **89.0%** (+20.9, verified) | **92.0%** (+8.7, verified) | **79.0%** (+6.3, verified) | **73.0%** (+6.3, verified) |
| Same, including past tasks it got wrong | +5.2, verified | −1.0, not verified | +3.3, verified | +1.3, not verified |
| 8 self-written lessons | +0.2, not verified | — | — | — |

Self-written lessons also gave no verified gain on FreshQA (146 questions × 3 seeds). Memory helped when tasks were
related (recurring diagnoses, intents, APIs), and only when the stored examples were correct.

**Evidence-governed memory** (noisy feedback: 25% of correct/wrong marks flipped; 300 fresh tasks × 3 seeds): keeping
a stored example only when repeated feedback supports it beat keeping every example on all 3 datasets (+7.0, +4.3,
+2.3 points, verified) and matched or beat single-mark filtering (+1.0 n.s., −0.2 n.s., +2.0 verified).

Caveat: these memory pilots used one model; the gate's 2× cost rule rejected memory that made DDXPlus prompts about
7× longer.

## Grading

| Result | Denominator |
|---|---|
| A second, reference-based grader agreed with the primary grader on 98.0–99.5% of answers | Search-QA answers across benchmarks |
| A reference-free LLM judge falsely accepted 41% of answers | Same answers |

## What is not shown

No independent replication yet; no simulation study of the full sequential procedure's error rates; evidence from two
small open models only. See [methodology.md](methodology.md#known-limitations).
