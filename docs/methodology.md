# Methodology

What VERDICT computes, why, and what is and is not established about it. Written for reviewers and for anyone choosing
a policy.

- [Estimand](#estimand)
- [Pairing](#pairing)
- [The noise floor](#the-noise-floor)
- [Posterior](#posterior)
- [The sequential policy](#the-sequential-policy)
- [Evidence gate](#evidence-gate)
- [Cost, harm and subgroups](#cost-harm-and-subgroups)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)

## Estimand

For a fixed task set T, model, provider and settings, the effect is the mean paired difference

Δ = (1/|T|) Σ_t ( ȳ_cand(t) − ȳ_parent(t) ),

where ȳ is the mean score of an arm on task t over its valid repeats. Δ is an effect **on the evaluated tasks**. It
says nothing about tasks outside that set unless the set was sampled from the population you care about.

Scores are in [0, 1]. Repairs, regressions and flip rates assume binary (or thresholdable) scores; for continuous
scores read the paired differences rather than the transition counts.

## Pairing

A comparison is valid only if parent and candidate ran on the same tasks with the same model, provider, output-token
limit and evaluation settings; the importer and `compare` check this and report invalid evidence otherwise. Invalid
executions (infrastructure failures) are excluded from both arms of that task, never scored 0. A per-comparison limit
on the invalid rate (`configs/validity.yaml`, 5%) stops decisions on unusable evidence. Asymmetric failure (the
candidate failing more often on hard tasks) can still bias a comparison; check invalid rates per arm.

## The noise floor

`calibrate` takes ≥ 2 repeat runs of the **unchanged** parent on the same tasks. For each block size n it resamples n
tasks, computes the mean paired difference between two repeats, and takes the 95th percentile of its absolute value
(Bayesian bootstrap, 10,000 draws). F_noise(n) is how large a "gain" an unchanged agent shows against itself on n
tasks. A candidate must clear it.

What the floor is **not**: it measures run-to-run (execution) noise on fixed tasks. It does not measure uncertainty
from which tasks happened to be sampled, and it is itself estimated from few repeats. Both points are open items
below.

## Posterior

The effect's posterior is a Bayesian bootstrap (Rubin, 1981) over the per-task paired differences: Dirichlet(1, …, 1)
weights on the observed differences, 10,000 draws, fixed seed. From it: P(Δ>0), P(Δ>F), P(Δ<0), P(Δ<−F) and a 95%
interval.

Known boundary behaviour: the Bayesian bootstrap only re-weights *observed* differences. If every non-zero difference
seen so far is positive, every draw is positive and P(Δ>0) = 1.000, even at n = 8. It does not mean an unseen
regression is impossible. The evidence gate below exists because of this.

## The sequential policy

Tasks are visited in a fixed seeded order in blocks (8 by default). After each block, the first matching rule decides:

1. **Invalid evidence** if pairing is broken or the invalid rate is over the limit.
2. **HARMFUL** if P(Δ < −F) ≥ 0.95, or the candidate costs more than `max_cost_ratio` × the parent and the pass rule
   is not met.
3. **PASS_GLOBAL** if P(Δ > 0) ≥ 0.95 and P(Δ > F) ≥ 0.50, no subgroup is harmed and cost is within limit.
4. **PASS_CONDITIONAL** if the gain is supported only in some predeclared subgroups.
5. **UNSUPPORTED** on futility (P(Δ>0) ≤ 0.05 and P(Δ>F) ≤ 0.10), at `max_n`, or when even all-repair remaining tasks
   could not pass (curtailment). Futility and curtailment are held while the running estimate is negative, so harm
   evidence can accumulate.
6. Otherwise **NEEDS_MORE_EVIDENCE**.

These thresholds are **policy rules**, chosen in advance. They are not calibrated long-run error rates. Checking after
every block is a form of optional stopping; the frequentist error rates of the complete procedure have not been
established (see [Roadmap](#roadmap)).

## Evidence gate

*Added after 0.2.0; ships as `moderate_v4`.* With a near-deterministic agent (the same answer on almost every repeat),
F_noise(n) = 0 and, by the boundary behaviour above, a single repair in the first block gives P(Δ>0) = 1.000 and a
pass at n = 8; a single regression gives HARMFUL; a block of ties triggers futility. In the thesis memory experiments
this made decisions depend on which 8 tasks came first: over 200 visiting orders, a candidate with a +8.7-point
verified gain passed only 55% of the time.

`evidence_gate: {min_n, min_discordant}` adds two conditions:

- no terminal decision before n ≥ `min_n` (unless the task budget ends sooner);
- PASS and the posterior harm rule also need at least `min_discordant` tasks where parent and candidate differ; at
  `max_n` without them the decision is UNSUPPORTED.

`moderate_v4` uses `min_n = 32` (four blocks), `min_discordant = 6` (the fewest discordant pairs at which an exact
one-sided sign test can reach p < 0.05) and `max_n = 120`. These values were fixed before the benchmark and the
confirmation run that evaluated them. On fresh data (3 datasets, 12 comparisons) it made 7 of 12 decisions right
against 5 of 12 for `moderate`, with no unsupported or harmful admission.

## Cost, harm and subgroups

- **Cost rule.** With per-execution `cost_usd`, the cost ratio is the candidate's mean cost per task over the
  parent's. Above `max_cost_ratio` (2.0 in `moderate`) a candidate is HARMFUL unless it already meets the pass rule.
  A change that multiplies prompt length (for example, retrieved memory) can be rejected on cost alone; set the limit
  your deployment can afford, before the evaluation.
- **Subgroups.** A stratum is harmed if P(Δ_s < −F) ≥ 0.90 with ≥ 8 tasks, supported with ≥ 16. Strata must be fixed
  before results are seen; conditional routing should be confirmed on held-out data.

## Known limitations

| Issue | Status |
|---|---|
| Optional stopping: error rates of the full sequential rule not validated | Open: simulation study planned |
| Floor = execution noise only; task-sampling uncertainty not separated | Open: two-way (task × repeat) bootstrap planned |
| Floor estimated from few repeats; its own uncertainty not reported | Open |
| Bayesian bootstrap overconfident at small n with one-sided outcomes | Mitigated by the evidence gate (`moderate_v4`); a smoothed Dirichlet-multinomial posterior is planned |
| Many candidates → more false admissions; no campaign-level control | Open: online FDR / e-value control planned |
| Subgroups discovered and used on the same data | Mitigated by policy (predeclared strata); held-out confirmation required by guidance, not enforced |
| Asymmetric invalidity between arms | Partly: combined limit only; per-arm gating planned |
| `pass.p_gain_gt_floor = 0.50` is permissive if the floor is read as a minimum meaningful effect | By design the floor is a noise bound, not a minimum effect; a separate `minimum_practical_effect` is planned |
| VERDICT misses small real gains (cautious by design) | Measured: across 34 stored comparisons it missed 8.4 of 12 verified gains while admitting 0.3 of 22 non-gains |

## Roadmap

1. **Operating characteristics by simulation:** false-admission, harmful-admission, power, stopping time and cost
   for the complete procedure under null, practical-null, benefit, harm, heterogeneity, missingness, grader noise and
   repeated-candidate scenarios; compared with keep-if-higher, fixed repeats, sign/permutation tests and confidence
   sequences.
2. Separate execution noise and task-sampling uncertainty; report floor uncertainty.
3. Explicit metric contract (binary / continuous / ordinal, minimum practical effect) stored in the card.
4. Campaign-level error control for adaptive candidate generation.
5. Per-arm invalidity gating and missing-data sensitivity.
6. A preregistered multi-domain validation and an independent replication.

References (checked against Crossref):

- Rubin, D. B. (1981). The Bayesian Bootstrap. *The Annals of Statistics*, 9(1). https://doi.org/10.1214/aos/1176345338
- Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2022). Always Valid Inference: Continuous Monitoring of A/B Tests.
  *Operations Research*, 70(3). https://doi.org/10.1287/opre.2021.2135
- Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021). Time-uniform, nonparametric, nonasymptotic
  confidence sequences. *The Annals of Statistics*, 49(2). https://doi.org/10.1214/20-AOS1991
