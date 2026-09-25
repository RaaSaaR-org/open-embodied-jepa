---
id: TASK-057
aliases:
- TASK-057
title: Locate the open-loop/closed-loop gap with cheap diagnostics on the development cohort
slug: locate-the-open-loop-closed-loop-gap-with-cheap-diagnostics
status: review
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- learning
- evaluation
sprint: ''
depends_on:
- "[[TASK-056]]"
due_date: ''
created: 2026-09-24
updated: 2026-09-25
---




# Locate the open-loop/closed-loop gap with cheap diagnostics on the development cohort

## Description

TASK-056 failed on its development stop rule: four arms, 64 development attempts, 0 full
successes. Its recorded candidate mechanism — smooth-L1 hedging against a `right_dz` target
saturated at 0.400 — is **not supported by the statistics it was read off**, and this task exists
because of that.

**Measured on the validation split, the hedging signature is absent offline**: saturated-row error
is lower than unsaturated, conditional means reach 88–98 % of the boundary on both signs, and the
predicted-to-**target** standard-deviation ratio is 0.952–0.980. The statistics the original
reading was taken from could not have shown this either way — an unconditional median necessarily
sits inside the non-saturated group at 44.503 % val saturation, and only the predictions' standard
deviation was ever computed, so "compressed" had no referent. **Whether dz behaviour caused the
closed-loop failure is unmeasured** and stays so, because the closed-loop command signs are
unrecoverable. The expert's own `right_dz` is a signed, near-symmetric bang-bang whose **mean is
−0.0012**, so the published comparison was never between the same quantities.

Two attempts were not **total** failures and this task leads with them: **A2 on seed 45100 and A3
on seed 45006 each reached the scorer's `grasp` stage, lifted the apple ~17 cm, and were still
holding it undropped at the step cap.** Neither reached `transport`, and **the task still has zero
learned successes** — 0/64 on development, as every generation of this line has been.

So this is a **diagnostic** task. It trains nothing, changes no model, and preregisters no use of
cohort C. It runs three cheap probes on the already-consumed development cohort — a signed
step-zero contrast against the scripted expert, a divergence trace, and a per-dimension expert
substitution sweep — and its single gate can fire the abandonment clause that
`docs/experiments/apple_policy_v1.md` §7 already preregistered.

Protocol: `docs/experiments/apple_policy_diagnostics_v1.md`.
Manifest: `benchmarks/manifests/apple-policy-diagnostics-v1.json`.

## Acceptance

- The preregistration and manifest merge on an independent reviewer's **reported** APPROVE, before
  any probe runs.
- Diagnostics run to completion **before** any edit to `src/embodied_jepa/policy.py`, whose
  `implementation_sha256` every trained checkpoint enforces.
- B1, B2 and B3 pass; B3 (the all-dimensions inert-probe tripwire) runs first.
- D1, D1-grasp, D2 and G-SUB are reported whatever they say, with the results document separating
  numbers from reading.
- Cohort C (45300–45339) is not simulated at any stage.
- If G-SUB fails, the abandonment clause fires as written and the line stops.
## Amendment 1 (2026-09-25, pre-run)

The merged preregistration named the wrong shadow expert: plain `OracleManipulationPolicy`
(805 commands) instead of `scripted.apple_collector_policy` (745), the policy every BC root was
collected with. Verified by bit-exact replay of 8 train roots, the corpus phase budgets and the
early-release signature. With the frozen expert, B3's `full` configuration scores 0/16 in a dry
run, so the run would have been void.

The whole-document sweep and a blind-predictor check of every absolute threshold found more:
- the frozen D1 fails for all four arms on recorded reset frames, so it cannot detect a pipeline
  defect. It is demoted to a reported D1-contrast and replaced by D1-pipeline, an
  observation-path equivalence test;
- the D1-grasp cut does not fire on the prior-only predictor. It moves to > -0.5;
- G-SUB's 8/16 does not discriminate for `dy`: a clock-only controller with `dy` substituted
  scores 9/16. Thresholds are now per candidate against that blind reference;
- `first_departure_step` was never defined. It is now defined;
- Table E mislabelled two phases, and the manifest had two stale entries.

Protocol §13 and `amendment_1` in the manifest have the details.
`exemption_spent` stays false. Numbers: `scripts/calibrate_policy_diagnostics.py`.

## Phase B: probe runner (2026-09-25)

Amendment 1 merged in PR #44 (`56194a5`). The runner is `scripts/diagnose_policy.py`. It runs
the stages in the amended order: B3, D1-pipeline, B1, the arms with nothing substituted (A2 first,
since that run is also B2), then D3. After that it evaluates every gate and the decision table.

- **Device:** CPU.
- **Inputs:** it refuses to run if any hashed input differs from the amendment's record.
- **Seeds:** development resets go only through `evaluate_policy.cohort_resets`. D1-pipeline
  has its own whitelist of 15 val roots.
- **Manifest:** the runner never writes to it.

The harness now records `hand_contact`, apple height and `dropped` after every step, and it
enforces the 300 s per-attempt cap.

`--smoke` checks the wiring on train root 48000 only. It confirmed that `full` commands match
`scripted_oracle` exactly and that the oracle's departure is 0.

## Phase C and D: run and results (2026-09-25)

Amendment 2 merged in PR #46 (`82eafa9`) and the runner in PR #45 (`e9f4670`), each on its
reviewer's reported APPROVE. The pre-run reviewer reported APPROVE at `e9f4670`.

The gated run took 46.5 min on CPU from a clean tree. Output is in
`outputs/task057-diagnostics/run-1`; hashes are in
`benchmarks/manifests/apple-policy-diagnostics-v1-results.json`.

**Outcome X. The abandonment clause fired.**
- The controls all passed: B3 16/16, B1 16/16 (hold and random 0), and B2 reproduced exactly.
- D1-pipeline passed for all four arms and D1-grasp fired for none, so clause (a) does not hold.
- G-SUB failed: the best candidate reached 1/16, against thresholds of 8–12.

Where it breaks (non-gating development cohort, n = 16 per configuration; the substitution
results are privileged diagnostics):
- **Approach, mostly.** Most attempts copy the expert's early rise and then never descend toward
  the apple: 61/64 never get 1 cm closer than at reset. The arms that descend on some resets are
  A2 on 1 and A3 on 3.
- **Carry, on the 2 grasping attempts.** The hand holds the apple on every step after the grasp
  but never moves it toward the plate.

`exemption_spent` stays false. Results are in `docs/experiments/apple_policy_diagnostics_v1_results.md`.

Acceptance:
- The prereg merged on a reported APPROVE.
- The diagnostics ran before any edit to `policy.py`.
- B1, B2 and B3 passed, with B3 first.
- D1, D1-grasp, D2 and G-SUB are reported.
- Cohort C was not simulated.
- The abandonment clause fired as written.
%% mc-links: [[TASK-056]] %%
