---
id: TASK-057
aliases:
- TASK-057
title: Locate the open-loop/closed-loop gap with cheap diagnostics on the development cohort
slug: locate-the-open-loop-closed-loop-gap-with-cheap-diagnostics
status: todo
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
updated: 2026-09-24
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
