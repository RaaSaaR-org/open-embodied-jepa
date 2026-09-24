---
id: TASK-058
aliases:
- TASK-058
title: Audit every accepted numerical claim in the experiment record against the code that computes it
slug: audit-every-accepted-numerical-claim-against-the-code-that-computes-it
status: todo
priority: 1
owner: ''
projects: []
customers: []
tags:
- research-integrity
- evaluation
- documentation
sprint: ''
depends_on: []
due_date: ''
created: 2026-09-24
updated: 2026-09-24
---


# Audit every accepted numerical claim in the experiment record against the code that computes it

## Description

This project's dominant defect is not arithmetic. It is that a number is restated **correctly**
and nobody checks what computed it. Every instance below was caught late, by accident, or by a
reader who happened to trace one field:

- The `0.78 cm` "perception" headline matched the committed JSON exactly and was confirmed by two
  readers. `world_model_v2.window_metrics` builds that field through `predicted_readouts`, i.e.
  `encode` → **`predict`** → `readout`, so it is a **rollout** output. The only directly-encoded
  quantity in that function is the persistence baseline. A pivot rested on it briefly.
- TASK-057's `§1.1(b)` claimed a hedging mechanism was contradicted by a per-dimension median that
  `cloning.action_error` computes **unconditionally over all rows**. With saturation below 50 %,
  that median necessarily sits inside the non-saturated group and is mathematically incapable of
  testing the claim. The companion "compressed standard deviation" had no referent at all: only
  the predictions' standard deviation was ever computed, never the target's.
- Every policy command statistic in `apple_policy_v1_results.md` passes through
  `np.abs` at `scripts/evaluate_policy.py`. **No sign survives anywhere**: 2 894 numeric values
  across all 64 TASK-056 attempts, zero negatives. Two published readings (a dz under-shoot, a
  `|grasp| = 1.0` "hand fully closed") were sign claims read off absolute values.
- `task056_handover.md` states the two **controls** reached `grasp` on 0/16 and gives no count for
  the two learned arms. A2/45100 and A3/45006 each reached `grasp`, lifted the apple ~17 cm, and
  were still holding it undropped at the step cap. The compression made the result read as
  uniformly inert.

## Scope

Walk the accepted numerical claims in `docs/experiments/` and `benchmarks/manifests/` and, for
each, record **what code produces it**. Not whether it matches the artifact — that check has
passed every time and caught none of the above.

For each claim: the file and line, the artifact field, the function that computes the field, the
population it ranges over, and whether the quantity the prose names is the quantity the code
returns. Where they differ, correct the prose or withdraw the claim; do not patch the number.

## Acceptance

- A committed audit document, one row per load-bearing claim, with the computing function named
  for every row.
- Every divergence either corrected in its source document or recorded as withdrawn, with the
  reason.
- Claims that cannot be traced to committed code are listed as **not citable** — the standard
  `apple_policy_diagnostics_v1.md` §9 already applies to Table A and
  `scripts/measure_policy_offline_conditionals.py`.
- No number is changed without the change being visible in the source document's history.

## Rules that apply

- Prefer reading the function over reading the docstring. A docstring asserting what a quantity
  means is not evidence that the code computes it.
- A guard that matches on vocabulary rather than on the condition passes any mutation fluent
  enough to reuse the vocabulary; the same holds for prose describing a metric.
- Where an absolute value, a pooled median or an unconditional aggregate destroys the distinction
  a claim depends on, say so explicitly — the claim is **unmeasured**, which is not the same as
  refuted.

## Notes

Raised during TASK-056/TASK-057 after the first four instances above were found in a single
session. Deliberately not scoped to one protocol: the pattern crosses TASK-048 through TASK-057.
