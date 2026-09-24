---
id: TASK-058
aliases:
- TASK-058
title: Audit every accepted numerical claim in the experiment record against the code that computes it
slug: audit-every-accepted-numerical-claim-against-the-code-that-computes-it
status: done
priority: 1
owner: ''
projects: []
customers: []
tags:
- governance
- evaluation
- docs
sprint: ''
depends_on: []
due_date: ''
created: 2026-09-24
updated: 2026-09-25
---




# Audit every accepted numerical claim in the experiment record against the code that computes it

## Description

This project's dominant defect is not arithmetic. It is that a number is restated **correctly**
and nobody checks what computed it. Every instance below was caught late, by accident, or by a
reader who happened to trace one field:

- The `0.78 cm` "perception" headline matched the committed JSON exactly (`palm_apple_median_m`
  0.00781842 in `apple-world-model-v3.json`) and passed more than one reading.
  `world_model_v2.window_metrics` builds that field through `predicted_readouts`, i.e.
  `encode` → **`predict`** → `readout`, so it is a **rollout** output. The only directly-encoded
  quantity in that function is the persistence baseline. A pivot rested on it briefly.
- TASK-056 recorded a hedging mechanism (`apple_policy_v1_results.md` §12/§16,
  `task056_handover.md` §5), and an **earlier, unmerged draft** of TASK-057's `§1.1(b)` hardened it
  to "contradicted". Both rest on a per-dimension median that `cloning.action_error` computes
  **unconditionally over all rows**: with saturation below 50 % the median is bracketed by the
  non-saturated rows — `min(non-sat) ≤ median ≤ max(non-sat)` — so it cannot isolate the saturated
  rows and is incapable of testing the claim either way. The companion "compressed standard
  deviation" had no referent: nothing in `cloning.py` computes the **target's** spread, only
  `predictions.std(axis=0)`. The committed `§1.1(b)` is the text that withdraws the conclusion.
- Every policy command statistic in `apple_policy_v1_results.md` passed through `np.abs` at
  `scripts/evaluate_policy.py`. **No sign survived**: 2 894 numeric values in the
  `command_statistics` blocks of all 64 TASK-056 attempts, zero negatives. (The four files entire
  hold 164 negatives, in `configured_bounds` and the reset coordinates — the population has to be
  named.) One directional reading was published off those numbers, the dz under-shoot; `|grasp|
  q50 = 1.0` is the second quantity the absolute value renders unreadable, not a second published
  claim. Commit `4c72ffa` restored a `signed` block alongside, so this is true of TASK-056's
  reports and no longer of the runner.
- The two learned arms' grasp counts **are** in `apple_policy_v1_results.md` §14/§16 ("1/16
  against 0/16"). `task056_handover.md` gives only the two **controls'** 0/16 and no count for the
  learned arms. What is in neither document is the lift height, the sustained contact or the
  undropped state: A2/45100 and A3/45006 each lifted the apple ~17 cm and were still holding it
  undropped at the step cap. The compression made the result read as uniformly inert.

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
- Claims that exist only in a git-ignored run artifact are listed as **not citable** — the
  standard stated in `apple_policy_diagnostics_v1.md` §9 and credited there to
  `apple_policy_v1.md` §9. Table A and `scripts/measure_policy_offline_conditionals.py` are what
  satisfying it looks like.
- No number is changed without the change being visible in the source document's history.

## Rules that apply

- Prefer reading the function over reading the docstring. A docstring asserting what a quantity
  means is not evidence that the code computes it.
- A guard that matches on vocabulary rather than on the condition passes any mutation fluent
  enough to reuse the vocabulary; the same holds for prose describing a metric.
- Where an absolute value, a pooled median or an unconditional aggregate destroys the distinction
  a claim depends on, say so explicitly — the claim is **unmeasured**, which is not the same as
  refuted.

## Known inherited wording

`apple_policy_diagnostics_v1.md` §1.1(b) says the unconditional median "sits inside the
non-saturated group". The bracketing statement above is the correct form; the verdict it supports
is unchanged. §1.1(b) is a merged, frozen preregistration, so correcting it is an amendment and
belongs to this audit's output rather than to a quiet edit.

## Notes

Raised during TASK-056/TASK-057 after the four instances above were found in a single session.
They span TASK-054 to TASK-057. Whether the pattern reaches further back is **not established** —
it is what this audit is for, and the scope is deliberately TASK-048 onward rather than one
protocol.

## Evidence log

- 2026-09-25: The audit was run as six independent corpus slices plus lead adjudication. The
  deliverable is `docs/experiments/claim_audit_v1.md`: 145 rows, each naming its computing
  function. Classes: 72 confirmed, 46 mislabelled, 11 wrong (1 of them already withdrawn at
  source), 16 unverifiable / not citable.
- **No preregistered gate verdict or pre-declared outcome changes.** The CEM abandonment (G2a,
  Outcome B) and the TASK-056 FAIL stand.
- **Withdrawn or corrected readings.**
  - P3's "no shortcut available" is wrong. A proprioception-only linear probe reads orient
    apple position to 1.01 cm, or 0.163 cm when fit on orient rows only.
  - The P2 threshold rationale is wrong for the relative quantity.
  - The dz under-shoot is withdrawn as unmeasured.
  - v3's "close needs no prediction" is wrong: the closure is rollout-planned.
  - The ejection mechanism is overstated.
  - G4 is passed by a constant in v2, v3 and v4.
  - v4's gate-count and ranking readings hold at h = 16 only.
  - The encoder/rollout decomposition is a difference and a ratio of medians, and its v2
    anchor was never committed.
  - E2's "90 % in the targeted term" is withdrawn.
  - In the README, "15 % of needed change" is wrong (it is 29 %), and the 0/150 lacked its
    guard-stop qualifier.
- **Cross-reference for TASK-057, not edited here.**
  - The shadow expert must be `scripted.apple_collector_policy`, with a budget of 745, not
    `OracleManipulationPolicy` with 805.
  - A label-based step-zero reconstruction puts the expert dx gap at 84 % of A0's D1 dx
    threshold (median over cohort D).
  - Table E's phase names 5 and 6 are wrong (they are `release_high` and `lower_open`).
  - The §1.1(b) wording should be "bracketed by".
- **Errata.** Dated errata blocks were added to 14 experiment docs, plus in-place corrections
  with a dated note in the README, an erratum in DECISIONS and MODELS, and erratum pointers
  in the TASK-049/050/051/052/054/055/056 logs.
- **Reproduction scripts.** These are in `benchmarks/audits/task058/`. They need the
  git-ignored data; no simulation or training was run.
- **Not edited.** CLAUDE.md (agent configuration, which repeats the 0/150 and "every attempt"
  wording), `src/` docstrings (implementation hashes), and manifests.
- 2026-09-25: Merged as PR #42 (https://github.com/RaaSaaR-org/open-embodied-jepa/pull/42),
  squash commit `34bac32`. CI was green on the final head `6c0b997`. An independent
  reviewer's first verdict was REQUEST_CHANGES: three blocking findings (the MVP guard-stop
  wording, the D1 consequence stated too strongly, and the README's 3.13 cm figure) and seven
  non-blocking ones. All were fixed in `6c0b997`. The re-review verdict was APPROVE.
- Known residuals, non-blocking and not fixed: the README's per-episode MVP medians and maximum
  (3–15, max 170) come from git-ignored `outputs/mvp-v0-*/episode-*/episode.json`, and the
  reviewer's S5-10 exact-definition figures come from an uncommitted scratch script. Both are
  **not citable**, and neither is yet in the audit's consolidated not-citable list.
- Handed to the TASK-057 owner, not done here: amend `apple_policy_diagnostics_v1.md` for
  S6-29 (shadow expert `apple_collector_policy`, 745 commands), the Table E phase names, the
  S6-26 "bracketed by" wording, and §1.1(c)/§1.5 "near-constant command" (magnitude only).
  The CLAUDE.md 0/150 and "every attempt" wording (S4-25, L-01) is left to the maintainer.
