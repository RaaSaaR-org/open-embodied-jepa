---
id: TASK-071
aliases:
- TASK-071
title: 'Apple first policy v2: train the first learned Apple-to-Plate policy on e9 demonstrations under apple-to-plate-v2'
slug: apple-first-policy-v2-train-the-first-learned-apple-to-plate-policy-on-e9-demons
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- learned-policy
- preregistration
sprint: ''
depends_on:
- "[[TASK-070]]"
due_date: ''
created: 2026-09-28
updated: 2026-09-28
---


# Apple first policy v2: train the first learned Apple-to-Plate policy on e9 demonstrations under apple-to-plate-v2

## Description

The owner's TASK-071 brief (2026-09-28, via the coordinator): train the project's first learned
Apple→Plate policy that succeeds. Carry TASK-067's preregistered design
(`docs/experiments/apple_first_policy_v1.md`, rulings R1–R7) over to `apple-to-plate-v2`, with
only these changes:
1. Task apple-to-plate-v2; success `apple_at_rest_v0`, the latched v1 scorer reported beside it.
2. Demonstrator and DAgger expert: e9, frozen (TASK-070).
3. A fresh e9 demonstration corpus under v2 (`apple-look-v1` is not valid for BC here), with
   hashes and splits by session.
4. Perception bars from e9's measured plate-error tolerance (TASK-070), re-derived through a
   calibration stage with v1's logic; no looser bars.
5. Fresh seeds for everything, disjoint from every declared or spent range.

Everything else stays as in v1 unless the owner rules a justified change; every deviation is in
the protocol's §3 table.

Protocol: `docs/experiments/apple_first_policy_v2.md`. Manifest:
`benchmarks/manifests/apple-first-policy-v2.json`. Code: `first_policy_v2.py`,
`first_policy_v2_runtime.py`, `first_policy_v2_model.py`, `scripts/run_first_policy_v2.py`,
`scripts/check_first_policy_v2_render.py`.

A success of the fallback F is "partially learned". Privileged or scripted successes (the corpus,
C0, B-oracle, A4-look) are never learned results. **Learned Apple→Plate is still 0 successes.**

## Seeds
- Gated: 51000–51999 (corpus 51000–51199; perception 51200–51583; C0 51584–51615; DAgger
  51616–51999). Development cohort D2: 52000–52015. Reserved: 52016–52099. Smoke and render
  check only: 52100–52199.
- The 2026-09-28 search (protocol §4.1) found no seed use in 51000–52999.

## Acceptance Criteria
- [ ] Preregistration PR (protocol, manifest, code, tests, smoke runs) merged by the owner on an
      independent reviewer's reported APPROVE.
- [ ] Render check IDENTICAL at 8 workers, and the pre-run reviewer's reported GO, before the
      gated run.
- [ ] Gated run from a clean checkout of `main`, to completion under the frozen caps (void rule:
      one repeat; a second V is INCONCLUSIVE).
- [ ] Results PR states the outcome plainly with counts, reviewed; the owner merges.

## Notes
- 2026-09-28: card opened under the owner's TASK-071 brief. Branch
  `feat/task-071-first-policy-v2`. Learned Apple→Plate is still 0 successes.

%% mc-links: [[TASK-070]] %%
