---
id: TASK-071
aliases:
- TASK-071
title: 'Apple first policy v2: train the first learned Apple-to-Plate policy on e9 demonstrations under apple-to-plate-v2'
slug: apple-first-policy-v2-train-the-first-learned-apple-to-plate-policy-on-e9-demons
status: done
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
updated: 2026-10-02
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
- [x] Preregistration PR (protocol, manifest, code, tests, smoke runs) merged by the owner on an
      independent reviewer's reported APPROVE (PR #96, squash `9e23ced`; owner rulings T71-R1 and
      T71-R2 recorded verbatim).
- [x] Render check IDENTICAL at 8 workers, and the pre-run reviewer's reported GO, before the
      gated run (the GO reached the task agent as a hand-back message; it was not relayed to the
      coordinator before the run started — see the results doc §6).
- [x] Gated run from a clean checkout of `main` (`9e23ced`), to completion under the frozen caps:
      run-1, no void, 6 215 s on the Mac (MPS).
- [x] Results PR states the outcome plainly with counts, reviewed; the owner merges (#97,
      `7a2dafd`).

## Notes
- 2026-09-28: card opened under the owner's TASK-071 brief. Branch
  `feat/task-071-first-policy-v2`. Learned Apple→Plate is still 0 successes.
- 2026-09-28: PR #96 merged (`9e23ced`) after rulings T71-R1 (a counted success needs a latched
  grasp) and T71-R2 (and a latched place). Pre-run reviewer reported GO; render check IDENTICAL.
- 2026-09-28: run-1 from `9e23ced` on the Mac, 08:35:44Z–10:19:19Z, report sha256
  `77aae207…d90b`. **Outcome M1-PASS**, carried P-3. Counted successes on D2 (of 16): P-0 4,
  P-1 9, P-2 15, **P-3 16**, C-3 3, **R-3 16**, A4-look 16, D-oracle-perc 16, B-oracle 16,
  B-replay 9, B-hold 0, B-random 0; no at-rest attempt went uncounted. An existence result on the
  development cohort only; encoder pretraining shows no measurable contribution (R-3 = P-3); the
  oracle arms gain nothing over the readout. The official learned Apple→Plate count on the frozen
  benchmark is still 0; cohort C untouched; M2 needs a separate authorization.
- 2026-09-28: the project paused, then moved to a Linux PC (RTX 5080), where the frozen v2 runner
  refuses to start (floor digest, MPS, frame hashes); a Linux replication is a separate task.
  Results PR: `results/task-071-run-1`.
- 2026-10-02: closed as done. The results merged in #97 (`7a2dafd`); the Linux replication and M2
  ran under TASK-072 (#100, #104). The status sentence is in `docs/DECISIONS.md` (decision
  2026-10-02, R7). `mc validate` passed with mc 0.1.14.
%% mc-links: [[TASK-070]] %%
