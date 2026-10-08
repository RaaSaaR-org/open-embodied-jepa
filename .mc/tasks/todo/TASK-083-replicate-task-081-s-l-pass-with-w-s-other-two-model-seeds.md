---
id: TASK-083
aliases:
- TASK-083
title: "Replicate TASK-081's L-PASS with W's other two model seeds (66801, 66802) under C1-M"
slug: replicate-task-081-s-l-pass-with-w-s-other-two-model-seeds
status: todo
priority: 1
owner: ''
projects: []
customers: []
tags:
- lewm
- apple-pnp
- preregistration
- replication
sprint: ''
depends_on:
- "[[TASK-081]]"
due_date: ''
created: 2026-10-08
updated: 2026-10-08
---

# Replicate TASK-081's L-PASS with W's other two model seeds (66801, 66802) under C1-M

## Description

TASK-081 closed **L-PASS** under C1-M with one model seed (W-66800, flagged
`last_two_triggered`): W 118/128 on 128 fresh gated resets, non-inferior to the non-learned H-rule
(125/128) within the allocated −16/128. R19.23 (3) recommended, before any broader claim, more
model seeds of W. This task is that replication, option (3a): TASK-077's other two seeds, 66801
and 66802, each W with its own N and its own R-S and R-N from TASK-080's Stage R. Nothing is
trained or fitted; CPU only.

**The preregistration (DRAFT, R21.1–R21.12, decided by Claude under owner delegation):**
[`docs/experiments/apple_lewm_seed_replication_v2.md`](../../../docs/experiments/apple_lewm_seed_replication_v2.md).
TASK-081's condition, controller (affine_local), τ_commit, arms, gates, bar (112/128), margin
(16/128) and ladder per seed, on one fresh cohort S of 128 resets (75200–75327) that both seeds
share with the seed-independent arms (L-rand, H-rule, H-sysid, privileged H-final(commit)).
Replication claim REP-PASS only if both seeds reach L-PASS individually (a conjunction; no pooling,
no rescue); REP-ONE / REP-NONE otherwise; no abandonment clause; what each row means for R7 is
declared in advance. W-66800 runs on the same resets, reported only. No Stage D. A new thin runner
(`scripts/run_lewm_rep_v2.py`), because TASK-081's pinned runner hard-codes seed 66800. Seeds
75000–75999 (debug 75900–75999), salts 8501–8512. About 75–90 min of CPU.

## Acceptance Criteria

- [x] DRAFT preregistration, independently reviewed (APPROVE) and merged (#173)
- [x] Stage 0 (R21.13–R21.16): frozen-block candidate, runner, tests, DRAFT manifest, debug smokes
  on 75910–75913, power with salt 8502; record
  `docs/experiments/apple_lewm_seed_replication_v2_stage0.md` (merged with its PR)
- [x] The freeze (R21.17–R21.18), independently reviewed and merged with its PR
- [ ] Stage S once, on an independent reported GO, CPU only
- [ ] Results PR with independently reviewed restatements; R7 changed by its own reviewed ruling
  as §10 declares

## Notes

TASK-082 closed at K0 (CAL-ESCALATE; R20.23–R20.25) and is independent of this task.
