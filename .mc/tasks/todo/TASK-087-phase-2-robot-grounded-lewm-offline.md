---
id: TASK-087
aliases:
- TASK-087
title: 'Phase 2: robot-grounded LeWM on play-v1, offline'
slug: phase-2-robot-grounded-lewm-offline
status: in-progress
priority: 2
owner: ''
projects: []
customers: []
tags:
- world-model
- lewm
- preregistration
- phase-2
sprint: ''
depends_on:
- "[[TASK-085]]"
- "[[TASK-086]]"
due_date: ''
created: 2026-10-09
updated: 2026-10-09
---



# Phase 2: robot-grounded LeWM on play-v1, offline

## Description

Phase 2 of [JEPA_ZERO_SHOT_PLAN.md](../../../docs/JEPA_ZERO_SHOT_PLAN.md): on frozen DINOv2 tokens
of the play corpus `play-v1` (TASK-085), train plain LeWM and robot-grounded variants (+ robot
state; + state, joint-change and inverse-dynamics losses), with state fusion + readout heads as a
control arm, and test whether a grounded version beats plain LeWM on action sensitivity and
short-step ranking on all seeds. Sim play only (TASK-086 dropped the real-data arm).

**The preregistration (R24.1–R24.14, DRAFT; frozen at the Stage 0 PR's merge):**
[`docs/experiments/grounded_lewm_v1.md`](../../../docs/experiments/grounded_lewm_v1.md).
Offline only; not an Apple→Plate experiment; R7 does not change.

## Acceptance Criteria

- [x] DRAFT preregistration independently reviewed (APPROVE at 497a551) and merged (#187)
- [ ] Stage 0 code, tests and debug smoke; independently reviewed and merged (freezes it)
- [ ] Stages F, B, T, V, F′, E run at the frozen revision; evidence under
  `~/develop/emai/evidence/task087-*` with `SHA256SUMS`
- [ ] Results (P2-VOID / P2-PLAIN-INVALID / P2-PASS / P2-SHORT / P2-FAIL) and DECISIONS entry,
  reviewed and merged; R7 unchanged
%% mc-links: [[TASK-085]] [[TASK-086]] %%
