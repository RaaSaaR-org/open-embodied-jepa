---
id: TASK-087
aliases:
- TASK-087
title: 'Phase 2: robot-grounded LeWM on play-v1, offline'
slug: phase-2-robot-grounded-lewm-offline
status: done
priority: 2
owner: ''
projects: []
customers: []
tags:
- world-model
- lewm
- preregistration
- p2-fail
- phase-2
sprint: ''
depends_on:
- "[[TASK-085]]"
- "[[TASK-086]]"
due_date: ''
created: 2026-10-09
updated: 2026-10-10
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
- [x] Stage 0 code, tests and debug smoke; independently reviewed (APPROVE at c7de73d) and merged (#188, `dab7475`); reader fix #189 (`bcda883`, R24.20)
- [x] Stages F, B, T, V, F′, E run at `bcda883`; evidence under
  `~/develop/emai/evidence/task087-*` with `SHA256SUMS`
- [x] Results (P2-VOID / P2-PLAIN-INVALID / P2-PASS / P2-SHORT / P2-FAIL) and DECISIONS entry,
  reviewed and merged; R7 unchanged

## Outcome

**P2-FAIL** (R24.21–R24.25). S and G beat plain LeWM on sensitivity and top-1 at h = 4 and 8 on
all seeds but fail one-step sensitivity (wrong / true below plain's); G's one-step top-1 is
higher (0.40–0.41 vs 0.35). Val report sha256 `dbc05e790ce503142b7e6de6af65af99a5b0510b3c171364e0fdeb4ac67cac38`
(recorded 2026-10-10T05:49:58Z, before the test split was read); test report `aed62c91…ca18`.
Evidence `~/develop/emai/evidence/task087-*`. Results:
[grounded_lewm_v1_results.md](../../../docs/experiments/grounded_lewm_v1_results.md).
S and G are not carried into Phase 3 as "the grounded model" (stop rule); R7 unchanged.
%% mc-links: [[TASK-085]] [[TASK-086]] %%
