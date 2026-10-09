---
id: TASK-085
aliases:
- TASK-085
title: 'Phase 1: MuJoCo play corpus play-v1 (right arm + right Dex3 hand; random, perturbed and failed play)'
slug: phase-1-mujoco-play-corpus-one-arm-one-dex3-hand
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- play-corpus
- data
- preregistration
- phase-1
sprint: ''
depends_on:
- "[[TASK-084]]"
due_date: ''
created: 2026-10-09
updated: 2026-10-09
---


# Phase 1: MuJoCo play corpus play-v1 (right arm + right Dex3 hand; random, perturbed and failed play)

## Description

Phase 1 of [JEPA_ZERO_SHOT_PLAN.md](../../../docs/JEPA_ZERO_SHOT_PLAN.md) ("Play corpus"): a
task-agnostic MuJoCo play corpus on G1 with the right arm and right Dex3 hand, four movable
objects (apple, cube, banana, can) and the plate, a privileged scripted play mixture (scripted,
perturbed and random modes; pick, push, poke and wander skills; deliberate failures), 20 Hz,
onboard 112 px camera, full robot state and a per-episode sidecar of object truth for labels.
3 200 episode seeds (≈ 15.8 h expected). Gate (plan): covers the Phase 3 test workspace and
≥ 20 % of episodes move an object.

**The preregistration (R23.1–R23.9, DRAFT; frozen at the Stage 0 merge):**
[`docs/experiments/play_corpus_v1.md`](../../../docs/experiments/play_corpus_v1.md).
Not an Apple→Plate experiment and not a learned result; R7 does not change.

## Acceptance Criteria

- [ ] DRAFT preregistration independently reviewed (APPROVE) and merged
- [ ] Stage 0: collector module, runner, gate script, tests, debug smoke (86000–86015);
  independently reviewed and merged (freezes the protocol)
- [ ] Corpus `data/play-v1` collected on the merged revision; evidence under
  `~/develop/emai/evidence/task085-*` with `SHA256SUMS`
- [ ] Results document and DECISIONS entry with the measured row, reviewed and merged; R7
  unchanged, stated explicitly
