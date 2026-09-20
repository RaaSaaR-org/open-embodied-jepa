---
id: TASK-014
aliases:
- TASK-014
title: Demonstrate native JEPA image-goal reaching in MuJoCo
slug: demonstrate-native-jepa-image-goal-reaching-in-mujoco
status: done
priority: 2
owner: ''
projects: []
customers: []
tags:
- m2
- mvp
sprint: ''
depends_on:
- "[[TASK-010]]"
- "[[TASK-011]]"
- "[[TASK-013]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---




# Demonstrate native JEPA image-goal reaching in MuJoCo

## Description

Train on the pilot, then run the first complete data-to-model-to-planner reach slice on the Mac.

## Acceptance Criteria

- [x] Archive checkpoint, data hashes, complete config, commands, and fixed reset/goal cohort.
- [x] Report one/multi-step prediction, collapse diagnostics, action ablations, CPU/MPS memory and synchronized planning latency.
- [x] Demonstrate reach progress and compare success with random/hold and scripted controls; diagnose failures before adding grasp complexity.

## Notes

- Milestone: m2.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
## Implementation evidence (ongoing)

V2 corrected target-space persistence and online/target collapse guards. Both native and LeWM selected noncollapsed action-sensitive checkpoints after 3,000 CPU updates. Each achieves 1/5 reach versus hold/random 0/5 and oracle 5/5; wide intervals preclude reliability claims. Historical collapsed v0 and invalid mixed-space v1 evidence remains preserved. See docs/experiments/reach_results.md and benchmarks/manifests/reach-development-results.json; full local logs outputs/reach-v2-{native,lewm}. Source for these historical checkpoints is commit 772b5ab.

Delivery PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 (draft; merge pending).
## Final measured evidence

Corrected v2 models each reached 1/5 development goals versus hold/random 0/5 and oracle 5/5. Their declared per-dimension online/target collapse guards and action/persistence controls pass. CPU process RSS, synchronized planner latency and horizon 1/4/8 diagnostics are retained; MPS capability/diagnostic checks are separate from the CPU experiment. This is weak observed progress, not established statistical superiority. See docs/experiments/reach_results.md and the versioned development manifest.

## Acceptance close-out

Deliverable acceptance and final audit passed. PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 was awaiting merge when this record was written; its GitHub state is authoritative. Closing this engineering task does not claim successful learned manipulation or physical hardware validation.
%% mc-links: [[TASK-010]] [[TASK-011]] [[TASK-013]] %%
