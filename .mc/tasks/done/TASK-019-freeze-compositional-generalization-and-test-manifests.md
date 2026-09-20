---
id: TASK-019
aliases:
- TASK-019
title: Freeze compositional generalization and test manifests
slug: freeze-compositional-generalization-and-test-manifests
status: done
priority: 2
owner: ''
projects: []
customers: []
tags:
- m4
- mvp
sprint: ''
depends_on:
- "[[TASK-010]]"
- "[[TASK-018]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---




# Freeze compositional generalization and test manifests

## Description

Reserve apple-to-plate combinations as described in the PRD and distinguish in-distribution evaluation from zero-shot composition tests.

## Acceptance Criteria

- [x] Publish seen/unseen object, receptacle, appearance, and pairing tables with immutable episode/goal/reset manifests.
- [x] Verify no session/frame/goal leakage and no task-specific training or normalization fitting on the held-out cohort.
- [x] Freeze evaluation seeds, episode counts, thresholds and planned comparisons before unsealing final results.

## Notes

- Milestone: m4.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
## Implementation evidence (ongoing)

Implementation in goals.py strictly verifies image, reset, threshold, runtime-source, asset and action hashes. Final protocol declared in docs/experiments/mvp_evaluation.md: 50 resets (20000–20049), three training seeds, same goals for both backends and controls, held-out Apple→Plate pairing only; no unseen-appearance claim. Data assembly preserves each source corpus’s original split assignments through checked DatasetStore.freeze_split_assignments; 36 dataset tests pass including official reader.

Delivery PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 (draft; merge pending).
## Deliverable review evidence

The frozen evaluation protocol and versioned manifests document seen appearances, held-out pairing, reset distribution and all 50 goals. An independent 5.15-second audit verified source membership, zero cross-split sessions/exact RGB matches, zero goal/corpus matches, no Apple→Plate episodes and training-only normalization. Near-duplicate semantic independence is not claimed. See benchmarks/manifests/mvp-leakage-audit.json.

PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3. Review status does not claim the merge has occurred.

## Acceptance close-out

Deliverable acceptance and final audit passed. PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 was awaiting merge when this record was written; its GitHub state is authoritative. Closing this engineering task does not claim successful learned manipulation or physical hardware validation.
%% mc-links: [[TASK-010]] [[TASK-018]] %%
