---
id: TASK-018
aliases:
- TASK-018
title: Build grasp, transport, release, and Apple to Plate curriculum
slug: build-grasp-transport-release-and-apple-to-plate-curriculum
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
- "[[TASK-009]]"
- "[[TASK-014]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---




# Build grasp, transport, release, and Apple to Plate curriculum

## Description

Extend reach through stable lift, transport, placement, release and dwell success, preserving right-arm-first control with both hands represented.

## Acceptance Criteria

- [x] Freeze geometric/contact thresholds, dwell times, timeout, reset distribution, and task versions before final evaluation.
- [x] A scripted controller validates reachability and scoring; boundary tests reject hovering objects or objects still held above the plate.
- [x] Record intermediate stage outcomes and representative successes/failures; expand training coverage without leaking reserved test combinations.

## Notes

- Milestone: m4.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.
## Implementation evidence (ongoing)

Current-code controller v1 had 0/6 full successes, faithfully retained. A separately declared v2 releases earlier after transport, avoiding compressed-thumb velocity stops without changing limits/physics; first two seen cube→plate trials now pass all ordered stages. A frozen supplementary 24-episode collection is authorized before final training, no held-out Apple→Plate samples. See docs/experiments/manipulation_controller_v{1,2}.md. Final thresholds/reset/goal manifest will be sealed before the comparison.

Delivery PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 (draft; merge pending).
## Deliverable review evidence

The final 50-goal manifest pins thresholds, resets, assets, actions and source before evaluation. A current valid-reset early-release probe achieved 4/4 scripted full successes with unchanged physics and limits. Supplementary collection retained 24 attempts, 18 recorded episodes and 9 successes, including release/retreat frames, with preserved 14/2/2 splits and no Apple→Plate samples. Scorer tests reject held, hovering and tossed-object shortcuts.

PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3. Review status does not claim the merge has occurred.

## Acceptance close-out

Deliverable acceptance and final audit passed. PR https://github.com/RaaSaaR-org/open-embodied-jepa/pull/3 was awaiting merge when this record was written; its GitHub state is authoritative. Closing this engineering task does not claim successful learned manipulation or physical hardware validation.
%% mc-links: [[TASK-009]] [[TASK-014]] %%
