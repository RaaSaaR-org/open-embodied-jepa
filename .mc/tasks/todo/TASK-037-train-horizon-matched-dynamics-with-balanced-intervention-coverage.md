---
id: TASK-037
aliases:
- TASK-037
title: Train horizon-matched dynamics with balanced intervention coverage
slug: train-horizon-matched-dynamics-with-balanced-intervention-coverage
status: review
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- world-model
- training
sprint: ''
depends_on:
- "[[TASK-035]]"
due_date: ''
created: 2026-09-21
updated: 2026-09-21
---



# Train horizon-matched dynamics with balanced intervention coverage

## Description

Matched branch data improved VAL H16 action assignment to73.87%, but raw endpoint error reduction was6.26%, below the frozen10% gate. H8 secondary results passed. Test the hypothesis that training at H16 with explicitly balanced intervention coverage improves longer-horizon causal prediction. A naive H16 uniform-window sampler would reduce each16-step branch from9 H8 windows to1 H16 window, diluting the intervention cohort; preserve half-batch intervention coverage explicitly.

## Acceptance Criteria

- [x] Add optional explicit split-scoped sampling groups to generic training, preserving the default uniform-window behavior; validate disjoint complete TRAIN/VAL group membership and no TEST leakage.
- [x] Preregister one bounded sensor_wm attempt: same architecture, seed0,3000 updates,B16,H16,600 wall seconds,16GiB;8 nominal/full-demo windows and8 intervention windows per train/validation batch, sampled uniformly within each group. Select by H16 normalized visual validation MSE averaged over rollout (the existing selector is named `raw_mse`).
- [x] Freeze group/config/data/source/protocol hashes; keep original checkpoints, data, failed gates and final cohort untouched.
- [x] Compare selected checkpoint against original and branch-v1 evidence using the unchanged matched causal metric and VAL H16 gate; preserve failure outcomes. No automatic physical run when gate fails.
- [ ] Independent code/scientific review, focused/full required checks, reproducible evidence, and reviewed PR delivery.

## Notes

Collector corpus: `data/apple-branches-v1`, manifest SHA256 `6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`. Implementation agent owns generic sampling and bounded supervisor profile/protocol; coordinator owns MC/Git and experiment authorization. Architecture and model implementation remain unchanged for strict checkpoint compatibility. This jointly changes horizon and sampling support; attribute any difference to that declared training configuration, not horizon alone.
## Implementation review

Explicit group sampling and the supervised profile were independently reviewed with no remaining blockers. Full optional/graphics suite:462 passed in10.15s before parent merge;58 affected sampler/supervisor/diagnostic checks passed after merge. Default window-sampling RNG behavior remains covered. Parent PR #5 is verified merged; this branch preserves its civil-wall diagnostic fix and delivered evidence. Full H16 training subsequently completed all 3,000 updates in 49.36 seconds; the selected update-3000 checkpoint passed the unchanged primary causal gate (89.40% assignment; 81.41% endpoint error reduction). See the H16 result report/manifest. Development control was interrupted by an explicit user pause and is preserved separately; the user has now authorized continuation.
%% mc-links: [[TASK-035]] %%

## Delivery

PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/7. Independent implementation and saved-evidence reviews completed; current revision CI passes on Linux and macOS including optional integration. Merge remains pending. Physical control remains TASK-033: the resumed comparison produced one completed learned failure, one persistence timeout and four unstarted budget-limited attempts; no grasp or placement. The causal gate does not establish physical success.
