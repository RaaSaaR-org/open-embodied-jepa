---
id: TASK-052
aliases:
- TASK-052
title: Attack world model v3 readout precision under motion and gate CEM candidate ranking
slug: attack-world-model-v3-readout-precision-under-motion-and-gate-cem-candidat
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- learning
sprint: ''
depends_on:
- "[[TASK-050]]"
due_date: ''
created: 2026-09-23
updated: 2026-09-23
---


# Attack world model v3 readout precision under motion and gate CEM candidate ranking

## Description
Follow-up to TASK-050, whose three arms all failed the preregistered offline gates. The measured
defect is readout precision under motion: on val at h=8 the palm-apple error is 0.77 cm over all
valid windows but 3.65 cm on the 1,462 moving windows, and a *directly encoded* observation is
already 3.26 cm off, so about 89 % of the rollout error is in the encoder/head rather than the
dynamics. Action information exists (the shuffled-action and sibling-divergence controls pass) but
the predicted approach cost ranks candidates at only rho ~= 0.16.

v3 attacks that defect with shared, backend-agnostic changes:

1. A two-camera model (onboard + FK-placed hand crop) fused in `models/base.py`, so the one-line
   `world_model.backend` swap invariant holds.
2. A readout loss weighted towards moving frames, plus auxiliary apple/palm world-position
   regression readouts derived from privileged labels (targets only, never inputs).
3. The absolute cost-calibration gate (v2 G6) replaced by the CEM-relevant metric: within-state
   Spearman rho and top-1 regret across sibling candidate actions from one pre-grasp state.
4. Carried over from the TASK-050 review: near-constant proprioception dimensions are no longer
   amplified by the 0.01 std floor, and `load_split` gets the test it never had.

OFFLINE only. No closed loop is started in this task; learned Apple->Plate stays at 0 successes.

## Acceptance Criteria
- [ ] Preregistration committed before the frozen runs: `docs/experiments/apple_world_model_v3.md`
      and `benchmarks/manifests/apple-world-model-v3.json` (arms, budgets, seed, selection rule,
      gates, pre-declared readings and next steps, disclosed pilots).
- [ ] Two-camera support is shared backend-agnostic code; the backend swap stays one config line;
      planners/evaluators touch latents only through declared readouts; test split never decoded.
- [ ] The proprioception std-floor amplification is fixed with a test; `load_split` has a test.
- [ ] Fresh pre-run review; blockers fixed; smoke on a tiny subset from the committed revision.
- [ ] The frozen arms run once each from a clean committed revision and are evaluated against the
      frozen gates.
- [ ] Fresh post-run verification; results doc reporting every gate honestly, failures kept; ruff,
      pytest, `mc validate`; PR with green CI and an independent review.
- [ ] PR merged and this task closed.

## Scope and resources
MPS (M5 Pro, 48 GB). Total training budget about 6 hours across arms, sequential, sharing the
machine with a CPU-bound TASK-051. Checkpoints and reports under the main checkout's ignored
`checkpoints/task052-wm-v3/` and `outputs/task052-wm-v3/`.

## Notes
- Branch `feat/task-052-world-model-v3` from main `34ebce9` (TASK-050 closure).
%% mc-links: [[TASK-050]] %%
