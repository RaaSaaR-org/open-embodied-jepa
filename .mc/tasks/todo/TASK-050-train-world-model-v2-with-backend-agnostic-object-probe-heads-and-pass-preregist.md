---
id: TASK-050
aliases:
- TASK-050
title: Train world model v2 with backend-agnostic object probe heads and pass preregistered offline gates
slug: train-world-model-v2-with-backend-agnostic-object-probe-heads-and-pass-preregist
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
- "[[TASK-048]]"
due_date: ''
created: 2026-09-22
updated: 2026-09-22
---



# Train world model v2 with backend-agnostic object probe heads and pass preregistered offline gates

## Description
Roadmap T3. Train world model v2 on the TASK-048 wide corpus `data/apple-wide-v1` and test,
OFFLINE on val, whether a learned model can predict the quantities the object-aware phase cost
needs (`object_ceiling.py`): palm-apple offset, apple height, apple-plate offset, contact/held,
from its own PREDICTED latents, without simulator truth at run time. LeWM (`models/lewm.py` over
the pinned upstream) is the product backend; native JEPA is the comparison. Shared,
backend-agnostic additions: state fusion (proprioception into the latent) and declared readout
heads (`models/readout.py`, `Capabilities.readouts`, `model.readout`). Privileged sidecar labels
are training targets and val scoring references only. No closed loop; learned Apple->Plate stays 0.

## Acceptance Criteria
- [ ] Preregistration committed before the frozen runs: `docs/experiments/apple_world_model_v2.md`
      and `benchmarks/manifests/apple-world-model-v2.json` (budgets, seed, selection rule, gates
      G1-G8, pre-declared readings, disclosed pilots).
- [ ] Shared readout/state-fusion code with tests; backend swap stays a one-line config change;
      planner/evaluator touch latents only through declared readouts; test split never decoded.
- [ ] Fresh pre-run review; blockers fixed; smoke on a tiny subset from the committed revision.
- [ ] Three frozen runs executed once each from a clean committed revision (LeWM/onboard,
      LeWM/hand-crop, native/onboard) and evaluated against the frozen gates.
- [ ] Fresh post-run verification; results doc reporting every gate honestly, including failures;
      ruff, pytest, `mc validate`; PR merged.

## Scope and resources
MPS (M5 Pro, 48 GB), about one hour per run plus evaluation. Checkpoints and reports under the
main checkout's ignored `checkpoints/task050-wm-v2/` and `outputs/task050-wm-v2/`.

## Notes
- Branch `feat/task-050-world-model-v2` from main `57f3169` (TASK-048/PR #18).
- Gates and budgets committed in `17bf5dd` before any pilot longer than 200 steps.
- Pilots (disclosed in the protocol): `smoke-a`, `smoke-full` (decode 11.5 s, 0.23 s/step LeWM),
  `pilot-a` (2,000 steps; showed that 19% of train frames have the apple on the floor, which led
  to the dropped-apple regression mask and the `apple_dropped` readout), `pilot-b` (2,000 steps
  after the fix: h=8 moving median 7.4-8.3 cm against 7.8-9.2 cm persistence).
- Pre-run review (fresh subagent): one blocker (the untrained step-0 state could be selected),
  fixed in `1a4c484` together with G5 moving to the lift cohort, image-only collapse statistics,
  `--require-clean`, provenance checks in `evaluate` and doc wording. Re-review CLEAR.
- Frozen protocol revision `3b6af0b`.
%% mc-links: [[TASK-048]] %%
