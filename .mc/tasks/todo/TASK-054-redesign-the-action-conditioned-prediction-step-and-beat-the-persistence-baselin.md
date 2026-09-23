---
id: TASK-054
aliases:
- TASK-054
title: Redesign the action-conditioned prediction step and beat the persistence baseline
slug: redesign-the-action-conditioned-prediction-step-and-beat-the-persistence-baselin
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
- "[[TASK-052]]"
due_date: ''
created: 2026-09-23
updated: 2026-09-23
---


# Redesign the action-conditioned prediction step and beat the persistence baseline

## Description
Follow-up to TASK-052, whose four LeWM arms all failed the preregistered offline gate set. The
merged encoder/rollout decomposition (`docs/experiments/apple_world_model_v3_results.md`) splits
the G1 error on identical moving validation windows:

| | encoded target | rollout (G1) | rollout excess |
|---|---|---|---|
| v2 LeWM/onboard (carried over, NOT recomputed) | 3.26 cm | 3.65 cm | 0.39 cm |
| v3 arm B (onboard) | 2.59 cm | 3.65 cm | 1.06 cm |
| v3 arm D (onboard, patch 8) | 2.47 cm | 3.30 cm | 0.83 cm |

The encoder improved 21-24 % while the rollout excess doubled to tripled. The error moved into
the action-conditioned prediction step, and G7a (siblings' own actions beat swapped ones) failed
on arms A and B, so the protocol's own earliest-failing-group rule selects branch 2: redesign the
action conditioning.

**Two constraints this task does not overstate away.** Arm D's encoded-target error alone
(2.47 cm) already exceeds the 1.5 cm G1 threshold, so a perfect predictor would still fail G1:
both terms fail and only one grew. And the v2 reference row is carried over from the v2 results
document, not recomputed, because the v2 checkpoints have an incompatible implementation hash.

v4 changes the prediction step and nothing else. Four LeWM arms, one camera (onboard), patch 14,
each one shared backend-agnostic config key away from the baseline, all defaults OFF so every
earlier model keeps its exact predictor and its exact loss:

- **E0** baseline: TASK-052 arm B's configuration re-trained at this revision (the control).
- **E1** `action_chunk: 4` - each step conditioned jointly on a chunk of four future actions.
- **E2** `multistep_tail_weight: 3.0` - the multistep loss ramped towards the late steps.
- **E3** `predictor_step_embedding: true` - a horizon-conditioned rollout instead of one shared
  single-step module applied autoregressively.

The primary gate is **G2a (rollout error divided by the model's own persistence readout) < 0.8**,
which has never passed (v2 0.835; v3 1.010 / 0.876 / 0.940 / 0.831). A new gate **G9** bounds the
rollout excess directly, since that is the term this task attacks. G1, G2b, G3, G4, G5, G6a/G6b,
G7a/G7b and G8a/b/c are carried forward verbatim from the v3 protocol; no carried threshold is
loosened.

OFFLINE only. No closed loop is started in this task; learned Apple->Plate stays at 0 successes.

## Acceptance Criteria
- [ ] Preregistration `docs/experiments/apple_world_model_v4.md` and frozen manifest
      `benchmarks/manifests/apple-world-model-v4.json` committed BEFORE any gated run.
- [ ] Each arm trained and evaluated exactly once, from a clean committed checkout, into a new
      outputs directory; no retuning after seeing gated results.
- [ ] Gate results reported per arm with the encoder/rollout decomposition, contrasts carrying
      both an iid and an episode-clustered paired-bootstrap interval with a stated method.
- [ ] `test_episodes_decoded == 0`; reset seeds 44000-44019 and TEST untouched; normalization
      fitted on train only.
- [ ] The pre-declared control-formulation clause restated and applied: if G2a does not go below
      0.8, behaviour cloning with the world model as a critic becomes the primary line and CEM
      over this cost is abandoned.
- [ ] Results document keeps every failure and negative result; no green CI or passing smoke is
      described as working manipulation.

## Notes
%% mc-links: [[TASK-052]] %%
