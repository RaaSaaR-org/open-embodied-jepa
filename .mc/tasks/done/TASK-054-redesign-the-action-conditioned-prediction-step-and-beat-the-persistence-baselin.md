---
id: TASK-054
aliases:
- TASK-054
title: Redesign the action-conditioned prediction step and beat the persistence baseline
slug: redesign-the-action-conditioned-prediction-step-and-beat-the-persistence-baselin
status: done
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

## Outcome
**All four arms FAILED the preregistered gate set, and the primary gate G2a failed on every
one: 0.8763 / 0.8814 / 0.8635 / 0.9036 against a threshold of 0.8.** None of the three
redesigns of the action-conditioned prediction step moved it below 0.8. By the pre-declared
reading this fires **Outcome B**: the control-formulation clause TASK-052 recorded is not
deferred again, behaviour cloning with the world model as a critic becomes the primary line,
and **CEM over this cost is abandoned**. Learned Apple->Plate remains at **0 successes**;
nothing in this task is a control or manipulation result.

Gates passed of 14: E0 (control) **10**, E1 (chunk) 9, E2 (tail) 9, E3 (step) 9 -- the
control passed more than every intervention. G9 (rollout excess <= 0.53 cm) failed on all
four: 1.058 / 1.118 / 0.903 / 0.796 cm.

**The headline is a replication result.** E0, arm B's configuration re-trained after
`models/base.py` gained three new keys, reproduced TASK-052 arm B **bit-for-bit**: all 13
carried gates identical as exact floats, all 11 decomposition quantities identical, and all
**170 weight tensors bit-identical** after 15,000 MPS training steps. The three new options
are provably inert at their defaults end to end, and E0 is a genuine control.

**Per-arm readings, each one factor from E0 and each a single seed:**
- **E1 action-chunk conditioning: no effect.** Rollout +0.010 cm, excess +0.060 cm, both
  intervals crossing zero; G6a collapsed 0.544 -> 0.287. It was measured with three future
  actions a horizon-8 CEM cannot supply (declared before the run) and still did not beat the
  control, which strengthens the negative result.
- **E2 stronger multistep weighting: right direction, far too small, not established.**
  Excess 1.058 -> 0.903 cm, a 14.6 % reduction where 49.9 % was needed; 90 % of its gain is
  in the targeted term. Its rollout contrast's clustered interval [-0.804, +0.132] crosses
  zero, and the protocol pre-declared the clustered reading wins.
- **E3 horizon-conditioned predictor: damaged the encoder.** Encoded target +1.023 cm and
  rollout +0.761 cm, the only contrast excluding zero under both bootstrap designs. Its
  excess is lowest (0.796 cm) only because its encoder degraded -- exactly the failure mode
  the preregistration warned G9 alone would hide. It is, however, the **only v4 arm to pass
  G7a** (0.7264), the action-sensitivity gate.

Also recorded: the frozen runner died silently after E0's training on a zsh bug (`status` is
a read-only special variable), was fixed and made resumable, and **nothing was retrained or
overwritten**; two preregistration defects found after launch were left in place and
reported rather than silently edited; and no interval is available for the G9 contrast
because the bootstrap's excess field is a different estimand.

Budget: 14,470 s = 4.02 h of MPS training against a 3.85 h estimate. `test_episodes_decoded`
is 0 in all eight reports; test and the frozen 44000-44019 cohort were never opened.

Protocol: `docs/experiments/apple_world_model_v4.md`.
Results: `docs/experiments/apple_world_model_v4_results.md`.
Manifest: `benchmarks/manifests/apple-world-model-v4.json`.
PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/29 (merged as `c5ec88c`).

The post-run verifier raised **eleven findings across five passes**; all were fixed before
merge. Three changed what the document claims rather than a digit: G7a's granularity (the
sole positive result rests on 3 sibling pairs of 106, with the control one pair short), the
trade-off section's rank correlation (recomputed with averaged ties over the seven distinct
arms it flips sign, -0.107 against +0.084), and a restatement of E2's shortfall that implied
a 7x requirement against the document's own 3.4x. Two false provenance claims were also
corrected. Every claim that was *computed* held under independent re-derivation; what failed
repeatedly was removing text a correction superseded.

Two findings are recorded elsewhere and are NOT the post-run verifier's. The pilot gate pass
sets and the missing Outcome A clause were found by the **pre-run** reviewer after the runs
had started, and are disclosed in the results document under *Two preregistration defects*.
The prediction/discrimination trade-off was **refuted by this author** before the verifier
saw it, when the proposed reading was checked against the artifacts and did not hold; the
verifier's finding was the narrower one that the correlation had been computed with
array-order tie-breaking on a double-counted arm.

## Acceptance Criteria
- [x] Preregistration `docs/experiments/apple_world_model_v4.md` and frozen manifest
      `benchmarks/manifests/apple-world-model-v4.json` committed BEFORE any gated run.
- [x] Each arm trained and evaluated exactly once, from a clean committed checkout, into a new
      outputs directory; no retuning after seeing gated results.
- [x] Gate results reported per arm with the encoder/rollout decomposition, contrasts carrying
      both an iid and an episode-clustered paired-bootstrap interval with a stated method.
- [x] `test_episodes_decoded == 0`; reset seeds 44000-44019 and TEST untouched; normalization
      fitted on train only.
- [x] The pre-declared control-formulation clause restated and applied: if G2a does not go below
      0.8, behaviour cloning with the world model as a critic becomes the primary line and CEM
      over this cost is abandoned.
- [x] Results document keeps every failure and negative result; no green CI or passing smoke is
      described as working manipulation.

## Notes
%% mc-links: [[TASK-052]] %%
