---
id: TASK-052
aliases:
- TASK-052
title: Attack world model v3 readout precision under motion and gate CEM candidate ranking
slug: attack-world-model-v3-readout-precision-under-motion-and-gate-cem-candidat
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

## Outcome
**All four arms FAILED the preregistered gate set. No arm passed the primary gate G1 and no arm
passed G2a, so by the pre-declared reading the closed loop does NOT start.** Learned Apple->Plate
remains at **0 successes**; nothing in this task is a control or manipulation result.

Gates passed of 13: A (primary, two cameras) 8, B (onboard only) 10, C (uniform readout) 10,
D (patch 8) 10. G1 (<= 1.5 cm): 6.62 / 3.65 / 6.11 / 3.30 cm. G2a (<= 0.8): 1.010 / 0.876 /
0.940 / 0.831 -- arm A is worse than its own persistence readout.

**Both v3 hypotheses were wrong in the direction they were predicted to help, and this is kept as
a negative result:**
- A<->B (the second camera, a true one-tensor ablation): adding the FK hand crop made G1 *worse*,
  3.65 -> 6.62 cm, and cut the latent effective rank 7.74 -> 5.12, at 1.7x peak RSS.
- A<->C (the v3 readout shaping -- motion weighting AND the auxiliary position targets, which this
  design cannot separate): removing it *improved* G6a 0.328 -> 0.516, G7a 0.613 -> 0.717 and even
  G1 6.62 -> 6.11 cm.
- B<->D (patch 14 -> 8, the only other controlled contrast): G1 3.65 -> 3.30 cm for 2.1x the wall
  clock, while losing G6a (0.544 -> 0.378). Within-run validation jitter exceeds this gap and there
  is one seed per arm, so it is not distinguishable from noise.
- Arm B reproduces TASK-050's G1 to within 0.03 mm (3.6481 vs 3.6453 cm) and is worse on G2a
  (0.876 vs 0.835): the v3 readout work did not move the headline number.

**Genuinely positive:** the CEM-relevant ranking metric improved a lot over v2. Median top-1 regret
is exactly 0 m in all four arms against a label-derived random-choice baseline of 8.44 mm, and G6a
passes for arms B (0.544) and C (0.516) against v2's cross-window rho of 0.165. The models order
candidate actions far better than they localise the apple. The cohort is 18 ranked groups / 68
pooled points from 20 val resets at h = 16 while the planner's horizon is 8, so this is necessary,
not sufficient, evidence -- and G6b alone discriminates nothing (it is 0.0 even for arm A).

**Encoder/rollout decomposition (added after the first merge; it changed the reading).** On the
identical 1,462 moving windows at h=8, the readout of a *directly encoded target frame* -- what a
perfect predictor could reach -- is 6.31 / 2.59 / 5.45 / 2.47 cm for A/B/C/D against v2's 3.26 cm,
while the rollout's excess over it is 0.31 / 1.06 / 0.66 / 0.83 cm against v2's 0.39 cm. The
one-camera encoders improved 21-24 % over v2 and the prediction step got two to three times worse:
the error moved out of the encoder and into the rollout. Re-derived here from the checkpoints with
`scripts/decompose_wm_v3_readout.py`; the rollout column reproduces each published G1 exactly.

**Pre-declared next steps, applied as written then corrected on the decomposition's evidence:**
branch 2 (redesign the action conditioning) is triggered for arms A and B, where G7a fails.
Branch 4's *premise* holds for arms C and D (G2a does not improve over v2's 0.835), but its
*inference* -- that the evidence points at the data, remedy an information-ceiling measurement --
is **withdrawn**: no ceiling has been demonstrated, since modest architectural changes improved
the encoder. Branch 2 is the primary next line, on one camera, preregistered anew. The
single-frame ceiling measurement is demoted to a cheap side-check rather than dropped, because
arm D's 2.47 cm encoded-target error alone still exceeds the 1.5 cm gate -- a perfect predictor
would fail G1 too. The protocol's control-formulation clause is **triggered** (G2a >= 0.8 on all
four arms) and sequenced behind the action-conditioning work; newly pre-declared: if that does not
move G2a below 0.8, behaviour cloning with the world model as a critic becomes the primary line
and CEM over this cost is abandoned.

An earlier version of the results document endorsed branch 4's inference. That was wrong, and
independent verification caught it, not the author.

## Acceptance Criteria
The title's "and pass preregistered offline gates" was an outcome, not a deliverable: no arm
passed. The criteria below are the deliverables; the gate outcome is recorded honestly above.

- [x] Preregistration committed before the frozen runs: `docs/experiments/apple_world_model_v3.md`
      and `benchmarks/manifests/apple-world-model-v3.json` (arms, budgets, seed, selection rule,
      gates, pre-declared readings and next steps, disclosed pilots).
- [x] Two-camera support is shared backend-agnostic code; the backend swap stays one config line;
      planners/evaluators touch latents only through declared readouts; test split never decoded
      (`test_episodes_decoded: 0` in all four run reports and all four gate reports).
- [x] The proprioception std-floor amplification is fixed with a test; `load_split` has a test.
- [x] Fresh pre-run review; blockers fixed; smoke on a tiny subset from the committed revision.
- [x] The frozen arms ran once each from clean committed revision `e2f8227` (`--require-clean`,
      `dirty: false`) and were evaluated against the frozen gates. No threshold was changed:
      `git diff e2f8227..HEAD` over the protocol and all five configs is empty, and the manifest's
      diff touches only the placeholder keys `results`, `overall`, `verification` and `status`
      (`frozen`, `arms`, `baselines`, `gate_definitions` byte-identical).
- [x] Fresh post-run verification; results doc reporting every gate honestly, failures kept; ruff,
      `ruff format --check`, 701 pytest passing, `mc validate`; PR
      https://github.com/RaaSaaR-org/open-embodied-jepa/pull/27 with green CI (all three checks
      pass) and an independent fresh-context review, which raised three blockers -- a verification
      command quoted as returning an empty diff when it does not, two wrong validation-curve
      numbers, and these acceptance boxes -- all fixed and re-checked.
- [x] PR #27 merged as `bce146e` (squash, green CI on all three checks, independent reviewer
      APPROVE reported before the merge), and this task closed by the follow-up PR that also
      records the encoder/rollout decomposition.

## Scope and resources
MPS (M5 Pro, 48 GB). Total training budget about 6 hours across arms, sequential, sharing the
machine with a CPU-bound TASK-051. Checkpoints and reports under the main checkout's ignored
`checkpoints/task052-wm-v3/` and `outputs/task052-wm-v3/`.

## Notes
- Branch `feat/task-052-world-model-v3` from main `34ebce9` (TASK-050 closure).
- Runs: revision `e2f8227`, Python source SHA-256 `476a527e...`, seed 0, MPS, 15,000 steps each,
  sequential A -> B -> C -> D. Elapsed 5,405 / 3,513 / 5,363 / 7,381 s against a frozen 10,800 s
  per-arm cap (6.02 h total, predicted 5.88 h). Selected steps 10,000 / 15,000 / 10,000 / 13,000.
- Artifacts (git-ignored, main checkout): `checkpoints/task052-wm-v3/*.{pt,latest.pt,run.json,
  metrics.jsonl}`, `outputs/task052-wm-v3/*-val-gates.json`, `outputs/task052-runner.log`.
- Results: `docs/experiments/apple_world_model_v3_results.md`; manifest results, overall and
  verification sections in `benchmarks/manifests/apple-world-model-v3.json`.
- Post-run verification re-derived every gate from the raw JSON and corrected three framing errors
  before the write-up: the patch size is 14 (not 16); arm D is onboard-only and extends arm B, so
  the resolution contrast is B<->D (not A<->D); and arm C removes the motion weighting together
  with the auxiliary readouts.
%% mc-links: [[TASK-050]] %%
