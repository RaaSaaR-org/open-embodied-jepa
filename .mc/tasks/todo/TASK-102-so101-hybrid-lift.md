---
id: TASK-102
aliases:
- TASK-102
title: 'SO-101 S5 (second study): hybrid closed loop on so101_cube_lift_v1 in simulation (CEM over TASK-100''s native_jepa model J approaches the pre-grasp image subgoal; a learned local grasp-and-lift policy G_loc grasps and lifts)'
slug: so101-hybrid-lift
status: in-progress
priority: 2
owner: ''
projects: []
customers: []
tags:
- so101
- world-model
- closed-loop
- behaviour-cloning
- preregistration
sprint: ''
depends_on:
- "[[TASK-100]]"
- "[[TASK-101]]"
due_date: ''
created: 2026-10-19
updated: 2026-10-20
---


# SO-101 S5 (second study): hybrid closed loop on so101_cube_lift_v1 in simulation

## Description

Opened by Claude under owner delegation (DECISIONS 2026-10-19 (by), R28.91–R28.97) after TASK-101
closed with no claim (S0-NULL, Δ-MIXED, A2-NOT-BINDING; never frozen). TASK-101's envelope showed
that the approach to the pre-grasp pose is tolerant (the follower lifted 32/32 up to 0.16 rad RMS
from it), while the grasp needs about 0.02 rad when the jaw closes where the arm stands, or
correction while closing. J's cost does not resolve that.

Controller **H**: phase 1 is TASK-101's CEM over J-100101 (settings unchanged; one of TASK-101's
four declared configurations, chosen on development cohort A by hand-overs by tolerance) toward the
pre-grasp image subgoal of the per-reset demonstration. On K′ no H8-mean episode came within
0.12 rad of subgoal 1 (median 0.45, best 0.21 rad), so the approach is the main risk, gated by
S0-APPROACH on hand-overs by tolerance. It hands over at τ_h (at most 0.12 rad, set on
development cohort P by a declared rule) or after a 60-command timeout. Phase 2 is **G_loc**, a
learned local grasp-and-lift policy: behaviour cloning (P-3's recipe) of an MLP on the frozen
DINOv2 g4 latent of both cameras plus proprioception and a clock. It is trained on grasped attempt
windows of `so101-play-v1`'s train split (episode-grouped; val for selection; test never opened).
Arms: H, G_loc-only, N_J+G, S+G (privileged approach), W+F (privileged grasp), F, H-pose+G, rand,
hold, and the replication seeds H-G1 and H-G2. The bar is
min(0.9 × min(r_SG, r_WF), max(0.50, LCB(H(D)))) of 128 fresh resets. Rows: VOID / HY-PASS /
HY-UNSPECIFIC / HY-LOW / HY-FAIL. There is no abandonment clause; a decomposition record comes
before any second variant.

**The preregistration (DRAFT):**
[`docs/experiments/so101_hybrid_lift_v1.md`](../../../docs/experiments/so101_hybrid_lift_v1.md).
Simulation only. Not an Apple→Plate experiment; R7 does not change (R28.9). Not LeWM: a pass is
named "a hybrid with a `native_jepa` world-model approach and a learned grasp policy succeeds"
(R28.96). R28.57's LeWM retrain stays a separate recommended task, not started. The owner can
override the ruling that opened this task (R28.97).

## Acceptance Criteria

- [ ] DRAFT preregistration independently reviewed and merged
- [ ] Stage 0 PR: code in new files only (`so101_hybrid.py`, `so101_grasp_policy.py`,
  `so101_hybrid_runtime.py`, `train_task102.py`, `run_task102.py`, `summarize_task102.py`,
  `run_task102_stage.sh`, manifest `benchmarks/manifests/so101-hybrid-lift-v1.json`). G1, TASK-100
  and TASK-101 pinned files and `config.py` byte-identical; core-import check passes. Tests; G_loc
  trained (3 seeds, primary by lowest val MSE); §7.3's checks in the ledger before counts; P
  (103000–103031) sets τ_h; A (103500–103531) chooses phase 1's configuration; D (103100–103163)
  runs every arm; bar, power and GPU projection; Stage 0
  row and record `docs/experiments/so101_hybrid_lift_v1_stage0.md`; independently reviewed and
  merged (the freeze, on S0-DONE only)
- [ ] Gated stage S on 103200–103327 (reserve 103400–103463) at the frozen revision in a clean run
  tree, every GPU job under `scripts/gpu_run.sh --wait --board`, within the 12 GPU h cap (Stage 0
  ≤ 5); evidence under `~/develop/emai/evidence/task102-*` with `SHA256SUMS`
- [ ] Results document and DECISIONS entry with the row, reviewed and merged; R7 unchanged

## Notes

- 2026-10-19, created with the DRAFT preregistration (branch `docs/so101-hybrid-draft`). Nothing
  trained, run or evaluated.
- Seeds 103000–103999 and salts 10110–10115 checked unused across all remote refs on 2026-10-19
  (`git grep -w` over `docs/`, `src/`, `scripts/`, `tests/`, `benchmarks/`, `configs/` and `.mc/`). The
  only matches are digits of two decimals in an apple results table, which are not seeds.
- Read only: TASK-100's checkpoints (`checkpoints/task100-run/`) and pretrained feature store
  (`outputs/task100-features-pretrained`), and `data/so101-play-v1` (HDD).
- 2026-10-20, Stage 0 (branch `feat/so101-hybrid-stage0`; record
  `docs/experiments/so101_hybrid_lift_v1_stage0.md`; DECISIONS (bz), R28.110–R28.118). G_loc
  trained (3 029 train windows; primary G-0, val MSE 0.0271). Every §7.3 check passed at `c586eb1`
  before any count. P: S+G 30/32 at r = 0, 31, 28, 19 at 0.04/0.08/0.12 rad, so τ_h = 0.08 rad.
  A: a_A = 0/32 in all four configurations (every approach timed out), so the **row is
  S0-APPROACH**. D did not run, nothing is frozen (manifest CANDIDATE), and the study escalates.
  1.246 GPU h. Evidence `~/develop/emai/evidence/task102-stage0/`.
- 2026-10-20, after S0-APPROACH (DECISIONS (ca), R28.119–R28.125; branch
  `docs/task102-approach-diag`): a development decomposition of the approach failure, Δa, is
  declared before any measurement in `docs/experiments/so101_approach_diagnostic_v1.md` (Λ0 A-log
  re-read; Λ1 cost landscape on cohort L 104000–104015; Λ2 J at three budgets and two privileged
  oracle costs in closed loop on L's slots 0–7; Λ3 per camera). Rows Δa-VOID / NOT-REPRODUCED /
  PLANNER-BOUND / GEOMETRY / PLANNER / MODEL / MIXED. Cap 2 GPU h. No controller variant is
  admitted by it.
