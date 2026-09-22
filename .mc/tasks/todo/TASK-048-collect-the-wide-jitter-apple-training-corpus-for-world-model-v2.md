---
id: TASK-048
aliases:
- TASK-048
title: Collect the wide-jitter Apple training corpus for world model v2
slug: collect-the-wide-jitter-apple-training-corpus-for-world-model-v2
status: review
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- data
sprint: ''
depends_on:
- "[[TASK-047]]"
due_date: ''
created: 2026-09-22
updated: 2026-09-22
---




# Collect the wide-jitter Apple training corpus for world model v2

## Description
Roadmap T2. The prior learned model (24x24 RGB MLP, 32 narrow-jitter episodes) cannot see the grasp. TASK-047 showed the wide distribution (apple +-3 cm, plate +-2 cm) is feasible for the scripted collector (8/8) and separates replay (2/8). Build the model-v2 TRAIN corpus on it: 200 new resets (48000-48199) with the PRIVILEGED scripted collector plus seeded perturbations, 3 grasp-phase branches per root from the pre-grasp state (incl. failed grasps), 112 px onboard RGB + 112 px robot-kinematics hand crop + proprio, outcome labels, simulator truth only in gated label sidecars, whole-reset train/val/test split. Not a learned result.

## Acceptance Criteria
- [x] Protocol `docs/experiments/apple_wide_collection_v1.md` and pre-run manifest `benchmarks/manifests/apple-wide-collection-v1.json` with acceptance checks A1-A13 frozen before any 48000-range seed is simulated.
- [x] Collector `scripts/collect_apple_wide.py`, robot-only `hand_crop.py`, gated `training_labels.py`, tests (reset-rule parity, seed-range disjointness, split, perturbation, branch modifications, label gating, crop geometry, import isolation).
- [x] Fresh pre-run review; blockers fixed; smoke on pilot seeds only.
- [x] Frozen run executed once from a clean checkout into new `data/apple-wide-v1` (+ `-work`); every outcome recorded.
- [x] Fresh post-run verification of numbers/hashes; results doc + manifest; ruff, pytest, `mc validate`; PR merged.

## Scope and resources
CPU MuJoCo, 12 worker processes, ~30-40 min, ~7 GB under the main checkout's ignored `data/`.

## Notes
- Branch `feat/task-048-wide-train-data` from main `61f8ec1`.
- Pilot design probes (seeds 48900-48931, scratch `outputs/task048-scratch/{smoke-a,pilot-a,pilot-b,pilot-c}`), disclosed in the protocol: pilot-a -> per-branch noise streams (identical sibling prefixes); pilot-b -> lower noise, wider weak_close/early_lift ranges (failures too one-sided); pilot-c final design: roots 20/32 success, grasp-phase failure fraction 0.52, 78/78 sibling pairs distinct, ~95 s/root with 12 workers.
- Frozen plan SHA-256 `15ed1a99e45114a5cec6013d345804ec561fad859dc3f0dd89dd93ec1e33062c`.
- Pre-run review (fresh subagent): blockers B1 (final run path never executed -> `finalize` subcommand, always-written report, smoke), B2 (branch kinds confounded with aim-offset roots -> decoupled slot, plan re-frozen `15ed1a99...062c`), B3 (pilot thresholds undisclosed) fixed in `d023a6b`; post-R1 smoke on pilot seeds 48900-48903 from clean `d023a6b` (integrity ok; finalize reproduced identical manifest). Re-review CLEARED (`32865f5`, docs-only).
- Frozen run executed once from clean `32865f5` into `data/apple-wide-v1` (+ `data/apple-wide-v1-work`): exit 0, 3345 s, 12 workers. **All acceptance checks A1-A13 passed.** Dataset manifest SHA-256 `028e130576c052437f7753d74dd64d80dabc1edb8085247e7f56bc71a0412184`, 3.40 GB, 797 episodes (200 root + 597 branch; 3 branches guard-stopped <8 commands), 205,519 transitions, 112 px onboard + hand crop. Root success 115/200, grasp-phase 329 successes / 468 failures (0.587). Splits 170/20/10 sessions (677/80/40 episodes), normalization train-only.
- Post-run verification (fresh subagent) recomputed all numbers: VERIFIED, two cosmetic notes applied.
- Evidence: `docs/experiments/apple_wide_collection_results_v1.md`, `benchmarks/manifests/apple-wide-collection-v1.json`. PRIVILEGED scripted-collector corpus; not a learned result; learned Apple->Plate stays 0.
%% mc-links: [[TASK-047]] %%
