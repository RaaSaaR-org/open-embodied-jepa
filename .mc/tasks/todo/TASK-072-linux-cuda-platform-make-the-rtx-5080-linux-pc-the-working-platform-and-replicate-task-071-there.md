---
id: TASK-072
aliases:
- TASK-072
title: 'Linux CUDA platform: make the RTX 5080 Linux PC the working platform, and replicate TASK-071 there'
slug: linux-cuda-platform-make-the-rtx-5080-linux-pc-the-working-platform-and-replicate-task-071-there
status: review
priority: 1
owner: ''
projects: []
customers: []
tags:
- platform
- cuda
- learned-policy
- preregistration
sprint: ''
depends_on:
- "[[TASK-071]]"
due_date: ''
created: 2026-09-28
updated: 2026-09-28
---


# Linux CUDA platform: make the RTX 5080 Linux PC the working platform, and replicate TASK-071 there

## Description

The owner decided on 2026-09-28 to switch directly to the Linux PC (Ubuntu 24.04, RTX 5080
16 GB, sm_120, torch 2.14.0+cu130 from the locked sync): "setup everything on the new pc - then
run and validate". Three stages, each a separate PR merged by the owner:

- **Stage A (software, not an experiment):** `cuda` as a supported device wherever cpu/mps are
  accepted; `auto` prefers cuda, then mps, then cpu; a cuda RNG branch in `rng_scope` and an
  `rng_cuda` checkpoint field; one deterministic CUDA setup in `embodied_jepa.devices`;
  CUDA-only tests that skip cleanly; Linux validation (ruff, full and graphics pytest, CPU
  smoke, CUDA smoke with bit-identical same-seed losses); docs. Frozen protocols keep their
  hardcoded MPS training device.
- **Stage B (preregistration):** a separately versioned protocol replicating TASK-071
  (`apple_first_policy_v2`) on the Linux PC with CUDA training; same seeds, arms, 36,000 s cap,
  corpus plan, readout, DAgger schedule and M1 gate on development cohort D2 (52000–52015);
  every deviation named; pins measured on Linux; v2's pins unchanged; the reading of a match or
  mismatch with TASK-071 run-1 declared before running. Cohort C untouched; M2 is out of scope.
- **Stage C:** pre-run review, owner notified, one gated run from a clean merged checkout on
  Linux (void rule: one from-scratch repeat; a second void is INCONCLUSIVE), reviewed results.

Learned Apple→Plate is still 0 successes on the frozen benchmark. A development-cohort result
from this task would be one run on D2, not LeWM, and not the frozen benchmark.

## Acceptance Criteria
- [x] Stage A PR merged by the owner on an independent reviewer's reported APPROVE, with the
      Linux validation evidence in the PR.
- [x] Stage B protocol and manifest merged by the owner on an independent reviewer's APPROVE.
- [x] Pre-run reviewer's reported GO, and the owner told, before the gated run (GO posted on
      PR #99, comment 5871360006; render check IDENTICAL at 16 workers, report sha256
      `52bf2188…4aa8`; the orchestrator was told before the run started).
- [x] Gated run from a clean checkout of the merged revision on Linux, to completion (run-1 from
      `db65816`, 13:56:57Z–14:17:31Z, 1 235 s, no void).
- [x] Results PR, reviewed, merged by the owner (#100, `3dcc6ba`).
- [ ] M2 (owner go-ahead 2026-09-28): authorization record, runner and tests merged on an
      independent reviewer's reported APPROVE (branch `feat/task-072-m2-cohort-c`).
- [ ] M2: fresh pre-run reviewer's reported GO, orchestrator told, one gated run on cohort C
      from a clean merged checkout, reviewed results PR.

## Notes
- 2026-09-28: card opened. Stage A branch `feat/task-072-cuda-support`.
- 2026-09-28 Stage A findings: (1) the `base.py` edit changes every VisualModel implementation
  hash; E0 (TASK-054/056/057), TASK-065 and TASK-066 checkpoints now load only at 9e23ced.
  `sensor.py` changed too (sensor_wm). Three guard tests said "revert, or retire
  deliberately"; the retirement is recorded in
  `benchmarks/manifests/task072-checkpoint-compatibility.json` and needs the owner's ruling.
  (2) Strict `use_deterministic_algorithms(True)` makes `native_jepa` untrainable on CUDA
  (`adaptive_avg_pool2d_backward_cuda` has no deterministic kernel), so the default is
  `warn_only=True`. LeWM's memory-efficient attention backward has a deterministic variant
  that only strict mode selects; LeWM protocols set strict first, and it is never downgraded.
  Same-seed bit-identity in warn-only mode is checked by `scripts/cuda_smoke.py`.
  (3) `policy.py` (ClonedPolicy) and the scripts pinned by the TASK-059/061 manifests stay
  cpu/mps; `first_policy*`, `latent_dynamics` keep their frozen MPS device.
  (4) Over ssh, rendering needs `MUJOCO_GL=egl`.
- 2026-09-28: Stage A merged as #98 (4f85f24) with owner ruling T72-R1. Stage B branch
  `feat/task-072-linux-replication`: protocol `apple_first_policy_v2_linux`
  (docs/experiments/apple_first_policy_v2_linux.md, manifest
  benchmarks/manifests/apple-first-policy-v2-linux.json); strict CUDA; 16 workers; floor digest
  re-pinned from a Linux measurement (546b9011…); reading REPLICATED iff M1-PASS and P-3 >= 14/16.
- 2026-09-28: Stage B merged as #99 (`db65816`). Pre-run reviewer reported GO (PR #99 comment
  5871360006), with two non-blocking items: stale TASK-071 wording in the frozen `void_rule` and
  `REPLICATION_RULE["V"]` (§14 vs §6), and flash/mem-efficient attention backends enabled but
  unused. Both are recorded in the results doc §7 and are not amended.
- 2026-09-28: run-1 from `db65816` on the Linux PC (strict CUDA, 16 workers), 13:56:57Z–14:17:31Z,
  1 235 s, report sha256 `87745284…ec78`. **Replication row REPLICATED; outcome M1-PASS**, carried
  P-3. Counted successes on D2 (of 16; Mac run-1 in brackets where different): P-0 2 (4), P-1 11 (9),
  P-2 15, **P-3 16**, C-3 7 (3), **R-3 16**, A4-look 16, D-oracle-perc 16, B-oracle 16,
  B-replay 10 (9), B-hold 0, B-random 0. P-3 succeeded on the same 16 resets as run-1. The corpus
  (129/200) and C0 match run-1 outcome by outcome. An existence result on the development cohort
  only. R-3 ties P-3, so there is no evidence that pretrained vision helps. The policy is not
  LeWM. The official learned Apple→Plate count on the frozen benchmark is still 0. Cohort C is
  untouched, and M2 needs a separate owner go-ahead. Results:
  `docs/experiments/apple_first_policy_v2_linux_results.md`,
  `benchmarks/manifests/apple-first-policy-v2-linux-results.json`; results PR branch
  `task072-results`.
- 2026-09-28: results merged as #100 (`3dcc6ba`). The owner gave go-ahead for **M2 on cohort C**
  (TASK-071 §12, unchanged) for TASK-072 run-1's P-3, through the normal pre-run review. Branch
  `feat/task-072-m2-cohort-c`: authorization record `docs/experiments/apple_first_policy_v2_m2.md`,
  manifest `benchmarks/manifests/apple-first-policy-v2-m2.json`, design
  `src/embodied_jepa/first_policy_v2_m2.py`, runner `scripts/run_first_policy_v2_m2.py` (modes
  preflight / smoke / run). The readouts were not stored by run-1, so they are refitted and must
  reproduce run-1's recorded facts exactly (G-repro). The cohort comes from the stored values in
  `apple-policy-v1.json`, never recomputed; this discharges the stored-values debt of
  `task056_handover.md` §7 with a behavioural test. Expected from D2: R-3 ties P-3, so G3 is
  likely to fail and the row would be M2-FAIL. `mc` is not installed on the Linux PC; MC
  validation was not run.
