---
id: TASK-072
aliases:
- TASK-072
title: 'Linux CUDA platform: make the RTX 5080 Linux PC the working platform, and replicate TASK-071 there'
slug: linux-cuda-platform-make-the-rtx-5080-linux-pc-the-working-platform-and-replicate-task-071-there
status: in-progress
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
- [ ] Stage A PR merged by the owner on an independent reviewer's reported APPROVE, with the
      Linux validation evidence in the PR.
- [ ] Stage B protocol and manifest merged by the owner on an independent reviewer's APPROVE.
- [ ] Pre-run reviewer's reported GO, and the owner told, before the gated run.
- [ ] Gated run from a clean checkout of the merged revision on Linux, to completion.
- [ ] Results PR, reviewed, merged by the owner.

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
