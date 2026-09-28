# Apple→Plate first learned policy v2, Linux replication: results (TASK-072)

**Replication row: REPLICATED. M1 row: M1-PASS**, with P-3 as the carried arm and **P-3 at
16/16** on the non-gating development cohort D2 (16 resets, 52000–52015) of `apple-to-plate-v2`.
The abandonment clause does not fire. The run was TASK-071's pipeline, re-run end to end on the
Linux PC (RTX 5080, strict-deterministic CUDA training, NVIDIA EGL rendering, 16 simulation
workers).

**Read this first.**
- **This replicates an existence result on the development cohort, and only that.** The
  protocol's declared reading (§5 of [`apple_first_policy_v2_linux.md`](apple_first_policy_v2_linux.md),
  fixed before the run) was REPLICATED iff M1-PASS and P-3 ≥ 14/16. Both hold. P-3 succeeded on
  the **same 16 resets** as the Mac's run-1. D2 was already spent by run-1, so this is not a fresh
  cohort.
- **The official learned Apple→Plate count on the frozen benchmark is still 0.** The frozen MVP
  benchmark (0/150 per model, v1 task) is unchanged. Cohort C (45300–45339) was not simulated.
  **M2, the gated claim test on cohort C, has not been run**; it needs a separate owner go-ahead.
- **The policy is a behaviour-cloning/DAgger MLP on a frozen DINOv2 readout, not LeWM.** No
  world model is in the loop.
- **There is no evidence that pretrained vision helps.** The random-init floor R-3 again tied
  P-3 at 16/16, on the same success set. The oracle arms (D-oracle-perc, A4-look, B-oracle) are
  again 16/16, the ceiling. Only the no-image control C-3 is below, at **7/16**; run-1 had 3/16.
- The early DAgger arms moved between platforms: P-0 went from 4 to 2 and P-1 from 9 to 11. P-2
  stayed at 15/16, but on a different set of resets. See §4.
- This is one run with one training seed per arm, on 16 resets. P-3 was carried by a selection on
  D2.

Protocol: [`apple_first_policy_v2_linux.md`](apple_first_policy_v2_linux.md), merged in PR #99 as
`db65816`. It replicates TASK-071 ([`apple_first_policy_v2.md`](apple_first_policy_v2.md);
results in [`apple_first_policy_v2_results.md`](apple_first_policy_v2_results.md)). The results
manifest is `benchmarks/manifests/apple-first-policy-v2-linux-results.json`; it was generated from
`report.json` and checked against it. This document does not amend the frozen protocol.

---

## 1. Provenance

| item | value |
|---|---|
| revision | `db65816eb56dadb73664f08978cadeb6e881b35f` (main after #99), clean checkout in the worktree `worktrees/task072-run` with its own `.venv`; `revision_at_end` the same; `tracked_tree_dirty: false` |
| command | `uv run --no-sync python scripts/run_first_policy_v2_linux.py run --output outputs/task072-first-policy-v2-linux/run-1 --checkpoints checkpoints/task072-first-policy-v2-linux/run-1 --corpus data/apple-look-v2-linux/run-1` |
| machine | the Linux PC: `Linux-7.0.0-34-generic-x86_64-with-glibc2.39`, Python 3.12.3, NumPy 2.5.3, torch 2.14.0+cu130, MuJoCo 3.13.0, `MUJOCO_GL=egl`. Training on **CUDA** (NVIDIA GeForce RTX 5080, capability 12.0, CUDA runtime 13.0, cuDNN 92400). Features and readouts on the CPU. **16** simulation workers |
| determinism | strict, at the preflight and at the end (`determinism_at_end`): deterministic algorithms on with `warn_only` false, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuDNN deterministic, benchmarking off, TF32 off, float32 matmul precision `highest`. No kernel raised |
| pre-run GO | a fresh pre-run reviewer, in its own temporary worktree at `db65816` (since removed), reported **`PRE-RUN: GO`**. It was posted on PR #99 as comment 5871360006, and the orchestrator was told before the run started. Its render check at 16 workers was **IDENTICAL**: 16 distinct workers, 32/32 seeds rendered on ≥ 2 workers, and the negative control raised G-frame at seed 52132. Render report sha256 `52bf2188a7e5b81e8d647e456c2470a072c144f647d13b1d00a8248c546e4aa8`. Two non-blocking items are recorded in §7 |
| started / ended | 2026-09-28T13:56:57Z to 14:17:31Z (UTC): **1 235 s** of the 36 000 s cap. Run-1 on the Mac took 6 215 s |
| report | `outputs/task072-first-policy-v2-linux/run-1/report.json`, sha256 `877452840d0d03520ec8644321ef32f6b19877caa679e77d173e3d2509edec78` |
| log, checkpoints, corpus | log `worktrees/task072-run-1.log` (one line: report written, outcome M1-PASS); `checkpoints/task072-first-policy-v2-linux/run-1/` (12 checkpoints; each sha256 is in the manifest and matches the report); `data/apple-look-v2-linux/run-1/` (manifest sha256 `67c342f6d37f…4f54`). All are git-ignored, on the Linux PC |
| guards | the preflight passed: G-platform (Linux, x86_64, egl), G-frozen, G-hash with 37 pins (TASK-071's 30 plus 7), clean tree, G-seeds, G-device (strict CUDA), G-weights (pretrained `3a697b87…af27`, Linux floor `546b9011…29d2`) and G-scene. At the end, all 37 pins were unchanged and the process was still strict. `non_finite_fields` was empty, and **`test_split_decoded` was false** (190 train + val episodes decoded) |
| seeds simulated | corpus 51000–51199, perception 51200–51583, C0 51584–51615, DAgger 51616–51999, D2 52000–52015. These are TASK-071's seeds, reused on purpose (protocol §2). Cohort C was not simulated. RNG seeds 7100–7106, unchanged |
| void | none; one run, no repeat |
| comparison target | TASK-071 run-1, `outputs/task071-first-policy-v2/run-1/report.json` (copied to the Linux PC), sha256 `77aae207…d90b`, which matches its results manifest |

## 2. Stages before the closed loop

**The corpus, `apple-look-v2-linux`.** It was collected inside the run by e9 under v2. It is
privileged scripted data, **not** a learned result. It has 200 roots, split 170 / 20 / 10. Its
manifest sha256 differs from run-1's, as it must: the frames are rendered by NVIDIA EGL.

| noise level | roots | counted success | latched v1 scorer | ended complete |
|---|---|---|---|---|
| 0 | 50 | 50 | 50 | 50 |
| 1 | 50 | 34 | 45 | 47 |
| 2 | 50 | 28 | 38 | 45 |
| 3 | 50 | 17 | 26 | 47 |
| all | 200 | **129** | 159 | 189 (11 guard refusals) |

**All 200 roots match run-1 root by root**: outcome, at rest, latched, split, termination and
step count. The scripted expert reads simulator truth and not the frames, so this is the same
physics giving the same outcomes. It is not an independent check of the learned stages. The train
split has 112 of 170 counted successes, as in run-1, and they form B-replay's library of 112
roots.

**Readouts.** They were fit on 426 post-look frames, and both chose the linear family. The
pretrained readout (P) chose λ_rel 0.001 (inner MSE 1.60e-5; run-1 1.97e-5). The random-init
floor (R) chose λ_rel 0.01 (inner MSE 2.38e-5; run-1 2.94e-5).

**C0: e9's tolerance, scripted and privileged (calibration only; successes of 32).** All 288
attempts match run-1 attempt by attempt.

| condition | counted success (= at rest here) | latched |
|---|---|---|
| reference | 32 | 32 |
| apple 0.5 / 0.8 / 1.0 / 1.2 cm | 32 / 32 / 31 / 31 | 32 / 32 / 31 / 31 |
| plate 1.0 / 1.5 / 2.0 / 2.5 cm | 29 / 30 / 23 / 21 | 32 / 32 / 31 / 31 |

The bars are run-1's: apple 0.75 cm (median) and 1.2 cm (p90), plate 1.5 / 1.5 cm.

**S0-P passed** on the 128 held-out perception resets (pretrained readout). The errors are
slightly smaller than run-1's:

| | median | p90 | run-1 median / p90 | bars (median / p90) |
|---|---|---|---|---|
| apple error | 0.25 cm | 0.54 cm | 0.27 / 0.54 cm | 0.75 / 1.2 cm |
| plate error | 0.06 cm | 0.10 cm | 0.07 / 0.13 cm | 1.5 / 1.5 cm |

The A4-look threshold T was **8** of 16, as in run-1.

**BC-0.** It has 116 449 training rows from the 170 train roots and 13 479 val rows from the 20
val roots. The mask dropped 1 936 train rows (apple dropped) and 405 (apple drift above 1 cm
before `close`), and 7 val rows. All of these equal run-1's, because the corpus physics matches.
**S0-D1 passed**: live and offline inputs identical, commands within 2.4e-7 on 8 seeds (run-1
5.4e-7), frames identical.

**Trainings** (CUDA, strict, 24–26 s each; run-1 took about 70 s each on MPS). Every training
selected an eligible checkpoint, with val MSE 0.0016–0.0042. The selected updates differ from
run-1's for 9 of the 12 trainings; C-0, C-1 and R-0 selected the same update. The values are
listed in the manifest.

**DAgger rollouts.** Each policy was in command on the iteration's 128 fresh resets. Counts are
counted successes of 128, with run-1 in brackets. No at-rest attempt went uncounted in any
iteration or arm (run-1 had 1, for C in iteration 3).

| iteration | P | R | C |
|---|---|---|---|
| 1 (P-0, R-0, C-0 in command) | 15 (23) | 15 (22) | 0 (0) |
| 2 (P-1, R-1, C-1) | 107 (91) | 122 (96) | 26 (1) |
| 3 (P-2, R-2, C-2) | 123 (124) | 121 (124) | 34 (33) |

These rollouts are training data, not evaluation. The DAgger schedule's row counts are in the
manifest; P-3 was trained on 392 369 rows (run-1: 391 473).

## 3. M1 on the development cohort D2 (16 resets, each arm once)

A **counted success** is `apple_at_rest_v0` after the 60-step settle **and** the latched `grasp`
and `place` stages, both reached during the attempt's commands (T71-R1, T71-R2). 95 % intervals
are Wilson intervals for k of 16. "Agree" is the number of the 16 resets on which the arm's
success or failure matches run-1's.

| arm | rung | counted success (run-1) | 95 % CI | agree with run-1 | at rest, not counted | latched grasp | latched v1 success | terminations | final distance, cm, all 16 attempts (q10 / q50 / q90) |
|---|---|---|---|---|---|---|---|---|---|
| P-0 | L1 | **2/16** (4) | 0.04–0.36 | 10 | 0 | 16 | 5 | step limit 16 | 3.68 / 98.4 / 159.4 |
| P-1 | L1 | **11/16** (9) | 0.44–0.86 | 8 | 0 | 16 | 15 | step limit 16 | 2.61 / 3.39 / 4.12 |
| P-2 | L1 | **15/16** (15) | 0.72–0.99 | 14 | 0 | 15 | 15 | step limit 15, guard 1 | 2.71 / 3.19 / 3.73 |
| **P-3 (carried)** | L1 | **16/16** (16) | 0.81–1.00 | **16** | 0 | 16 | 16 | step limit 16 | 2.34 / 2.84 / 3.09 |
| C-3 (no-image control) | L1 | 7/16 (3) | 0.23–0.67 | 12 | 0 | 7 | 7 | guard 8, step limit 8 | 3.05 / 18.0 / 24.7 |
| R-3 (random-init floor) | L1 | 16/16 (16) | 0.81–1.00 | 16 | 0 | 16 | 16 | step limit 16 | 2.36 / 2.85 / 3.21 |
| A4-look (readout → e9) | L3, not learned | 16/16 (16) | 0.81–1.00 | 16 | 0 | 16 | 16 | policy complete 16 | 3.16 / 3.39 / 3.89 |
| D-oracle-perc (P-3 fed true xy) | L4 | 16/16 (16) | 0.81–1.00 | 16 | 0 | 16 | 16 | step limit 16 | 2.31 / 2.86 / 3.30 |
| B-oracle (e9 from truth) | L4 | 16/16 (16) | 0.81–1.00 | 16 | 0 | 16 | 16 | policy complete 16 | 3.27 / 3.47 / 3.88 |
| B-replay (nearest corpus root) | L4 | 10/16 (9) | 0.39–0.82 | 13 | 0 | 11 | 11 | policy complete 12, guard 4 | 2.45 / 3.36 / 20.2 |
| B-hold | L4 | 0/16 (0) | 0.00–0.19 | 16 | 0 | 0 | 0 | step limit 16 | 15.1 / 18.3 / 21.2 |
| B-random | L4 | 0/16 (0) | 0.00–0.19 | 16 | 0 | 0 | 0 | step limit 16 | 15.1 / 18.3 / 21.2 |

- **The harness is valid.** B-oracle is 16/16 (≥ 14 required), and B-hold and B-random made 0
  grasps.
- **No at-rest attempt went uncounted** in any arm. Every learned counted success latched grasp at
  steps 263–270 and place at steps 610–666, before the settle. Every learned attempt made zero
  privileged reads.
- **`decide_m1` gave M1-PASS**, with P-3 as the carried arm (the most D2 successes; ties go to the
  later k). F was not trained or run, because it runs only on M1-MOTOR.
- **`read_replication` gave REPLICATED**: outcome M1-PASS, P-3 16 ≥ 14. The report's signed
  differences from run-1 are P-0 −2, P-1 +2, C-3 +4, B-replay +1, and 0 for every other arm.
- **P-3 per reset:** 16 of 16 resets agree with run-1. Both runs succeeded on all 16.
- **The declared descriptive facts** (protocol §5; observed, not gates):
  - R-3 is within 2 of P-3 (the difference is 0).
  - C-3 is at least 8 below P-3 (16 − 7 = 9; run-1 had 13).
- **Paired on the same 16 resets** in this run (descriptive only; M2's gates are for cohort C):
  - P-3 vs C-3: 9 resets where only P-3 succeeds, 0 where only C-3 does (exact two-sided McNemar
    p = 0.0039).
  - P-3 vs R-3: 0 and 0; the success sets are identical.
  - P-3 vs B-replay: 6 and 0 (p = 0.031).
- **B-replay's nearest roots differ from run-1's** on 10 of 16 resets. They are chosen from the
  readout's estimates, which differ between the runs. The library of 112 roots is the same.

## 4. Reading (interpretation, not measurement)

1. **The development result replicates across platforms.** The whole pipeline ran again on a
   different CPU architecture, renderer and training device: corpus, readouts, BC, three DAgger
   iterations and M1. The carried policy again left the apple at rest on the plate, after a
   latched grasp and place, on all 16 D2 resets, the same 16 as on the Mac. So run-1's P-3 16/16
   was not a Mac, Apple GL or MPS artefact. It remains **an existence result on a non-gating
   cohort that both runs have now used**, with P-3 selected on it. It is not evidence that "a
   learned policy works". Only M2 on cohort C can make that claim, and M2 has not been run.
2. **Still no evidence that pretrained vision helps.** R-3, which reads random-init DINOv2
   tokens, again succeeded on exactly P-3's 16 resets. Its DAgger rollouts were comparable to
   P's (15 vs 15, 122 vs 107 and 121 vs 123 of 128). The oracle arms are again at the ceiling, so
   perception is not the binding constraint on D2, and D2 cannot rank these arms above P-3. The
   TASK-071 reading carries over: if cohort C agrees, M2's G3 (P − R-3 ≥ +8) would fail.
3. **The no-image control is less stable than the learned arms.** C-3 went from 3/16 to 7/16. C's
   DAgger iteration-2 rollouts went from 1 to 26 of 128. C-3 is still clearly below P-3 on the same
   resets (9 vs 0), so the per-reset estimates still matter. But the C arm's level moves a lot
   between two runs of the same design. That is a reason to treat any single C-3 number, on D2 or
   on C, as having wide run-to-run spread beyond its Wilson interval.
4. **Early DAgger iterations vary; the final one did not.** P-0 (2 vs 4) and P-1 (11 vs 9) agree
   with run-1 on only 10 and 8 of 16 resets, and P-2 differs on 2 resets. By P-3 both runs are at
   16/16. The corpus, C0 and BC-0 inputs are identical outcome by outcome. What differs is the
   rendered frames, and hence the features and readout estimates, and the training device and
   numerics. One run cannot separate these sources (protocol §1).
5. **What the corpus and C0 agreement means.** The scripted expert's outcomes are identical on
   Linux and the Mac, root by root and attempt by attempt. This is consistent with the bring-up's
   "physics outcomes match within 2 µm" (TASK-071 results §7). It checks the simulator, not the
   learned stages.

## 5. Caveats kept in the record

- **Development cohort only.** M1 on D2 is an existence result. D2 was spent by run-1 and reused
  here by design, and P-3's 16/16 is a selected maximum, not an unbiased estimate.
- **Pretraining is not shown to help.** R-3 ties P-3 (16/16, identical success sets).
- **The frozen benchmark is unchanged.** Learned Apple→Plate on the frozen v1 benchmark is still
  0 successes (0/150 per model). This run is on `apple-to-plate-v2` only.
- **Not LeWM.** The policy is DINOv2 features, then a readout, then a BC/DAgger MLP. No world
  model is in the loop.
- **M2 has not been run.** Cohort C (45300–45339) was not opened. M2 needs a separate owner
  go-ahead.
- **Small scale.** One run, one training seed per arm, one encoder, one floor seed, one camera
  (112 px onboard), 16 resets.
- **Not learned results.** The corpus, C0, A4-look, B-oracle, B-replay and D-oracle-perc are
  privileged, scripted, replayed or substituted (rungs L3–L4). The DAgger labeller is privileged at
  training time only.
- **Platform comparison is on outcomes, not bytes.** The Linux and Mac runs are not bit-identical
  and were not meant to be (protocol §3 rows 4–5). Their features, readouts and checkpoints all
  differ.

## 6. Process

- The pre-run reviewer's **reported** GO, delivered as a message and posted on PR #99, preceded
  the run. The orchestrator was told before the run started. This closes TASK-071's process gap
  (its results §6).
- The run was made from a clean checkout of merged `main` (`db65816`) and ran to completion under
  the frozen caps. It was not void, so no repeat was made.
- Nothing was re-thresholded, retrained or re-selected after the numbers were seen.

## 7. The pre-run reviewer's two non-blocking items (recorded, not amended)

1. **Stale TASK-071 wording in two frozen strings.**
   - The frozen block's `void_rule`, inherited from TASK-071, still says "a second V closes
     **TASK-071** as INCONCLUSIVE".
   - `REPLICATION_RULE["V"]` in `first_policy_v2_linux.py` cites "protocol §14". That is TASK-071's
     section number; this protocol's void rule is §6.

   Neither string changes behaviour. The run was not void, so neither was exercised. Both are
   frozen and pinned, so they are not edited here. A correction, if wanted, goes through the
   protocol's amendment log (§11, still empty) in a separately reviewed change.
2. **Flash and memory-efficient attention backends are enabled but unused.** `determinism`
   reports `sdp_backends` with `flash` and `mem_efficient` true, both at the preflight and at the
   end. They did not matter here: the policy is an MLP, the DINOv2 features run on the CPU, and
   strict mode raises rather than falling back to a nondeterministic kernel. A future protocol
   that runs attention on CUDA should state its backends explicitly.

**Learned Apple→Plate on the frozen benchmark is still 0 successes.** On the non-gating
development cohort of `apple-to-plate-v2`, the TASK-071 existence result (P-3 16/16) now
reproduces on the Linux PC, on the same 16 resets. The random-init floor ties it, and M2 has not
been run.
