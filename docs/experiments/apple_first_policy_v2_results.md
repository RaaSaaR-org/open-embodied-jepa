# Apple→Plate first learned policy v2: results (TASK-071)

**Outcome: M1-PASS.** On the non-gating development cohort D2 (16 resets, 52000–52015) of
`apple-to-plate-v2`, every behaviour-cloning/DAgger policy P-k reached at least one counted
success. The carried arm is **P-3, with 16/16**. The abandonment clause does not fire.

**Read this first.**
- **This is an existence result on the development cohort only** (v1 ruling R3, carried). It is
  the first time a learned (rung L1) policy in this project has picked the apple, carried it and
  left it at rest on the plate. It is a **"learned policy with a DINOv2 encoder"**, not LeWM
  driving the robot (R4).
- **The official learned Apple→Plate count on the frozen benchmark is still 0.** The frozen MVP
  benchmark (0/150 per model, v1 task) is unchanged, cohort C (45300–45339) is untouched, and M2,
  the gated claim test, has not been run. It needs a separate authorization.
- **Two findings limit what the development result says about vision:**
  - **R-3 = P-3 = 16/16.** The same pipeline on the seed-0 *random-init* DINOv2 tokens did
    exactly as well. On D2, encoder pretraining shows no measurable contribution.
  - **The oracle arms gain nothing over the readout.** P-3's head fed the *true* positions
    (D-oracle-perc) and the scripted expert fed the readout's estimates (A4-look) are also 16/16.
  - Only the no-image control **C-3 = 3/16** is clearly below, so the per-reset estimates matter;
    how good they have to be is not resolved by this cohort.
- One run, one seed, 16 resets; P-3 was selected on D2 (a declared selection).

Protocol: [`apple_first_policy_v2.md`](apple_first_policy_v2.md) (PR #96, merged as `9e23ced`),
with owner rulings T71-R1 and T71-R2 recorded verbatim in its §3. Results manifest:
`benchmarks/manifests/apple-first-policy-v2-results.json`. The frozen protocol is not amended by
this document.

---

## 1. Provenance

| item | value |
|---|---|
| revision | `9e23cedb1fc8a5012cd1fdf4661aa4537b9189fa` (main after #96), clean detached checkout; `revision_at_end` the same; `tracked_tree_dirty: false` |
| command | `PYTHONPATH=src caffeinate -i uv run --no-sync python scripts/run_first_policy_v2.py run --output outputs/task071-first-policy-v2/run-1 --checkpoints checkpoints/task071-first-policy-v2/run-1 --corpus data/apple-look-v2/run-1`, from the task agent's worktree at that revision (the main checkout's `.venv`; `PYTHONPATH=src` because that venv's editable install points at the main checkout's `src`) |
| machine | the Mac: macOS 26.5.1 arm64, Python 3.12.13, NumPy 2.5.3, torch 2.14.0, MuJoCo 3.13.0; training on **MPS**; features on CPU; 8 simulation workers; kept awake with `caffeinate -i` |
| pre-run GO | a fresh pre-run reviewer, in its own temporary worktree at `9e23ced`, reported **`PRE-RUN: GO`** (§6). Its render check at 8 workers was **IDENTICAL** (32/32 seeds on ≥ 2 workers; negative control raised G-frame), report `outputs/task071-scratch/prerun-render-1/report.json`, sha256 `f87bda94…04ba4` |
| started / ended | 2026-09-28T08:35:44Z to 10:19:19Z (UTC, machine clock): **6 215 s** of the 36 000 s cap |
| report | `outputs/task071-first-policy-v2/run-1/report.json`, sha256 `77aae2077f5e1070f9cdc63b9f228f32515713e869b999b5303a5134c101d90b` |
| log, checkpoints, corpus | `outputs/task071-first-policy-v2-run-1.log`; `checkpoints/task071-first-policy-v2/run-1/` (12 checkpoints); `data/apple-look-v2/run-1/` (all git-ignored) |
| guards | preflight passed: G-frozen, G-hash (30 pins), clean tree, G-seeds, G-device (MPS), G-weights (digests `3a697b87…af27` pretrained, `3d305f9c…c9db` floor), G-scene. G-look held on every attempt; G-privileged held on every learned attempt; the end-of-run G-hash held on all 30 pins; `non_finite_fields` empty; **`test_split_decoded: false`** (190 train + val episodes decoded) |
| seeds simulated | corpus 51000–51199, perception 51200–51583, C0 51584–51615, DAgger 51616–51999, D2 52000–52015. Cohort C was not simulated. |
| void | none; one run, no repeat |

## 2. Stages before the closed loop

**The corpus `apple-look-v2`** (e9 under v2, collected inside the run; privileged scripted
data, **not** a learned result). Manifest sha256 `18d30b1a1e17…3a1b`, 200 roots split 170 / 20 /
10 by reset.

| noise level | roots | counted success | latched v1 scorer | ended complete |
|---|---|---|---|---|
| 0 | 50 | 50 | 50 | 50 |
| 1 | 50 | 34 | 45 | 47 |
| 2 | 50 | 28 | 38 | 45 |
| 3 | 50 | 17 | 26 | 47 |
| all | 200 | **129** | 159 | 189 (11 guard refusals) |

Every at-rest root was a counted success. The train split has 112 / 170 counted successes; they
form B-replay's library.

**Readouts.** Fit on 426 post-look frames (170 corpus train roots + 256 perception-train
resets). Both chose the linear family, with λ_rel 0.001 (pretrained, P) and 0.01 (random-init
floor, R).

**C0: e9's tolerance, scripted and privileged (calibration only; successes of 32).**

| condition | counted success (= at rest here) | latched |
|---|---|---|
| reference | 32 | 32 |
| apple 0.5 / 0.8 / 1.0 / 1.2 cm | 32 / 32 / 31 / 31 | 32 / 32 / 31 / 31 |
| plate 1.0 / 1.5 / 2.0 / 2.5 cm | 29 / 30 / 23 / 21 | 32 / 32 / 31 / 31 |

The rule gave apple bars 0.75 cm (median) and 1.2 cm (p90), and plate bars 1.5 / 1.5 cm: v1's
rule allowed a plate p90 of 1.5 cm (2.0 cm fails at 23/32), and the TASK-070 ceiling is also
1.5 cm.

**S0-P passed** on the 128 held-out perception resets (pretrained readout):

| | median | p90 | bars (median / p90) |
|---|---|---|---|
| apple error | 0.27 cm | 0.54 cm | 0.75 / 1.2 cm |
| plate error | 0.07 cm | 0.13 cm | 1.5 / 1.5 cm |

The A4-look threshold T was **8** of 16.

**BC-0.** 116 449 training rows from the 170 train roots and 13 479 val rows from the 20 val
roots. The mask dropped 1 936 train rows (apple dropped) and 405 (apple drift above 1 cm before
`close`), and 7 val rows (apple dropped). **S0-D1 passed:** live and offline inputs identical, commands within 5.4e-7 on 8
seeds, frames identical.

**Trainings** (MPS, about 70 s each). Every training selected an eligible checkpoint; val MSE
0.0016–0.0040.

**DAgger rollouts** (each policy in command on the iteration's 128 fresh resets; counted
successes of 128; at-rest attempts not counted: 0 everywhere except 1 for C in iteration 3):

| iteration | P | R | C |
|---|---|---|---|
| 1 (P-0, R-0, C-0 in command) | 23 | 22 | 0 |
| 2 (P-1, R-1, C-1) | 91 | 96 | 1 |
| 3 (P-2, R-2, C-2) | 124 | 124 | 33 |

These rollouts are training data, not evaluation.

## 3. M1 on the development cohort D2 (16 resets, each arm once)

A **counted success** is `apple_at_rest_v0` after the 60-step settle **and** the latched
`grasp` and `place` stages reached during the attempt's commands (T71-R1, T71-R2). 95 %
intervals are Wilson intervals for k of 16.

| arm | rung | counted success | 95 % CI | at rest, not counted | latched grasp | latched v1 success | terminations | final distance, cm, all 16 attempts (q10 / q50 / q90) |
|---|---|---|---|---|---|---|---|---|
| P-0 | L1 | **4/16** | 0.10–0.49 | 0 | 16 | 8 | step limit 16 | 2.65 / 46.5 / 164.5 |
| P-1 | L1 | **9/16** | 0.33–0.77 | 0 | 15 | 14 | step limit 15, guard 1 | 2.98 / 3.63 / 4.41 |
| P-2 | L1 | **15/16** | 0.72–0.99 | 0 | 16 | 15 | step limit 16 | 1.86 / 3.03 / 3.49 |
| **P-3 (carried)** | L1 | **16/16** | 0.81–1.00 | 0 | 16 | 16 | step limit 16 | 2.22 / 2.89 / 3.22 |
| C-3 (no-image control) | L1 | 3/16 | 0.07–0.43 | 0 | 5 | 5 | guard 11, step limit 5 | 3.69 / 17.2 / 24.2 |
| R-3 (random-init floor) | L1 | 16/16 | 0.81–1.00 | 0 | 16 | 16 | step limit 16 | 2.10 / 3.01 / 3.46 |
| A4-look (readout → e9) | L3, not learned | 16/16 | 0.81–1.00 | 0 | 16 | 16 | policy complete 16 | 3.11 / 3.39 / 3.58 |
| D-oracle-perc (P-3 fed true xy) | L4 | 16/16 | 0.81–1.00 | 0 | 16 | 16 | step limit 16 | 2.43 / 2.80 / 3.22 |
| B-oracle (e9 from truth) | L4 | 16/16 | 0.81–1.00 | 0 | 16 | 16 | policy complete 16 | 3.27 / 3.47 / 3.88 |
| B-replay (nearest corpus root) | L4 | 9/16 | 0.33–0.77 | 0 | 11 | 11 | policy complete 12, guard 4 | 2.45 / 3.61 / 20.8 |
| B-hold | L4 | 0/16 | 0.00–0.19 | 0 | 0 | 0 | step limit 16 | 15.1 / 18.3 / 21.2 |
| B-random | L4 | 0/16 | 0.00–0.19 | 0 | 0 | 0 | step limit 16 | 15.1 / 18.3 / 21.2 |

- **The harness is valid:** B-oracle 16/16 ≥ 14, and B-hold and B-random 0 grasps.
- **Final distances** are the latched scorer's apple–plate distance at each attempt's last
  step, over all 16 attempts, including those that ended on a guard refusal.
- **No at-rest attempt went uncounted** in any arm: every at-rest attempt also latched grasp and
  place before the settle. Every learned counted success latched grasp at steps 264–286 and
  place at steps 606–677, before the settle (which starts after the policy's 740 commands;
  e9's own clock ends at 725).
- **`decide_m1` gave M1-PASS**, carried arm P-3 (the most D2 successes; ties go to the later k).
  F was not trained or run (it runs only on M1-MOTOR).
- **Paired, on the same 16 resets** (descriptive only; M2's gates are for cohort C):
  - P-3 vs C-3: 13 resets only P-3 succeeds, 0 only C-3 (exact two-sided McNemar p = 2.4e-4).
  - P-3 vs R-3: 0 and 0; identical success sets.
  - P-3 vs B-replay: 7 and 0 (p = 0.016).
- P-0's median final distance of 46.5 cm (q90 164.5 cm) means that, before DAgger, many of its
  attempts moved the apple far off the plate after the grasp.

## 4. Reading (interpretation, not measurement)

1. **M1-PASS is an existence result on the development cohort.** A learned policy (L1: every
   command after the look and before the task's settle from a trained network, zero privileged
   reads) left the apple at rest on the plate after a carried placement on 16 of 16 D2 resets,
   and each DAgger iteration improved on the last (4 → 9 → 15 → 16). This is the project's first
   learned Apple→Plate success of any kind. It is on `apple-to-plate-v2` (apple condim 6), on a
   non-gating cohort, with the carried arm selected on that cohort. **The official learned count
   on the frozen benchmark is still 0**, and nothing here is evidence about the v1 task.
2. **Encoder pretraining shows no measurable contribution on D2.** R-3, whose readout reads
   random-init DINOv2 tokens, succeeded on exactly the same 16 resets as P-3. This matches
   TASK-063/064's caveat that random-init token floors already meet the readability bars. If the
   same holds on cohort C, M2's G3 fails with its declared reading: "a learned visuomotor policy
   works; encoder pretraining contributes nothing measurable".
3. **The oracle arms gain nothing over the readout.** Feeding P-3's head the true positions
   (D-oracle-perc) and feeding the readout's estimates to the scripted expert (A4-look) both give
   16/16, the ceiling. The readout's held-out errors (apple median 0.27 cm, plate median 0.07 cm)
   are far inside e9's measured tolerance, so perception is not the binding constraint on this
   cohort. The ceiling also means D2 cannot rank these arms above P-3.
4. **The per-reset estimates do matter.** The no-image control C-3, with the estimates fixed to
   the train mean, reached 3/16, ended 11 of 16 attempts on a guard refusal and left the apple a median 17.2 cm from the plate centre. The resets vary
   by up to ±3 cm (apple) and ±2 cm (plate), more than e9's plate tolerance, so a blind policy
   mostly fails. What D2 does not show is whether pretrained features are needed to read the
   positions: random-init features were enough.
5. **What would make a claim.** "A learned policy works" needs M2 on cohort C (40 resets): P-3
   ≥ 17/40 and strictly more than B-replay (G1), P − C-3 ≥ +8 (G2), P − R-3 ≥ +8 (G3), and the
   other gates. On D2 the P-3 − R-3 difference is 0, so G3 is the gate this run gives reason to
   expect to fail. M2 needs a separate authorization.

## 5. Caveats kept in the record

- One run, one training seed per arm, one encoder, one floor seed, one camera (112 px onboard),
  16 development resets.
- P-3 was carried by a selection on D2; its 16/16 is a selected maximum, not an unbiased estimate.
- The corpus, C0, A4-look, B-oracle, B-replay and D-oracle-perc are privileged, scripted, replayed
  or substituted (rungs L3–L4). None of them is a learned result.
- v2's friction values are the scene's declared ones, not measured apple data (TASK-070).
- The DAgger labeller is privileged at training time only (R2 (e)).

## 6. Process note: the pre-run GO

The pre-run reviewer's `PRE-RUN: GO` reached the task agent as that reviewer's final hand-back
message. The task agent started run-1 on it, and the run began at 08:35:44Z. **The GO was not
relayed to the coordinator before the run started.** The coordinator learned of it only from the
task agent's report after the run. The rule that a gated run starts only on a reported GO was
met; the relay was not. **From now on the task agent tells the coordinator before any gated run
starts.**

## 7. Platform note (after this run; dev-only evidence)

After run-1 the project moved to a Linux PC (RTX 5080). There the frozen v2 runner refuses to
start, as designed:
- **G-weights:** the random-init floor encoder's weight digest differs on x86 (`546b9011…` on
  Linux, against the pinned `3d305f9c…`); the seed-0 initialisation differs in low-order float
  bits there.
- **G-device:** non-smoke runs require MPS.
- **G-frame:** post-look frame hashes differ between Apple GL and NVIDIA EGL.

Physics outcomes match within 2 µm. A Linux replication therefore needs a new, separately
reviewed protocol version, which is a separate task. The evidence is in `outputs/linux-bringup-1/`
on the Linux PC, as the coordinator reported it. It is **development-only evidence**, is not part
of this run, and was not examined from the Mac. The frozen protocol is not amended here.

**Learned Apple→Plate on the frozen benchmark is still 0 successes.** On the non-gating
development cohort of `apple-to-plate-v2`, the first learned successes exist (P-3 16/16, an
existence result).
