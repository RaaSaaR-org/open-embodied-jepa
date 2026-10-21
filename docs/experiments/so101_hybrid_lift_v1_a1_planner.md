# TASK-102 A1: the planner grid on Q, row A1-MODEL-BOUND (record)

**Status: development record of amendment A1's planner grid. Row: A1-MODEL-BOUND (§18.4 row 3).
Escalated with this record. Stage 0′ does not run.** Declaration:
[so101_hybrid_lift_v1.md §18](so101_hybrid_lift_v1.md), committed and pushed (`45aca25`) before any
measurement. Rulings R28.131–R28.140 (declaration) and R28.141–R28.147 (this record) in
[DECISIONS.md](../DECISIONS.md), (cc) and (cd), recorded by Claude under owner delegation.

**Nothing here is a result.** It is not a gate and makes no claim for or against J, LeWM, the
hybrid or the task. Simulation only, `so101_cube_lift_v1`, development cohort Q (105000–105007,
no reserve drawn) and debug seeds 105900–105901 (plumbing and timing only, never read). No TASK-102
protocol seed was drawn. The world model is `native_jepa` J-100101, not LeWM. O-pose reads
simulator state. It is a planner check, not a controller. R7 does not change.

## Summary

- **With the jaw scripted open, the planner reaches the arm pose when the cost is exact.** O-pose
  handed over by tolerance (arm RMS ≤ 0.08 rad for 3 consecutive commands) on **8/8 Q resets in 15
  of the 16 configurations** (exact 95 % interval 0.631–1.000 each) and 7/8 in the 16th
  (`last-H16-k4-w0`, 0.473–0.997). So all 16 configurations were admitted. Median closest arm RMS
  was 0.017–0.064 rad by configuration, median commands to the hand-over 14.5–42.5.
- **The scripted jaw did what it was declared to do.** At every hand-over of every arm (O-pose and
  J), the measured jaw was within 0.15 rad of subgoal 1's jaw. The median jaw error at the hand-over
  was 0.000 rad. Under Δa the jaw had never come within 0.15 rad.
- **J did not.** At 256 × 6, J handed over by tolerance on **at most 1/8** in every configuration
  (1/8 in `last-H8-k1-w1` and `last-H16-k4-w0`, 0/8 in the other 14; 1/8 has an exact interval of
  0.003–0.527). The median closest arm RMS was 0.132–0.534 rad. At the one larger budget §18.3 step
  3 allows (1 024 × 12 on `last-H8-k1-w1`, the first in §18.4's order), J handed over on **3/8**
  (0.085–0.755), median closest 0.182 rad.
- **Row (first match): A1-MODEL-BOUND.** A1-VOID no; A1-PLANNER-STILL-BOUND no (16 admitted);
  **A1-MODEL-BOUND fires**: J < 4/8 in every admitted configuration, the larger budget included.
  Under §18.4 row 3, R28.57's LeWM retrain, or another model change, becomes the recommended next
  step, with its own ruling and preregistration. No Stage 0′ runs. P′, A′ and D′ stay unused.
- **GPU: 0.338 GPU h of lock time** (5 jobs, two of them debug), against the 3.0 GPU h cap. O-pose
  ran on the CPU (414 s wall for 128 episodes, 8 workers).

## 1. What ran (revision `2f8459d`, clean tracked tree, every counted job)

| job | what | lock | outcome |
|---|---|---|---|
| goals Q | follower v1 on 105000–105007 | 7 s | **8/8 kept**, no reserve drawn; goal store `97974987…3d02`; subgoal 1's jaw 1.7453 rad on all 8 (so the scripted gripper command is +1) |
| O-pose grid | 16 configurations × 8, CPU, 8 workers | — (414 s wall) | 128 episodes |
| J grid | 16 configurations × 8, 6 workers | 1 046 s | 128 episodes |
| J larger budget | `last-H8-k1-w1` at 1 024 × 12, 3 workers | 150 s | 8 episodes |
| debug | goals 105900–105901; O-pose and J timing, 8-command timeout, two configurations | 14 s | plumbing only |

**Guards held.** The tracked tree was clean and at `2f8459d` at the start and end of every counted
job. Every pin held at the start and end of every job: TASK-100's frozen-files check, TASK-101's
pinned and composed files, TASK-102's `CANDIDATE` manifest files, Δa's code and documents and
`schema_planning.py`. G-projection held on every executed command (every executed command equalled
its scored projection, k = 4 chunks included). G-jaw held on every plan (every planned gripper
command was the scripted +1). G-rollout held on every O-pose command: after each executed command
the copy's re-execution matched the transport's full qpos and qvel bit for bit (2 797 commands).
No candidate rollout tripped the measured-velocity stop, and no execution was `stopped`. A unit test
holds the arm-channel planner's cold path to the pinned `SchemaCEMPlanner` bit for bit.

## 2. O-pose (§18.3 step 1; privileged; 8 resets each, τ_h = 0.08 rad, T₁ = 60)

| configuration | h (by tolerance) | exact 95 % | closest arm RMS, median (min–max) | commands to hand-over, median | arm RMS at hand-over, median | s / command (median) |
|---|---|---|---|---|---|---|
| last-H8-k1-w0 | 8/8 | 0.631–1.000 | 0.057 (0.046–0.062) | 24.0 | 0.057 | 1.11 |
| last-H8-k1-w1 | 8/8 | 0.631–1.000 | 0.034 (0.014–0.046) | 18.5 | 0.035 | 1.09 |
| last-H8-k4-w0 | 8/8 | 0.631–1.000 | 0.050 (0.034–0.055) | 22.0 | 0.050 | 0.31 |
| last-H8-k4-w1 | 8/8 | 0.631–1.000 | 0.033 (0.019–0.043) | 18.0 | 0.036 | 0.31 |
| last-H16-k1-w0 | 8/8 | 0.631–1.000 | 0.061 (0.050–0.077) | 42.5 | 0.062 | 2.21 |
| last-H16-k1-w1 | 8/8 | 0.631–1.000 | 0.035 (0.006–0.055) | 25.5 | 0.035 | 2.20 |
| last-H16-k4-w0 | **7/8** | 0.473–0.997 | 0.064 (0.049–0.091) | 39.5 | 0.066 | 0.58 |
| last-H16-k4-w1 | 8/8 | 0.631–1.000 | 0.038 (0.023–0.058) | 25.0 | 0.038 | 0.62 |
| mean-H8-k1-w0 | 8/8 | 0.631–1.000 | 0.024 (0.009–0.037) | 15.5 | 0.024 | 1.11 |
| mean-H8-k1-w1 | 8/8 | 0.631–1.000 | 0.016 (0.007–0.022) | 14.5 | 0.017 | 1.11 |
| mean-H8-k4-w0 | 8/8 | 0.631–1.000 | 0.027 (0.016–0.032) | 16.5 | 0.027 | 0.31 |
| mean-H8-k4-w1 | 8/8 | 0.631–1.000 | 0.022 (0.017–0.029) | 15.5 | 0.022 | 0.32 |
| mean-H16-k1-w0 | 8/8 | 0.631–1.000 | 0.041 (0.031–0.050) | 18.0 | 0.041 | 2.21 |
| mean-H16-k1-w1 | 8/8 | 0.631–1.000 | 0.028 (0.012–0.040) | 16.0 | 0.028 | 2.20 |
| mean-H16-k4-w0 | 8/8 | 0.631–1.000 | 0.041 (0.027–0.049) | 19.5 | 0.041 | 0.58 |
| mean-H16-k4-w1 | 8/8 | 0.631–1.000 | 0.034 (0.020–0.048) | 16.5 | 0.034 | 0.60 |

Every O-pose hand-over had the measured jaw within 0.15 rad of subgoal 1's (median error 0.000
rad). Reported only (no rule reads it): the step-mean cost and the warm start each shortened the
approach and brought the arm closer under the exact cost.

## 3. J (§18.3 steps 2–3; 8 resets each, the same seeds and planner seeds)

| configuration | h (by tolerance) | exact 95 % | closest arm RMS, median (min–max) | s / command (median) |
|---|---|---|---|---|
| last-H8-k1-w0 | 0/8 | 0.000–0.369 | 0.275 (0.157–0.494) | 0.25 |
| **last-H8-k1-w1** | **1/8** | 0.003–0.527 | 0.181 (0.062–0.350) | 0.20 |
| last-H8-k4-w0 | 0/8 | 0.000–0.369 | 0.287 (0.101–0.339) | 0.06 |
| last-H8-k4-w1 | 0/8 | 0.000–0.369 | 0.220 (0.124–0.340) | 0.06 |
| last-H16-k1-w0 | 0/8 | 0.000–0.369 | 0.339 (0.151–0.410) | 0.37 |
| last-H16-k1-w1 | 0/8 | 0.000–0.369 | 0.150 (0.105–0.247) | 0.38 |
| last-H16-k4-w0 | 1/8 | 0.003–0.527 | 0.288 (0.073–0.422) | 0.10 |
| last-H16-k4-w1 | 0/8 | 0.000–0.369 | 0.132 (0.067–0.397) | 0.10 |
| mean-H8-k1-w0 | 0/8 | 0.000–0.369 | 0.534 (0.265–0.644) | 0.20 |
| mean-H8-k1-w1 | 0/8 | 0.000–0.369 | 0.406 (0.208–0.529) | 0.20 |
| mean-H8-k4-w0 | 0/8 | 0.000–0.369 | 0.401 (0.182–0.544) | 0.06 |
| mean-H8-k4-w1 | 0/8 | 0.000–0.369 | 0.358 (0.161–0.477) | 0.06 |
| mean-H16-k1-w0 | 0/8 | 0.000–0.369 | 0.444 (0.223–0.616) | 0.37 |
| mean-H16-k1-w1 | 0/8 | 0.000–0.369 | 0.353 (0.109–0.457) | 0.38 |
| mean-H16-k4-w0 | 0/8 | 0.000–0.369 | 0.353 (0.236–0.446) | 0.10 |
| mean-H16-k4-w1 | 0/8 | 0.000–0.369 | 0.319 (0.186–0.399) | 0.09 |
| **last-H8-k1-w1 @ 1 024 × 12** | **3/8** | 0.085–0.755 | 0.182 (0.049–0.341) | 0.67 |

Every J episode that did not hand over timed out at 60 commands. The two 1/8 hand-overs were at
arm RMS 0.067 and 0.073 rad. At the larger budget the three hand-overs (105000, 105003, 105007)
came after 55, 36 and 36 commands, at closest arm RMS 0.061, 0.062 and 0.049 rad. Of the five
timeouts, one came to 0.114 rad and four stayed 0.25–0.34 rad away. §18.4's order put
`last-H8-k1-w1` first: it ties `last-H16-k4-w0` on J's count (1/8) and has more O-pose hand-overs
(8 against 7).

*Reported only:* under the exact cost, the `mean` form was better than `last` in every pairing,
whereas under J it was worse in every pairing (median closest 0.319–0.534 rad against
0.132–0.339 rad). The warm start lowered the closest median in all 8 pairings, under J as under O-pose.

## 4. The row (§18.4, first match)

1. A1-VOID: every guard held, every job completed, 8/8 kept, 0.338 ≤ 3.0 GPU h. No.
2. A1-PLANNER-STILL-BOUND: 16 of 16 configurations admitted (O-pose ≥ 7/8). No.
3. **A1-MODEL-BOUND: the most J hand-overs in any admitted configuration was 1/8 at 256 × 6 and
   3/8 at 1 024 × 12, both below 4/8. Fires.**

**Row: A1-MODEL-BOUND.** Under §18.4 row 3 the study escalates. No Stage 0′ runs, and P′, A′ and D′
stay unused. **R28.57's LeWM retrain, or another model change, becomes the recommended next step**,
with its own ruling and preregistration. With Δa (R28.126–R28.128), the readings are these. With
the jaw scripted, this planner reaches the pose with an exact pose cost in every configuration
tried. With J's predicted image cost it does not, in any of them, even at four times the samples
and twice the iterations. Δa's Λ1 showed J ranking distance correctly along straight lines and
among executed candidates. This record shows that this is not enough to steer CEM to 0.08 rad from
the reset pose in closed loop.

## 5. Deviations (disclosed)

1. **The O-pose `mean`-against-`last` and warm-start readings in §2 and §3** were read after the
   declaration. No rule reads them.
2. **Workers per job were set by memory**: 8 (O-pose, CPU), 6 (J at 256 × 6) and 3 (J at
   1 024 × 12). Each episode runs in a fresh process, so this affects timing only.
3. **The image cost was not measured as an oracle** (Δa's O-img) under A1. §18.3 did not declare
   it. So this record does not separate J's prediction from the image-goal cost's geometry with the
   jaw scripted. Δa's Δa-GEOMETRY reading (O-img 0/8 with the jaw failing) predates the scripted jaw.
   §6 says what this leaves open.

## 6. What this does not show

- A development reading on 8 resets, one run, in simulation. The bars (7/8, 4/8) were declared, not
  calibrated. J's 3/8 at the larger budget has an exact interval of 0.085–0.755, so a true rate at or
  above one half is not excluded.
- **It does not separate J's prediction from the image-goal cost itself.** A perfect model of the
  frames (Δa's O-img) was not re-run with the jaw scripted. A model change is recommended because
  §18.4 says so, but whether the binding component is J's prediction or the image cost's precision
  near 0.08 rad is not measured here. The next study's declaration should include an O-img arm with
  the jaw scripted, so it can tell the two apart.
- It does not test any planner family other than CEM, any budget above 1 024 × 12, or any cost
  other than J's latent error to subgoal 1's latent.
- No test of LeWM. No world-model claim in either direction. R7 does not change.

## 7. Evidence

`~/develop/emai/evidence/task102-a1/` (27 files; `SHA256SUMS` `ab6f9b34…489a`): `summary.json`
(`35a579b6…b89c`), the two partial summaries (after O-pose, after J), every job's `report.json` and
`progress.jsonl`, the goal reports and job records of Q and debug, the stage logs, the GPU ledger
lines and `episode_npz.sha256` (the SHA-256 of every per-episode array file). Outputs: the shared
checkout's `outputs/task102-a1/` (HDD). Code: `so101_hybrid_a1.py`, `so101_hybrid_a1_runtime.py`,
`scripts/run_task102_a1.py`, `scripts/summarize_task102_a1.py`, `scripts/run_task102_a1_stage.sh`,
`tests/test_so101_hybrid_a1.py` (12 tests), all new.
