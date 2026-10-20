# TASK-102 Δa: the approach-failure decomposition, row Δa-PLANNER-BOUND (record)

**Status: development diagnostic record. Row: Δa-PLANNER-BOUND (§6 row 3, first match; row 4,
Δa-GEOMETRY, also fired). Escalated with this record.** Declaration:
[so101_approach_diagnostic_v1.md](so101_approach_diagnostic_v1.md), committed and pushed (`fbab0b0`)
before any measurement. Rulings R28.119–R28.125 (declaration) and R28.126–R28.130 (this record)
in [DECISIONS.md](../DECISIONS.md), (ca) and (cb), recorded by Claude under owner delegation.

**Nothing here is a result.** It is not a gate and makes no claim for or against J, LeWM, the
hybrid or the task. Simulation only, `so101_cube_lift_v1`, development cohort L (104000–104015)
and debug seeds 104900–104901 (plumbing and timing only, never read). No TASK-102 seed was drawn.
R7 does not change. The world model is `native_jepa` J-100101, not LeWM. The oracle arms O-img
and O-pose read simulator state. They are diagnostic instruments, not controllers.

## Summary

- **No arm handed over by tolerance on any of its 8 resets.** That includes both privileged
  oracle costs. h = 0/8 for J-ref, J-big16, J-big32, O-img and O-pose (exact 95 % interval
  0.000–0.369 each). Every episode ran to the 60-command timeout.
- **The jaw is what fails, in every arm.** No episode of any arm brought the jaw within 0.15 rad
  of subgoal 1's jaw at any command (0 of 40 episodes). The median absolute jaw error over the
  episode was 0.90–0.95 rad by arm. Subgoal 1's jaw is open (1.745 rad), and the gripper target
  can move 0.25 rad per command.
- **The arm, by contrast, gets close when the cost is good.** With the exact pose cost (O-pose),
  the closest arm RMS to subgoal 1 was 0.035–0.069 rad on all 8 resets (median 0.060), and the arm
  stayed within τ_h = 0.08 rad for at least 3 consecutive commands on 7/8 (a reading added after
  the declaration; no row reads it). With the true-frame image cost (O-img), the median was
  0.076 rad (2/8 held for 3 commands). For J it was 0.170 rad at TASK-101's budget (J-ref; 0/8),
  0.149 rad at 1 024 × 12 with H = 16 (J-big16; 2/8) and 0.260 rad at H = 32 (J-big32; 0/8).
- **Rows, first match:** Δa-VOID no; Δa-NOT-REPRODUCED no (J-ref 0/8 < 4); **Δa-PLANNER-BOUND
  fires (O-pose 0/8 < 6/8)**; Δa-GEOMETRY also fires (O-img 0/8 < 6/8), but it is not first.
- **Λ1: the image-goal cost is not flat, and J follows it.** Along straight joint-space lines
  from the reset pose (d₀ 0.67–1.16 rad) to subgoal 1, the true-frame cost and J's predicted hold
  cost resolve every band from 0.08 to 0.64 rad (d_flat > 0.64 for both; per-seed Spearman ρ with
  distance 1.000 in median for both). Among 64 executed candidates at far states, J's cost ranks
  the pose-law approach candidate first (median rank 0) and correlates with the true-frame cost at
  ρ_JT = 0.67. Δρ = 0.11 (ρ_true 0.44, ρ_J 0.33 at H16-last), below Δa-MODEL's 0.20.
- **Λ3: the overview half carries the arm pose further out than the wrist half.** On the
  true-frame cost the overview half resolves every band (d_flat > 0.64), and the wrist half
  stops at 0.40 rad.
- **Λ0** (Stage 0's 128 A episodes, re-read): the true-frame cost fell along CEM's own
  trajectories (median within-episode ρ with arm RMS 0.57–0.70). J's planned cost was 0.59–0.72
  of the true-frame cost at the same step, so J's plans were optimistic.
- **GPU: 0.810 GPU h of lock time** (11 jobs, including a probe, debug runs and two out-of-memory
  debug attempts), against the 2.0 GPU h cap. O-pose and Λ0 ran on the CPU.

## 1. What ran (revision `5e5ffdc`, clean tracked tree, every job)

| job | what | lock | outcome |
|---|---|---|---|
| goals L | follower v1 on 104000–104015 | 10 s | **16/16 kept**, no reserve drawn; goal store `4a317e57…` |
| Λ1 landscape | 16 seeds, 4 workers | 190 s | 16 seeds; 9 demonstration states and 12 line states each (0 excluded; largest cube displacement 0.11 mm); 5 candidate states × 64 executed candidates |
| Λ2 J-ref + O-img | slots 0–7, 6 workers | 1 786 s | 16 episodes |
| Λ2 J-big16 | slots 0–7, 2 workers | 290 s | 8 episodes |
| Λ2 J-big32 | slots 0–7, 1 worker (memory) | 544 s | 8 episodes |
| Λ2 O-pose | slots 0–7, CPU, 8 workers | — (137 s wall) | 8 episodes |
| Λ0 | CPU | — | 4 × 32 episodes re-read |
| debug and probe | 104900–104901 goals, landscape and approach timing (3 commands per episode) | 97 s | plumbing only. Two approach attempts ran out of GPU memory with J-big32 alongside other workers, so J-big32 then ran alone |

**Guards held.** G-replay: 16/16 demonstration replays were equal to the goal store. G-rollout:
in Λ1 the approach candidate was executed through the real transport at every candidate state
and matched the copy bit for bit (1 280 commands). In Λ2, after every oracle command the copy's
re-execution matched the transport's full qpos and qvel bit for bit (960 commands), and a unit
test checks the same thing. G-projection held on every executed command. No candidate rollout
exceeded the measured-velocity stop. Pins: TASK-100's frozen-files check, TASK-101's pinned and
composed files, and TASK-102's CANDIDATE manifest files were unchanged at the start and end of
every job (`hr.check_pins`).

## 2. Λ2: the closed loops (§4; 8 resets each, phase 1 alone, τ_h = 0.08 rad, T₁ = 60)

| arm | cost | budget | h (by tolerance) | closest arm RMS, median (min–max) | jaw error at the closest state, median | arm ≤ 0.08 for ≥ 3 commands¹ | jaw ever within 0.15¹ | s / command (median) |
|---|---|---|---|---|---|---|---|---|
| J-ref | J, H16-last | 256 × 6 | **0/8** | 0.170 (0.097–0.717) | −0.82 | 0/8 | 0/8 | 0.43 |
| J-big16 | J, H16-last | 1 024 × 12 | **0/8** | 0.149 (0.050–0.327) | −0.79 | 2/8 | 0/8 | 0.91 |
| J-big32 | J, H32-last | 1 024 × 12 | **0/8** | 0.260 (0.151–0.789) | −0.85 | 0/8 | 0/8 | 0.94 |
| O-img | true-frame image cost (privileged) | 256 × 6 | **0/8** | 0.076 (0.025–0.975) | −1.00 | 2/8 | 0/8 | 18.8 |
| O-pose | true pose cost (privileged) | 256 × 6 | **0/8** | 0.060 (0.035–0.069) | −0.83 | **7/8** | **0/8** | 2.18 |

¹ Added after the declaration, from the same logs. Reported only; no row reads it.

J-ref's closest median (0.170 rad) is lower than Stage 0's 0.354 rad on A, with the same
configuration. These are different seeds, and both are development readings.

**A reading (not a row): the planner leaves the jaw for later.** O-pose's planned cost at its
last commands was about 0.1–0.3. Under its definition that means a planned end state with the
jaw within about 0.07 rad. Yet the measured jaw stayed 0.57–1.26 rad from the goal (reset 104000, every third command). The gripper
moves up to 0.25 rad per command, so a jaw error of about 1 rad can be closed in the last 4 of 16
planned commands. A `last` cost then does not depend on the jaw command that is actually
executed first, and replanning every command postpones the opening indefinitely. The executed
jaw commands look like noise (for example, requested jaw values between −0.66 and +0.73, sampled every sixth command, on reset
104000). The same mechanism is plausible for the arm's last few hundredths of a radian, which
the arm reached but did not always hold. It is consistent with Stage 0's report that the jaw
was 0.64–0.95 rad off at J's closest arm states. It is an inference from the logs, not a
measured intervention.

## 3. Λ1: the landscape (§3; 16 seeds)

**Band ordering on the line paths** (fraction of 16 seeds with the cost higher at the band's
upper edge; resolved at ≥ 0.75):

| cost | 0.08–0.12 | 0.12–0.16 | 0.16–0.24 | 0.24–0.32 | 0.32–0.40 | 0.40–0.48 | 0.48–0.64 | d_flat | ρ(line), median | ρ(demonstration path), median |
|---|---|---|---|---|---|---|---|---|---|---|
| true, full | 1.00 | 0.94 | 0.94 | 1.00 | 1.00 | 0.88 | 0.94 | **> 0.64** | 1.000 | 0.975 |
| true, wrist | 1.00 | 0.94 | 1.00 | 0.88 | 0.94 | 0.69 | 0.56 | **0.40** | 0.979 | 0.966 |
| true, overview | 1.00 | 0.88 | 0.94 | 1.00 | 1.00 | 0.88 | 1.00 | > 0.64 | 1.000 | 0.957 |
| J-hold H16-last, full | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.94 | 1.00 | **> 0.64** | 1.000 | 0.957 |
| J-hold H16-last, wrist | 1.00 | 1.00 | 1.00 | 0.94 | 1.00 | 0.75 | 0.88 | > 0.64 | 1.000 | 0.957 |
| J-hold H16-last, overview | 1.00 | 0.94 | 1.00 | 1.00 | 1.00 | 0.94 | 1.00 | > 0.64 | 1.000 | 0.949 |

The bands below 0.08 rad (0–0.02, 0.02–0.04, 0.04–0.08) were also resolved, at 0.94–1.00 for
both full costs. The other J-hold forms and horizons gave the same picture (d_flat > 0.64 on the
full latent; H8-mean's wrist half stopped at 0.40). The summary has every band's median increment
and its ratio to the increments' IQR. For the full true-frame cost that ratio falls from 2.7 to
0.5 from the inner to the outer bands, so the cost resolves distance less sharply far out. Note
that the line paths interpolate the jaw together with the arm.

**Executed candidates** (64 per state, H16-last, full latent; medians over 16 seeds):

| state (nominal d) | ρ_true | ρ_J | ρ_JT | J's regret (rad) | rank of the approach candidate by J / by the true cost |
|---|---|---|---|---|---|
| 0.08 | 0.57 | 0.33 | 0.67 | 0.080 | 1 / 0 |
| 0.16 | 0.55 | 0.34 | 0.66 | 0.000 | 0 / 0 |
| 0.32 | 0.44 | 0.35 | 0.69 | 0.000 | 0 / 0 |
| 0.48 | 0.28 | 0.24 | 0.73 | 0.000 | 0 / 0 |
| d₀ (reset) | 0.45 | 0.35 | 0.57 | 0.000 | 0 / 0 |
| **far (64 readings)** | **0.44** | **0.33** | 0.67 | | Δρ = **0.11** |

The approach candidate (the pose law, which reaches the subgoal within 16 commands, d_end ≈ 0.0002
rad) is the best of the 64 by both costs at every far state. J agrees with the true-frame cost on
which candidate is best, but it ranks the other 63 less well (ρ_J below ρ_true at every state).
Per camera at H16-last far: wrist ρ_true 0.32 and ρ_J 0.28; overview 0.39 and 0.29.

## 4. Λ0: Stage 0's A logs (§2; reported only)

| config | ρ(C_true, arm RMS) median (full / wrist / overview) | C_true full at reset → closest → last | planned / true cost, median |
|---|---|---|---|
| H8-last | 0.68 / 0.60 / 0.69 | 1.557 → 1.060 → 1.178 | 0.62 |
| H8-mean | 0.57 / 0.40 / 0.61 | 1.557 → 1.140 → 1.357 | 0.72 |
| H16-last | 0.70 / 0.50 / 0.72 | 1.557 → 0.995 → 1.076 | 0.59 |
| H16-mean | 0.65 / 0.55 / 0.64 | 1.557 → 1.142 → 1.371 | 0.68 |

The closest arm RMS medians re-computed from the logs (0.384, 0.482, 0.354, 0.519 rad) equal Stage
0's record.

## 5. The row (§6, first match)

1. Δa-VOID: no guard failed, 16/16 kept, every job completed, 0.810 ≤ 2.0 GPU h. No.
2. Δa-NOT-REPRODUCED: J-ref 0/8 < 4/8. No.
3. **Δa-PLANNER-BOUND: O-pose 0/8 < 6/8. Fires.**
4. (Δa-GEOMETRY: O-img 0/8 < 6/8 also fires, but it is not first.)

**Row: Δa-PLANNER-BOUND.** Under §6 it leads to a planner change first, re-measured with
O-pose. As declared, no reading of the image cost or of J follows from the rows. The record adds
what the logs show about why (§2 above): **the failure is the jaw under a terminal (`last`) cost
with a horizon long enough to postpone the jaw**, not the arm's reach. The candidates the
evidence points to, each needing its own ruling, declaration and fresh seeds:

- **A planner change measured first with O-pose** (§6 row 3): a cost that reads every step
  (TASK-101's `mean` form, or the minimum over the horizon) or a shorter horizon, so the first
  command matters; or committing more than one command; or a warm start. Only once O-pose hands
  over can O-img and J be read.
- **Taking the jaw out of phase 1:** open the jaw by a scripted rule during the approach, or hand
  over on the arm alone and let G_loc open the jaw. With the arm-only reading, O-pose would have
  handed over on 7/8, O-img on 2/8 and J-big16 on 2/8 (post hoc; not a row). G_loc was trained
  from pre-grasp windows with the jaw open, so an arm-only hand-over changes its input
  distribution. That would need a P-like check.
- Beside that, the reported Λ1 and Λ3 readings say that the image cost and J's prediction of it
  carry distance information out to the reset pose, best in the overview half. They do not show
  that this suffices for 0.08 rad once the planner is fixed: O-img's arm closest median was
  0.076 rad, but only 2/8 held for 3 commands. R28.57's LeWM retrain is **not** indicated by this
  record as the next step.

## 6. Deviations (disclosed)

1. **The arm-only and jaw readings in §2's table were added after the declaration.** No row reads
   them.
2. **Workers per job were set by GPU memory.** Two debug attempts ran out of memory with J-big32
   beside other workers, so the gated jobs used 6 workers (J-ref, O-img), 2 (J-big16) and 1
   (J-big32). Each episode runs in a fresh process, so this affects timing only.
3. **O-img's and O-pose's `distance` returns the last-step cost at every step**, as declared (only
   the last step is read by the planner). Seconds per command for O-img (18.8 s) include the GPU
   contention of the shared job.
4. **Λ1's line states interpolate the jaw** together with the arm (as declared), so their cost
   readings mix arm and jaw distance.

## 7. What this does not show

- A development reading on 16 (Λ1) and 8 (Λ2) seeds, one run, in simulation. The thresholds were
  declared, not calibrated. 0/8 has an exact one-sided 95 % upper bound of 0.31.
- It does not show that any planner change would hand over. It also does not show that the image
  cost or J would be enough once the jaw is handled: Δa-GEOMETRY also fired, and it says the
  true-frame cost did not carry this planner to the hand-over either.
- The postponement mechanism is inferred from logs (planned against executed jaw), not tested by
  intervention.
- No test of LeWM. No world-model claim in either direction.

## 8. Evidence

`~/develop/emai/evidence/task102-approach-diag/` (86 files; `SHA256SUMS` `2170336e…e808`):
`summary.json` (`e9eeaa5f…6fbf`), `lambda0.json`, the goal report (goal store `4a317e57…06a5`),
the landscape report and its 16 per-seed arrays, each Λ2 job's report, progress and per-episode
logs, the stage logs and the GPU ledger lines. Outputs: the shared checkout's
`outputs/task102-approach-diag/` (HDD). Code: `so101_approach_diag.py`,
`so101_approach_diag_runtime.py`, `scripts/run_task102_approach.py`,
`scripts/summarize_task102_approach.py`, `scripts/run_task102_approach_stage.sh`,
`tests/test_so101_approach_diag.py` (16 tests), all new.
