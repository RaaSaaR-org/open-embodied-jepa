# TASK-101 A2: the follower's pose-error envelope and the switching tolerance (record)

**Status: development measurement record (§19 of [so101_closed_loop_lift_v1.md](so101_closed_loop_lift_v1.md),
amendment A2, R28.83–R28.89). Row: A2-NOT-BINDING; τ_s stays 0.04 rad; escalated to the owner.**
Not a gate, no claim, not a learned result: every episode here is the privileged scripted
follower v1. Simulation only, `so101_cube_lift_v1`, not Apple→Plate. R7 does not change.
Recorded by Claude under owner delegation (DECISIONS 2026-10-19 (bx), R28.90).

**Commit order.** The rule (§19.4) and the code were committed in `ea19cbb` and pushed before the
run. The run started at 19:36:48Z at that revision, from a clean tracked tree, and ended at
19:38:43Z at the same revision with the tree still clean. This record is a later commit.

## What ran

- **Cohort E 102000–102031**: the unperturbed goal demonstration was kept on 32 of 32 seeds; the
  reserve 102032–102047 was not used.
- **Guards:** G-zero held on all 96 (seed, variant) pairs: with δ = 0 every variant executed the
  demonstration's commands bit for bit. No episode ended before its hand-over command, and the
  revision and tree were unchanged at the end. **No guard failed.**
- **Achieved perturbation:** the median measured arm RMS to q_k after the hand-over command equals
  the declared level to within 2 × 10⁻⁵ rad at every level and reading, so joint limits and contact
  did not change the dose.
- **Compute:** CPU only, 115 s, 32 demonstrations and 32 × 3 × 7 = 672 follower episodes. No GPU.

## Envelope (lift successes of 32; exact 95 % intervals)

| level (rad RMS) | "pre" (subgoal 1; rule) | "carry" (subgoal 2; rule) | "reaim" (subgoal 2; reported only) | level passes? |
|---|---|---|---|---|
| 0.02 | 32/32 [0.891, 1.000] | 32/32 [0.891, 1.000] | 32/32 [0.891, 1.000] | **yes** |
| 0.04 | 32/32 [0.891, 1.000] | **20/32** [0.437, 0.789] | 31/32 [0.838, 0.999] | no |
| 0.06 | 32/32 [0.891, 1.000] | 14/32 [0.264, 0.623] | 29/32 [0.750, 0.980] | no |
| 0.08 | 32/32 [0.891, 1.000] | 10/32 [0.161, 0.500] | 26/32 [0.636, 0.928] | no |
| 0.12 | 32/32 [0.891, 1.000] | 4/32 [0.035, 0.290] | 28/32 [0.710, 0.965] | no |
| 0.16 | 32/32 [0.891, 1.000] | 2/32 [0.008, 0.208] | 25/32 [0.600, 0.907] | no |

Every failure was a timeout (the cube not held up through the scorer's window): 110 in "carry",
21 in "reaim", none in "pre".

## The rule, applied mechanically (§19.4)

1. A2-VOID: no guard failed.
2. The largest level L with every level up to L passing (both "pre" and "carry" ≥ 30/32) is
   **0.02 rad**. L < 0.06, so **A2-NOT-BINDING: τ_s stays 0.04 rad**, K″ does not run, and the
   owner chooses between a hybrid local grasp controller and R28.57's LeWM longer-budget retrain.
   **The current 0.04 rad tolerance itself fails the 30/32 bar** ("carry" 20/32), which §19.4 asks
   to be recorded.

## What the row's name does and does not mean

The row's name is the declared one. The measurement says something more specific, recorded here
as a reading (nothing selects on it):

- **At the pre-grasp pose the tolerance is loose for the follower.** "pre" lifted 32/32 at every
  level up to 0.16 rad, because the follower re-aims its descent from simulator truth. Loosening
  the switch off subgoal 1 would cost the follower nothing. It would let W leave subgoal 1 from
  where Δ found J's cost minimum (about 0.05 rad off).
- **At the grasp pose the task needs far more precision than 0.04 rad.** With the jaw closing
  where the arm is ("carry"), the lift rate falls from 32/32 at 0.02 rad to 20/32 at 0.04 rad and
  2/32 at 0.16 rad. That is the joint-space counterpart of S2's Cartesian grasp-offset envelope
  (7.5 mm toward the fixed jaw, at least 20 mm away; asymmetric), which a random joint-space
  direction samples on both sides. If the arm is corrected back while the jaw closes
  ("reaim", reported only), 0.04 rad gives 31/32 and 0.16 rad still 25/32.
- So **the switching tolerance is not the knob that can rescue the closed loop**: no single
  tolerance both lets W move off the pre-grasp pose where J's cost settles (needs ≥ about
  0.05 rad) and keeps the grasp reliable when the jaw closes where the arm stands (needs about
  0.02 rad). Whatever closes the jaw has to place it to about 0.02 rad RMS, or correct the pose
  while closing. Δ found J's cost minimum about 0.05 rad off the pre-grasp pose, which the
  "carry" reading says is too coarse at the grasp pose. This is the case for the two options the
  rule names. A hybrid local grasp controller would do the correction. A model change would aim to
  move the cost's minimum onto the pose.

## Limitations (§19.7)

τ_s is calibrated on the follower, a privileged controller that re-aims from simulator truth; a
tolerance it absorbs is necessary for the closed loop, not sufficient. The switching rule changes
when the controller moves on, not what counts as a lift: the simulator-truth scorer is unchanged.
One direction per seed and subgoal, 32 seeds, two hand-over points, declared levels and bar. K′'s
logs were not re-read. A development measurement, simulation only.

## Evidence

Revision `ea19cbb` (pinned file hashes in `envelope.json`: `so101_switch_envelope.py` `6cf863d6…`,
`measure_task101_envelope.py` `857fc734…`, `so101_followers.py` `fe086b89…`,
`so101_tasks_lift_v1.py` `5bb47aa7…`, `so101_simulation.py` `a5143d9e…`). Run output
`outputs/task101-envelope/` in the worktree `worktrees/task101-a2`; copied to
`~/develop/emai/evidence/task101-envelope/`: `envelope.json` (`0355aba9…6c70`, every episode's
seed, δ, outcome, hand-over error and executed-command hash), `applied_actions.npz` (`a08aaea0…`),
`run.log` and `SHA256SUMS` (`a536ac96…5371`).
