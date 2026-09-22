# Apple privileged simulator-rollout planning ceiling v1: negative result

**This is a NON-LEARNED privileged diagnostic ceiling, not a learned result.** It
used exact MuJoCo rollouts as the planner's forward model. That is allowed only
because it measures a ceiling.

**The preregistered primary gate failed.** The ceiling reached the scorer's grasp
stage on **0/4** development resets, where ≥ 2/4 was required. Every scorer stage
was false on every reset.

With exact dynamics, the unchanged TASK-043 state-goal scaffold and CEM planner
ended every attempt with `goal_stall`:

- 43000 and 43001 stalled on goal 0.
- 43002 and 43003 advanced to goal 1 and stalled there.

The readings are **conclusive**: all 4 attempts were counted, provenance was
valid, and the rollouts were exact. `state_goal_tracking_inadequate` is **true**:
all 4 resets stalled before the first close goal (index 13). Per the
preregistration, **the planner/goal design must change before any further model
training.** The dynamics model is not the only bottleneck.

## Frozen execution

The run executed once from a clean checkout of the reviewed source
`733bfdca972e001a83835a8f4b193807f980e4c6`. It used the
[preregistered protocol](apple_privileged_ceiling_v1.md), including revision R1,
and its exact frozen command. There were no retries, reruns or parameter
changes, and the final cohort (44000–44019) and TEST were not run.

- All 4 attempts started and completed.
- The evaluator exited 0 with report status `completed` and valid provenance.
- Global wall time was **371.964 s** of 3,600 s.
- Each attempt was allocated its full 840 s and used 62.6–119.0 s. None timed
  out.

Frozen inputs (unchanged from TASK-043):

- Checkpoint `0192b99a…a5a5`. Only retrieval, TRAIN normalization and the state
  metric came from it; its dynamics were not used.
- Corpus manifest `6e9a5bcb…0331`.
- Action manifest `f247effe…38b1`.
- Asset manifest `2421e194…e993`.
- TRAIN state-goal library `state_goals.npz` `839190fc…f88d` and
  `state_calibration.json` `d8dae051…8d75`. These are byte-identical to TASK-043,
  so the goals and tolerances are the same.

The [compact manifest](../../benchmarks/manifests/apple-privileged-ceiling-v1.json)
records full hashes, the ceiling gate, per-attempt records and SHA-256 of the
core artifacts under `outputs/apple-privileged-ceiling-v1/`. Logs and the source
snapshot are excluded.

## Every attempt

Retrieval selected the same TRAIN demonstrations as in TASK-043. Goal indices
are 0-based, and the first close goal is 13 for all four. "Commands on goal"
counts acknowledged commands while that goal was pursued. "Stages" counts
leading true scorer stages. The "min" distance includes the final observation
at termination. "Median plan" is the controller's planning time only;
observe + plan medians were 0.953–0.974 s.

| Reset | Retrieved demo | Commands (per goal) | Last goal | Goal-0 tol. | Measured goal-0 dist. start → min | Stages | Termination | Parity checks / mismatches | Median plan (s) | Wall (s) |
|---|---|---|---:|---:|---|---:|---|---|---:|---:|
| 43000 | apple-42029 | 64 (g0: 64) | 0 | 0.0166 | 0.101 → 0.018 | 0 | goal_stall | 63 / 0 | 0.942 | 62.6 |
| 43001 | apple-42025 | 64 (g0: 64) | 0 | 0.0058 | 0.113 → 0.024 | 0 | goal_stall | 63 / 0 | 0.947 | 62.8 |
| 43002 | apple-42008 | 121 (g0: 57, g1: 64) | 1 | 0.0183 | 0.124 → ≤ 0.0183 (advanced) | 0 | goal_stall | 120 / 0 | 0.958 | 119.0 |
| 43003 | apple-42021 | 119 (g0: 55, g1: 64) | 1 | 0.0124 | 0.101 → ≤ 0.0124 (advanced) | 0 | goal_stall | 118 / 0 | 0.964 | 118.2 |

- **No rejections.** No candidate rollout was rejected on any reset.
- **Exact parity.** Across 364 checks, the measured state always equalled a
  simulated first step exactly.
- **Planning time.** The median was 0.94–0.96 s per command (0.95–0.97 s
  including observation), and the maximum observe + plan time was 1.14 s. That is
  about 12–13× the learned mode's time and under the 5 s deadline.
- **Scorer state.** No reset had hand contact. After settling by 1.2 mm at the
  first step, the apple stayed at rest, 0.174–0.182 m from the plate.

## Gate and readings

From `report.json["ceiling_gate"]`, with 4/4 attempts counted:

- **Primary gate: failed.** 0 grasp resets (≥ 2 needed), 0 summed ordered stages,
  0 full successes. `rollouts_exact` was true, with 0 mismatches.
- **`state_goal_tracking_inadequate`: true.** 4/4 resets were still pursuing a
  goal before the close goal (index 13), against a threshold of ≥ 3/4.
- **`arm_pose_goals_insufficient_for_grasp`: false.** 0 resets advanced past the
  close goal. This question was never tested, because the ceiling never got that
  far.
- **Interpretation (computed):** "planner/goal design must change before further
  model training".

## Diagnostics (not gate evidence)

The exact rollouts suggest a likely reason the scaffold fails even with perfect
dynamics. This is an interpretation, not a tested mechanism.

**Planned endpoints were within tolerance, but the robot never arrived.** On each
stalled goal, the selected H16 endpoint (an exact prediction) was within the goal
tolerance in 20–81% of searches:

| Reset | Goal | Share of searches | Median selected cost | Tolerance | Closest measured distance |
|---|---:|---:|---:|---:|---:|
| 43000 | 0 | 77% | 0.0103 | 0.0166 | 0.0183 |
| 43001 | 0 | 20% | 0.0103 | 0.0058 | 0.0243 |
| 43002 | 1 | 81% | 0.0102 | 0.0184 | 0.0263 |
| 43003 | 1 | 67% | 0.0104 | 0.0121 | 0.0125 |

The closest measured distance includes the observation at termination. The
measured state never entered the tolerance on those goals.

The pattern is consistent with receding-horizon **endpoint chasing**:

- The cost scores only the state 16 steps ahead.
- Only the first action is executed before replanning (commitment 1).
- So every search again plans to arrive in 16 steps, and nothing rewards arriving
  sooner.

What the measured distances did:

- Closest approach was 1.0–4.2× the tolerance.
- The distance oscillated widely: the median was 4.1–35× the tolerance, with
  peaks of 11–76×. Only 0–17% of commands were within 2× of the tolerance.
- At 43000, 43001 and 43002, the distance was still decreasing when the
  64-command bound ended the goal.
- At 43003, the distance to goal 1 came within 0.0005 of its tolerance, then
  diverged to 0.139 at the stall.

**Tight tolerances contribute.** At 43001, the goal-0 tolerance is 0.0058, and
only 20% of the exact plans even predicted an endpoint inside it.

**Only goal 0 was ever reached.** The two resets that advanced needed 55–57
commands for goal 0, against 16 frames in the demonstration. Neither advanced
past goal 1. In TASK-043, `demo_replay` executed
the same goals open loop and completed 3/4 resets.

## Interpretation and limits

This is n = 4 development evidence under one frozen configuration. It says the
TASK-043 failure is **not only** a learned-dynamics problem. Even with exact
dynamics, the same scaffold (endpoint-only H16 state cost, commitment 1, TRAIN
q90 tolerances with dwell 1, and a 64-command stall bound) cannot track the
demonstration's state goals past goal 1.

Retraining the dynamics alone (multi-step proprio rollout loss or grasp-phase
data) would therefore not be expected to pass the TASK-043 gate. The learned
model's forecast bias remains a real, separate defect (TASK-043 diagnostics).

The ceiling does not show what *would* work. The following candidate changes
are untested and each needs its own preregistered ceiling test:

- a time-indexed or running cost over the horizon instead of the endpoint only;
- longer commitment;
- horizon-shrinking toward the goal;
- tolerance/dwell choices.

TASK-033/TASK-034 physical acceptance stays open. Learned Apple→Plate remains at
zero successes. The final cohort and TEST were not touched.

## Recommended next step (not started)

Redesign the planner/goal scaffold under the **privileged ceiling first**, then
return to the learned model. Two candidate protocols:

1. **Time-indexed tracking (preferred).** Score the whole H16 rollout against the
   demonstration's interpolated per-frame state trajectory: a running cost, not
   only the endpoint. Advance goals by time/progress, with the stall bound kept.
   Gate it with the same privileged ceiling on the same four development resets.
2. **Commitment or shrinking horizon.** Execute a larger share of each exact plan,
   or target the goal at min(H, frames remaining). This removes the endpoint
   chasing and tests whether tolerances alone still block progress.

Only a scaffold that passes the privileged ceiling (grasp on ≥ 2/4) should be
paired with learned dynamics. At that point, dynamics retraining (multi-step
proprioceptive rollout loss and grasp-phase data) becomes the next learned test.
