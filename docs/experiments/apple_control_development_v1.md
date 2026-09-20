# Apple sensor-world-model control v1 — development preregistration

Execute only after the first registered sensor-model training attempt completes
and its validation evidence is audited. This is task-specific simulation control
with an explicit demonstration-derived image-waypoint scaffold. It is not the
historical common CEM benchmark or a zero-shot JEPA/LeWM result.

Before any rollout, commit the runner and this protocol, pin the sealed
`data/apple-task-v1` manifest and selected `apple-sensor-v1/sensor.pt` hash, and
save the full resolved configuration, source snapshot and waypoint calibration.
No final-test seed is used in this development experiment.

## Fixed six-attempt comparison

- Reset seeds **43000, 43001**, each with modes **learned, persistence,
  dynamics_shuffle**, in that order. Six intended attempts, no retries.
- Apple center (0.34, −0.18), plate center (0.49, −0.09), independent uniform
  ±0.006 m XY jitter; draw apple first, then plate from the reset-seeded RNG.
- Unchanged MuJoCo physics, camera, action manifest and ordered AppleToPlateTask
  scoring. Evaluator truth never advances the controller or ranks its actions.
- Bounded shooting MPC: horizon 4, 16 candidates, 2 search rounds, proposal
  standard deviation 0.15, minimum 0.05, at most 1,000 accepted commands.
  Left pose deltas fixed at zero, left grasp −1; right pose deltas in [−0.5, 0.5]
  and right absolute grasp in [−1, 1]. Embodiment projection is mandatory.
- Offline control deadline 5 seconds, including observe/plan work before execution;
  abort a missed deadline without sending its stale action.
- A single **600 true-wall-second** supervisor budget includes configuration,
  calibration, loading, rendering and all attempts. Count incomplete/unstarted
  attempts explicitly. No extension or replacement run in the same directory.
- CPU, four Torch threads. Preserve per-attempt logs, pending/executed actions,
  projected/scored/accepted command agreement, selected predicted cost, candidate
  cost spread, waypoint index, timings, stop reason and all physical task stages.

## Training-only scaffold and controls

Select the lexicographically first successful **nominal TRAIN** apple episode.
Use its RGB frames every 10 accepted transitions plus its final frame. Goals never
contain target joint state. Candidate demonstration proposals use applied-action
windows lying entirely within the preceding 10 training transitions; they are
always projected and rescored alongside alternatives. No proposal executes via
an unscored fallback.

For each waypoint calibrate tolerance from training pixels before physics:
maximum of (a) 1.1 times the largest image distance within its ±2-frame local
neighborhood, (b) median positive adjacent-frame distance in the selected demo,
(c) 1.1 times the 90th percentile distance from corresponding frames in other
successful nominal TRAIN episodes with frame coverage, and (d) 1e-12. Store all
calibration sources and values. Record adjacent-waypoint distances and overlapping
tolerances: overlap may make ordered dwell behave like a temporal scaffold and
must be considered when interpreting model ablations. Require three consecutive
observed-image matches to advance; no scorer flag or scripted phase clock drives
advancement. This is a declared task decomposition, not long-horizon foresight.

Learned mode ranks projected action sequences through the trained transition
model. Persistence gives all candidates the current image cost and uses seeded
tie selection. Dynamics shuffle breaks the association between predicted costs
and candidate actions. All modes retain the same image scaffold, proposal source,
budgets and mechanics. A successful scaffold or proposal alone does not establish
that the world model contributes; matched ablations are necessary evidence.

## Interpretation and next gate

Report every attempt, including zero success, source-integrity failure, guard stop,
deadline and budget truncation. Inspect waypoint stalls and prediction/control
agreement. Two development resets cannot establish reliability. If results fail,
write the diagnosis and preregister the next development change before rerunning;
do not adjust this protocol retrospectively. The fresh 20-reset cohort remains
unexecuted until a development configuration is selected and frozen.

Planned output: `outputs/apple-control-development-v1/`. The working-MVP target
remains 16/20 complete successes on the later frozen cohort, with all failures in
the denominator and evidence that learned dynamics improve control.

## Recorded before the first control rollout

The phase diagnostic completed all 24 groups. Validation prediction improved over
persistence in every measured phase, but the small shuffled-action advantage was
inconsistent and negative in several close/lift cohorts. The model has **not**
passed a robust phase-specific action-conditioning gate. This six-attempt run
therefore remains a limited failure/control diagnostic, not progression to the
fresh final cohort or a claim of learned task readiness. Its previously declared
parameters and controls remain fixed; no phase result changes the controller.
