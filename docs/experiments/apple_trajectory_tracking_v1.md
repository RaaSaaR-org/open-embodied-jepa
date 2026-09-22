# Apple demonstration trajectory tracking v1 — preregistration

Prospective TASK-045 protocol. It was committed before any development attempt of
this scaffold. Stage 1 is a **NON-LEARNED privileged diagnostic ceiling**. Its
outcomes are never learned results and never count toward TASK-033/TASK-034.
Stage 2 is a learned comparison. It runs **only if** stage 1 passes its gate.

## Question and hypothesis

TASK-044 ([protocol](apple_privileged_ceiling_v1.md),
[results](apple_privileged_ceiling_results_v1.md)) gave the TASK-043
state-goal scaffold exact MuJoCo-rollout dynamics. It still stalled on goal 0 or 1
on 4/4 development resets, with 0 scorer stages. Its exact 16-step plans often
ended inside the goal tolerance, but the executed state only came within
1.0–4.2× of the tolerance and oscillated. The suspected cause is **endpoint
chasing**: the cost scored only step 16, only one command ran before replanning,
and the goal's target moved every replan.

**Hypothesis.** Track the retrieved TRAIN demonstration's per-frame right-arm+hand
position trajectory instead of discrete endpoint goals. Score the whole H16
rollout against a time-indexed reference, and advance the reference by measured
progress along the demonstration. With the same CEM budget (16 candidates, 2
iterations, commitment 1, H16) and the same privileged exact-rollout forward
model as TASK-044, this reaches the scorer's grasp stage on **≥ 2/4** of resets
43000–43003.

## What changes and what does not

Only the goal representation, the cost aggregation and the progress rule change.
The following are identical to TASK-044:

- the retrieval rule;
- the TRAIN candidate set;
- the privileged forward model (`privileged_rollout.py`, unchanged: twin
  restore, per-step re-projection, unchanged `execute`, rejection penalty,
  runtime parity check);
- the CEM sampling (hold, warm/search-best and Gaussian perturbations);
- the proposal std 0.15 (min 0.05);
- the RNG seeding (the reset seed);
- the action bounds and initial grasps;
- the mandatory projection, embodiment, guards and physics;
- the unchanged ordered `AppleToPlateTask` scorer.

Nothing in `sensor.py`, `base.py`, `state_goal_sensor.py`,
`privileged_rollout.py`, `waypoint_planning.py`, the scorer or the guards
changes. The tracking controller is a new, isolated module,
`src/embodied_jepa/trajectory_tracking.py`. It selects only through
`evaluate_apple.py --goal-kind trajectory`. The image and state goal kinds keep
their plans, controllers, traces and gates, and tests check this.

**No action seeding.** Neither the CEM mean nor any candidate is seeded with the
demonstration's recorded actions. So no "reference actions open-loop" control is
needed. The planner must produce every command. (TASK-043's `demo_replay`, 3/4
successes, already measures open-loop replay of the whole demonstration.)

## Design (fixed before any development attempt)

### Reference (TRAIN only)

- **Retrieval, per reset:** the same rule as TASK-043/044. From the 16
  successful NOMINAL apple/plate TRAIN episodes, take the one whose frame-0 RGB
  has the smallest frozen sensor image distance to the live initial RGB. Ties go
  to the lexically first ID. Only the initial RGB is used.
- **Rows:** the demonstration's measured 14 right-arm+hand joint positions (the
  TASK-043 fields), taken at kept frames.
- **Dead-time removal (keyframing).** Frame 0 and the final frame are always
  kept. Another frame is kept only when its state distance to the last kept
  frame exceeds `0.1 × the demonstration's median positive adjacent-frame state
  distance`. That median is the existing TRAIN `adjacent_distance_floor`, about
  1.4–1.7e-4.
- **Justification (TRAIN data and first principles; no development outcome).**
  All 16 TRAIN demonstrations contain long spans where the full 86-D robot state
  does not move while scripted commands continue. The main spans are frames
  144–210 (descend) and 332–405 (lift): about 178 frames per demonstration have
  an adjacent distance below 0.1× the floor, and the longest run is 72–74
  frames. A time-indexed tracker cannot sense progress through a span where the
  target does not change. The measured-progress rule would then have to "wait"
  without any state signal, and could stall on dead time.
- **Size and sensitivity.** Keyframing keeps 338–342 of 502 frames. The first
  closing frame falls at reference row 141–142, and the demonstration's recorded
  grasp at row 195–199. The kept-frame count is insensitive to the fraction:
  0.01× keeps 366–371 frames and 0.5× keeps 309–314. The value 0.1 was chosen
  as "one order of magnitude below typical per-frame motion", not tuned.

### Cost

For candidate `k` with predicted states `ŝ_1..ŝ_16` (the state after executing
step `h` of the candidate), with `t` the current reference index and `L` the
number of rows:

`C_k = (1/16) Σ_{h=0..15} d(ŝ_{h+1}, R[min(t+1+h, L−1)])`

Here `d` is the model's own `distance` output at horizon step `h` against the
single-row goal `R[j]`. For the privileged model and for
`state_goal_sensor_wm_v1`, this is the same TRAIN-normalized 14-field mean
squared error used by TASK-043/044.

- The cost goes through the generic model contract only: `encode`,
  `encode_goal({"state": [1,14]})`, `predict`, and `distance` returning
  `[1,K,H]`. So the learned model plugs in unchanged.
- For the privileged model, a rejected rollout step still costs `1000 + last
  reached distance`. That keeps rejected candidates last.

### Progress (measured, monotone)

At every observation, the controller computes the measured state distance
`d(s, R[j])` for `j ∈ [t, min(t+16, L−1)]`. It sets `t` to the **latest** `j`
that attains the minimum.

- `t` never decreases, and it moves by at most 16 rows per command.
- The window equals the planning horizon, because the planner is never asked to
  look further ahead.
- No tolerance or dwell is used. Scorer truth never drives progress.

### Termination

- `reference_complete`: `t` reaches the final row.
- `reference_stall` (a **failure**): after at least 64 acknowledged commands,
  the current index is fewer than 16 rows ahead of the index measured 64 commands
  earlier. This is the TASK-043 rate (16 frames within 64 commands, 4× slack
  over the demonstration's pace), applied as a sliding window.
- `step_limit` after 1,000 commands.
- The attempt also ends on scorer success. The scorer never drives the
  controller.

## Frozen inputs (unchanged from TASK-043/044)

- Corpus `data/apple-branches-v1`, manifest SHA-256
  `6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`.
- Checkpoint `checkpoints/apple-branches-h16-sensor-v1/sensor.pt`, SHA-256
  `0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5`.
  - Stage 1 uses it only for retrieval, TRAIN normalization and the metric.
  - Stage 2 also uses its dynamics.
- Action manifest `configs/g1_sim_action.json` and the pinned G1/Dex3 assets.
- The TRAIN state-goal library that the runner writes for retrieval:
  - `state_goals.npz` is expected to be byte-identical to TASK-043/044
    (`839190fc…f88d`).
  - `state_calibration.json` is expected to be byte-identical too
    (`d8dae051…8d75`).
  - The TRAIN smoke reproduced both hashes.
  - New artifacts: `tracking_references.npz` and `tracking_calibration.json`.
    They are hashed into `resolved_plan.json` and re-verified around every
    attempt.

## Stage 1 — privileged trajectory-tracking ceiling

- Development resets **43000, 43001, 43002, 43003**, mode `privileged_rollout`,
  `--goal-kind trajectory`: **4 attempts**. No retries, replacement seeds or
  reruns.
- **Budget:**
  - H16, K16, 2 rounds, commitment 1.
  - At most 1,000 commands and a 5 s per-command observe/plan deadline.
  - **600 s per attempt** and **3,600 s global**, including preparation and
    finalization. The existing `min(attempt cap, global remaining)` allocation
    applies.
  - CPU, single-threaded MuJoCo, Torch capped at 4 threads.
- **Time risk (declared).** Each command simulates 512 control steps, about 1 s,
  so the 600 s cap allows roughly 570 commands.
  - The demonstration's recorded grasp is at reference row 195–199.
  - Tracking at 1 row per command would reach it in about 200 commands.
  - The cap binds before grasp only if average progress falls below about
    0.35 rows per command.
  - The stall rule tolerates progress down to 0.25 rows per command. So a slow
    but non-stalling attempt can end with `attempt_timeout`. That counts as 0
    stages, with its latched stages reported as uncounted.
- **Primary gate (`report.json["tracking_ceiling_gate"]`):**
  - the ceiling reaches the scorer's **grasp** stage on **≥ 2/4 resets**;
  - AND zero rollout parity mismatches occur in counted attempts.
  - Only attempts with status `completed`, a termination other than
    `runtime_error`/`deadline_miss`/`attempt_timeout`, and valid provenance are
    scored. `reference_stall`, `reference_complete` and `step_limit` are clean
    terminations whose scored stages count. Missing or failed attempts count as
    0. Any provenance failure fails the gate.
- **Reported per reset:**
  - retrieved demonstration and termination;
  - commands, last reference index, reference length, first-close and
    demo-grasp reference rows;
  - ordered stages, grasp, success and parity counts.
- **Readings.** They are computed from counted attempts only. They are asserted
  only when all 4 attempts count, provenance is valid and rollouts are exact.
  Otherwise the result is *inconclusive*.
  - `trajectory_tracking_inadequate`: the gate failed, and on ≥ 3/4 resets the
    last reference index was below the first-close row without a grasp. Tracking
    cannot follow the demonstration to the grasp, even with exact dynamics.
  - `arm_pose_tracking_insufficient_for_grasp`: the gate failed, and on ≥ 3/4
    resets the last reference index reached the demonstration's recorded-grasp
    row without a scored grasp. This does not establish accurate joint tracking
    or identify missing object information as the cause. The recorded TRAIN
    scorer labels
    (`stage_scores`) define this row. They are used **only** for this post-hoc
    reading, never in planning, progress or cost.
- **Interpretation:**
  - *Pass.* The trajectory-tracking scaffold is adequate under perfect
    dynamics. This supports the revised scaffold as a whole; cost, progress and
    keyframing changed together, so it does not isolate endpoint chasing. Run
    stage 2.
  - *Fail (conclusive).* Do not pair this scaffold with learned dynamics. The
    readings above say which part failed.
  - *Inconclusive.* Report it; there is no design verdict and no stage 2.
- **Comparison with TASK-044.** It is the same resets, forward model and CEM
  budget; only the scaffold differs. The goal is not the same object (row track
  vs. tolerance goals), so compare scorer stages only, not goal indices.

## Stage 2 — learned comparison (conditional, declared now)

Run **only if** `tracking_ceiling_gate.primary_gate_passed` is true. It runs
once, after stage 1's results are recorded.

- Modes `learned`, `dynamics_shuffle`, `persistence`, with
  `state_goal_sensor_wm_v1` over the frozen checkpoint.
  - The ablations are defined as in TASK-043: shuffle permutes feasible cost
    assignments each round; persistence gives all candidates the current
    measured tracking distance.
  - The same 4 resets, reset-major: **12 attempts**.
- **Budget (TASK-043):** H16/K16/2 rounds/commitment 1, at most 1,000 commands,
  a 5 s deadline, **200 s per attempt**, **3,600 s global**.
- **Primary gate (`report.json["tracking_gate"]`):**
  - `learned` reaches grasp on **≥ 2/4** resets;
  - AND `learned` summed ordered stages are **strictly greater** than both
    `dynamics_shuffle` and `persistence`.
  - The same counting rules apply.
- **Readings:**
  - `learned_dynamics_failure_under_tracking`: conclusive, failed, and `learned`
    stalled before the first-close row on ≥ 3/4 resets.
  - `no_model_contribution`: `learned` scored ≥ 1 stage but not strictly more
    than the stronger control.
- **Interpretation:**
  - A pass is n = 4 development evidence for a learned tracking controller, not
    final acceptance.
  - A fail after a ceiling pass motivates inspection of learned forecasts and
    control traces. It does not uniquely isolate dynamics error: stage budgets,
    closed-loop state distributions and per-step projection also differ. A
    retraining experiment requires a separate prospective justification.
- If stage 1 does not pass, stage 2 is **not run**, and this is recorded.

## Allowed pre-physics steps and recorded smoke

Software checks only. A software smoke ran on **TRAIN reset 42000**, not a
development reset:

- Setup: scratch output, input-hash verification stubbed, stage-1 privileged
  mode, `max_steps` 64. The source was parent `b2e0789` plus the uncommitted
  TASK-045 working tree.
- Retrieval selected `apple-42000`. Its reference has 340 rows, first close row
  142, and recorded-grasp row 198.
- The regenerated `state_goals.npz` and `state_calibration.json` matched
  TASK-044 byte for byte.
- Planning took a median of 0.985 s per command, and observe + plan peaked at
  1.11 s.
- Runtime parity matched exactly on 63/63 checks, and no candidates were
  rejected.
- After 64 commands (`step_limit`), the reference index was 40, with measured
  tracking distance around 0.003–0.016.

The smoke is a runtime check, not evidence. No parameter was changed after it.
The keyframe fraction, window, stall rule, budgets and gates were fixed before it
ran.

## Frozen run commands

Execute each **once**, from a clean checkout of the reviewed commit, into a
directory that does not exist yet.

Stage 1 (always):

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_apple.py \
  --dataset data/apple-branches-v1 \
  --checkpoint checkpoints/apple-branches-h16-sensor-v1/sensor.pt \
  --output outputs/apple-trajectory-tracking-ceiling-v1 \
  --stage development --seeds 43000 43001 43002 43003 \
  --modes privileged_rollout \
  --goal-kind trajectory --goal-stall-limit 64 --no-proposals \
  --horizon 16 --stride 16 --dwell 1 --candidates 16 --iterations 2 \
  --commitment-steps 1 --max-steps 1000 --attempt-max-seconds 600 \
  --max-seconds 3600 --control-timeout 5
```

Stage 2 (only if stage 1's `tracking_ceiling_gate.primary_gate_passed` is
true):

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_apple.py \
  --dataset data/apple-branches-v1 \
  --checkpoint checkpoints/apple-branches-h16-sensor-v1/sensor.pt \
  --output outputs/apple-trajectory-tracking-learned-v1 \
  --stage development --seeds 43000 43001 43002 43003 \
  --modes learned dynamics_shuffle persistence \
  --goal-kind trajectory --goal-stall-limit 64 --no-proposals \
  --horizon 16 --stride 16 --dwell 1 --candidates 16 --iterations 2 \
  --commitment-steps 1 --max-steps 1000 --attempt-max-seconds 200 \
  --max-seconds 3600 --control-timeout 5
```

`--stride 16` sets the stall rule's minimum advance and the retrieval library's
goal spacing. `--dwell 1` is used only to rebuild the identical retrieval
library; tracking has no dwell. Every outcome, including failures, stalls and
timeouts, will be recorded in `apple_trajectory_tracking_results_v1.md` and
`benchmarks/manifests/apple-trajectory-tracking-v1.json`. The final cohort
44000–44019 and TEST stay untouched.

## Known risks declared before outcomes

- **Arm pose is not object pose.** Tracking the demonstration's joints can miss
  the apple, whose reset position differs by up to ±6 mm. The
  `arm_pose_tracking_insufficient_for_grasp` reading exists for this.
- **Time cap.** A slow but steady tracker can hit the 600 s cap before the grasp
  row (see stage 1).
- **Receding-horizon lag.** With commitment 1, the reference ahead of `t` moves
  only when measured progress moves. If the robot lags, the target does not run
  away, but the tracker may progress below the demonstration's pace.
- **Keyframing removes waiting.** Removing dead time also removes any settling
  the scripted demonstration did while stationary. The robot state does not
  change there, but object/contact settling may still matter. Its effect is
  untested.
- **Short run.** n = 4 development resets per stage.

## Pre-run review revision R1

Before any development attempt, independent review narrowed the causal claims.
Reference-index advancement does not prove accurate joint tracking, stage-1
success cannot isolate which scaffold change mattered, and stage-2 failure
cannot uniquely identify dynamics error. Existing diagnostic reading keys are
heuristic labels, interpreted only as their explicit index/stage predicates.
No numeric gate, seed, controller parameter, budget or run command changed.
