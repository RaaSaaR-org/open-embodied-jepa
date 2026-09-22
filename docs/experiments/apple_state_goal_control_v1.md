# Apple demonstration-state-goal MPC v1 — preregistration

Prospective TASK-043 protocol, committed before any physical attempt on the
development resets named below. It declares a **new hypothesis**,
"demonstration-state-goal MPC", not a retune of the image-goal controller.

## Status of prior evidence and disclosure

No learned run has passed orient/descend: the best image-goal attempt reached
goal index 5 (TRAIN frame about 140 of 501) and every scorer stage has been false
for every learned attempt so far. TASK-041/042 VAL analysis, which this design
**was informed by**, found that the frozen H16 sensor model ranks action siblings
far better by predicted arm pose (89.8%) than by predicted 24×24 RGB (73.4%),
and a fixed visual/pose hybrid failed its +5 pp gate (+4.30 pp). Visual forecasts
are biased (hold is predicted to improve the image cost while measured cost
worsens). The current image scaffold also has goal stride 28 > H16, 39/52 goals
inside a neighbour's tolerance, and goal-indexed demonstration proposals that
replay the same window whenever goals stop advancing.

Because TASK-041/042 VAL numbers motivated choosing pose over pixels, those VAL
results cannot confirm this hypothesis. Only the new physical comparison below
is evidence for or against it, and it is n=4 development evidence.

## Hypothesis

The frozen H16 `sensor_wm` checkpoint drives orient → descend → close to a scored
grasp when MPC plans on its **predicted right-arm + right-hand joint positions**
against **TRAIN demonstration state goals spaced one horizon (16 frames) apart**,
with waypoint progress measured from **observed proprioception**.

## Frozen inputs

- Corpus `data/apple-branches-v1`, manifest SHA-256
  `6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`, unchanged
  splits (410 TRAIN / 147 VAL / 3 TEST episodes).
- Checkpoint `checkpoints/apple-branches-h16-sensor-v1/sensor.pt`, SHA-256
  `0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5`. **No
  retraining.** Its normalization covers exactly the TRAIN split (enforced by
  the evaluator before model use).
- Action schema `ee_delta_grasp_v0`, action manifest `configs/g1_sim_action.json`,
  pinned G1/Dex3 assets; physics, guards and the ordered `AppleToPlateTask`
  scorer are unchanged.

## Model: `state_goal_sensor_wm_v1`

A frozen composition (`src/embodied_jepa/models/state_goal_sensor.py`) wrapping
the unchanged sensor checkpoint through its existing strict loader. `sensor.py`,
`base.py`, the generic CEM planner core, task scoring and physical guards are
not modified. Predictions are byte-identical to the child's.

- Goal fields: 14 named radian positions, resolved by name — right shoulder
  pitch/roll/yaw, elbow, wrist roll/pitch/yaw, and right hand thumb 0/1/2,
  index 0/1, middle 0/1.
- Normalization: the child's TRAIN-population mean/scale for those fields
  (`training_population_moments_sensor_grid_v1`). Nothing is fitted on VAL/TEST
  or on control outcomes.
- Planning cost: `C = mean_f((q̂_f − μ_f)/σ_f − (g_f − μ_f)/σ_f)²` at the H16
  endpoint, where q̂ is the model's predicted normalized position and g the
  demonstration goal.
- Visual endpoint MSE against the goal frame's RGB is computed and logged per
  search round as a **diagnostic only, with fixed weight 0**.
- Progress (`observed_distance`) applies the same normalized metric to the
  **measured** current positions. No object pose, contact or score enters.

## Goals, retrieval and tolerances (TRAIN only)

- Candidate demonstrations: every successful NOMINAL apple/plate TRAIN episode
  (16 on this corpus). The same set is the cross-demonstration reference
  population for tolerances.
- Retrieval, per reset: after reset, the demonstration whose frame-0 RGB has the
  smallest frozen sensor image distance to the live initial RGB; ties go to the
  lexicographically first ID. Retrieval uses the initial RGB only — no state,
  object pose or score. All distances are saved per attempt in `retrieval.json`.
- Goals: measured 14-field positions of the retrieved demonstration at frames
  16, 32, …, 496 and the final frame 501 (32 goals). Frame 0 is excluded.
- Tolerance per goal: the existing TRAIN-only recipe applied to the state metric,
  `max(1.1·max d(s[t±2], s[t]), median positive adjacent-frame distance (floor
  1e-12), 1.1·q90 same-frame distance over references with frame coverage)`.
  Dwell **1** observation.
- Stall rule: if **64** acknowledged commands pass on one goal without
  advancing, the attempt ends with `goal_stall`. This counts as a failure; goals
  are never skipped.
- **No demonstration action proposals.** They were a named scaffold confound,
  and `demo_replay` measures the demonstration's own contribution separately.
  Candidates are hold, warm/search-best and Gaussian perturbations only.

Pre-run calibration, allowed by this protocol and done once on the frozen
checkpoint before any development attempt: all 16 candidate libraries have
32 goals each, and 16–20 of the 32 goals per demonstration fall inside a
neighbour's tolerance under the state metric. This overlap is reported now as a
known scaffold property. It does **not** permit changing tolerances.

## Modes and controls

Run reset-major in this order: `learned`, `dynamics_shuffle`, `persistence`,
`demo_replay`.

- `learned`: CEM ranks projected candidates by the state-goal cost of the frozen
  predictions.
- `dynamics_shuffle`: same forecasts, with the feasible candidate↔cost
  assignment permuted each round (seeded).
- `persistence`: all candidates get the current observed distance; seeded
  unbiased tie selection. The world model is not called.
- `demo_replay`: a **NON-LEARNED reference**. It replays the retrieved
  demonstration's recorded actions open loop, clipped to the same action bounds
  and passed through the same mandatory projection. It never calls the world
  model and stops with `demo_exhausted` after the recorded actions run out.

## Fixed configuration and budget

- Development resets **43000, 43001, 43002, 43003** only; 4 modes = **16
  attempts**. No retries, replacement seeds or reruns. The final cohort
  44000–44019 and TEST episodes stay untouched.
- H16, 16 candidates, 2 CEM rounds, proposal std 0.15 (minimum 0.05),
  commitment 1, initial grasps (−1, −1), existing action bounds (left pose 0,
  left grasp −1, right pose ±0.5, right grasp [−1, 1]).
- At most 1,000 acknowledged commands per attempt, a 5 s observe/plan deadline,
  200 s per attempt, 3,600 s global including preparation and finalization.
  CPU, up to four Torch threads. Existing `min(attempt cap, global remaining)`
  allocation. Unstarted or killed attempts stay in the denominator as failures.

## Gates and interpretation (fixed before outcomes)

"Ordered stages" = the number of leading true stages of the unchanged scorer
(reach, grasp, transport, place, release; 0–5). "Grasp" is the scorer's
contact-lift stage.

- **Primary gate:** `learned` reaches grasp on **≥ 2/4 resets** AND `learned`
  summed ordered stages over the 4 resets are **strictly greater** than both
  `dynamics_shuffle` and `persistence`. The evaluator writes this computation to
  `report.json["state_gate"]`. Missing or failed attempts count as 0 stages:
  only attempts with status `completed`, a termination other than
  `runtime_error`/`deadline_miss`/`attempt_timeout`, and valid provenance are
  scored (`goal_stall`, `step_limit`, `waypoints_complete` and `demo_exhausted`
  are clean terminations whose scored stages count). Any provenance failure makes
  the primary gate false.
- **Secondary (report only):** ≥ 1 full Apple→Plate success in `learned`.
- Falsification readings:
  - `learned` stalls before the close goals on ≥ 3/4 resets → model/planner
    failure under state goals. A "close goal" is one whose preceding 16 recorded
    demonstration actions include a right-grasp command above −1. It is computed
    from the retrieved TRAIN demonstration only (`first_close_goal_index` in the
    calibration and attempt report). "Stalls before" = no scored grasp and
    `last_goal_index < first_close_goal_index` (missing/failed attempts count).
  - `learned` passes descend but has 0 grasps → grasp-precision bottleneck.
    "Passes descend" = the unchanged scorer's `reach` stage on ≥ 1 reset (goal
    advancement is not used: goals 7–12 overlap their neighbours, see R1).
  - `dynamics_shuffle` ≈ `learned`, or `demo_replay` ≥ `learned` with no
    measurable model contribution → the scaffold (goals/tolerances), not the
    world model, explains the result. With integer stage sums, "≈" means
    `learned` is not strictly above `dynamics_shuffle`; "no measurable model
    contribution" means `learned` is not strictly above both controls. This
    reading applies only when `learned` scored ≥ 1 stage.
  - All three readings are written to `report.json["state_gate"]["falsification"]`.
- A pass supports only this development hypothesis. It is not final acceptance
  or reliability. `demo_replay` success is never a learned result.

## Allowed pre-physics steps

Only calibration and software checks. A one-attempt software smoke on a **TRAIN
reset** (not 43000–43003) is allowed to verify runtime and must be recorded.
Recorded smoke: TRAIN reset 42000, source at parent `5d73e56` plus the
uncommitted TASK-043 working tree, `max_steps=40`, scratch output only,
input-hash verification stubbed. `learned` and `demo_replay` each completed 40
commands without error in about 4.1 s and 0.8 s; retrieval selected `apple-42000`.
The smoke is a runtime check, not evidence. Its trajectory is not used to
change any parameter.

## Pre-run revision R1 (independent review, before any development attempt)

An independent software/scientific review of `17a9ac9` ran before any attempt on
43000–43003. It changed **no** gate threshold, reset, mode, budget, tolerance or
controller parameter. Changes:

1. `state_gate` now honours "missing or failed attempts count as 0 stages". The
   original code scored any record carrying a `score`, including child failures,
   killed attempts reconstructed from a partial trace, runtime errors and
   provenance-invalid runs.
2. The falsification readings are computed by the evaluator. Each demonstration
   records `first_close_goal_index`, each state attempt report copies it, and
   "passes descend" and "≈" have operational definitions (see above).

Offline review checks, TRAIN nominal demonstrations only (16 episodes; no
simulator, VAL, TEST or development reset):

- Tolerances: every demonstration has 32 goals, and 16–20 goals per
  demonstration overlap a neighbour. The cross-demonstration q90 term sets 470
  of the 512 thresholds, and the ±2-frame term sets the other 42. Goal-0
  thresholds are 0.0048–0.018, against a frame-0→16 distance of 0.096–0.128
  (4–15%). Overall, threshold ≥ segment length for 38.7% of goals, concentrated
  at goals 7/9–12 just before the first close goal. Goal 13 (frame 208) is the
  first close goal in all 16 demonstrations. `demo_replay` needs no clipping
  (max clip 0).
- Forecast bias: this was checked on all 16 goal-0 windows, not one. From frame
  0, the model predicts **hold** at 0.015–0.024 from goal 0, while the
  demonstration's own 16 actions are predicted at 0.034–0.044. Measured hold
  stays at 0.096–0.128. Over all 496 16-frame goal windows, the demonstration's
  actions are predicted closer than hold in only 64.5%.

Assessment of the declared risks. None is a pre-run blocker, so no tolerance was
changed:
(a) The tight goal-0 tolerance comes from the TRAIN demonstrations' own
same-frame agreement. The demonstrations meet it by construction, and
`demo_replay` measures whether it is achievable. Loosening it now would be
informed by the TRAIN smoke trajectory, which this protocol forbids. Per-step
`observed_distance` traces still show whether a stalled attempt approached its
goal.
(b) This is a property of the frozen model, and retraining it is out of scope. It
makes a goal-0 `goal_stall` under `learned` likely. That outcome would be an
informative model/planner failure, not an artifact of the scaffold.
(c) The primary gate and all readings use scorer stages against controls that
share the scaffold. Goal index is never success evidence.
(d) Declared. `demo_replay` quantifies it.

## Frozen run command

Execute once, from a clean checkout of the reviewed commit:

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_apple.py \
  --dataset data/apple-branches-v1 \
  --checkpoint checkpoints/apple-branches-h16-sensor-v1/sensor.pt \
  --output outputs/apple-state-goal-control-v1 \
  --stage development --seeds 43000 43001 43002 43003 \
  --modes learned dynamics_shuffle persistence demo_replay \
  --goal-kind state --goal-stall-limit 64 --no-proposals \
  --horizon 16 --stride 16 --dwell 1 --candidates 16 --iterations 2 \
  --commitment-steps 1 --max-steps 1000 --attempt-max-seconds 200 \
  --max-seconds 3600 --control-timeout 5
```

The runner snapshots its source, freezes checkpoint/corpus/action/asset hashes in
`plan.json`, writes the TRAIN state-goal library (`state_goals.npz`,
`state_calibration.json`) and their hashes into `resolved_plan.json` before any
attempt, and re-verifies them around every attempt. Any code or input change
invalidates the run. Record every outcome, including failures, timeouts and
stalls, in `apple_state_goal_control_results_v1.md` and a compact manifest under
`benchmarks/manifests/`.

## Known risks declared before outcomes

- Tight early tolerances. The first goal's tolerance is about 0.005, while the
  initial distance is about 0.11 normalized units. The goal may be unreachable
  with ±0.5 per-step pose deltas, which would end the attempt in `goal_stall`.
- Pose forecasts ranked siblings well offline, but a closed loop can compound
  errors. In all 16 goal-0 windows (R1), the demonstration's own first 16 actions
  are predicted *farther* from goal 0 than hold. This is a warning sign, not a gate.
- Arm/hand configuration does not encode the apple's position. A matched pose can
  still miss the apple, and the scorer, not goal completion, defines success.
- Neighbour-overlapping tolerances can let goals advance without real
  progress. `dynamics_shuffle` and `demo_replay` exist to expose this.
