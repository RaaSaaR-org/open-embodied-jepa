# Apple privileged simulator-rollout planning ceiling v1 — preregistration

Prospective TASK-044 protocol. It was committed before any development attempt of
this mode. It measures a **NON-LEARNED privileged diagnostic ceiling**. Its
outcomes are never learned results, never count toward TASK-033/TASK-034, and
never enter a learned-result report.

## Question

TASK-043 ([protocol](apple_state_goal_control_v1.md),
[results](apple_state_goal_control_results_v1.md)) found that learned
demonstration-state-goal MPC with the frozen H16 `sensor_wm` stalls on goal 0 on
4/4 development resets. So do `dynamics_shuffle` and `persistence`. Learned
predicted costs (median 0.044–0.069) were far below the measured distances
(median 0.126–0.228). The non-learned open-loop `demo_replay` succeeded on 3/4
resets.

With **perfect dynamics**, can the **same** state-goal scaffold and the same CEM
planner progress through the scorer's grasp stage? The answer separates
learned-model error from goal/planner design.

## Why simulator truth is permitted here, and only here

AGENTS.md forbids simulator truth in planning cost. This mode breaks that rule
on purpose, and only because it is a declared diagnostic ceiling. It asks what the
planner could achieve if the forward model were exact. Isolation:

- The implementation is `src/embodied_jepa/privileged_rollout.py`. It is **not
  registered** in `MODELS`, so no model config can select it. Construction
  requires an explicit `acknowledge_privileged_ceiling=True`.
- The evaluator builds it only for `--modes privileged_rollout`. `make_plan`
  rejects that mode when it is combined with any other mode, used at the final
  stage, or used without `--goal-kind state`. A worker refuses to run it unless
  the plan declares `privileged_ceiling: true`.
- A ceiling report carries `ceiling_gate` and **no** `state_gate`. The learned
  `state_gate` ignores `privileged_rollout` records.
- No learned path changes: `sensor.py`, `base.py`, `state_goal_sensor.py`, the
  `WaypointController`/CEM core and its defaults, the scorer, the embodiment,
  its guards, and the behaviour of every existing mode are untouched. Tests
  cover this isolation.

## Ceiling design

Everything below is identical to the TASK-043 `learned` mode except the forward
model.

- **Goals.** For each reset, the TRAIN demonstration is retrieved by initial RGB
  with the frozen sensor image distance, exactly as in TASK-043. Its
  right-arm+hand qpos goals are taken every 16 frames plus the final frame (32
  goals). The same TRAIN-only tolerances apply, dwell is 1, and 64 acknowledged
  commands on one goal without advancing ends the attempt with `goal_stall` (a
  failure). There are no demonstration proposals and no goal skipping.
- **Planner.** The unchanged `WaypointController` runs with ablation `learned`,
  H16, 16 candidates, 2 CEM rounds, proposal std 0.15 (min 0.05), commitment 1,
  initial grasps (−1, −1), the same action bounds, the mandatory live
  projection, and the same RNG seeding (the reset seed).
- **Candidate scoring (privileged).**
  1. At each search, the full live MuJoCo state (`mj_copyData`, joint targets,
     plate body pose) is saved. It must equal the planning observation exactly,
     or the attempt errors.
  2. For each candidate, the saved state is restored into a separate
     non-rendering twin simulator: same scene XML, model sizes, timestep,
     substeps, action manifest and state schema.
  3. The candidate's 16 projected actions are executed step by step through the
     twin's own mandatory `project_candidates` and the unchanged
     `G1Embodiment.execute`, with no rendering. This is exactly what the live
     loop does with each command.
  4. The cost is the frozen state-goal model's own `observed_distance` from the
     reached state to the goal, at the H16 endpoint. This is the same
     TRAIN-normalized 14-field metric that measures waypoint progress.
  5. If the twin rejects a step (projection infeasible, a guard, or a MuJoCo
     instability), that candidate stops. Its rejected steps cost
     `1000 + distance of the last state it actually reached`, so it ranks after
     every executable candidate.

  The live simulator is never modified by planning.
- **Runtime conformance.** At every search, the evaluator checks that the
  measured state equals one of the previous search's simulated first steps
  exactly (`previous_search_first_step_exact_match`). This is logged per
  command.
- **Diagnostics** logged per command: rejected candidates and their reasons,
  steps changed by per-step re-projection, and rollout seconds.

## Frozen inputs

These are unchanged from TASK-043:

- Corpus `data/apple-branches-v1`, manifest SHA-256
  `6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`.
- Checkpoint `checkpoints/apple-branches-h16-sensor-v1/sensor.pt`, SHA-256
  `0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5`. It is used
  only for retrieval, TRAIN normalization and the state metric. Its dynamics are
  **not** used.
- Action manifest `configs/g1_sim_action.json`, pinned G1/Dex3 assets, and the
  unchanged `AppleToPlateTask` scorer.

## Budget

The planning budget matches TASK-043: H16, K16, 2 rounds, commitment 1, at most
1,000 acknowledged commands, the 64-command stall bound, and the 5 s
per-command observe/plan deadline.

The wall caps differ, because exact rollouts are slower than the learned
forward pass. Each command simulates 2 × 16 × 16 = 512 control steps. In the
recorded TRAIN-reset smoke below, planning took a median of 0.96 s per command,
against the learned mode's 0.074–0.080 s median in TASK-043.

**The ceiling cannot meet the learned mode's per-command time.** It remains well
under the shared 5 s deadline.

Declared caps:

- **840 s per attempt** and **3,600 s global**, including preparation and
  finalization.
- 4 attempts on CPU (single-threaded MuJoCo, Torch capped at 4 threads).
- The existing `min(attempt cap, global remaining)` allocation applies.

At about 1 s per command, an attempt that never stalls reaches roughly 800
commands before its cap, not 1,000. In TASK-043, `demo_replay` first scored
grasp at command 268–272, so this cap is not expected to bind before a grasp
under normal progress. An attempt that ends by `attempt_timeout` still counts
as a failure (0 stages) for the gate. Its latched stages are reported
separately as `reported_ordered_stages_uncounted`.

## Design and run

- Development resets **43000, 43001, 43002, 43003** only, mode
  `privileged_rollout`: **4 attempts**.
- No retries, replacement seeds or reruns.
- The final cohort 44000–44019 and TEST are untouched.

## Gate and interpretation (fixed before outcomes)

"Ordered stages" means the leading true stages of the unchanged scorer (reach,
grasp, transport, place, release). The evaluator writes
`report.json["ceiling_gate"]`.

Only attempts that meet all of the following are scored:

- status `completed`;
- termination not `runtime_error`, `deadline_miss` or `attempt_timeout`;
- valid provenance.

Missing, failed or killed attempts count as 0 stages. Any provenance failure
fails the gate.

- **Primary gate:** the ceiling reaches the scorer's **grasp** stage on
  **≥ 2/4 resets**.
- **Reported per reset:** retrieved demo, termination, commands, last goal
  index, first close-goal index, ordered stages, grasp and success.
- **Interpretation:**
  - *Pass.* The goal/planner design is adequate under perfect dynamics, so the
    learned dynamics are the bottleneck. Recommended next step: retrain the
    dynamics with a multi-step proprioceptive rollout loss and/or grasp-phase
    data, under a new preregistration that includes a closed-loop forecast-bias
    check.
  - *Fail.* The planner/goal design must change before more model training.
    Two sub-readings are computed:
    - `state_goal_tracking_inadequate`: the gate failed and the ceiling stalled
      before the close goal on ≥ 3/4 resets. Tolerances, the stall bound or the
      CEM budget cannot track the state goals even with exact dynamics.
    - `arm_pose_goals_insufficient_for_grasp`: the gate failed, the ceiling
      reached or passed the close goal, and it did not grasp on ≥ 3/4 resets.
      Arm+hand pose goals lack the object information needed to grasp.
- The ceiling is **never** compared to learned results as if it were one. A pass
  is not evidence of learned manipulation.

## Allowed pre-physics steps and recorded smoke

Only software checks are allowed. A software smoke ran on **TRAIN reset 42000**,
not on a development reset:

- Scratch output only, input-hash verification stubbed, `max_steps` 64.
- Source: parent `cb81c22` plus the uncommitted TASK-044 working tree.
- Retrieval selected `apple-42000`.
- Planning took a median of 0.96 s per command (max 0.99 s).
- Runtime parity matched exactly on 63/63 checked commands.
- The goal-0 observed distance fell from 0.113 to a minimum of 0.0093, against
  a goal-0 tolerance of 0.0052. It did not advance within 64 commands.

A first smoke build executed each candidate's projected sequence without per-step
re-projection. Its later rollout steps were often rejected by the embodiment's
joint-rate guard (279 rejections in 40 commands), because a whole-sequence
projection assumes perfect tracking. The live loop re-projects every command, so
the rollout was corrected to do the same. After the fix, 0 candidates were
rejected.

This is a fidelity fix to the forward model, disclosed here. It is not a change
to goals, tolerances, the planner or its budget. No scaffold parameter was
changed because of either smoke. The smokes are runtime checks, not evidence.

## Frozen run command

Execute once, from a clean checkout of the reviewed commit, into a directory
that does not yet exist:

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_apple.py \
  --dataset data/apple-branches-v1 \
  --checkpoint checkpoints/apple-branches-h16-sensor-v1/sensor.pt \
  --output outputs/apple-privileged-ceiling-v1 \
  --stage development --seeds 43000 43001 43002 43003 \
  --modes privileged_rollout \
  --goal-kind state --goal-stall-limit 64 --no-proposals \
  --horizon 16 --stride 16 --dwell 1 --candidates 16 --iterations 2 \
  --commitment-steps 1 --max-steps 1000 --attempt-max-seconds 840 \
  --max-seconds 3600 --control-timeout 5
```

The runner snapshots its source (including `privileged_rollout.py`) and freezes
the input hashes and the TRAIN state-goal library before any attempt. It
re-verifies them around every attempt.

Every outcome will be recorded in `apple_privileged_ceiling_results_v1.md` and
`benchmarks/manifests/apple-privileged-ceiling-v1.json`, including failures,
stalls and timeouts.

## Known risks declared before outcomes

- **Tight goal-0 tolerances.** They are 0.005–0.018. The TRAIN smoke came within
  1.8× of its tolerance but did not reach it. A goal-0 stall under perfect
  dynamics is plausible. That outcome would be informative: the scaffold's
  tolerance/stall design, not the model, blocks progress.
- **Endpoint-only cost.** CEM scores only the H16 endpoint. With exact dynamics
  it can still pick a sequence whose first step is poor. Only the first step is
  executed before replanning, as in the learned mode.
- **Arm pose is not object pose.** Matching the demonstration's arm/hand qpos
  can still miss the apple. Grasp is judged only by the scorer.
- **Short run.** n = 4 development resets.
