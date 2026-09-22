# Apple hybrid phase control v1 — preregistration

Prospective TASK-046 protocol. It is committed before any development attempt of
this scaffold. Every arm is a **NON-LEARNED diagnostic** under the privileged
exact-rollout ceiling or open-loop demonstration replay. No outcome here is a
learned result, and none counts toward TASK-033/TASK-034.

**Attribution rule (fixed now).** In the hybrid arms, every command after the
handoff is a recorded TRAIN demonstration action replayed open loop. Any grasp,
transport, place, release or success that latches after the handoff is therefore
attributed to the **demonstration's actions**, not to any world model, planner
cost or learned component. Before the handoff, commands come from CEM over exact
MuJoCo rollouts (privileged simulator truth), not from a learned model either.

## Question and hypothesis

TASK-045 ([protocol](apple_trajectory_tracking_v1.md),
[results](apple_trajectory_tracking_results_v1.md)) tracked the retrieved TRAIN
demonstration's right-arm+hand trajectory with exact MuJoCo-rollout dynamics on
development resets 43000–43003:

- it latched `reach` on 4/4 resets but `grasp` on 0/4;
- the apple fell off the table without being lifted on 4/4;
- the measured tracking distance was about 0.012–0.015 (median) before the close
  row and rose to 0.066–0.157 between the close row and the recorded-grasp row.

Open-loop `demo_replay` of the same retrieved demonstrations from reset
(TASK-043, [results](apple_state_goal_control_results_v1.md)) grasped on 4/4 and
succeeded on 3/4 of the same resets.

**Hypothesis (primary).** Privileged trajectory tracking up to the retrieved
demonstration's first close row, then open-loop execution of the demonstration's
own remaining recorded actions from the matched frame, reaches the scorer's
`grasp` stage on **≥ 2/4** of resets 43000–43003.

**Disclosure.** This hypothesis was motivated by TASK-045's development outcomes
on these same four resets (it is that report's recommended next step). This is
therefore development-stage diagnostic evidence on already-seen resets, not a
held-out test. The handoff rules below are fixed from TRAIN data and first
principles; no development trace was inspected to choose them.

## Arms (one command, 12 attempts)

| Arm | Mode | Before handoff | Handoff row | After handoff | Role |
|---|---|---|---|---|---|
| Primary | `privileged_hybrid` | TASK-045 tracker, exact rollouts | first close row | demo actions, open loop | **gate** |
| Reference | `demo_replay` | — | — (from reset) | demo actions, open loop | same-code reference |
| Secondary | `privileged_hybrid_early` | TASK-045 tracker, exact rollouts | close row − 16 | demo actions, open loop | own reading only |

The secondary arm and `demo_replay` never change the primary gate.

## What changes and what does not

Unchanged from TASK-045 (same code, same parameters):

- retrieval rule and TRAIN candidate set (16 successful NOMINAL apple/plate
  TRAIN episodes; nearest frame-0 RGB under the frozen sensor image distance);
- keyframed TRAIN reference (fraction 0.1), cost rule, measured-progress rule,
  stall rule (64 commands, 16 rows), final-row handling;
- the privileged forward model (`privileged_rollout.py`: twin restore, per-step
  re-projection, unchanged `execute`, rejection penalty, runtime parity check);
- CEM (H16, K16, 2 rounds, commitment 1, std 0.15 → min 0.05, reset-seeded RNG),
  action bounds, initial grasps, projection, embodiment, guards, physics and
  the unchanged ordered `AppleToPlateTask` scorer.

Nothing in `sensor.py`, `base.py`, `state_goal_sensor.py`,
`trajectory_tracking.py`, `privileged_rollout.py`, `waypoint_planning.py`, the
scorer or the guards changes. The new code is an isolated wrapper,
`src/embodied_jepa/hybrid_phase.py`, selected only through
`evaluate_apple.py --goal-kind hybrid`. The image, state and trajectory goal
kinds keep their plans, controllers, traces and gates; tests check this.

## Design (fixed before any development attempt)

### Handoff

- `handoff_row = max(0, first_close_reference_index − rows_before_close)`, with
  `rows_before_close = 0` for the primary arm and `16` (= the planning horizon
  H and the tracker's progress window) for the secondary arm.
- `first_close_reference_index` is TASK-045's TRAIN-derived close row: the first
  keyframed reference row at or after the demonstration's first frame produced
  by a recorded right-grasp command above −1.
- At every observation before the handoff, the wrapper evaluates the tracker's
  own measured-progress rule (monotone, windowed, latest minimum; it consumes no
  randomness). If the measured index is **≥ `handoff_row`**, planning stops for
  the rest of the attempt. Otherwise the unchanged tracker plans and the
  command it returns is executed. Pre-handoff commands are therefore identical
  to the unwrapped tracker's commands (tested).
- The handoff check runs before the tracker's own termination checks, so a
  reset whose measured index reaches the handoff row always hands off.
- Scorer truth, object pose and TRAIN scorer labels never drive the handoff.

### Replay after the handoff

- With `t` the measured index at the handoff and `f = frames[t]` its original
  demonstration frame, the wrapper replays recorded actions `f, f+1, …, T−1`
  (action `i` turns demonstration frame `i` into `i+1`), one per command.
- Each command goes through the same bounds clip and mandatory candidate
  projection as `demo_replay`; an infeasible projection is a runtime error.
- No model, rollout or cost is evaluated after the handoff.
- The attempt ends with `demo_exhausted` after the last action, with `success`
  if the scorer's success latches earlier, or with `step_limit` at 1,000 total
  commands (tracked + replayed).

### TRAIN-only justification of the two handoff rows

From the 16 TRAIN references built by the unchanged TASK-045 preparation
(no development data):

- The first closing frame is **211** in all 16 demonstrations; the close row is
  141 (2 demos) or 142 (14 demos), and its frame is exactly 211.
- The row before the close row is at frames 195–206. Frames about 144–210 are
  the demonstration's blocked descent: the arm commands continue while the 14
  tracked positions barely move, so keyframing removed most of that span.
- Row `close − 16` is at frame **136–137**, before that descent.

Primary (close row): this is the hypothesis as stated. The tracker has followed
the approach and the first closing motion to frame 211; replay then starts at
action 211 or later, so the replayed segment is the demonstration's own
closing, lifting, transport, placing and release.

Secondary (close row − 16), a first-principles alternative declared now:

1. **The planner never optimizes against a closing row.** The tracker's H16
   cost at index `t` targets rows `t+1 … t+16`. Every command executed before
   the handoff was planned at an index `< close − 16`, so all of its targets were
   `< close`. No executed tracked command was chosen to follow closing motion.
2. **The replay restores the demonstration's descent timing.** Replay starts
   at frame 136–137, so it includes the whole blocked descent (≈ frames
   144–210) that keyframing removed from the tracker, and all closing actions.

The value 16 is the existing horizon/window; it was not tuned, and no other
offset is run.

## Frozen inputs (unchanged from TASK-043/044/045)

- Corpus `data/apple-branches-v1`, manifest SHA-256
  `6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`.
- Checkpoint `checkpoints/apple-branches-h16-sensor-v1/sensor.pt`, SHA-256
  `0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5`. It is
  used only for retrieval, TRAIN normalization and the tracking metric; its
  dynamics are never used.
- Action manifest `configs/g1_sim_action.json` and the pinned G1/Dex3 assets.
- The preparation regenerates `state_goals.npz`, `state_calibration.json`,
  `tracking_references.npz` and `tracking_calibration.json`. They are expected
  to be byte-identical to TASK-045 (`839190fc…f88d`, `d8dae051…8d75`,
  `48418ea3…18ca`, `68b408d0…aabf`); they are hashed into `resolved_plan.json`
  and re-verified around every attempt.

## Budget

- One command, **12 attempts in mode-major order**: `privileged_hybrid` on
  43000–43003, then `demo_replay` on 43000–43003, then
  `privileged_hybrid_early` on 43000–43003. No retries, replacement seeds or
  reruns.
- H16, K16, 2 rounds, commitment 1, at most 1,000 commands, 5 s per-command
  observe/plan deadline, **840 s per attempt**, **3,600 s global** including
  preparation and finalization, with the existing
  `min(attempt cap, global remaining)` allocation. CPU, single-threaded MuJoCo,
  Torch capped at 4 threads.
- **Primary protection.** Mode-major order runs all four primary attempts
  first: 4 × 840 s = 3,360 s, so the primary arm cannot be shortened by the
  global cap unless preparation exceeds about 230 s (TASK-045 preparation took
  about 10 s).
- **Expected time (from TASK-045 timings, used only for budgeting).** The TASK-045
  tracker reached the close row after 222–249 commands at about 1 s each;
  replay commands take milliseconds (TASK-043 `demo_replay`: about 7.4 s per
  501-command attempt). Expected totals are roughly 250–300 s per primary
  attempt, under 10 s per `demo_replay` attempt, and 200–260 s per secondary
  attempt, about 2,100 s overall.
- **Declared risk.** The secondary arm runs last and receives whatever global
  budget remains. An attempt that is not started, globally shortened or timed
  out counts as 0 stages even if a grasp had latched (its latched stages are
  reported as uncounted), and makes that arm's reading inconclusive.

## Primary gate (`report.json["hybrid_ceiling_gate"]`)

- `privileged_hybrid` reaches the scorer's **grasp** stage on **≥ 2/4** resets;
- AND zero rollout parity mismatches occur in its counted attempts;
- AND provenance is valid.

Only attempts with status `completed`, a termination other than
`runtime_error`/`deadline_miss`/`attempt_timeout`, and valid provenance are
scored. `demo_exhausted`, `success`, `reference_stall` and `step_limit` are clean
terminations whose scored stages count. Missing or failed attempts count as 0.

Readings are computed from counted attempts only, and asserted only when all 4
primary attempts count with valid provenance and exact rollouts; otherwise the
primary result is **inconclusive**:

- `tracking_failed_before_handoff`: the gate failed and on ≥ 3/4 resets the
  attempt ended without a handoff and without a grasp.
- `replay_from_tracked_state_insufficient_for_grasp`: the gate failed and on
  ≥ 3/4 resets the attempt handed off, replayed, and did not grasp.

## Secondary arm reading (`readings.handoff_comparison`)

Asserted only when both hybrid arms are complete (all 4 attempts counted,
provenance valid, rollouts exact); otherwise "inconclusive". It never changes the
primary gate. With "grasp" meaning grasp on ≥ 2/4:

| Primary | Secondary | Reading |
|---|---|---|
| grasp | grasp | handoff timing within the last pre-close horizon is not critical |
| no | grasp | tracking over the last pre-close horizon (or skipping the dead-time descent) is associated with losing the grasp |
| grasp | no | the longer open-loop segment from the earlier tracked state is associated with losing the grasp |
| no | no | open-loop demonstration actions from a tracked approach state do not reproduce the grasp at either handoff |

These are associations over n = 4; the two arms differ both in the tracked
segment and in the replayed segment, so neither row isolates one mechanism.

## Reference arm: `demo_replay`

The same-code open-loop replay from reset, identical to TASK-043's mode (same
retrieval, same actions, same bounds and projection). TASK-043 recorded, on
43000–43003: grasp 4/4, success 3/4 (43002: 3 ordered stages,
`demo_exhausted`). The rerun is cheap (about 30 s in total) and checks that the
reference still holds under the current code. It is reported per reset, and
any difference from TASK-043 is reported as a parity finding. It never enters
the primary gate. If it is incomplete or differs, the TASK-043 values are cited
with that caveat.

## Diagnostics (not gate evidence)

- Per reset and arm: retrieved demonstration, termination, handoff row, measured
  handoff row and frame, handoff command, tracked and replayed commands,
  ordered stages, grasp/success, parity counts, wall time.
- Pre-handoff parity with TASK-045: the primary and secondary arms run the same
  tracker code with the same seed. Their pre-handoff commands are expected to
  equal TASK-045's first commands on the same reset. Any divergence (for
  example from nondeterminism) is reported; it does not invalidate the run.
- Scorer stage timing relative to the handoff, apple height and `dropped`.

## Interpretation (fixed now)

- **Pass (conclusive).** Under exact dynamics, a tracked approach to the close
  row followed by the demonstration's own closing actions reaches grasp. The
  grasp is attributed to the **replayed demonstration actions**, not to a model.
  Together with TASK-045 this is consistent with closed-loop replanning during
  contact (or its timing) breaking the grasp that the demonstration achieves;
  it does not isolate that mechanism. **Next learned replacement:** keep the
  replayed close segment fixed and replace only the **privileged forward model
  of the approach phase** with the learned `state_goal_sensor_wm_v1` dynamics
  (learned tracking to the same handoff, with `dynamics_shuffle`/`persistence`
  controls), in its own preregistered task. Only after that should a learned
  component (for example an object-aware cost or learned closing) replace the
  open-loop close segment, because replay is demonstration memory, not a model.
- **Fail with `replay_from_tracked_state_insufficient_for_grasp`.** Open-loop
  demonstration closing from the tracked approach state does not reproduce the
  grasp, although the same actions from reset do (reference arm). The tracked
  approach state differs from the demonstration's in a way that matters (arm
  pose relative to the apple, hand state or skipped descent timing; not
  isolated). Do not pair this scaffold with learned dynamics. Next: an
  object-aware approach/close cost under the ceiling (TASK-045 candidate 2).
- **Fail with `tracking_failed_before_handoff`.** This would contradict TASK-045,
  which passed the close row on 4/4; treat it as an implementation or
  determinism finding and investigate before any new design.
- **Inconclusive.** Report it; no design verdict.
- In every case, learned Apple→Plate remains at zero successes, TASK-033/034 stay
  open, and the final cohort 44000–44019 and TEST stay untouched.

## Allowed pre-physics steps and recorded smoke

Software checks only, plus one software smoke on **TRAIN reset 42000** (not a
development reset), recorded below after it runs. No parameter may change
after the smoke; the handoff rows, budgets, gates and readings above are fixed
before it runs.

## Frozen run command

Execute **once**, from a clean checkout of the reviewed commit, into a directory
that does not exist yet:

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_apple.py \
  --dataset data/apple-branches-v1 \
  --checkpoint checkpoints/apple-branches-h16-sensor-v1/sensor.pt \
  --output outputs/apple-hybrid-phase-ceiling-v1 \
  --stage development --seeds 43000 43001 43002 43003 \
  --modes privileged_hybrid demo_replay privileged_hybrid_early \
  --goal-kind hybrid --goal-stall-limit 64 --no-proposals \
  --horizon 16 --stride 16 --dwell 1 --candidates 16 --iterations 2 \
  --commitment-steps 1 --max-steps 1000 --attempt-max-seconds 840 \
  --max-seconds 3600 --control-timeout 5
```

`--stride 16` is the stall rule's minimum advance and the retrieval library's
goal spacing; `--dwell 1` only rebuilds the identical retrieval library. Every
outcome, including failures, stalls and timeouts, will be recorded in
`apple_hybrid_phase_results_v1.md` and
`benchmarks/manifests/apple-hybrid-phase-v1.json`.

## Known risks declared before outcomes

- **Arm pose is not object pose.** The tracked approach carries no apple
  information; the reset jitter is ±6 mm. Replayed actions are open loop.
- **Handoff state mismatch.** At the handoff the robot state is only the nearest
  reference row within the window, not the demonstration state; the replayed
  delta actions start from wherever the tracker left the arm and hand.
- **Primary skips the descent.** The primary replay starts at frame ≥ 211 and so
  does not replay the demonstration's blocked descent (≈ frames 144–210); the
  secondary arm does.
- **Contact before the handoff.** If the tracker disturbs the apple before the
  measured index reaches the handoff row, replay starts from a disturbed scene.
- **Short run.** n = 4 development resets per arm.
