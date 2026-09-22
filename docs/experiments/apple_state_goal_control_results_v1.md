# Demonstration-state-goal apple control v1: negative development result

**The preregistered primary gate failed.** `learned` reached grasp on **0/4**
resets, and its summed ordered stages (0) were not above `dynamics_shuffle` (0)
or `persistence` (0). All 12 attempts of those three modes stopped with
`goal_stall` on **goal 0** after exactly 64 acknowledged commands. Every scorer
stage was false. The measured right-arm+hand state never came within the goal-0
tolerance. The secondary outcome, at least one full learned Apple→Plate success,
was not met (0/4). This is not working learned manipulation.

The **non-learned** `demo_replay` reference replayed the retrieved TRAIN
demonstration's recorded actions open loop. It completed the unchanged scorer on
**3/4** resets and reached grasp + transport on the fourth. That is a scripted
reference, not a learned result. It shows that the retrieved goals are physically
achievable from these resets, and that the learned controller fails far earlier
than the scaffold's limit.

## Frozen execution

The run executed once from a clean checkout of the reviewed source
`a90bf5b48cb3326935d2222d4074607df1b0c33b`, using the
[preregistered protocol](apple_state_goal_control_v1.md) including revision R1
and its exact frozen command. There were no retries, replacement seeds, parameter
changes, or final-cohort (44000–44019) or TEST execution. All 16 planned attempts
started and completed. The evaluator exited 0 with report status `completed` and
provenance valid. Global wall time was **129.985 s** of the 3,600 s budget. Each
attempt took 6.8–7.4 s of its 200 s allocation, so no attempt was shortened or
timed out.

Frozen inputs, unchanged and re-hashed after the run:

- Checkpoint `0192b99a…a5a5`.
- Corpus manifest `6e9a5bcb…0331`.
- Action manifest `f247effe…38b1`.
- Asset manifest `2421e194…e993`.
- The TRAIN state-goal library `state_goals.npz` `839190fc…f88d` and
  `state_calibration.json` `d8dae051…8d75`. Both were written before any attempt,
  and the runner verified them around every attempt.
- `plan.json` `fc515aa7…d1`, `resolved_plan.json` `b0cf4205…af`, and
  `report.json` `42e91b61…55ee`.

The [compact manifest](../../benchmarks/manifests/apple-state-goal-control-v1.json)
records full hashes, per-attempt scores and SHA-256 of 70 core artifacts under
`outputs/apple-state-goal-control-v1/`: attempt reports, traces, retrievals and
start records, plus plan, report, goals and calibration. Logs and the source
snapshot are excluded.

## Every attempt

Retrieval selected a TRAIN demonstration from the initial RGB alone. The selected
frame-0 image distances were 0.00012–0.0013. Runner-ups were 0.0010–0.0028, so
the margins are narrow at 43000 and 43002 (runner-up within 1.3×). All selected
demonstrations have 32 goals, and the first close goal is index 13 (frame 224;
goals are 0-based from frame 16. The protocol's R1 note says "frame 208", which
is an off-by-one in that note only). "Stages" counts leading true scorer stages (reach,
grasp, transport, place, release).

| Reset | Retrieved demo | Mode | Commands | Last goal | Goal-0 tol. | Measured dist. start → min → stall | Stages | Termination | Wall (s) |
|---|---|---|---:|---:|---:|---|---:|---|---:|
| 43000 | apple-42029 | learned | 64 | 0 | 0.0166 | 0.101 → 0.098 → 0.167 | 0 | goal_stall | 7.18 |
| 43000 | apple-42029 | dynamics_shuffle | 64 | 0 | 0.0166 | 0.101 → 0.097 → 0.309 | 0 | goal_stall | 7.25 |
| 43000 | apple-42029 | persistence | 64 | 0 | 0.0166 | 0.101 → 0.101 → 0.245 | 0 | goal_stall | 6.89 |
| 43000 | apple-42029 | demo_replay | 501 | — | — | — | 5 (success) | success | 7.36 |
| 43001 | apple-42025 | learned | 64 | 0 | 0.0058 | 0.113 → 0.098 → 0.198 | 0 | goal_stall | 7.13 |
| 43001 | apple-42025 | dynamics_shuffle | 64 | 0 | 0.0058 | 0.113 → 0.102 → 0.320 | 0 | goal_stall | 7.12 |
| 43001 | apple-42025 | persistence | 64 | 0 | 0.0058 | 0.113 → 0.079 → 0.797 | 0 | goal_stall | 6.82 |
| 43001 | apple-42025 | demo_replay | 501 | — | — | — | 5 (success) | success | 7.34 |
| 43002 | apple-42008 | learned | 64 | 0 | 0.0183 | 0.124 → 0.115 → 0.363 | 0 | goal_stall | 7.14 |
| 43002 | apple-42008 | dynamics_shuffle | 64 | 0 | 0.0183 | 0.124 → 0.124 → 0.468 | 0 | goal_stall | 7.10 |
| 43002 | apple-42008 | persistence | 64 | 0 | 0.0183 | 0.124 → 0.124 → 0.810 | 0 | goal_stall | 6.81 |
| 43002 | apple-42008 | demo_replay | 501 | — | — | — | 3 | demo_exhausted | 7.35 |
| 43003 | apple-42021 | learned | 64 | 0 | 0.0124 | 0.101 → 0.092 → 0.253 | 0 | goal_stall | 7.10 |
| 43003 | apple-42021 | dynamics_shuffle | 64 | 0 | 0.0124 | 0.101 → 0.057 → 0.290 | 0 | goal_stall | 7.23 |
| 43003 | apple-42021 | persistence | 64 | 0 | 0.0124 | 0.101 → 0.083 → 0.240 | 0 | goal_stall | 6.82 |
| 43003 | apple-42021 | demo_replay | 501 | — | — | — | 5 (success) | success | 7.36 |

At 43002, `demo_replay` reached reach, grasp and transport. The apple ended
0.021 m from the plate with no support dwell when the 501 recorded actions ran
out (`demo_exhausted`), so place and release stayed false. No attempt had a
runtime error, deadline miss, timeout, drop or provenance failure.

## Gate, secondary outcome and falsification readings

From `report.json["state_gate"]`, with 16/16 attempts counted:

- **Primary gate: failed.** Learned grasp resets = 0 (needed ≥ 2). Summed ordered
  stages: learned 0, dynamics_shuffle 0, persistence 0, demo_replay 18 (a
  non-learned reference, not part of the gate).
- **Secondary:** learned full successes = 0.
- **Model/planner failure under state goals: triggered.** `learned` stalled
  before the close goal (index 13) on **4/4** resets without a grasp; the
  threshold is ≥ 3/4. It never left goal 0.
- **Grasp-precision bottleneck: not triggered.** The scorer's reach stage was
  reached on 0/4 learned resets.
- **Scaffold explains the result: not applicable (false).** The reading applies
  only when learned scored ≥ 1 stage. `demo_replay` (18 stages) shows that the
  goals and tolerances are reachable when the demonstration's actions are
  executed.

## Diagnostics (not gate evidence)

- **Forecast bias.** The model predicted endpoint costs far below the measured
  distances it produced. Median selected cost for learned was 0.044–0.069,
  against median measured distance 0.126–0.228. As an approximate check, the
  measured distance 16 commands later exceeded the cost selected at step k in
  84–96% of learned decisions. This repeats R1's offline warning: hold and small
  motions are predicted to approach goal 0 while the measured state does not.
- **Distances grew in every mode.** Learned ended closer to goal 0 than
  `dynamics_shuffle` on 4/4 resets and closer than `persistence` on 3/4 resets.
  That ordering reflects model ranking, not task progress. No closed-loop mode
  came within 3× of any goal-0 tolerance. The closest point was 0.057 for
  shuffle at 43003, against a tolerance of 0.012.
- **Search.** Learned and shuffle used 64 searches × 2 rounds × 16 candidates =
  2,048 model candidate evaluations per attempt. Persistence scored the same
  number of candidate slots with the constant observed distance and made no
  model calls. Learned round cost spreads were non-degenerate (median
  0.016–0.028). Median planning time was 73.6–79.9 ms, with a maximum of 99.3 ms,
  well under the 5 s deadline. The per-round visual diagnostic was logged at
  weight 0 and did not affect selection. The evaluator does not record worker
  CPU seconds.
- **Actions.** Mandatory projection changed 17–29 of 64 planned commands per
  closed-loop attempt, which is expected feasibility projection. All applied
  actions matched the projected commands. `demo_replay` needed no clipping or
  projection change on any of its 2,004 commands.

## Interpretation and limits

This is n=4 development evidence under one frozen configuration. It refutes the
TASK-043 hypothesis as stated: planning on the frozen H16 model's predicted
right-arm+hand qpos against TRAIN state goals does not even reach the first
16-frame goal. The mechanism matches the declared risk (b): the frozen forecasts
are biased toward "stay near the start", so CEM chooses actions whose measured
effect moves away from the goal. The tight goal-0 tolerances (risk a) play a role
too, but no mode got within 3× of any tolerance. Loosening them by less than 3×
would not have advanced goal 0, and no loosening was done.

Open-loop replay of the nearest TRAIN demonstration succeeds on 3/4 of these
development resets. So these resets are close to the TRAIN distribution, and a
successful learned controller on them would not by itself show generalization.
Any future physical claim must beat `demo_replay` or use resets where replay
fails.

TASK-033/TASK-034 physical acceptance stays open. Learned Apple→Plate remains at
zero successes. The final cohort and TEST were not touched.

## Recommended next step (not started)

The failure is in the model's action-conditioned state forecasts, not in the
goal scaffold. The next step should therefore address the dynamics or bound what
planning could achieve with correct dynamics. It should not retune costs or
tolerances. Two fallbacks were recorded:

1. **Privileged MuJoCo-rollout planning ceiling (recommended first).** Run the
   same CEM/state-goal scaffold with simulator rollouts as the forward model, on
   the same four development resets and budgets, labelled as privileged.
   - If it passes, the planner/scaffold is sufficient and the learned dynamics
     are the bottleneck.
   - If it also fails, the cost/goal design needs work before more training.

   It is cheap and needs no new data.
2. **Grasp-phase data plus retraining with object-relative palm offsets.** Only
   if the ceiling passes: collect grasp-phase branch data and retrain with
   object-relative palm targets. The aim is to remove the hold-bias and add the
   object information the arm-only state lacks. This needs a new preregistration,
   including a closed-loop forecast-bias check before physics.
