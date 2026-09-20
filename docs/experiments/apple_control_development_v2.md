# Apple sensor-world-model control v2 — conditional preregistration

Prospective TASK037 follow-up, written before the H16-trained model's causal
assessment or any v2 control rollout. The previous control diagnostic remains
0/6 placements, and the H8-trained branch model's primary H16 causal gate remains
failed. This protocol does not authorize a rollout by itself. Execute only after
the coordinator confirms that the new, already selected TASK037 checkpoint passes
the **unchanged primary VAL H16 causal gate** and separately authorizes this run.
A failed, incomplete or integrity-invalid diagnostic means no v2 rollout.

## Conditional gate and frozen inputs

Use the sealed `data/apple-branches-v1` corpus, manifest SHA-256
`6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`, with its unchanged
parent/session partitions. The candidate checkpoint is the single selected
`checkpoints/apple-branches-h16-sensor-v1/sensor.pt` from
[apple_branch_training_h16_v1.md](apple_branch_training_h16_v1.md). Selection is
by the registered normalized H16 validation prediction MSE, not by control or
matched-diagnostic outcomes. No alternate checkpoint may be substituted to pass.

Apply [apple_branch_diagnostics_v1.md](apple_branch_diagnostics_v1.md) unchanged:
at least six informative VAL roots across two parent sessions, root-mean ranking
accuracy >= 0.70, raw visual endpoint error reduction R >= 0.10, and root-bootstrap
95% lower bound of R strictly above zero. Keep the same pair screen, tie rule,
2,000 bootstrap samples and seed 20260921. H8/TRAIN results and parent-session
intervals remain secondary/sensitivity evidence; they cannot replace the gate.
A pass permits this limited development experiment, not a manipulation claim.

Before any physics, commit this protocol and record the passing diagnostic's
registration/report hashes, selected checkpoint SHA-256, source revision, exact
command and this protocol's SHA-256 in the coordinator's launch record. The
existing runner then freezes dataset/checkpoint/action/assets/source identities
in `plan.json`, snapshots its runtime, and saves TRAIN calibration, image/proposal
arrays and their hashes in `resolved_plan.json` before starting any attempt.
The passing checkpoint hash must match the runner's registered checkpoint hash.
Do not modify runtime, checkpoint, corpus, protocol or resolved calibration during
execution. Preserve all earlier outputs; refuse an existing v2 output directory.

## Fixed six-attempt comparison

Use reset seeds **43000 and 43001**, each with modes **learned, persistence,
dynamics_shuffle**, in that order: six intended attempts, with no retries,
replacement seeds, extensions, outcome-based early cancellation or reruns.

- Existing fixed-base G1/Dex3 MuJoCo physics, onboard 96×96 RGB, 14D action schema,
  simulation-only conversion and ordered `AppleToPlateTask` scorer stay unchanged.
- Apple center (0.34, −0.18), plate center (0.49, −0.09), independent uniform
  ±0.006 m XY jitter. Use the reset-seeded RNG, drawing apple first and plate second,
  exactly as control v1. Identical reset coordinates apply across modes per seed.
- Shooting MPC: **H16**, **16 candidates**, **2 search rounds**, proposal standard
  deviation **0.15**, minimum **0.05**, and initial grasps (−1, −1). Preserve the
  existing seeded proposal/tie/shuffle RNG behavior and warm-start handling.
- Left pose deltas remain zero, left grasp −1; right pose deltas remain bounded
  to [−0.5, 0.5], right absolute grasp to [−1, 1]. Candidate projection is mandatory.
  Every feasible candidate is rescored; no unscored demonstration fallback exists.
- At most **1,000 acknowledged commands per attempt**. The **5-second** observe/
  plan deadline remains: stop before sending a stale command on deadline failure.
- **600 true-wall seconds total**, including setup, TRAIN calibration, checkpoint
  loading, rendering, all attempts and finalization. The existing supervisor
  reserves five seconds for final accounting. CPU, up to four Torch threads.

Attempt order is fixed even if the longer horizon makes the budget insufficient.
Every planned attempt remains in the denominator, including rejected commands,
execution errors, deadline misses, killed or unstarted attempts. An incomplete
comparison is negative/inconclusive evidence and does not authorize an extension.

## TRAIN image goals and explicit action proposals

Select the lexicographically first successful **nominal TRAIN apple/plate**
demonstration. On the frozen corpus this is `apple-42000`, with 502 sensor frames
and 501 transitions. Use every **28th frame**, always including the final frame:
0, 28, 56, 84, 112, 140, 168, 196, 224, 252, 280, 308, 336, 364, 392, 420, 448, 476,
501. These 19 RGB goals are shared by all modes and resets. No goal joint position,
object pose, collector phase or task score enters planning or waypoint advancement.
Advance only after **three consecutive observed-image matches**; there is no
elapsed-time phase switch. Goal-sequence completion is not physical success.

Keep the existing TRAIN-only threshold rule, evaluated once with this checkpoint
before physics: maximum of 1.1 times the largest selected-demo ±2-frame image
distance, the median positive adjacent-demo-frame distance (floor 1e-12), and
1.1 times the 90th percentile same-frame image distance from successful nominal
TRAIN references with actual frame coverage. There is no end clamping or VAL/TEST
calibration. Store all reference IDs/frames/distances and adjacent-goal overlap
flags. Overlap is diagnostic evidence, not permission to change thresholds.

At goal frame t, proposal windows are all complete H16 applied-action sequences
starting in `[max(0,t−28), t−16]`. Ordinary goals therefore have **13 demonstration
proposals**, alongside hold and warm/search-best, leaving **one Gaussian candidate
per search round**. All proposals pass the same bounds, projection and model
scoring. Frame 0 has no demonstration proposal and therefore 14 Gaussian slots.
The final goal has an irregular 25-frame gap, but the existing runner uses its
fixed 28-frame lookback, so frame 501 also has 13 proposals (starts 473 through
485). It does not shorten the lookback to the previous waypoint interval.

This is a **joint controller configuration change** from v1: horizon 4→16,
waypoint stride 10→28, and consequently seven→thirteen ordinary demonstration
proposals and seven→one Gaussian slots. The upstream TASK037 model also changes
training horizon and sampling balance. Cross-version improvement cannot be
attributed to horizon alone. Longer prediction spans 0.8 seconds while regular
goal spacing represents 1.4 seconds of demonstration time; image-goal progress
can still be nonmonotonic or unreachable by the selected short action sequence.
Coarser goals may skip helpful visual constraints. More proposal coverage also
makes the explicit demonstration scaffold stronger. These risks are declared
before outcomes, not resolved by assuming the longer horizon will succeed.

## Causal controls, compute and interpretation

Learned mode ranks projected candidate sequences using the trained dynamics.
Persistence assigns tied current-image costs and seeded unbiased tie selection.
Dynamics shuffle permutes the association between candidates and predicted costs.
All three retain the same goals, proposal source, projection, limits, search
budget and reset cohort. Realized candidates diverge after different commands;
this is a closed-loop comparison, not replay of a common action tape.

Record selected proposal origin, requested/projected/applied actions, feasible
counts, candidate cost spread, selected cost, goal index, observed image distance,
per-command and projection timing, and all evaluator-only task stages. H16 creates
up to 512 candidate time steps across two rounds per command, four times the H4
configuration's 128. Projection/backtracking may dominate and the fixed wall or
control deadline may truncate the cohort. Do not reduce horizon/candidates or
change proposals during this run to rescue completion.

Report success and ordered stages for all six attempts, including failures and
uncertain executions. Inspect goal stalls, overlap, proposal dependence and
model/control ablations. Six development attempts over two resets cannot establish
reliability; a successful proposal scaffold alone cannot establish a world-model
contribution. The historical 0/6 result is retained unchanged. The fresh final
20-reset cohort **44000–44019 remains unexecuted** and requires a later frozen
selection and explicit authorization; do not use `--stage final` here.

## Compatibility inspection and conditional launch template

Read-only code/metadata inspection confirms the current `evaluate_apple.py` accepts
H16/stride28 through existing CLI arguments. The unchanged sensor backend supports
`max_horizon=64`; its image-only goal and observed-image progress APIs satisfy the
controller. The branch corpus retains `ee_delta_grasp_v0`, `g1_sim_action_v0` and
`g1_dex3_proprio_v0`. The runner loads weights with `weights_only=True`, verifies
corpus/split/action hashes, state schema and exact TRAIN normalization IDs, then
strictly validates the model configuration/source/preprocessing. Additional
sampling-group provenance in a TASK037 checkpoint is retained and does not alter
these contract checks. No source change is needed for this configuration.

This inspection has not loaded the future H16 checkpoint, calibrated goals or run
physics. Actual checkpoint compatibility remains a fail-closed prerequisite of
the runner once that checkpoint exists. The following command is a **conditional
launch template**, not an instruction to bypass the causal gate or coordinator:

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_apple.py \
  --dataset data/apple-branches-v1 \
  --checkpoint checkpoints/apple-branches-h16-sensor-v1/sensor.pt \
  --output outputs/apple-control-development-v2 \
  --stage development --seeds 43000 43001 \
  --modes learned persistence dynamics_shuffle \
  --max-seconds 600 --max-steps 1000 \
  --stride 28 --dwell 3 --horizon 16 --candidates 16 --iterations 2 \
  --control-timeout 5
```
