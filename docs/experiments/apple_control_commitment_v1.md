# Apple control with four-command commitments v1 — prospective design

TASK-039 hypothesis: repeatedly replacing a good H16 plan after its first command
can prevent useful later commands from executing. TASK-038's fixed-state audit
showed lower measured H16 goal distance for winning plans at roots 119/300, with
no scored-versus-applied action difference. At root 550 the winner was rejected
on command 10, so this does not justify unguarded open-loop execution. Test a
four-command commitment with fresh measured feedback and feasibility validation.
This is a new development experiment, not a completed or final-MVP result.

## Frozen comparison

Retain the selected H16 sensor checkpoint SHA256
`0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5`, source corpus
SHA256 `6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`,
TRAIN demonstration selection, image-goal threshold rule, stride 28, dwell 3, H16,
16 candidates, two search rounds, noise 0.15/minimum 0.05, normalized action bounds,
physics and ordered task scoring. No retraining, privileged-state planning,
threshold relaxation, scripted fallback or final-test seeds.

Exactly six attempts, in original order:
43000 learned, persistence, dynamics_shuffle; then 43001 learned, persistence,
dynamics_shuffle. The controller parameter change is commitment length from 1 to 4;
resource allocation also changes from the prior shared budget to per-attempt caps. All three modes
share the same proposal mechanism, fresh projection, four-command commitment,
image progress and reset distribution. Learned dynamics still rank all 16
candidates at each full planning call; persistence uses ties and dynamics_shuffle
breaks action/cost association before the same commitment. This comparison tests
whether model ranking contributes under the new execution frequency; comparing
with prior v2 is exploratory because resource allocation differs and prior attempts
were interrupted/time-limited. This is not a pure one-variable historical comparison.

Maximum 1,000 accepted commands per attempt; original per-control deadline 5 seconds.
Global 600 true-wall seconds includes setup, every subprocess and finalization.
Each attempt receives at most 90 true-wall seconds INCLUDING model load/reset and
its final report. Never lend unused time to a later attempt or extend an attempt.
Reserve five global seconds for finalization. Preparation and parent integrity
checks consume the same global budget; there is no separate preparation limit or
guarantee that six full 90-second slots remain. At each attempt launch allocate
`min(90, remaining global budget)` and log requested cap, allocated cap, global
remaining seconds and whether allocation was shortened. Launch while positive
budget remains; otherwise retain an explicit not-started status. Budget combinations
whose summed attempt caps exceed the global cap remain valid censored comparisons.
A full-slot supervisor termination is explicit `attempt_timeout`; a globally
shortened hard cutoff is a global timeout. A child soft cutoff reserves
`min(5, allocated_seconds / 10)` seconds for its final report (90 gives 85 active
seconds). The parent kills at the allocated cap, which includes imports, model
load/reset and reporting. Ordinary success/rejection ends earlier. All six planned
attempts remain in the denominator, including missing/incomplete attempts. These
limits do not guarantee completion of 1,000 commands.
Freeze source, exact runtime/config/checkpoint/data/waypoint hashes and this
protocol before any physics. No execution before independent review/authorization.

## Generic controller change

Add `WaypointConfig.commitment_steps: int = 1`, restricted to 1..horizon, and a
matching evaluator `--commitment-steps` option. Default1 follows the existing RNG,
proposal, full-planning, pending-acknowledgement and rejection path unchanged.
No model-specific branch is added to the planner.

Every `step()` still requires a fresh synchronized increasing observation, updates
image-goal distance and consecutive dwell once, and stops on completed waypoints.
An advance to the next image goal immediately clears any commitment and causes
full planning for that new goal. Task scorer evaluation remains in the evaluator,
after EVERY attempted execution; it never drives image-goal advancement.

A full planning call remains the existing 16-candidate/two-round search. After
successful acknowledgement, cache up to the first remaining 3 commands of the
winning PROJECTED sequence. Before each cached command, project ONLY that next command (K1,H1) again from
the new measured robot snapshot. Require a valid feasible CandidateProjection
and exact array equality between its projected command and the cached original
model-scored command. Do not silently execute a newly clipped/changed cached
action; abandon the commitment and perform one ordinary full candidate search.
This avoids distant uncommitted H16 limits preventing safe near-term commands.

A false feasible flag OR any change to the next cached action clears it and
permits exactly ONE
ordinary full search on the same fresh observation. Projection exceptions or
malformed results remain errors, with no hidden fallback. Waypoint advance,
completed commitment or an accepted action differing from the freshly projected
request clears the cache before the next planning call. ACTUAL execute rejection
terminates the episode as today; it never bypasses the increasing-observation
contract or retries a measured joint-rate violation at unchanged simulation time.

Acknowledgement remains mandatory after every command. Update last grasps from
ACTUAL applied action. Keep the normal H16 warm sequence separately from the
short committed suffix: for a cached decision, pass the current full H16 warm sequence (whose first action matches the cache)
to the existing acknowledgement/shift operation. Never
shrink warm-start shape to H15/H3. After an accepted cached command, drop its first
action from the original scored commitment. Retain model-scored original
sequence and requested sequence separately for attribution and trace provenance.

## Required traces and validation

For every command record decision kind (new plan or commitment), stable plan ID,
origin plan observation timestamp, originating goal/index/candidate source and
round, commitment action index, original sampled/model-scored action, refreshed
projected request and actual applied action, projection differences/time,
remaining commitment length and abort reason. Record the full selected requested
and projected H16 sequence on its original full-planning record. Cached decisions
perform zero candidate model evaluations and do not consume planner RNG. Their
inherited forecast cost must be labeled with its originating observation/horizon;
never present it as a newly scored prediction at the current observation.

Report plan counts, realized commitment lengths, per-command observed progress,
waypoint/stage/full-task success, all guard stops/timeouts, projection changes,
CPU/wall timings and action-cost ablations. Do not infer success from fewer model
calls or image progress alone. Verify first-step/default 1 behavior and RNG draws
against existing tests, default CLI budget behavior, initial/subsequent cached
projection, early waypoint advance, exact pending acknowledgement, warm H16
shape, cached infeasibility/projection-change fallback-once, actual rejection termination and
per-attempt/global/suspend-clock limits with synthetic fixtures. No physical run
is part of these software checks.

The 5-second per-command deadline includes fresh observation, image progress,
cached projection validation AND any fallback full search together. Fallback
search is not recursive step(): do not advance dwell twice, recheck the same
observation as if newer, or consume RNG for the rejected cached command.

## Frozen command (execute only after review and source freeze)

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_apple.py \
  --dataset data/apple-branches-v1 \
  --checkpoint checkpoints/apple-branches-h16-sensor-v1/sensor.pt \
  --output outputs/apple-control-commitment-v1 \
  --stage development --seeds 43000 43001 \
  --modes learned persistence dynamics_shuffle \
  --horizon 16 --stride 28 --dwell 3 --candidates 16 --iterations 2 \
  --commitment-steps 4 --attempt-max-seconds 90 --max-seconds 600 \
  --max-steps 1000 --control-timeout 5
```

The resolved plan and command traces bind checkpoint, corpus, source snapshot,
action/assets and calibrated TRAIN waypoint hashes. The preregistration is frozen
by the reviewed source commit; results belong in a separate report. No TEST images
or final cohort observations are used to select this configuration.
