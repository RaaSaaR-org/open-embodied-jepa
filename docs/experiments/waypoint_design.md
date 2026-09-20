# Dense image-waypoint learned MPC

The controller receives an ordered list of RGB images from declared training demonstrations. It receives no goal joint positions, object poses, phase labels, task scores, or privileged transition triggers. `ImageWaypoint` records a training source ID, image threshold, consecutive-observation dwell, and optional normalized action proposals. The integration runner must verify source membership in the sealed training split; the controller rejects explicitly non-training sources.

`WaypointController.step(observation, projector)` returns one projected action and a decision trace. The caller executes that action once and calls `acknowledge(ExecutionResult)` before requesting another decision. Rejections terminate control; a new step requires an increasing observation timestamp. The default limit is 1,000 acknowledged commands. The runner owns deadlines, transport exceptions, safe stopping, and task evaluation.

## Choosing actions

Defaults are horizon four, 16 candidates and two search rounds. Horizon, candidate budget and bounds are explicit configuration, so longer horizons can be tested under a separately declared budget. Each round includes a hold request, shifted previous-plan proposal, optional demonstration proposals, and Gaussian perturbations. Perturbation variance resets each observation and retains a floor within optimization. Every candidate, including demonstration actions, passes through the embodiment projector and then the opaque world model's `encode`, `predict`, `encode_goal` and `distance` methods. Infeasible sequences cannot win. No action prior or demonstration bonus enters the cost; terminal predicted image-goal distance alone ranks feasible sequences.

Demonstration proposals are an explicit action prior in the candidate distribution, not a trajectory executor. They must have the configured horizon and normalized 14-dimensional schema. Their source, use and limits must be reported. After additional transport clipping, the next hold request uses acknowledged grasps; warm starts remain proposals and are projected/scored again against fresh observations.

## Advancing image goals

The required `progress_distance(current_images, goal_images)` callback accepts images only. Wire it to the model's `observed_distance` method; no latent payload inspection or zero-action prediction substitutes for observed image distance. Advance at most one waypoint per observation after its distance threshold has held for the declared consecutive-observation dwell. Time passage, executed action count, oracle phase labels and task stages cannot advance goals. Completing the waypoint list is a controller termination condition, not a claim of physical task success.

Calibrate threshold scales using training images before development evaluation and save the exact procedure, values and source hashes. There is deliberately no guessed threshold default. Root protocol selection owns image spacing and dwell. The runner must report when nearby waypoints are visually ambiguous, rather than infer progress from hidden state.

## Scientific controls and traces

`learned` uses dynamics scores normally. `dynamics_shuffle` permutes the association between feasible candidate sequences and their learned costs using the declared seed; the permutation is recorded. `persistence` assigns the observed current-to-goal image distance to every feasible candidate without calling dynamics. Exact cost ties use seeded random selection, avoiding a deterministic preference for demonstration or hold candidates. All modes retain the same proposals, projection, image progress rule and budgets. Changed actions naturally produce different future observations and warm starts.

Traces include waypoint index/source/image hash, observed distance and dwell, candidate costs and feasible counts, cost spread/standard deviation, score permutation, selected proposal origin and round, sampled/projected/applied actions, projection/planning time, and execution status. These support auditing whether dynamics ranking changes behavior. Persistence and shuffled-dynamics controls are necessary: successful demonstration-informed control alone would not establish that the model matters.

## API and validation

```python
controller = WaypointController(
    model, waypoints,
    progress_distance=model.observed_distance,
    config=WaypointConfig(horizon=4, candidates=16, iterations=2),
)
decision = controller.step(observation, robot.project_candidates)
if decision.action is not None:
    trace = controller.acknowledge(robot.execute(decision.action))
```

Eleven lightweight opaque-model tests pass, including acknowledgement-clock freshness. They verify opposite learned dynamics select opposite actions from identical demonstration proposals, projection precedes scoring, invalid candidates are masked, consecutive observed-image dwell controls progress, fresh observations/acknowledgements/budgets are enforced, transport clipping informs future holds, ablations are reproducible, persistence does not call dynamics, nonfinite inputs fail, and exploration variance does not collapse across steps. These are software evidence, not learned MuJoCo manipulation results.


## Frozen evaluation runner

`scripts/evaluate_apple.py` prepares and supervises evaluation in isolated child processes. Development seeds are43000–43004; final seeds44000–44019 are opt-in and require an explicit development-selection manifest. The initial development protocol selects seeds43000/43001 and modes learned/persistence/dynamics_shuffle, with horizon4,16candidates,two rounds,1,000commands, five-second control deadlines and a600-second total wall budget. Seed and mode subsets are explicit arguments saved before execution. Default task reset centers are object(.34,−.18), plate(.49,−.09), with independent ±.006m jitter. All modes receive identical resets. The controller uses only the right arm and hand under the same normalized bounds; candidate projection is mandatory.

Before physics, select the lexicographically first successful **nominal training** apple/plate demonstration. Goals occur every10frames and include the final frame. Default dwell is three consecutive observed-image matches. Candidate proposals consist only of complete action windows within the preceding10demonstration frames; frame0 has no proposal. Stored robot states and phase labels never become goals or transition conditions.

Thresholds are calibrated before evaluation using only training RGB. For each goal, take the maximum of (a)1.1times the largest image distance from its±2-frame neighborhood, (b)the median positive adjacent-frame distance in the selected demo, with a1e−12floor, and (c)1.1times the90th percentile of corresponding-frame distances from successful nominal training episodes. An episode contributes only where that observed frame exists; no end clamping or extrapolation is used. Save all source episode IDs, indices, individual distances, quantiles, thresholds and image hashes. Also record previous/next waypoint distances, threshold-overlap flags and counts. Overlapping goals can make dwell a temporal scaffold; these diagnostics are disclosure, not an automatic threshold adjustment. Persistence/shuffle ablations remain necessary.

Preparation writes `calibration.json`, `waypoints.npz` and `resolved_plan.json` before constructing a physics environment. The parent snapshots Python source, configuration and asset manifests, verifies checkpoint/dataset/split/action/state/normalization provenance, and hashes generated artifacts. Checkpoint, dataset files, source, action/asset manifests, registered upstream asset files, calibration and waypoints are checked before and after attempts. Changes invalidate the comparison and prevent remaining attempts.

The supervisor clock starts before NumPy/runtime imports and input hashing. It counts host suspension and reserves five seconds of the declared budget for finalization. A killed process group retains flushed command-pending/result events, so incomplete execution is marked uncertain rather than omitted. Every planned attempt remains in the denominator, including preparation failures, rejected commands, timeouts and unstarted attempts. Per-attempt supervisor status/return code overrides an apparently completed worker report if the child failed afterward. The CLI exits2for infrastructure failures, integrity changes, deadline misses or incomplete comparisons; completed physical task failures remain normal completed negative results. No completed waypoint list is automatically scored as a successful manipulation.

Runner/controller validation:27lightweight tests pass in0.10seconds, without physics or training experiments. Coverage includes train-only nominal selection and calibration, exact action-window provenance, matching reset cohorts, input drift after a completed attempt, nonzero child exits, hard-wall process-group termination, partial trace recovery, control deadline rejection before actuation, retained timing after acknowledgement, and exit-code semantics.

Example development command, to run only after the coordinator freezes the protocol and checkpoint:

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_apple.py \
  --dataset data/apple-v1 --checkpoint checkpoints/apple/sensor.pt \
  --output outputs/apple-development-v1 --stage development \
  --seeds 43000 43001 --modes learned persistence dynamics_shuffle \
  --max-seconds 600 --max-steps 1000 --stride 10 --dwell 3
```

Paths in this example are placeholders; the runner requires real sealed data and compatible checkpoint artifacts, refuses existing outputs, and performs no fallback to a scripted oracle. No physics evaluation was executed while implementing this runner.
