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

Ten lightweight opaque-model tests pass in 0.05 seconds. They verify opposite learned dynamics select opposite actions from identical demonstration proposals, projection precedes scoring, invalid candidates are masked, consecutive observed-image dwell controls progress, fresh observations/acknowledgements/budgets are enforced, transport clipping informs future holds, ablations are reproducible, persistence does not call dynamics, nonfinite inputs fail, and exploration variance does not collapse across steps. These are software evidence, not learned MuJoCo manipulation results.
