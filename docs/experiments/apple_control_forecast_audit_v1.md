# Apple control forecast audit v1

Prospective diagnostic conditional on the completed matched control comparison
confirming the observed stall. The coordinator must authorize execution after
review and commit. No execution, physics, inference, or training occurs while
preparing these files. Existing control/data/checkpoint artifacts stay immutable.

## Frozen observations and budget

Use ORIGINAL `outputs/apple-control-development-v2`, never mutable resumed
outputs. Its interrupted `43000-learned/trace.jsonl` has exactly 574 acknowledged
commands, steps 0–573, with SHA256
`921eacf54ca7cb131ef5fe1abb7f99e32af11bb5423dc56fae459664286df4c2`.
The checkpoint SHA256 is
`0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5`;
its source corpus is `apple-branches-v1`, SHA256
`6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`.
Use its saved full runtime source, waypoint calibration, resolved controller,
reset, model configuration and CPU backend, with four Torch threads. Hash them
alongside this script/protocol, repository revision and package environment before
launch; verify original Python/assets/data payloads and input hashes afterward.
Do not decode any TEST images. Hashing stored TEST payloads is integrity checking.

One attempt, 180 true-wall seconds total including startup/finalization, 120 child
CPU seconds hard limit. Stop work at 165 wall or 115 CPU seconds; kill the worker at
175 wall seconds, reserving five seconds for supervisor finalization. No retries,
extensions, training, altered physics, goal thresholds, or success criteria.

## Replay and exactly nine branches

Replay the original frozen controller and physics ONCE in order, planning every
step and executing its selected command, up to pre-command roots 119, 300 and 550.
Original cumulative controller planning time through 550 commands was about
48 seconds; this excludes full physics/rendering replay cost. The completed
1,000-command learned rollout took about 172 seconds end to end, so this audit
may approach its budget. These observations do not authorize any extension. At every step compare goal identity,
dwell, selected origin/round/index, candidate feasible masks/costs/permutations,
requested/projected first action and acknowledged applied action against the
saved trace. Numeric costs/observations use atol 1e-6, rtol 1e-5; applied actions use
atol 1e-6, rtol 0. Observation/acknowledgement timestamps use atol 1e-9, rtol 0.
Goal image hash and advancement flags, plus discrete values, must match exactly.
Compare ordered scoring with explicit missing/null/type checks too.
Any mismatch aborts the audit without relaxing tolerances or constructing a proxy
state. Roots are development controller states, not selected successful outcomes.

Wrap the actual projector to retain both rounds' requested and returned projected
H16 candidate sequences, without consuming RNG or changing its result. At each
root recover: (1) the selected winning sequence, (2) the least-cost feasible
training demonstration sequence across both rounds, ties by earlier round/index,
and (3) the scored grasp-preserving hold sequence from slot 0. All use the SAME
frozen image goal. Missing/infeasible candidates are explicit failures; no fabricated
fallback. Confirm the recovered winner's first action matches the controller and
that scoring each recovered sequence reproduces its saved endpoint score.

Snapshot full MuJoCo data (including solver warm-start), controller targets,
embodiment grasp/stopped state, ordered scorer state, and controller RNG, warm
sequence, goal/dwell/counters, pending acknowledgement and immutable goal cache.
Persist the integration state, controller state and sensor observation. Restore
these between siblings and require identical RGB, proprio, mask and timestamp.
Refresh observation-derived projection caches and wall time after restoration.
Do not call controller.step/acknowledge during an open-loop branch. After all three
branches, restore the root and controller's post-planning pending state, execute
and acknowledge the ORIGINAL first action once, then continue the original replay.
The last root does not require an extra original command after its branches.

Execute each projected H16 sequence open-loop through the unchanged embodiment.
Store the requested command and every ACTUAL applied command with its resulting
RGB/state/mask/time. A pending record precedes execute; atomic sensor/action files
retain acknowledged prefixes. Rejections and runtime errors retain those prefixes;
never pad them to H16. Simulator truth is used only for the unchanged scorer, never
as model input or as a branch-selection cost.

## Attribution and reporting

From the same initial measured RGB/proprio, forecast each originally scored
projected sequence. After execution, separately forecast the exact accepted
ACTUAL-applied sequence from that same initial latent. This is diagnostic use of
recorded action labels, not future state supplied to the model. No refitting occurs.
Report H1 and H16 when the corresponding sensor frame exists; missing endpoints
remain unavailable. Report measured and predicted image-goal distance, RGB-grid
MSE after undoing the model's frozen normalization, qpos MSE (rad²), qvel MSE
(rad²/s²), action mismatch counts/magnitudes, source candidate/round, and outcomes.
Retain both sets of predictions and measured frames as artifacts. Both horizon
keys are explicit: unavailable endpoints/predictions carry a reason. Raw RGB error
refers to the model's resized RGB grid, not unresized camera pixels; no clipping of
predicted pixels hides extrapolation.

The first forecast measures scored-plan versus actual physics agreement. The
actual-action reforecast helps assess whether action projection/tracking changes
explain that disagreement. It is not a unique causal decomposition of servo error,
unobserved state and model bias. Compare predicted/actual progress of the winner,
demonstration and hold only at matched roots/horizons. Nine branches from three
correlated roots do not establish general reliability or manipulation success.
Report every intended branch, durable prefixes, replay failures, incomplete
endpoints, CPU/wall time and post-run integrity status. No acceptance metric
selects a new checkpoint or authorizes an automatic controller change.

On interruption, recover only contiguous, validated durable frame/action files;
report their confirmed applied-command count. A pending command indicates possible
additional execution only when its index exceeds the durable prefix, or when
journal evidence is malformed/gapped. Never count an acknowledged durable frame
as unknown solely because its pending marker remains on disk.
