# Apple aligned sensor cost v1: integration and conditional control protocol

Prospective TASK-042 protocol. Implementation is authorized; no corpus calibration, new inference or physical experiment has started. Commit the reviewed source and this protocol before execution.
TASK041 supports an image-to-arm bridge on three held-out parents; it does not
establish manipulation success or that an arm-only goal represents an apple on a
plate. Retain visual evidence and the frozen world model, change one model-owned
cost through a compact composition backend, and gate a small physical diagnostic
on an offline check of the actual combined cost.

## Fixed composition and image-only goal semantics

Frozen backend key: `aligned_sensor_wm_v1`; implementation confined to a new
`src/embodied_jepa/models/aligned_sensor.py`, plus lazy registration and focused
tests. Do not modify `sensor.py` or `base.py`, whose implementation hashes protect
the existing checkpoint. Do not modify the generic waypoint planner.

The composition owns the immutable trained `SensorWorldModel`, the completed
frozen TASK041 neural head, exact named seven-joint mapping, two TRAIN calibration
scalars, and its own latent-owner token. Public constructor follows the existing
registry signature `(state_schema, device='cpu', seed=0, config=None,
metadata=None)`. First supported/validated device is CPU4; do not advertise an
untested MPS path. No training is performed: `train_step` explicitly rejects use
of this frozen composition rather than silently returning a fake update.

Let x(I) be the frozen normalized 24×24 RGB feature grid, and g(I) the frozen
neural head's seven normalized joint-position means. The head receives x(I) only;
its uncertainty output is ignored by the cost. Let a predicted sensor state
contain x_hat and q_hat in that same frozen normalization. Define:

```
D_visual(pred, goal) = mean((x_hat - x(I_goal))**2)
D_pose(pred, goal)   = mean((q_hat - g(I_goal))**2)
C(pred, goal)        = 0.5 * D_visual / s_visual + 0.5 * D_pose / s_pose
```

Each mean is within its own feature group, so 1,728 image dimensions do not
implicitly outvote seven position dimensions. No goal robot-state array, velocity,
phase, scorer value, object pose or hand-written target enters the inference API.
No uncertainty weighting, adaptive mixing, clipping of forecast error or VAL-tuned
coefficient is introduced. Frozen predictions can still be wrong; the extra pose
term is not a guard or oracle.

API behavior:

- `encode(images, RobotState)` delegates observation encoding to the child and
  returns an opaque composition-owned state carrier.
- `encode_goal(goal_images)` returns an opaque carrier containing the child image
  goal latent and g(goal image). It accepts images only, including batch checking.
- `predict(z, actions)` unwraps a validated state, delegates unchanged recursive
  prediction to the child, and wraps the result. Same actual normalized14D action
  schema, horizon/history limits and chunking as the existing sensor model.
- `distance(prediction, goal)` checks owner/type/batch/horizon compatibility and
  returns finite nonnegative float32 costs `[B,K,T]` using the formula above.
- `observed_distance(current_images, goal_images)` is ALSO image-only:
  `0.5*mean((x_current-x_goal)^2)/s_visual +
  0.5*mean((g(current_RGB)-g(goal_RGB))^2)/s_pose`.
  Current measured q is not smuggled into this callback. It therefore uses inferred
  current pose, while predicted-future cost uses predicted q. This distinction
  is deliberate and must remain documented; head error can affect arrival.
- `.load()` invalidates old composition-owned latents, including image goals;
  child-owned latents are not accepted directly by the wrapper. Inference runs
  under no-grad/eval. No network requests or optional asset fetching occur.

Exact positions (radians) are the seven TASK041 fields: right shoulder pitch,
roll, yaw; elbow; wrist roll, pitch, yaw, each named
`right_*_joint.position`. Resolve the full names from the schema, require the
expected radian units and compare the saved field mapping; current indices22–28
are validation facts, not a positional assumption. Pixel preprocessing and all
means/scales are the child checkpoint's TRAIN-only values.

## Fixed TRAIN-only scale calibration

Use all 26 ORIGINAL TRAIN episodes in `apple-branches-v1`, including the failed
42003 demonstration and its extra phases. Exclude all branch episodes from this
scale estimate: deliberately injected branch distributions should not overweight
eight of the parents. Exclude every VAL/TEST payload from calibration.

For a parent with T transitions, use the nonoverlapping complete stride28 pairs
`(t,t+28)` for `t = 0,28,56,...` with `t+28 <= T`. Do not append a shorter final
pair or select frames by success, phase or measured error. Record the exact pair
ledger, session/parent IDs and hash before loading frames. Compute:

```
v_parent = median_pairs mean((x(I_t)-x(I_t+28))**2)
p_parent = median_pairs mean((q_norm_measured(t)-q_norm_measured(t+28))**2)
s_visual = median_over_26_parents(v_parent)
s_pose   = median_over_26_parents(p_parent)
```

The q labels here are measured TRAIN positions used only to calibrate a unit of
pose change; no measured goal q enters inference. The observed image-only metric
uses the same s_pose with inferred means instead of measured q. This is not a
claim that those quantities are exactly equal. Keeping the measured-pose scale
makes the calibration independent of neural overconfidence or inflated errors.

Retain zero-distance pairs in the median; do not silently drop settled phases.
Require each declared parent to have a complete pair and both final scales to be
finite and strictly greater than zero. Degenerate calibration blocks this version;
there is no post-result floor, positive-pair filter, weight sweep or replacement
statistic. Save all parent medians and selected distances, resulting scalars,
normalization/head/model identifiers and recipe version. Fixed weights are exactly
0.5/0.5 after this calibration; they are never adjusted using VAL or control traces.
A preparation cap of 120 true-wall seconds is allocated to attempt
streaming all original TRAIN episodes, without a guarantee or retry.

This scale measures typical adjacent demonstrated change, not physical importance,
observability or task correctness. In particular, equal arm configurations can
coexist with different hand/object outcomes. The visual term is retained to carry
that information, but its sufficiency still needs physical evidence.

## Portable wrapper artifact and strict compatibility

Use an ordinary safe-loadable Torch checkpoint envelope so existing
`evaluate_apple.checkpoint_model()` can continue to instantiate through the registry.
The envelope retains `backend`, `format_version`, `seed`, `config`, full state/action
schemas and metadata fields `dataset_hash`, `split_hash`, `action_hash` and the
child's normalization episode IDs (exactly the frozen TRAIN split). Add a versioned
composition manifest identifying immutable bundle members by relative filename
and SHA256:

1. byte-identical original sensor checkpoint;
2. completed TASK041 `head.pt` (2,000 updates, exact architecture/field names,
   normalized feature spec and training sample-ledger hash);
3. TRAIN calibration JSON and pair ledger;
4. TASK041 producer registration/seal references proving the head belongs to this
   world-model checkpoint, dataset, split, source and preprocessing.

Store these in a new bundle directory; never modify historical artifacts. `.load`
resolves only declared relative bundle paths, verifies every digest before use,
then delegates the child checkpoint to its existing strict loader. It validates
head shapes/config/completion, seven names/units, normalization identity, camera,
TRAIN membership, preprocessing and calibration recipe/positive scalars. Refuse
mismatched head/model/corpus or missing children; never reconstruct or refit an
artifact as fallback. The wrapper has its own implementation hash, independent of
the unchanged child implementation hash. The packaging script/configuration and
protocol hashes are included in provenance. Load must remain compatible with
`torch.load(..., weights_only=True)` and require no arbitrary object unpickling.

The neural head's small Sequential architecture should be constructed locally in
the new adapter with exactly the saved parameter names; do not import a training
CLI to perform inference. Verify numerical mean equivalence on synthetic inputs
and later during authorized packaging. For `.save()`, copy immutable bundle inputs
and write a fresh envelope atomically, rejecting overwrite; no optimizer is
invented for the frozen composition. Constructor metadata expectations are checked
on load exactly as the existing evaluator expects.

## Offline gate of the actual combined metric, before physics

Use TASK041's saved VAL feature/label cache, image-goal estimates and per-root
frozen sensor predictions. Recompute hybrid costs from the saved arrays and the
new immutable TRAIN scales; no dynamics inference, fitting, new VAL sample
selection or simulator execution is needed. Keep the same 18 roots, eight action
siblings, cross-parent same-phase eight goal images per root, pair eligibility,
0.5 tie handling and goal→root→parent aggregation. Retain measured-q label ordering
as an arm-alignment criterion, not full object/task success.

The frozen hybrid must improve PRIMARY H16 predicted ranking over the original
pixel metric by at least5 percentage points, with strictly positive delta on EACH
of the three VAL parents and at least two informative roots per parent. Report
absolute ranking, its change from neural-q-only ranking, H8, measured-state costs,
coverage, and uncertainty intervals descriptively. No weight/scale changes after
this gate. For the measured-state hybrid diagnostic, use measured visual endpoint and measured endpoint q against inferred goal q. This differs from deployed image-only observed_distance, which infers current q from its image. The reused TASK-041 VAL gate is a development/model-selection check, not independent held-out confirmation.

The already frozen neural head's TASK041 image-to-q eligibility must
remain verified by artifact provenance; do not rerun selection. If the hybrid
fails or coverage/artifacts are incomplete, stop before physical control.

Also verify cost implementation matches the saved-array formula exactly within
`rtol=1e-5, atol=1e-7` on saved float32 operands, with finite outputs and correct dimensions.
A small test fixture checks that changing only the visual component still changes
hybrid cost, so arm similarity cannot erase visible object differences. Such a
software test is not evidence that actual object outcomes are recognized.

Limit this offline hybrid audit to60 true-wall seconds, with inputs and results
sealed before any physical launch. No passed offline screen automatically changes
the user-facing model or physical acceptance criterion.

## One-reset physical feasibility test (conditional on gate/review)

Use the existing generic evaluator and planner, new wrapper checkpoint/backend,
and unchanged proposal mechanism, stride28, dwell3, H16, K16, two rounds, original
noise/bounds and five-second command deadline. Reuse the existing TRAIN-only
threshold calibration rule under the new `observed_distance`; do not reuse old
pixel thresholds or tune new ones from VAL/control outcomes. The goal source
remains the same first successful nominal TRAIN demonstration. Physics, guards,
ordered task scores and reset distribution remain unchanged.

Exactly reset43000 with three modes in original order: learned, persistence,
dynamics_shuffle. Commitment length1; the known negative always-four policy is
not retained. Maximum1,000 accepted commands per attempt, per-attempt300 true-wall
seconds, global1,000 true-wall seconds including preparation/finalization, CPU4.
Keep the generic `min(attempt cap, global remaining)` allocation rule and record
all three planned statuses/prefixes. No final cohort, hidden scripted action,
threshold adjustment, time extension or retry. No on-policy data is folded into
training during this comparison.

Learned mode uses the unchanged sensor forecasts ranked by hybrid cost;
persistence keeps the same candidate mechanism but disables action-dependent
forecast ranking; dynamics_shuffle permutes scored action/cost assignments as
before. The image goal metric and all other configuration are common across
modes. The world-model contribution therefore remains testable despite the
supervised image-goal head. Historical comparisons are exploratory because this
resource allocation and the earlier commitment experiment differ; this is n=1
correlated development evidence, not statistical manipulation acceptance.

Report every ordered physical stage, full task success, executed commands,
waypoint progress/dwell, origins/cost spread, guard stops, deadline/resource
censoring and timing. Learned success with failed controls is encouraging only;
equal behavior across modes does not demonstrate useful action-conditioned model
control. Failure preserves its earliest unmet stage and does not trigger an
unregistered weight or proposal search. Final MVP acceptance remains outstanding.

## Review and tests before experiment execution

Parent owns numeric protocol/MC/Git and authorization. This task implements only the new adapter, registry/config hook, packaging/scale script,
offline audit and focused tests. Verify strict bundle corruption/incompatibility,
TRAIN-only calibration membership, independent latent ownership and reload,
image-only goal/progress signatures, frozen child prediction equivalence, finite
batch/chunk costs, equal calibrated weighting and a visual-only distinction case.
No model/planner/physics changes are needed for this design.

## Execution order and stopping rules

1. Review and commit the model composition, immutable bundle builder, saved-array audit, tests and protocol. Preserve TASK-041 artifacts byte-for-byte.
2. Run one TRAIN-only bundle preparation within 120 true-wall seconds. Retain all 26 declared parents and zero-distance pairs. Invalid or zero scales stop the version without a fallback statistic.
3. Run one saved-array hybrid audit within 60 true-wall seconds. Require the fixed primary gate and complete provenance. Failure or insufficient evidence stops before physics; weights and scales cannot be changed within this attempt.
4. Only after a passing reviewed offline result, register the exact bundle SHA and evaluator configuration and run the single three-mode development comparison within 1,000 wall seconds. No TEST or final cohort access. Preserve all three intended attempts and their censoring.

No training, threshold tuning, retries, or budget extension occurs in this task. Stage caps include imports and artifact finalization; reserve five seconds in each offline stage and classify a late completion as incomplete. Physical execution retains the existing evaluator's independently reviewed timeout rules. Code or dependency changes between dependent stages invalidate the run. A negative completed research task never closes TASK-033's physical success requirement.
