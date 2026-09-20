# Apple manipulation: diagnosis and sensor-world-model route

This is a development design review, not a successful manipulation result. The
historical common-mode native/LeWM results and original Apple→Plate holdout remain
unchanged. The new objective explicitly permits **task-specific Apple→Plate
training** to establish learned control before claiming compositional transfer.
The proposed sensor-space backend is separately named `sensor_wm`; it is not an
unsupervised JEPA result or an upstream LeWM improvement.

## What the evidence establishes

| Finding | Evidence and consequence |
|---|---|
| Proprioception is discarded | `models/base.py:VisualModel.encode` validates the robot state but returns only image embeddings. Both historical backends use that implementation. Their predictors cannot distinguish otherwise similar images with different joint velocities or tracking lag. This is a design limitation, not a broken advertised visual-only baseline. |
| Planning looks only 0.2 seconds ahead | `configs/mvp_common.yaml` uses H4, while the action manifest specifies 0.05 seconds per command. Successful scripted cube trials first reached/grasped/transported at approximately 11/14.25/21.75 seconds and released around 24 seconds. Terminal whole-image distance over 0.2 seconds supplies no demonstrated incentive to perform this ordered sequence. Increasing H without corresponding training is not a remedy. |
| The training support is insufficient for apple success | The original 184-episode corpus excludes Apple→Plate. The release-v1 corpus has nine cube successes, zero apple successes, and no successful release-v1 validation episode. Successful cube trajectories do not establish that this hand configuration can manipulate apples. |
| Noncollapse does not imply useful state | Final target effective ranks were 2.59–3.56 despite passing per-coordinate variance guards. Native prediction beat persistence; all three LeWM checkpoints were worse than persistence. Shuffled-action error establishes sensitivity but not reliable counterfactual ranking or grasp awareness. |
| Feasibility was necessary but insufficient | TASK-029 removed joint-rate stops in seven completed pairs and achieved exact accepted-action agreement for 666 commands. There was one reach event, no grasp, and two LeWM deadline misses. Purely increasing training cannot eliminate repeated IK/backtracking cost. |

Sources: [training results](mvp_training_results.md), [release collection](manipulation_controller_v2.md),
[feasibility results](feasibility_results_v1.md),
[base model](../../src/embodied_jepa/models/base.py),
[native model](../../src/embodied_jepa/models/native.py), and
[LeWM adapter](../../src/embodied_jepa/models/lewm.py).

Independent read-only inspection of `data/mvp-v0/meta/jepa_manifest.json` found
146/19/19 train/validation/test episodes and 33,623/4,169/4,519 frames.
Training pair counts are cube→target 37, apple→bowl 39, banana→bowl 34,
cube→plate 36. Validation includes six apple→bowl episodes but no Apple→Plate.
These are metadata counts, not additional model-selection measurements. No test
images were decoded and no historical split was reassigned for this diagnosis.

Several mechanisms remain **hypotheses**, not proven causes: background pixels
dominating a small object, ambiguity between closing and slipping, and poor
counterfactual action support in mostly scripted trajectories. Evaluate these
directly rather than asserting them from low latent loss. Uniform window sampling
weights long holds more heavily than brief contact/release events; new corpus
sampling should disclose and balance those phases.

## Smallest implemented model

`models/sensor.py:SensorWorldModel` uses the existing world-model API. Its opaque
current latent concatenates a fixed spatial RGB grid with normalized observed
joint positions/velocities. Default RGB resolution is 24×24, giving 1,728 visual
features and 1,814 total features for the 86-field G1 state. A two-layer,
256-unit MLP learns the residual transition conditioned on the actual 14D
executed action. It predicts **both** future visual and proprioceptive features;
recursive inference never reads future measurements, simulator state, contact
truth or evaluator stages. There is no hidden episode history or previous-action
cache in v1. This omission is explicit; add resettable sensor history only if
observed-state aliasing fails the diagnostic gate.

The encoder is fixed preprocessing, not a learned perception model. Learning is
in the action-conditioned visual/proprioceptive dynamics. Image-only goal cost
compares predicted visual features with the supplied image. It never uses a
demonstration's goal joints, substitutes simulator kinematics for prediction,
or calls a scripted controller at inference. `observed_distance(images, goals)`
measures actual image progress for the waypoint controller without pretending a
zero action is a hold or inspecting a backend's latent payload.

`fit_normalization(batches, training_episode_ids=...)` must run once on declared
training episodes before initial validation. It rejects foreign episode IDs and
incomplete declared coverage, stores the training IDs/frame counts, and freezes
image, state and residual scales in checkpoint buffers. The caller remains
responsible for supplying IDs from the sealed training split. Nonoverlapping
chunks minimize duplicated boundary frames; supplied frame frequency determines
the population moments. All proprioceptive fields must be observed in v1.

Training combines one-step normalized residual error with recursive sequence
error, weighting visual and proprioceptive groups separately. Each coordinate's
residual scale comes from training transitions, with explicit floors. This avoids
rewarding an almost-static prediction solely because 50 ms RGB changes are small.
Validation reports visual prediction/persistence/zero-action/shuffled-action
errors and proprioceptive error. Fixed inactive background coordinates are not
representation collapse; do **not** use the historical JEPA collapse selector.
Generic `raw_mse` selection can consume this backend, but passing a validation
prediction/persistence and action-ranking gate is a separate requirement.

Save/load preserve optimizer state, normalization, source/config/schema identity,
RNG state and provenance. Sensor preprocessing is explicitly recorded instead of
inheriting the visual baseline's `ignored_visual_baseline` checkpoint label.

## Model-based control and new data

First demonstrate a mechanically valid apple grasp/lift/transfer/release with
unchanged physical limits and the ordered task scorer. A privileged collection
policy is acceptable for obtaining demonstrations and a positive control; its
performance is never attributed to a learned model. A concurrent mechanics probe
reported 2/2 successes with a −0.015 m forward-palm offset. That early report
motivates a fresh corpus; this document does not independently certify those
rollouts or extrapolate their reliability.

Collect a new, versioned Apple→Plate development corpus with complete accepted
actions, T+1 RGB/state observations, phase labels and failed prefixes. Include
successful release and negative/recovery transitions in both training and
validation by a **prospective** collection/split design. Never move an already
observed test episode into training to repair a failed validation gate. Vary
object/container positions beyond a single trajectory, and record the support of
each variation. Synthetic auxiliary truth labels, if later introduced, require
an explicitly supervised-perception variant and must never enter deployment
inputs. They are not required by the implemented fixed-feature backend.

Use dense **training-image-only** waypoints spanning approach, contact, lift,
transfer and release. A useful initial spacing is 5–10 accepted 20 Hz commands;
calibrate spacing and advancement thresholds on validation before freezing the
new evaluation. Waypoint ordering supplies a task-specific temporal scaffold,
which must be named in every result. Advance on observed visual agreement rather
than elapsed scripted phase time or ground-truth stage flags. Bound stalls and
record failure; do not silently skip an unreachable waypoint.

CEM must score action sequences by genuine learned recursive predictions and
execute the projected winning first action. H8/H16 span 0.4/0.8 seconds; start
with a small candidate count after the feasibility performance gate and increase
only within a declared budget. No amount of short-horizon planning provides
40-second foresight; the image-waypoint scaffold supplies that decomposition.
Training-demonstration or learned-policy proposals are permitted only as explicit
candidate proposals, with alternative candidates and model rescoring. Replaying
the proposal unchanged is a control, not proof of useful world-model planning.

## Staged gates and bounded first attempt

The following thresholds are **proposed acceptance gates**, not achieved results.
The coordinator must freeze the concrete cohort, budgets and selection rule
before running the corresponding experiment.

1. **Mechanics/data:** full ordered Apple→Plate success is recorded under existing
   physics/guards. New train and validation sessions both contain complete
   trajectories and failures. Hash and inspect every episode; preserve all
   attempted resets in the denominator.
2. **Sensor visibility:** distinguish approach/closed-lift/released images in
   validation using only RGB and measured state. Check feature distance to dense
   waypoints, occluded apple cases and visually similar wrong states. If 24×24
   loses necessary detail, freeze a higher resolution or declared camera crop;
   do not tune against the final cohort.
3. **Dynamics:** with frozen train-only normalization, predict better than
   persistence and shuffled actions at the actual planning horizon, separately
   around contact, lift and release. Run real held-out action branches to check
   predicted candidate ordering; arbitrary output sensitivity is insufficient.
   Reject a model that wins average error only during long stationary segments.
4. **Closed-loop validation:** complete ordered pick-and-place using observed
   waypoint advancement and model-selected actions. Store sampled/projected/
   accepted commands, selected costs, stage outcomes and deadline/stop records.
   With identical proposals, run shuffled-action-model and disabled-model
   controls. If they succeed equally, report the scaffold/proposal explanation;
   useful world-model control has not been isolated.
5. **Fresh final cohort:** after all development, freeze 20 new, session-disjoint
   resets/goals. A concrete user-goal target is at least **16/20 full successes**
   under the unchanged ordered scorer, with the same frozen protocol and reported
   uncertainty. Retain failures and compare the frozen ablations. This would be
   task-specific simulation evidence, not zero-shot Apple→Plate generalization,
   20 Hz wall-clock control, unseen appearances or hardware validation.

The coordinator's [first training protocol](apple_sensor_training_v1.md) fixes
1,800 true-wall seconds, 3,000 updates, batch 16, H8, hidden size 256, CPU four
threads, and raw validation MSE selection, with persistence/shuffle gates.
The bounded attempt includes preprocessing, validation and serialization; a
16 GiB host-memory cap is proposed. Phase-balanced validation is recommended
for a separately declared follow-up if the first cohort cannot cover contact and
release; do not alter the existing protocol after observing its outcomes.
Select on the fixed validation cohort;
save both best and latest checkpoints. First measure a small update/rollout
smoke within the cap. Do not silently extend an unsuccessful run. This is a
budget proposal, not a throughput or success promise. Dense waypoint selection,
control horizon and proposal ablations require a separately frozen evaluation
protocol. Match those choices when later comparing native/LeWM adaptations.

## TASK-024 assessment

JEPA-WMs may help investigate spatial representation and established planning
designs. Its official release lists DINOv2/DINOv3 encoders and predictors trained
for other environments; none is a demonstrated G1/Dex3 Apple→Plate model.
Its source license is CC-BY-NC 4.0, and encoder/checkpoint terms need separate
review. These observations were checked against the [official repository](https://github.com/facebookresearch/jepa-wms)
and [license](https://github.com/facebookresearch/jepa-wms/blob/main/LICENSE)
on 2026-09-20. No source, encoder or checkpoint was downloaded or adopted here.

TASK-024 is useful as an isolated optional research adapter after an exact
revision/artifact/CPU-MPS compatibility audit. It is **not a prerequisite** for
fixing missing apple data, absent proprioception, temporal decomposition or
actuator feasibility. Adding another adapter alone does not close those gaps.
There is no evidence in this audit that its pretrained dynamics transfer to this
robot or that its full training stack is validated on this Mac.

## Validation performed for this change

`PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_sensor_model.py` passed
**7 tests in 1.37 seconds**. Tests include a tiny synthetic action-conditioned
process whose held-out prediction error falls below 10% of persistence and
shuffled-action baselines; train-only normalization isolation; proprioception
usage; image-only goals; causal/chunked rollouts; exact CPU optimizer resume;
checkpoint/schema/provenance handling; missing-state rejection; and actual MPS
training/prediction/checkpoint roundtrip on this Mac. Synthetic learning tests
establish software behavior only. No apple training run or learned physical
success is claimed by this implementation evidence.
