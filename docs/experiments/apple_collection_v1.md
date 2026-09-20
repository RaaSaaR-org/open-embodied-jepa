# Apple task collection v1 — preregistration

This is a new **task-specific development corpus**, not an addition to the original
held-out Apple→Plate benchmark. Its privileged scripted policy supplies training
examples only; simulator object truth, phase targets and scorer output are never
canonical model inputs. Original MVP datasets, split assignments and results stay
unchanged. A later learned-control evaluation needs independently frozen reset
seeds and criteria.

Freeze **32 attempts**, reset seeds **42000–42031**, with independent uniform
±0.006 m XY jitter around apple (0.34, −0.18) and plate (0.49, −0.09). RNG draws
object XY then plate XY from `numpy.default_rng(seed)`. No reset rejection is
resampled. Use the successful forward-palm offset **−0.015 m**, original
EarlyReleaseOracleManipulationPolicy with opening ramp **0.08**, and the additional
transfer X shift selected by the separately preregistered six-attempt landing
probe. The collector requires that report's exact SHA-256 and a nonempty successful
selection. No collection starts before selection, review and coordinator approval.

The completed landing probe selected **−0.035 m** by the declared success/margin
rule. Its report SHA-256 is
`4463e9557fd3ca5c1083534b6792a9675aebe5f0d29e9b69562881a841130b8f`.

Each four consecutive attempts has two nominal policies, one position perturbation
policy and one closure perturbation policy: **16 nominal / 8 position / 8 closure**.
Perturbed policies alternate four nominal-recovery commands and four perturbed
commands. A perturbation vector is constant across its active four-command block;
the next active block receives a new deterministic draw from
`SeedSequence([reset_seed, accepted_step // 8, 731])`.

- Position variant adds uniform normalized right XYZ noise in **[−0.04, +0.04]**
  and right rotation noise in **[−0.02, +0.02]**; grasps are unchanged.
- Closure variant adds uniform right-grasp noise in **[−0.08, +0.08]**; other
  action channels are unchanged.
- Clip the final requested 14D vector to [−1, +1]. The base policy continues
  feedback from measured robot pose and accepted grasp. No oracle object feedback
  is newly introduced after its initial reset target computation.

Physics, contact parameters, robot gains, actuator limits and ordered
AppleToPlateTask thresholds remain unchanged. Stop at observed success, command
rejection, policy completion (**745 commands maximum**), or resource cap. Retain
all failures and usable prefixes; do not preferentially retain nominal successes.
Minimum stored prefix length is **8 applied transitions**, matching prior data
quality practice. Shorter attempts remain in the collection report.

The shared budget is **600 true-wall seconds**, including rendering and writes;
stop issuing commands at **540 seconds** to reserve 60 seconds for sealing,
normalization and train-waypoint export. A subprocess supervisor enforces 600
seconds using `max(wall, monotonic elapsed)`. Timeouts may leave an unsealed partial
corpus; such artifacts are explicitly incomplete and must not be used for training.
No retries or budget extensions. Source is copied to an immutable adjacent runtime
snapshot so unrelated development cannot change the collection implementation.
The supervisor starts its wall budget before source copying/setup. A frozen plan,
protocol and mechanics-selection copy are hashed in the snapshot before physics
initialization. Verify every snapshot file before and after the worker; a changed
source/configuration/asset-manifest/protocol/selection invalidates the run. Child
failure or timeout returns nonzero. Reports list all 32 planned attempts, including
interrupted attempts with unknown execution counts and unstarted attempts. A
sealed dataset with unfinished waypoint export remains `training_ready=false`.

Store canonical DatasetStore RGB **96×96**, **86 named measured proprioceptive
entries**, masks, timestamps, normalized **applied 14D actions** and physical raw
actions at **20 Hz**. Preserve T+1 frames/states for T applied commands. Metadata
records phase, nominal/perturbation/recovery class, perturbation seed/bounds, base
oracle proposal, perturbed execute request, stop reason, final ordered score and
session ID. There is no world-model score in oracle collection, so model-scored
actions are explicitly null with that reason. Rejected requests are reported
separately and never labeled as applied transitions.

Freeze whole-session train/validation/test splits with **seed 42** and ratios
**0.8/0.1/0.1**, explicitly passing `heldout_combinations=()` for this task-specific
corpus. Fit canonical state/action normalization on train only. Do not reshuffle
based on observed success. Report successful counts separately by split; if a split
lacks useful successful trajectories, report the limitation and require a new
declared collection iteration. Phase labels support later sampling/diagnostics;
this collector does not claim uniform phase coverage or balanced training windows.
Readiness additionally requires at least one successful train and validation
episode. This minimum is an explicit readiness gate, not a reliability claim.

Export dense image waypoints at every **10 executed transitions** plus the final
frame, only from **successful TRAIN episodes**. The manifest records source train
episode ID, frame index, timestamp, phase, image path and SHA-256. Goals contain
images only, never future robot state or simulator truth. All exported images
remain traceable to train episodes and the final dataset manifest hash.

After explicit coordinator approval:

```sh
PYTHONPATH=src .venv/bin/python scripts/collect_apple.py \
  --mechanics-report outputs/apple-mechanics-v1/report.json \
  --mechanics-sha256 4463e9557fd3ca5c1083534b6792a9675aebe5f0d29e9b69562881a841130b8f \
  --output data/apple-task-v1
```

At registration no collection has run. Results will be recorded separately from
the immutable pre-execution plan saved as `collection_plan.json`.
