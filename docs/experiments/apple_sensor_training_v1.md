# Apple sensor-world-model training v1 — preregistration

This new task-specific experiment follows the mechanics/data gate. It does not
replace or reinterpret the historical native-JEPA/LeWM zero-shot benchmark.
Training may start only after the apple corpus is sealed, source implementation
and this protocol are committed, and an execution registration records exact
source, dataset/split/action, model configuration and protocol hashes.

## Hypothesis and model

Measured robot joint positions/velocities plus spatial RGB features make local
contact-control dynamics more observable than the former single-frame visual-only
latents. A compact learned residual transition predicts both sensor components
under each normalized executed action. Fixed image features are explicitly
engineered perception, not a pretrained encoder or a newly learned JEPA encoder.
The model never receives simulator object poses, contact flags or task scores.
Its image-goal distance never invents goal joint state.

## Fixed initial attempt

- Backend `sensor_wm`; seed 0; CPU, four Torch threads.
- New `data/apple-task-v1` sealed train/validation partitions; no test decoding.
- Fit model normalization exactly once, using every training transition only.
- 3,000 optimization updates; batch size 16; recursive horizon 8.
- Model's committed default architecture/configuration, with the complete resolved
  mapping archived before execution; no changes during this attempt.
- Validation every 300 updates, four fixed batches each at horizons 1, 4 and 8.
  Seeds and cohort hashes follow the generic training runner.
- Select the checkpoint with minimum validation prediction MSE. The feature basis
  is fixed; this is not the learned-representation collapse selector. Report image
  and state errors separately, alongside persistence and action-shuffle controls.
- Maximum 1,800 true-wall seconds, including loading, normalization, training and
  validation; 16 GiB host-memory guard. A supervisor terminates execution at the
  hard wall cap and preserves partial outputs. Do not retry an interrupted run in
  the same output directory.
- Write `checkpoints/apple-sensor-v1/sensor.pt` plus the runner's latest, curve and
  report artifacts. Existing paths are refused; negative results are retained.

Uniform training-window sampling gives every recorded transition equal chance;
collection deliberately balances nominal and perturbed episodes. This does not
imply equal time in each manipulation phase. Record actual per-phase coverage and
flag insufficient failed-action support as a limitation rather than claiming a
balanced phase sampler that is not implemented.

## Gates before fresh task testing

The selected model must have finite recursive predictions, respond to changed
candidate actions, and improve validation prediction over persistence and shuffled
actions on the declared horizon. Report failures without silently choosing another
selector. Offline validation does not establish successful control: dense TRAIN
image waypoints and candidate planning next undergo separate bounded development
trials with complete stage metrics and dynamics ablations. Final test resets and
thresholds will be frozen before final evaluation, after development concludes.
