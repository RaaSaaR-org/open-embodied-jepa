# Apple matched-branch training v1

Prospective TASK035 training protocol, written before branch collection or training. Question: does adding sustained matched action alternatives improve causal prediction under the existing sensor model and optimizer recipe? The architecture, seed, step count and training horizon remain identical to the first task-specific run; this isolates the changed training distribution. Fresh final control seeds remain untouched.

## Frozen method and budget

Use only a sealed, reviewed `data/apple-branches-v1` corpus produced by `apple_branches_v1.md`. The coordinator must first inspect replay/restoration agreement, split inheritance, incomplete prefixes and observed endpoint divergence. No training on an incomplete or integrity-failed corpus. Record its exact manifest and source/protocol hashes in the supervisor registration before optimization.

Run `scripts/train_apple_sensor.py --profile branches_v1` with exact `--dataset-sha256`, `--source-sha256`, and `--protocol-sha256`. Output is a new immutable directory `checkpoints/apple-branches-sensor-v1`. One run: sensor_wm defaults, CPU four threads, seed0, 3,000 updates, batch16, horizon8, validation every300 updates with four batches, raw-MSE checkpoint selection and16GiB memory guard. Total parent-supervised budget600 true-wall seconds includes startup, configuration resolution, fitting and finalization. No retry or extension within this protocol.

TRAIN-only normalization is refitted from the new TRAIN partition; full original demonstrations and added branches are sampled using the existing uniform-window sampler. Branch roots are correlated siblings, not independent episodes for uncertainty claims. All siblings retain their parent's session and split. TEST is never decoded. Raw prediction-MSE selection is kept fixed for this data-only comparison and is explicitly insufficient for passing the causal gate.

## Assessment and decision

Compare the selected model and the original frozen sensor-v1 model on the identical new TRAIN/VAL branch cohorts using `apple_branch_diagnostics_v1.md`. Each checkpoint retains its own source-corpus normalization; no fitting on VAL, TEST or branch diagnostic outputs. Report matched within-root action assignment, visual/proprio endpoint error, future visibility, ties and task-phase counts at H8/H16. Also report persistence and wrong-action predictions. Original v1 results remain unchanged.

A useful causal signal requires lower endpoint error with actual actions than sibling actions and correct ordering of observed goal distances on distinguishable branch futures, rather than only lower unconditional prediction loss. Before any further physical controller experiment, inspect the preregistered diagnostic and declare whether its gate passed. Any controller change, alternate selection rule, additional training run or architecture change is a separate prospective experiment; no final-MVP claim follows from this training alone.
