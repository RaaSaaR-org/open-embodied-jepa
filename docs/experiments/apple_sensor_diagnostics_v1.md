# Apple sensor model phase diagnostics — preregistration

The selected sensor model beats persistence in aggregate validation, but its
reported shuffled-action advantage is about1%. This read-only development
diagnostic asks whether that weak advantage hides phase-specific action-ranking
or observation-visibility gaps. It neither fits a model nor changes checkpoint
selection, physical scoring, goals or sealed test outcomes.

Freeze the selected seed0 checkpoint at
`checkpoints/apple-sensor-v1/sensor.pt`, SHA-256
`3a5c5e7ba77556b6ce4251f05af72dc86df0121807d913ca8b444db60026e3ee`,
and `data/apple-task-v1` manifest SHA-256
`d70edd9daaeec58e80a4e6a743980c049335a03f2b32b7d42bbac5ec5bc579df`.
The checkpoint is selected from the prior declared validation procedure; this
phase analysis cannot substitute a checkpoint or retune the model afterward.

For **TRAIN and VAL separately**, evaluate horizons **4 and 8** in each of these
six phases: orient, descend, close, lift, transfer and release-high. `release_high`
is the explicit operational definition of release for this analysis; the single
failed episode's `lower_open` tail is excluded rather than silently mixed into it.
Every candidate window must have all H action labels in the same phase, stay
within one episode, and include H+1 sensor observations. Select at most **32
windows without replacement** per split/phase/horizon from sorted episode IDs and
ascending start indices. RNG is `SeedSequence([123, split_index, phase_index, H])`,
using the declared order above and TRAIN=0/VAL=1. Preserve selected indices and
cohort hashes. Scarce or unavailable groups remain explicit; do not replace them
with another phase. The design has **24 groups** and at most768 windows.

Use the unchanged model's diagnostics to report normalized visual prediction,
persistence, shuffled-action and zero-action MSE, learned proprioception error,
and action-effect RMS. Also report prediction/persistence ratio and relative
shuffled-action improvement. The model's shuffle cyclically shifts flattened
batch/time actions by one; report actual shuffled-action RMS, fraction of changed
commands and per-channel action standard deviations so near-identical actions
cannot masquerade as a strong negative control. Counterfactual physical futures
are unavailable in this corpus, so this diagnostic alone cannot establish correct
ranking of substantially different actions.

Report consecutive full-frame RGB RMS on [0,1] pixels as a crude observable-motion
measure. It is not object segmentation, contact visibility, or proof that the
task-relevant apple pixels are represented. No simulator truth enters this
diagnostic's model input or its phase cohort selection beyond recorded collector
phase labels used only to stratify evaluation.

Instantiate `sensor_wm` from its stored configuration through the common registry.
Strict load checks must pass for implementation/action/state schema. Verify
dataset/split/action metadata and normalization's exact training episode set.
Check checkpoint/dataset hashes before and after. Only selected TRAIN/VAL episodes
may be decoded and each is cached once. **No TEST image decoding or test windows**;
DatasetStore may hash stored test payload files for integrity without decoding.
Recheck all payload hashes at completion, along with source revision, Python
source digest, model implementation, diagnostic script and protocol hashes. Any
drift marks the diagnostic incomplete rather than silently accepting mixed code.

Resource budget: CPU four Torch threads, hard **60 child CPU seconds** and **120
true-wall seconds**, with internal execution cutoffs55 CPU/115 wall to preserve
partial reports. No retries or extensions. The outer supervisor counts imports,
setup and finalization, terminates on wall exhaustion, records incomplete planned
groups and returns nonzero. Existing outputs are refused. This is a bounded
diagnostic, not training or a new final-control comparison.

```sh
PYTHONPATH=src .venv/bin/python scripts/diagnose_apple_sensor.py \
  --output outputs/apple-sensor-phase-v1
```

Commit this protocol and runner before execution. At registration, only prior
aggregate training metrics are known; these phase cohorts have not been scored.
Results and subsequent collection hypotheses belong in a separate results report.
