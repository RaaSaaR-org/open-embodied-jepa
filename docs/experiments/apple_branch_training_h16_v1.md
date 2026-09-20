# Apple matched-branch H16 training v1

Prospective TASK037 protocol. The prior H8-trained branch model passed the
secondary H8 criterion but its primary VAL H16 endpoint error reduction was 6.26%,
below the frozen 10% gate. No controller experiment follows that failure. Test
whether matching the training horizon to H16 while retaining substantial action
intervention coverage improves the primary causal metric. This jointly changes
horizon and sampling distribution; any improvement cannot be attributed to
horizon alone. Existing results and checkpoints remain immutable.

## Frozen inputs and method

Use only `data/apple-branches-v1`, manifest SHA256
`6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`.
Require its completed collection, integrity and action-contrast evidence. Preserve
all parent/session split assignments. Never decode TEST or use final control seeds.

One run through `scripts/train_apple_sensor.py --profile branches_h16_v1`, output
`checkpoints/apple-branches-h16-sensor-v1/`: unchanged sensor_wm architecture,
optimizer and configuration defaults; seed 0; CPU four threads; 3,000 updates;
batch 16; H16; validation every 300 updates, four batches; best checkpoint chosen
only by unadjusted H16 validation prediction MSE, averaged across all 16 recursive
future steps in the normalized visual representation. This is not endpoint-only
error or raw-pixel MSE; the existing selector name is `raw_mse`. Preserve best and
latest checkpoints.
The 600-second parent-supervised true-wall budget includes configuration,
normalization, training and finalization; memory ceiling 16 GiB. No retries or
extensions. Failures/timeouts remain results and do not authorize control.

Before optimization derive and write `sampling_groups.json`, then register its
exact file SHA256, canonical content hash, groups, quotas, model configuration,
data/split/action/source/protocol hashes, and full command. For BOTH TRAIN and VAL,
group 0 contains all original demonstration episodes; group 1 contains episodes
with `metadata.parent_episode_id`. The original group includes recorded perturbed
and failed demonstrations, not just successful nominal trajectories. Every branch
must retain an original parent in its own split and the parent's session. Group
membership must cover each allowed split exactly once, with no duplicates,
empty groups or TEST members. Every group must provide H16 windows.

Each batch contains eight original-demonstration windows and eight branch windows.
Choose windows uniformly with replacement inside each group, then shuffle batch
order. The validation cohort is fixed by the existing deterministic validation
seed rule and uses the same 8+8 quotas; checkpoint selection therefore cannot
mostly measure long demonstrations. At H16 each complete 16-transition branch
contributes one eligible window. At shorter prespecified diagnostic horizons,
retain equal group quotas and sample uniformly among their eligible windows.
Normalization retains the existing all-TRAIN-transition population fit; balancing
loss exposure does not change normalization semantics. No VAL fitting occurs.

The generic runner's default remains uniform over all allowed windows with the
same RNG trajectory. Explicit groups require exact coverage, equal integer batch
quotas and at least one window in every group; reject incompatible inputs instead
of silently dropping groups or redistributing quota. Reports/checkpoint metadata
record the group configuration, canonical hash and sampler RNG state. The
supervisor checks the frozen group file and child report after optimization.

## Unchanged causal decision

After training, separately authorize the same matched diagnostic defined in
`apple_branch_diagnostics_v1.md`. Use all planned TRAIN/VAL roots and matched
actual-action sibling comparisons, with no new root selection or thresholds.
Primary VAL H16 requires at least six informative roots across two parent
sessions, root-mean action ranking >= 0.70, raw visual endpoint error reduction
R >= 0.10, and root-bootstrap 95% lower bound of R strictly above zero. Retain
2,000 bootstrap draws with seed 20260921, the same informative-pair screen and tie
rule, and parent-session cluster intervals as sensitivity evidence. H8 and TRAIN
remain secondary; three VAL sessions still limit confidence.

Raw validation loss chooses the checkpoint; matched diagnostic outcomes never
choose an alternate checkpoint. Report previous sensor-v1 and branches-v1
results alongside this new attempt without rewriting them. A causal diagnostic
pass permits discussion of a separately frozen development controller experiment,
not a claim of manipulation success. A failed primary gate stops that progression.
