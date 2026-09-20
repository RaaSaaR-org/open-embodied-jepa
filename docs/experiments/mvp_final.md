# Final MVP experiment

## Preregistered training protocol

This section fixes training and checkpoint selection before the final corpus and
evaluation are run. The question is whether the unchanged compact native JEPA
and actual pinned LeWM models learn useful action-conditioned visual predictions
from one shared simulation corpus. Earlier reaching pilots and controller trials
are development evidence; their outcomes motivated the corpus, not a claim of
independent generalization. Failure to outperform persistence or to control the
robot remains a negative result.

The final source list is `data/reach-pilot-v0`, `data/manipulation-pilot-v0`, and
`data/manipulation-release-v1`. The new release corpus must pass recording,
action-schema and sealed-split validation. Keep all failed trials; do not filter
episodes by downstream learned-model results. Controller diagnostic v1/v2 trials
are excluded. Assemble to `data/mvp-v0`; the coordinator must confirm its exact
manifest hash before training. No final corpus hash is asserted until that freeze
occurs. Model implementation is frozen at commit `75ca290`.

The assembled corpus is now frozen at manifest SHA-256
`200246b34bd18cb30c8c711879c42a5ed2f31a62c99b96c1b347e37aa57512b7`:
184 episodes, 42,127 transitions, 146 training / 19 validation / 19 test / zero
holdout episodes. Canonical normalization lists exactly the 146 training episodes.
Its split-and-policy hash is
`5a4bb221b021adea9d3ba5e56a634fbe5e4dae8f77372f8ed6fe5eca2e5cf4f6`.
Source manifests, in assembly order, are:

- Reach pilot: `9a5256cc2070d2bc8c44f44b741830e428a33682c5baf0719c7973c62231804f`.
- Manipulation pilot: `8152aa39ce50de0bc37d639051836310e7c2056be932c4d33b00ca58c62d19ec`.
- Release v1: `f156c1b9816eefffbca929f0fe20413efce4334c9fd25760d57b764e1b991980`.

Release v1 recorded 18 episodes from 24 fixed attempts, preserving all recorded
failed prefixes. Its nine successful episodes inherit eight train / zero validation /
one test assignments. This limitation is preserved, not repaired by reshuffling
after observing success; no test performance influences training or selection.

`scripts/assemble_mvp_data.py` creates a new output directory through the public
DatasetStore APIs. It preserves every original train/validation/test/holdout
assignment and original session ID, prefixes episode IDs with the source manifest
hash, and records source paths, manifest/split hashes, original IDs and licenses.
Cross-source session conflicts, duplicate sources, incompatible state/action/FPS/
camera schemas, and Apple→Plate episodes are rejected. Assembly copies all
partitions for archival fidelity; training decodes **only train and validation**.
Source episodes are never altered, splits are never refitted, and an existing
output is never overwritten. Assembly fits canonical state/action normalization
statistics using only the inherited training partition after sealing splits;
model image preprocessing and train-only latent scaling remain model-owned.

The fixed six runs interleave native JEPA then LeWM for each seed **0, 1, 2**:

- **3000 updates**, batch **16**, horizon **4**, CPU **four threads** per run.
- Unchanged registry model defaults; visual-only H1 inputs, no architecture or
  optimizer changes, no warm starts and no per-backend tuning.
- **600 seconds maximum per run**, further limited by the remaining shared
  **1080-second (18-minute)** experiment budget. No extensions or retries.
- Maximum host RSS budget **16 GiB**, monitored by the existing runner; this is
  an observed process high-water threshold, not an OS-enforced allocation limit.
- Validation at step zero, every **100 updates**, and the final step; fixed four
  batches of 16 windows for each available horizon **1/4/8**. Validation sampling
  uses `seed + 10001 + horizon`, identical between backends for each seed.
- Selector **noncollapsed_relative**, metric definition **2**, at horizon four:
  both online and target mean latent standard deviation must be at least **0.1**
  and each collapsed fraction at most **0.05**. Among eligible states minimize
  prediction MSE divided by `max(target-space persistence MSE, 1e-12)`; ties retain
  the earlier checkpoint. No eligible checkpoint means selection failure, with
  latest weights retained if runner finalization completes.

Persistence compares target-encoder current and future images in the same space.
Record prediction, shuffled-action and persistence errors; online and target
collapse statistics; target effective rank; checkpoint step and all eligibility
rejections. Raw latent losses and normalized ratios from different learned spaces
do not establish between-model superiority. Report all three seeds individually
and aggregate their variation; do not select the best seed after evaluation.

`scripts/train_mvp.py` requires the confirmed dataset SHA-256, refuses existing
output directories, records commands, source/script/protocol hashes and run order,
and refuses subsequent runs if the source or dataset changes. Source code and
this protocol must remain stable during execution. Each run has its own process,
log, metrics, run report and best/latest checkpoints. Completed outputs and
failures remain in place. Supervisor timeout may leave an incomplete child report
or no latest checkpoint; the outer report explicitly records this failure rather
than claiming successful finalization. Budget timing includes child imports and
finalization. The supervisor polls every 0.1 seconds and kills at the boundary;
OS scheduling/termination adds a small unavoidable overrun. Both clocks use
`max(wall-clock delta, monotonic delta)` so host suspension consumes budget.
Child reports additionally distinguish process CPU time from elapsed time.

After source review and corpus confirmation, commands take this form (replace
the SHA with the actual frozen hash):

```sh
.venv/bin/python scripts/assemble_mvp_data.py \
  --source data/reach-pilot-v0 --source data/manipulation-pilot-v0 \
  --source data/manipulation-release-v1 --output data/mvp-v0
.venv/bin/python scripts/train_mvp.py --dataset data/mvp-v0 \
  --dataset-sha256 200246b34bd18cb30c8c711879c42a5ed2f31a62c99b96c1b347e37aa57512b7 \
  --output checkpoints/mvp-v0
```

No training or final evaluation has been run under this protocol at registration.
Outputs use `checkpoints/mvp-v0/seed-{0,1,2}/{native_jepa,leworldmodel}.pt` with
adjacent latest checkpoints, reports, curves and logs. Evaluation cohorts, goals,
controls, planner settings and physical success criteria are preregistered
separately in [mvp_evaluation.md](mvp_evaluation.md).
