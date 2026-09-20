# Frozen MuJoCo MVP comparison

**All 400 planned attempts completed. Learned Apple→Plate manipulation was unsuccessful: 0/150 for each model, with 0/50 for both hold and random controls.** The software/data/training/evaluation path is implemented and reproducible; this experiment does not demonstrate useful full-task control.

The [protocol](mvp_evaluation.md) was fixed before evaluation. Six eligible checkpoints, three training seeds per backend, were evaluated on the same 50 frozen resets and image goals. All runs used the same dataset, actions, task, CPU device and CEM budget. A serial supervisor finished in **595.082 s**, within its 1,800 s limit, with no missing attempts, retries, deadline misses or orchestration failures. Physical guard stops count as task failures.

## Physical outcomes and compute

| Policy | Training seed | Full success | Reach / grasp / transport / place / release | Termination | Plan p50 / p95 ms | Peak host RSS MB |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| native_jepa | 0 | 0/50 | 0 / 0 / 0 / 0 / 0 | stopped: 50 | 0.964 / 1.052 | 1840.0 |
| leworldmodel | 0 | 0/50 | 0 / 0 / 0 / 0 / 0 | stopped: 50 | 7.011 / 7.264 | 1888.0 |
| native_jepa | 1 | 0/50 | 0 / 0 / 0 / 0 / 0 | stopped: 50 | 0.976 / 1.054 | 1829.0 |
| leworldmodel | 1 | 0/50 | 0 / 0 / 0 / 0 / 0 | stopped: 50 | 6.996 / 7.295 | 1881.4 |
| native_jepa | 2 | 0/50 | 0 / 0 / 0 / 0 / 0 | stopped: 50 | 0.967 / 1.039 | 1836.6 |
| leworldmodel | 2 | 0/50 | 0 / 0 / 0 / 0 / 0 | stopped: 50 | 7.004 / 7.297 | 1888.4 |
| control:hold | — | 0/50 | 0 / 0 / 0 / 0 / 0 | step_limit: 50 | not applicable | 1812.3 |
| control:random | — | 0/50 | 0 / 0 / 0 / 0 / 0 | stopped: 50 | not applicable | 1753.4 |

Each 0/50 row has a 95% Wilson interval of approximately **[0, 7.13%]** on its own fixed reset distribution. The same 50 physical resets are reused across training seeds; the descriptive 0/150 backend totals are not 150 independent environments and receive no pooled binomial interval. These zero success rates do not rank the learned representations or establish real-world reliability. All stage counts are also zero.

All 300 model episodes stopped on **right joint rate limit**. Native executed 1,117 / 620 / 670 commands across seeds 0/1/2; LeWM executed 239 / 706 / 461. Each run attempted another 50 plans whose commands were rejected. Hold executed all 805 steps in every episode; random also stopped under guards. No limit was relaxed and no failed episode was removed.

CEM evaluates candidate normalized commands before state-dependent actuator feasibility. The preregistered limitation is broader than grasp clipping: the observed model failures are arm IK target-rate rejections. A candidate can score well in latent prediction and still be infeasible for the current robot. This is the immediate observed failure mode; these results do not isolate it as the only cause of unsuccessful learning/control. TASK-029 tracks common pose/grasp feasibility projection and a separately declared comparison.

Planning latency includes the fixed 64 candidates × 3 iterations at horizon 4, with four Torch CPU threads. It excludes rendering and physical execution. Both backends ran serially after training; development process/RSS measurements and one Mac are not a hardware deployment guarantee. Process RSS includes dataset validation, imports and model setup. Forbidden-contact classification is unavailable and recorded as null with a reason, not zero collisions.

## Model evidence and interpretation

The [training report](mvp_training_results.md) records all six 3,000-update runs, their selected checkpoint hashes, validation curves and controls. Native prediction improved on its own persistence and shuffled-action baselines for all three seeds. LeWM improved on shuffled actions but was worse than persistence for every seed. All selected models passed the declared per-dimension online/target collapse guards; low effective ranks of 2.59–3.56 remain a limitation. Neither passing those guards nor prediction improvement implied successful control.

The corpus contains 184 episodes and 42,127 transitions, with inherited 146/19/19 partitions. Apple→Plate is an unseen pairing of individually seen procedural objects/containers and appearances. The independent leakage audit found no cross-split session or exact RGB overlap and no goal/corpus matches. It did not test semantic near-duplicates. No successful supplementary release episode falls in validation; its original assignment was retained.

The privileged collection controller establishes mechanical feasibility on seen pairs: 4/4 fixed valid-reset probe successes and 9/24 supplementary collection successes. It is not a learned policy and was not substituted into the final comparison. Earlier reach v2 results (1/5 per model, hold/random 0/5) are separate development evidence, not this full-task result.

## Reproduction and retained artifacts

```sh
.venv/bin/python scripts/evaluate_mvp.py
.venv/bin/python scripts/summarize_mvp.py
```

These commands assume the recorded corpus, selected checkpoints and frozen goals exist. They refuse existing output paths. Use a fresh checkout/output cohort for a new experiment; never overwrite this comparison or present an inspected cohort as a newly sealed test. The training and evaluation protocols contain the preparation commands. For a small fresh-data software demonstration, run `scripts/reproduce_smoke.py` as described in [clean_reproduction.md](clean_reproduction.md).

The versioned [machine-readable comparison](../../benchmarks/manifests/mvp-results-v0.json) includes selected horizon 1/4/8 diagnostics, all hashes, runtime summaries, exact commands, candidate throughput, replans and RSS. The summarizer independently validates all 400 episode records, matches evaluated checkpoints to selected training weights, checks the unchanged Python source hash, and verifies identical data/split/action/goal images and common planner settings.

Full local run directories are `outputs/mvp-v0-{native_jepa,leworldmodel}-seed{0,1,2}/`, `outputs/mvp-v0-hold/`, and `outputs/mvp-v0-random/`. Each retains run/config/dependency/training-report JSON, `episodes.jsonl`, summary, and per-reset initial/goal/final images. Model runs save every fifth observed frame. `episode-20000/` is a fixed representative first reset, not a selected success. Large data/checkpoints/rollouts remain local; public source contains their compact manifests and reports.

Executed source: `e57e28b53eb682102b882dbada937fbcefffb01f`; Python tree SHA-256 `0e9adee4051660dfe03695ef0e7f18e1432a996d6a4833357e580afa06319bb0`. The dirty flag reflects concurrent non-source documentation/artifact preparation, with source, goal and training protocol identities unchanged during all runs.
