# Evaluation and acceptance evidence

## Common-mode protocol

Freeze dataset/split hashes, robot and action manifest, simulator/assets, task resets, camera views, image-goal pool, planning budget, seeds, and metric code. Change `world_model.backend` only in the shared experiment config; adapter defaults and checkpoint selection resolve inside that backend and are archived. Both models train on the same canonical corpus. Log architecture, preprocessing, optimizer settings, and training compute separately, since they need not match.

Use the same CEM candidate count, horizon, iterations, elite fraction, action penalties, and timeout. Report actual wall time and hardware alongside equal candidate budgets. A faster backend does not get more candidates in common mode. Native mode is a separately labeled future experiment, never pooled into common-mode tables.

Proposed smoke cohort: 5 fixed resets, one training seed. Proposed final cohort: 50 fixed resets for each of 3 training seeds, shared across models; expand if intervals are too wide. Freeze this protocol before final evaluation. Report denominators, failures, timeouts, and confidence intervals; never count only successful runs.

## Metrics

| Group | Measurements |
| --- | --- |
| Model | One-step and multi-step error at horizons 1/4/8, per-dimension latent variance, effective rank, action sensitivity, training duration, peak memory (CUDA VRAM, MPS allocated memory, or process RSS, explicitly labeled) |
| Runtime | Encode/predict/plan latency p50/p95, candidates per second, deadline misses, executed steps, replans |
| Robot task | Reach/grasp/transport/place/full success, episode time, collisions or limit violations, termination reason |
| Controls | Random bounded actions, hold action, shuffled-action prediction, scripted/oracle controller to validate task reachability |

Raw latent distances are not comparable across independently learned feature spaces. Use them for within-backend learning/collapse diagnostics and compare physical task success and throughput across backends. Low prediction loss alone is insufficient evidence of useful dynamics. Action-shuffling and zero-action controls should expose whether the predictor ignores actions.

## Task semantics to freeze in TASK-018

Reach means end-effector position within a configured tolerance of the target for a fixed dwell period. Grasp means stable object lift above its initial support; transport means the held object enters the target region. Place means the object is supported within the plate region after release. Full success requires the object to remain on the plate for a dwell interval with no hand contact. Use simulator geometry/contact ground truth solely for evaluation, not model observations or planning cost.

Exact distances, heights, dwell time, timeout, and collision policy must be calibrated to the assets and frozen before model comparison. TASK-018 records these values and boundary tests. For hardware, record manual or sensor-based scoring and its uncertainty separately.

## Completion versus scientific result

Engineering acceptance requires passing API/data/adapter checks, reloadable trained models, finite action-dependent predictions, a model-independent closed-loop planner, complete Apple → Plate evaluations for both models, and reproducible result records. The initial reach slice should outperform random/hold controls on the fixed validation cohort; failure triggers model/data diagnosis before adding grasp complexity.

PRD §19 sets no numerical final success threshold. Do not invent a promised full-task success rate. Report observed rates and intervals, including zero-shot failures. Any product-level target added later must be recorded before the sealed final test, rather than chosen after seeing results.

## Result artifacts (TASK-017)

Write `run.json`, `episodes.jsonl`, `summary.json`, resolved configuration, dependency snapshot, and rollout references under `outputs/<run_id>/`. Schema version 1 must require run ID, timestamp, source revision, backend/checkpoint hash, common/native mode, dataset/split/action hashes, environment/device, simulator engine/version, planner budget, train/eval seeds, task version, per-stage success, termination reason, timings, and replans. Missing measurements use null plus a reason, never a fabricated zero. Store model metrics separately when they are not meaningful per episode.

TASK-023 assembles an acceptance report linking every PRD criterion to commands, checks, checkpoints, run IDs, and results. No physical outcome may be inferred from a simulator result.

## Executed MVP protocol

The proposed cohort above was adopted as 50 fixed Apple→Plate resets for each of three training seeds per model, plus 50 hold and 50 random controls. The preregistered [training protocol](experiments/mvp_final.md) and [evaluation protocol](experiments/mvp_evaluation.md) define exact budgets, reset distributions, limitations and failure accounting. The goal manifest and compact corpus manifest are versioned under `benchmarks/manifests/`; large local payloads are kept out of Git. [Clean reproduction](experiments/clean_reproduction.md) separately verifies installation and a small end-to-end run.

**Measured outcome: 0/50 in every one of the eight runs**, 400 attempts recorded, with no backend beating the hold or random controls ([results](experiments/mvp_results.md)). The semantics above still govern any new comparison.

## Task-specific apple protocols since the MVP

Later work is a separately labelled task-specific mode on the apple corpus, not this common-mode benchmark, and must never be pooled into a common-mode table. Each protocol preregisters its own gates, thresholds and decision rule before the run, and its results document records the outcome including failures; both live under [docs/experiments/](experiments/) with a machine-readable manifest under `benchmarks/manifests/`. Every gated **learned** closed-loop attempt in that line has failed its declared gate, and **learned Apple→Plate is still 0 successes**. (Some non-learned privileged-ceiling diagnostics did pass their gates; they establish that a design is adequate under exact dynamics, never that a learned policy works.) At TASK-054 CEM over the world-model cost was abandoned as the primary control line in favour of behaviour cloning with the world model as a critic, and no further predictor-architecture protocol is preregistered. That behaviour-cloning line was preregistered and run as TASK-056 ([protocol](experiments/apple_policy_v1.md)) and diagnosed in TASK-057 ([results](experiments/apple_policy_diagnostics_v1_results.md)), where its own pre-declared abandonment clause fired: it stops on this corpus (`apple-wide-v1`) and this camera (112 px onboard), and no third control formulation is preregistered on them. There is currently no primary control line. The next task is a perception/data task, TASK-059 ([information-ceiling probe](experiments/apple_info_ceiling_v1.md)), whose thresholds and outcomes are preregistered in its own protocol; see [DECISIONS.md](DECISIONS.md).
