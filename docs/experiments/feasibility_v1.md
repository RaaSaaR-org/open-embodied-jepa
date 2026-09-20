# TASK-029 feasibility projection diagnostic — preregistration

This bounded development experiment asks whether scoring embodiment-projected
actions reduces joint-rate rejection and requested/applied mismatch relative to
the original CEM behavior. It does not retrain models or replace the frozen MVP
comparison. All inspected MVP goals and outcomes are now development evidence.
Reusing them here is not an independent test of generalization.

## Fixed paired comparison

- Both frozen **training-seed-1** checkpoints, native JEPA and actual LeWM; no
  architecture, optimizer, weight or checkpoint-hash edits. Seed 1 is fixed before
  this diagnostic, not selected after seeing its projection result.
- Environment seeds **20000, 20001, 20002, 20003, 20004**, with original frozen
  Apple→Plate goal pixels and resets. The same physical proxy, thresholds, action
  manifest and CPU four-thread setting apply to every arm of the comparison.
- Projection **off/on** for each backend/reset pair: **20 planned episodes**.
  Deterministic order: environment seed, native then LeWM, projection off then on.
- Original CEM horizon **4**, **64** candidates, **3** iterations, **8** elites,
  original action bounds and defaults, **5-second** control deadline. Each episode
  stops after at most **100 executed steps**. This is a shortened diagnostic,
  not the original 805-step full-task benchmark.
- One benchmark subprocess per episode. A single **600-second true-wall budget**
  covers setup and the entire comparison, measured as the maximum of wall and
  monotonic deltas. The supervisor kills an active child at the remaining budget;
  0.1-second polling and OS termination may add a small overrun. There is no extra
  per-episode timeout, retry, budget extension or result-dependent early success
  rule. If slow early episodes consume the budget, later planned episodes remain
  explicitly unstarted; incomplete pairs cannot establish an improvement.

Only `planner.project_candidates` differs within each matched pair. The opt-in
path asks the embodiment to project candidates through its joint/grasp limits,
then scores those projected actions with the unchanged world model. No privileged
object state enters prediction or planning costs. Keep the original requested
candidate, model-scored candidate and accepted action in the trace.

Both conditions use the same **revised execution runtime**, including synchronized
robot-only snapshot kinematics. This removes stale derived transforms left by live
MuJoCo integration from execution's feasibility check. Projection-off is therefore
the paired control under the revised runtime, not a claim to reproduce historical
MVP trajectories exactly. Neither condition uses object truth for projection.

## Frozen inputs and goal lineage

The committed `PINNED_INPUTS` mapping in
[evaluate_feasibility.py](../../scripts/evaluate_feasibility.py) is the executable
input manifest. It also pins both inherited base configuration files and the robot
asset manifest. Principal SHA-256 values are:

| Input | SHA-256 |
|---|---|
| Shared corpus | `200246b34bd18cb30c8c711879c42a5ed2f31a62c99b96c1b347e37aa57512b7` |
| Native seed-1 checkpoint | `62125697464d56bcb79ece37008d6afd81b62505f0fab369b5aaa13510b1c33a` |
| LeWM seed-1 checkpoint | `ab00894e48c1fac0c862fa5b37cb648f0d673704d6afb839980c2f45611b2dd5` |
| Action manifest file | `f247effe9a9cd5e9ae3a9ad8e13ed036fa652f5a0d39f96b14c98979642c38b1` |
| Original 50-goal manifest | `eef30af440d40064efecc98a84a68d4388f9f60d1151104b4859a8f9934ee38e` |

The goal loader requires exact seed lists and runtime hashes. Therefore the runner
creates explicit one-seed **development reuse** manifests, preserving original
reset/threshold fields and pixel hashes while binding runtime hashes to the new
feasibility implementation. It records the parent manifest hash and original
runtime hashes, copies the original image bytes, and validates each derived
manifest through the normal loader. This is a declared runtime rebinding, not a
claim that the historical frozen runtime was unchanged. Original files remain
untouched. The root coordinator must review/commit the implementation and supply
its exact Python source hash before execution. Pinned manifest/checkpoint/config
hashes, generated configs, derived goal manifests/images and that source hash are
checked before and after each episode; a change invalidates the run and stops the
remaining comparison. Dataset payload integrity is checked by the benchmark's
normal DatasetStore loader. Reports record script and protocol hashes as well.

## Predeclared metrics and interpretation

For every completed episode record joint-rate stops, any guard/runtime stop,
executed steps, termination reason, final reach/grasp/transport/place/release stage
flags, and full task success at the 100-step cap. `step_limit` is an ordinary
budget termination, not a guard stop. `success` at this cap is recorded as observed;
failure at the cap does not establish failure at the original longer horizon.

Record planning, projection and whole-control latency distributions (count, mean,
median, 95th percentile and maximum). For action comparisons report paired-step
counts and mean/maximum absolute normalized-coordinate differences across 14D:

- `sampled_action` versus `projected_action`: original CEM request versus the
  candidate scored by the model, available only in the projection-on condition.
- `requested_action` versus `action`: actual execute request versus accepted action.
- `sampled_action` versus `action`: original CEM request versus accepted action.

Projection-off has no `projected_action`; absent values are null with a missing
reason, never invented zeros. Rejected commands have no applied action and are
excluded from action-error averages but remain in stop counts. Timeouts or failures
without a completed episode record retain unknown metrics; no fabricated partial
execution count is inferred from absent traces. Every planned run gets a status,
including attempted, incomplete and unstarted outcomes. Report paired differences
for completed matched pairs and their counts, alongside all failures. Five reused
resets per backend are a small diagnostic sample; no broad success or significance
claim follows from fewer stops alone, especially if projection misses deadlines.

## Execution and artifacts

No experiment has run under this protocol at registration. After coordinator
authorization, run from the reviewed worktree with its source hash:

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_feasibility.py \
  --source-sha256 REVIEWED_PYTHON_SOURCE_SHA256 \
  --output outputs/feasibility-v1
```

The runner explicitly sets child `PYTHONPATH` to this worktree's `src`, avoiding a
shared editable virtualenv importing an older checkout. Existing output directories
are refused. `comparison.json` contains all 20 planned records and metrics;
per-attempt logs, generated configs/goals, benchmark run manifests and completed
episode traces remain adjacent. Keep negative and incomplete outcomes. Summarize
results in a separate document so this protocol's preregistered bytes are preserved.
