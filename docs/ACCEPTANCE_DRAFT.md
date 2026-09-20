# MVP acceptance audit — development snapshot

Audit date: 2026-09-20. Reviewed the working implementation based on Git
`9afff34922100e4de3ba77178fe4bd74240ad0a6`, including explicitly identified
uncommitted work. This is a draft while collection and model experiments continue;
it does not close MissionControl tasks or approve a PR. Later evidence must update
the affected rows rather than silently supersede negative experiments.

**The complete MVP validation is not yet demonstrated.** Shared software, real
MuJoCo data, two trained adapters, and common image-goal execution exist. Useful
learned reaching remains an open gate; the frozen Apple→Plate comparison and clean
Mac reproduction have not been demonstrated by the artifacts inspected here.
PRD §19 specifies no numerical final success threshold. A negative completed
benchmark can be valid evidence; an unrun benchmark cannot.

Statuses mean **met** within the stated local scope, **unmet** when an observed
requirement is absent or contradicted, and **unverified** when execution/review
evidence is still missing. Mock or fixture evidence only satisfies software
criteria. Hardware/Isaac preparation is intentionally sufficient for their local
MVP tasks; physical execution and Isaac parity remain future tasks.

The reviewer independently inspected training, model/benchmark integration and
previous simulation fixes. The reviewer authored the contracts, dataset and
hardware mock components earlier and the result-schema correction during this
audit; those sections are evidence checks, **not independent code approval**.

## Evidence checked

- Independently executed `.venv/bin/pytest -q tests/test_training.py`: **9 passed**
  on the v1 runner before the v2 metric changes. No test/holdout decoding or
  train/validation session-leakage defect was found in that runner.
- Independently executed `.venv/bin/pytest -q tests/test_config.py
  tests/test_planning.py tests/test_task.py tests/test_benchmark.py`: **38 passed**.
- Executed `.venv/bin/pytest -q tests/test_result_schema.py tests/test_benchmark.py`:
  **41 passed** after implementing strict schema validation. Ruff format/check
  passed on those owned files. Both complete `outputs/reach-v1-*/run.json` and
  `episodes.jsonl` pairs passed the new validator; v0 records lack explicit stage
  nulls and remain historical rather than being rewritten.
- Rechecked `.venv/bin/pytest -q tests/test_result_schema.py tests/test_benchmark.py
  tests/test_training.py tests/test_models.py`: **66 passed** after the v2
  target-space correction and benchmark validator wiring (before adding two
  additional schema cases, which passed in the 41-case check above). Inspected
  the corrected same-target-space persistence and online/target eligibility code.
- Inspected `outputs/integration-results-local.xml`: **223 cases, 222 passed,
  1 graphics opt-in skip, no failures/errors**. This is an existing local result,
  not a newly reproduced clean environment or evidence for subsequent changes.
  It includes the actual pinned LeRobot reader for 1D, 2D and 86D state schemas,
  CPU/MPS model checks and SDK2 mocks. Relevant commands and individual tests are
  linked below; graphics-dependent evidence is separate.
- Earlier independent simulation regression checks confirmed that grip→stop→grip
  respects the per-step joint bound and a π orientation error is not falsely
  accepted as a converged IK solution. These are software/physics checks, not
  hardware calibration.
- Subsequently executed `JEPA_TEST_RENDER=1 .venv/bin/pytest -q tests/test_goals.py
  tests/test_config.py tests/test_planning.py`: **75 passed**, including one real
  frozen image/reset for each task, matching manipulation-training XY sampling,
  hash/path/reset/target rejection, inherited goal paths, requested/applied
  clipping distinctions and uncertain-execution trace retention. Repeated OpenGL
  rendering varied by one intensity level in three channels; exact reset state
  reproduced, and stored goal-image bytes remain strictly hash checked on use.

All `data/`, `outputs/` and `checkpoints/` references below are local, ignored
artifacts. Versioned metadata in `benchmarks/manifests/` preserves selected hashes
and summaries; it does not make the large payloads available to another checkout.

## PRD §19

| Criterion | Status | Evidence and scope |
| --- | --- | --- |
| G1 + Dex3 embodiment adapter exists | **Met** | `src/embodied_jepa/{simulation,embodiment}.py`, `configs/g1_sim_action.json`; `tests/test_simulation.py`, `tests/test_embodiment.py`. Dual-arm/hand MuJoCo control, named state, limits and synchronized observations. Physical calibration is excluded. |
| LeRobot-compatible G1 dataset loads | **Met** | `data/reach-pilot-v0/meta/jepa_manifest.json`; `data.py`, `tests/test_data.py::test_official_lerobot_reader`; `docs/DATA_FORMAT.md`. Genuine v3 PNG-in-Parquet profile read by pinned upstream, including G1's 86D schema; not arbitrary LeRobot-version compatibility. |
| `native_jepa` trains on the dataset | **Met** | `training.py`, `models/native.py`; `checkpoints/reach-pilot-v0/native_jepa.run.json` and v1 counterpart report 500 completed CPU updates on actual collected frames. Training execution does not establish useful dynamics. |
| At least one external JEPA backend is integrated | **Met** | `models/lewm.py` imports pinned unmodified LeWM classes; `scripts/fetch_lewm.py`, `docs/MODELS.md`, `tests/test_models.py`; actual corpus training report at `checkpoints/reach-pilot-v1/leworldmodel.run.json`. |
| Both implement the same WorldModel API | **Met** | `contracts.py`, `models/base.py`, `tests/test_models.py::test_shared_api_recursive_chunks_and_image_goal`; common canonical batches, opaque latents, costs and checkpoint methods. |
| Both support action-conditioned future prediction | **Met** | `models/{native,lewm}.py`, recursive-prediction and action-sensitivity tests; future state is not supplied to inference. Useful physical prediction remains unverified and has failed important controls in native pilots. |
| CEM works with both without model-specific code | **Met** | `planning.py`; `tests/test_planning.py`, both `outputs/reach-v1-*/episodes.jsonl` traces. Synthetic known-dynamics progress verifies planner mechanics separately from learned control. |
| Goal-image planning works in simulation | **Unmet** for the documented useful-control gate | Both adapters execute image-goal MPC, but both noncollapsed-selected v1 reach runs are 0/5; native v0's 3/5 used a collapsed representation. `docs/EVALUATION.md` requires meaningful reach progress before grasp expansion. No validated learned full-task outcome exists yet. |
| Identical benchmark can run against both | **Met** for development reach | `configs/reach_pilot_v1{,_lewm}.yaml`, `config.py`, `benchmark.py`; same seeds, action/planner bounds and budgets. Full manipulation evaluation is still unrun. |
| Results stored in common format | **Met** for development records | Both v1 runs contain common run/config/episode/summary/image artifacts. `result_schema.py` strengthens new-record validation; older v0 omissions remain explicitly historical. Final-cohort records are absent. |
| Backend changed through configuration only | **Met** | Backend-only inherited YAML and `tests/test_config.py::test_backend_only_override_keeps_shared_experiment`; checkpoint/default differences resolve per backend without changing shared source. |

## TASK-006–023 acceptance mapping

Numbers 1–3 below correspond in order to each task's three acceptance bullets in
`.mc/tasks/`. A task with an unmet or unverified bullet is not fully accepted.

| Task | Criterion 1 | Criterion 2 | Criterion 3 | Evidence / reproduction command |
| --- | --- | --- | --- | --- |
| 006 configuration | **Met:** rejects unknown/incompatible/missing requirements | **Met:** inheritance, device/seed/checkpoint resolution and saved config | **Met:** backend-only override | `config.py`, `registry.py`, `tests/test_config.py`; `.venv/bin/pytest -q tests/test_config.py`; reach v1 resolved configs. |
| 007 canonical data | **Met:** T+1 images/states, T actions, masks and episode-local windows | **Met:** timestamp/schema/action/end-row validation | **Met:** deterministic grouped splits, train-only statistics and hashes | `data.py`, `tests/test_data.py`, `docs/DATA_FORMAT.md`; `LEROBOT_SOURCE=third_party/lerobot .venv/bin/pytest -q tests/test_data.py`. |
| 008 embodiment actions | **Met:** both side mappings, units, round trips and saturation | **Met for simulation:** reachable IK and declared hand synergy; stale/unreachable rejection | **Met:** limits/interpolation/applied reports and separate experimental joint schema | `embodiment.py`, `contracts.py`, `configs/g1_sim_action.json`; `.venv/bin/pytest -q tests/test_embodiment.py`. Simulation values are not validated hardware calibration. |
| 009 MuJoCo scene | **Met:** dual sides and synchronized camera/state API | **Met:** seeded resets/image goals; scoring truth isolated | **Met:** scripted reach with separate evaluator access | `simulation.py`, `collection.py`; `.venv/bin/pytest -q tests/test_simulation.py tests/test_embodiment.py`; opt in with `JEPA_TEST_RENDER=1`; `outputs/reach-v0-oracle/`. |
| 010 first shared corpus | **Met locally:** hashed manifests and 80/10/10 sessions; public payload availability **unverified** | **Met:** coverage/failures/timing/storage reported | **Met:** same batches, applied normalized actions plus raw physical values | `collection.py`, `audit.py`, `benchmarks/manifests/reach-pilot-v0*.json`, `data/reach-pilot-v0/`; reach protocol commands in `docs/experiments/reach_pilot.md`. Grasp/left-arm/rotation coverage is absent in this corpus. |
| 011 native model | **Met:** full API, recursive inference without future state | **Met for software:** finite CPU/MPS fixtures and diagnostics; v0/v1 defect retained, v2 corrected metrics tested | **Met:** checkpoint state and exact next CPU update/reload | `models/{base,native}.py`, `tests/test_models.py`; `.venv/bin/pytest -q tests/test_models.py`; v0/v1 training reports. Fixture passing is not a real-corpus noncollapse certificate. |
| 012 CEM | **Met:** known-dynamics progress and bounded actions | **Met:** seeded elites/chunks/finite rejection/inference | **Met:** model-independent budget and run latency/RSS | `planning.py`, `tests/test_planning.py`, reach run manifests/summaries; `.venv/bin/pytest -q tests/test_planning.py`. RSS is process host peak, not per-planner allocation. |
| 013 MPC | **Met after correction:** requested/applied actions separately logged and clipping test passes | **Met:** stale/deadline/runtime/rejection/stop paths exercised, uncertain execution retains attempted action | **Met:** one action then observe; opaque model state | `planning.py`, `tests/test_planning.py`; historical v1 traces contain only applied actions, while the corrected producer records both. |
| 014 learned reaching | **Met locally:** checkpoints/data/config/commands/development goals archived | **Unmet as trustworthy native prediction evidence:** v1 mixed online/EMA persistence was invalid; corrected rerun pending | **Unmet:** useful noncollapsed native progress not demonstrated; diagnosis documented | `docs/experiments/{reach_pilot,reach_pilot_v1,reach_pilot_v2,reach_results}.md`; v0/v1 artifacts. Do not treat collapsed native 3/5 as this gate passing. |
| 015 external backend | **Met:** actual pinned adapter/API/action prediction/goal/checkpoints | **Met:** actual finite CPU/MPS training/reload; real-corpus budget recorded | **Met:** shared data/planner/sim; no future robot state | `models/lewm.py`, `tests/test_models.py`, `checkpoints/reach-pilot-v1/leworldmodel.run.json`, `docs/LEWM_SPIKE.md`; `.venv/bin/pytest -q tests/test_models.py tests/test_training.py`. |
| 016 conformance/swap | **Met for interface:** common finite/action-dependent/reload checks | **Met:** backend-only configuration change | **Met for reach:** matching data/actions/goals/seeds/planning budget | `tests/test_models.py`, `tests/test_config.py`, both reach-v1 configs/run records. No cross-model raw latent-score ranking is justified. |
| 017 benchmark records | **Met for software:** strict nested schema wired/tested, recursive source hashes and budget metadata implemented | **Unverified after correction:** new runner archives linked training reports; new actual run still needed | **Met for retained reach failures:** denominators, intervals, null reasons; historical omissions labeled | `benchmark.py`, `result_schema.py`, `tests/test_result_schema.py`, `tests/test_benchmark.py`; `.venv/bin/pytest -q tests/test_result_schema.py tests/test_benchmark.py`. Historical runs retain their original omissions. |
| 018 full-task curriculum | **Unmet:** threshold code exists, but final cohort/reset/timeout freeze not recorded | **Met for software + privileged feasibility:** ordered scoring and false-positive tests; two seen-pair successful oracle traces | **Unverified:** initial stage outcomes recorded; expanded manipulation corpus still collecting/unsealed at audit | `task.py`, `scripted.py`, `tests/test_task.py`, `outputs/manipulation_oracle_suite/{suite_report,scorer_replay}.json`, `docs/experiments/manipulation_pilot.md`. Privileged controller is not a learned-policy result. |
| 019 generalization freeze | **Unmet:** no immutable complete appearance/pairing/goal/reset test manifest inspected | **Unverified:** development split exclusions exist; final frame/session/goal isolation not yet auditable | **Unmet:** final seeds/counts/comparisons not sealed before an actual final run | `docs/DATA_PLAN.md`, `docs/EVALUATION.md` specify intent; `data/manipulation-pilot-v0/collection_plan.json` excludes apple→plate. Publish final manifest before evaluation and use no final outcomes for tuning. |
| 020 full common benchmark | **Unmet:** two matched full-task checkpoints/configs/evaluation cohorts absent | **Unmet:** no complete learned full-task/stage/zero-shot/controls table | **Unmet:** reach development report exists; full comparison/rollouts absent | `docs/experiments/reach_results.md` is explicitly a development negative-results report. Record all planned attempts or an explicit incomplete status; do not substitute oracle successes. |
| 021 SDK2 preparation | **Met:** pinned named mappings, calibration/time/channel requirements | **Met for mocks:** malformed/limit/stale/disconnect/enable/stop tests | **Met:** no SDK dependency or physical execution route | `hardware.py`, `tests/test_hardware.py`, `docs/HARDWARE.md`; `.venv/bin/pytest -q tests/test_hardware.py`. Real CRC/serialization/controller ownership/actuation remain TASK-026. |
| 022 future ports | **Met as specification:** assets/cameras/actions/API/parity checklist | **Met:** explicit missing-value templates and staged commissioning | **Met:** Mac scope separated; TASK-025/026 linked | `docs/ISAAC_PORT.md`, `docs/HARDWARE.md`. No Isaac runtime selected or tested; no hardware evidence claimed. |
| 023 final audit/reproduction | **Unverified:** no clean Mac end-to-end small train/reload/swap/simulation run inspected | **Met as a draft inventory:** every criterion mapped; open rows remain open | **Unverified for final deliverable:** inventories/limits exist, but finalized measured comparison and durable artifact access incomplete | This document; `docs/{SETUP,DEPENDENCIES,RESOURCES,HARDWARE,ISAAC_PORT}.md`, `.github/workflows/integration.yml`; clean reproduction must produce its own immutable evidence. |

## Current measured outcomes

The reach corpus has 100 episodes, 4,836 transitions and 4,936 observations at
20 Hz, split 80/10/10 by session. Manifest SHA-256:
`9a5256cc2070d2bc8c44f44b741830e428a33682c5baf0719c7973c62231804f`.
Only right-arm XYZ varies; hands remain open and rotations/left-arm commands are
constant. Coverage records 35 rate-limit stops and 65 budget-ended episodes.
This corpus cannot validate learned grasp/release or comprehensive 14D behavior.

| Development policy | Reach successes | Interpretation |
| --- | --- | --- |
| Scripted oracle | 5/5 | These five goals are physically reachable in the simulator. |
| Hold / bounded random | 0/5 each | Controls; all random attempts stopped under guards. |
| Native v0 | 3/5 | Selected collapsed latent fraction 1.0; not evidence of useful learned dynamics. |
| LeWM v0 | 0/5 | Failed learned reaching retained. |
| Native v1 | 0/5 | Online collapse guard passed, but persistence used mismatched online/EMA spaces; apparent prediction advantage invalid. |
| LeWM v1 | 0/5 | All attempts stopped; same selected update as v0. |

All use seeds 10000–10004. Wilson 95% intervals are [0, 0.434] for 0/5,
[0.231, 0.882] for 3/5 and [0.566, 1] for 5/5. Tiny development cohorts and
iterative model selection preclude a strong generalization comparison. V1 CPU
planning p50/p95 is approximately 1.00/1.11 ms native and 7.48/8.38 ms LeWM;
concurrent development activity means these are recorded development timings,
not isolated throughput measurements.

Each v0/v1 training run completed 500 updates with seed 0, batch 16, horizon 4;
native took about 33 seconds and LeWM about 16 seconds. Reported host peak RSS
was approximately 672–724 MB. Reports identify the checkout as dirty and retain
source-content hashes. These are one-seed CPU experiments, not a final repeated
training comparison. Preserve the historical model source when reproducing old
strict implementation-hash checkpoints.

The privileged manipulation suite reports 2/5 seen-pair successes, with three
velocity-limit failures. Replayed ordered scoring confirms reach→grasp→transport
→place→release for the two successful traces; replay velocities were reconstructed
from logged positions rather than recorded directly. This demonstrates limited
scripted feasibility, not Apple→Plate zero-shot success. The manipulation dataset
had 11 written episodes and no frozen split at one audit snapshot; counts may
increase during collection. Its final report and hash require a separate audit.

## Independent training review and remaining work

The inspected v1 runner checks sealed nonempty train/validation splits, rejects
episode/session overlap with test/holdout, decodes only train and validation,
samples uniformly within episodes, and preserves masks/applied actions. Validation
cohorts are deterministic and separate from training sampler RNG. Best/latest
checkpoints, partial-budget outcomes and numerical failures are retained. The
model and sampler RNG are serialized; the CLI does not claim a resume command.
No concrete leakage defect was found in those paths. Unit fixtures establish the
mechanics, not the scientific usefulness of the selected robot model.

The v0 raw-MSE selector legitimately selected a collapsed state under its declared
rule, demonstrating why tiny prediction loss is insufficient. The v1 selector's
online-only guard and mixed-space persistence made native prediction improvement
misleading. The documented v2 remedy compares persistence in target space and
checks both online and target collapse. The modified code and corresponding
model/runner tests were subsequently inspected and passed in the 66-case run.
No new concrete leakage defect was found. Resulting real-data runs still require
review; a software regression pass does not supply their outcome.

The following fixes or evidence are necessary before calling final acceptance:

1. **Finish the scientific gates.** Inspect corrected native target-space/action
   controls and bounded reaching outcomes; retain failures. Seal the manipulation
   corpus, verify executed-action coverage and all exclusions, then freeze final
   pairs/appearance variants, reset/goal hashes, counts, seeds and comparisons.
   Complete both models' full common benchmark with controls and uncertainty.
2. **Finish schema and provenance integration.** The previous benchmark validator
   accepted malformed nested environment/planner values and arbitrary hash strings.
   `result_schema.py` is now wired into the producer and tested. Source
   dirty/content identity, recursive model-file hashes, an archived training
   report path/hash, the actual episode budget and source lookup independent of
   caller directory are now implemented. Exercise a new actual run and inspect
   those records. Do not rewrite historical artifacts to pretend these fields
   were measured originally.
3. **Verify both command meanings in final records.** The producer now preserves
   requested and applied actions, including uncertain execution, and focused
   clipping/failure tests pass. Dataset transitions correctly use applied actions.
   Inspect new final traces to confirm this evidence rather than attributing new
   fields to historical runs that did not record them.
4. **Reproduce a clean installation.** At inspection `docs/SETUP.md` omitted the
   `data` extra, and `docs/DATA_FORMAT.md` named nonexistent `data-compat` instead
   of `compatibility`; both documentation errors have now been corrected. Fetch
   exact upstream/assets and perform small training/reload/backend swap/rendered benchmark in a clean Mac
   checkout, and preserve logs/dependency inventory. The existing warm `.venv`
   and integration XML do not establish this criterion. The new optional CI job
   is useful but its presence alone is not a passing remote execution.
5. **Make evidence reproducible outside this workspace.** Commit coherent source
   and frozen manifests; provide either licensed artifact retrieval with hashes
   or bounded complete generation commands. Match final source/config/data/action/
   checkpoint/goal hashes across compared runs. Root-owned MC updates should use
   `mc show TASK-023`, `mc validate`, `mc index`, and leave open criteria unchecked.

The software-only SDK2/Isaac preparation is acceptable within the stated local
scope. No network command, real camera alignment, hardware calibration, stop
response, Isaac import/render/physics parity or physical task validation has been
performed or inferred from this audit.
