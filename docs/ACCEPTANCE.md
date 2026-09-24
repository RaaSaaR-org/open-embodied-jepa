# MVP acceptance and measured limitations

> **This is a dated snapshot, not the current status.** It audits the local engineering
> MVP as of 2026-09-20 and is not rewritten. For where the project is now — including the
> TASK-054 decision to abandon CEM over the world-model cost as the primary control line —
> see the [README status section](../README.md#status--2026-09-24) and
> [DECISIONS.md](DECISIONS.md). Nothing below has been superseded in the direction of a
> better outcome: **learned Apple→Plate is still 0 successes.**

Evidence date: 2026-09-20. **The local engineering MVP and its preregistered
evaluation are complete; learned Apple→Plate manipulation failed.** All eight
50-episode runs completed, each with **0/50 full-task successes and zero stage
successes**. All 400 attempts and their negative outcomes are retained. The earlier
[development audit](ACCEPTANCE_DRAFT.md) remains a historical snapshot, including
the failures that prompted subsequent fixes.

The implementation provides a Mac-native MuJoCo research framework with real
G1/Dex3 dynamics, LeRobot-compatible data, two interchangeable trained world
models, common image-goal CEM/MPC, and auditable results. This is distinct from
reliable learned manipulation. PRD §19 specifies no numerical full-task success
threshold; completing an honestly reported negative experiment can satisfy an
engineering evaluation requirement. It cannot establish task robustness.

**Met** means supported within the stated software/simulation scope. It does not
mean learned manipulation succeeded, real hardware is ready, or an external
maintainer approved the PR. No physical robot or Isaac runtime validation is
claimed.

## Evidence anchors

| Evidence | Recorded result |
| --- | --- |
| [Clean Mac reproduction](experiments/clean_reproduction.md), [machine-readable record](../benchmarks/manifests/clean-reproduction.json) | Detached `4d2588d93dcbc7e5ed07623fe2bd2c7cceb24381`, fresh `.venv`, independently fetched pinned sources/assets; **329 tests passed, zero failures/skips**, lint and formatting passed. Same Mac and cached package wheels; not a second machine. |
| Clean integration smoke | Fresh 12-episode/336-transition dataset; 100 updates per backend; checkpoint reload and backend-only swap; two five-step image-goal episodes per backend, **0/2 successes each**, all step-limit endings. Software integration evidence only. |
| [Frozen corpus manifest](../benchmarks/manifests/mvp-corpus-v0.json) | **184 episodes, 42,127 transitions**, 146/19/19 train/validation/test sessions. Source assignments preserved; failed valid prefixes retained. |
| [Leakage audit](../benchmarks/manifests/mvp-leakage-audit.json) | All 42,311 stored RGB observations checked: no cross-split session or exact-frame overlap, no goal image matches to any corpus split, no Apple→Plate episode in the corpus, normalization fit IDs equal training IDs. Near-duplicate/semantic similarity was not assessed. |
| [Frozen goals](../benchmarks/manifests/mvp-goals-v0.json), [evaluation protocol](experiments/mvp_evaluation.md) | 50 fixed resets, seeds 20000–20049, hashed RGB/reset/threshold/action/asset/runtime identity; same cohort for both models and controls. Only the Apple→Plate pairing is held out; appearances and individual object/container kinds are seen. |
| [Final training protocol](experiments/mvp_final.md) | Six runs completed, three seeds per backend, 3,000 updates each; total true wall time **770.600 s**, within the 1,080 s shared budget. All six selected checkpoints satisfy the declared online/target noncollapse guards. |
| [Runtime review](reviews/runtime-review.md) | No concrete scoped integration blocker at `main...829cf4c`; 116 targeted tests passed with two graphics opt-in skips, preceded by 75 graphics-enabled goal/config/planner checks. Authorship limits are explicit. |
| [Full common benchmark](experiments/mvp_results.md), [machine-readable results](../benchmarks/manifests/mvp-results-v0.json) | **400/400 attempts recorded**, eight complete runs in **595.082 s** of the 1,800 s wall budget. Every run is 0/50; all learned/random runs stopped under right-joint target-rate guards, while hold reached the step limit. |

Corpus SHA-256:
`200246b34bd18cb30c8c711879c42a5ed2f31a62c99b96c1b347e37aa57512b7`.
Goal manifest SHA-256:
`eef30af440d40064efecc98a84a68d4388f9f60d1151104b4859a8f9934ee38e`.

Clean test XML at `outputs/clean-reproduction/tests.xml` was independently checked
for all 329 cases. Both relocated smoke run/episode records also pass the current
strict result validator and identify the clean source revision with
`source_dirty=false`, including archived training-report hashes.

Final training/evaluation used revision
`e57e28b53eb682102b882dbada937fbcefffb01f` with one unchanged runtime source hash.
Final checkout metadata records `dirty=true` during documentation/tracker work;
this is distinct from the clean reproduction above. The audit independently
validated all 400 episode schemas, recomputed all eight summaries from episode
records, checked recorded run/episode hashes and all 400 goal/initial PNG hashes,
confirmed equal runtime environments, and reproduced the versioned summary using
`scripts/summarize_mvp.py` without writing or rerunning experiments.

Large datasets, checkpoints, logs and images remain local under ignored `data/`,
`checkpoints/` and `outputs/`; versioned manifests preserve hashes and summaries.
They are not presented as downloadable public artifacts. The documented clean
smoke generates its own small dataset and checkpoints; final experiment generation
commands and source provenance are recorded separately.

## PRD §19 — eleven criteria

These rows assess executable engineering capability. Final evaluation evidence
is additionally mapped in TASK-020 below; quantitative claims use measured
outcomes rather than test pass counts.

| Criterion | Status | Evidence |
| --- | --- | --- |
| G1 + Dex3 embodiment adapter exists | **Met, simulation** | `src/embodied_jepa/{simulation,embodiment}.py`, `configs/g1_sim_action.json`; real dual-arm/hand control, synchronized observations and failure guards; `tests/test_simulation.py`, `tests/test_embodiment.py`. |
| LeRobot-compatible G1 dataset loads | **Met** | `data.py`, frozen corpus, `tests/test_data.py::test_official_lerobot_reader`; actual pinned LeRobot v3 reader loads the PNG-in-Parquet profile and named 86D state. Compatibility is scoped to the pinned format/reader. |
| `native_jepa` trains on the dataset | **Met** | `models/native.py`, `training.py`; three final 3,000-update runs and clean 100-update reproduction. |
| At least one external JEPA backend is integrated | **Met** | `models/lewm.py` uses pinned actual LeWM classes; three final 3,000-update runs and clean source-fetch/train/reload. |
| Both implement the same WorldModel API | **Met** | `contracts.py`, `models/base.py`, common conformance tests in `tests/test_models.py`. |
| Both support action-conditioned future prediction | **Met, interface and measured sensitivity** | Recursive inference requires no future robot state; action-shuffle/zero-action diagnostics and model tests. Nonzero sensitivity alone does not prove useful dynamics. |
| CEM works with both without model-specific code | **Met** | `planning.py`; known-dynamics tests, both clean real-simulation runs and development benchmarks. |
| Goal-image planning works in simulation | **Met, executable with limited reach progress** | Both adapters execute shared image-goal MPC. Corrected noncollapsed v2 models each reached **1/5** development goals; controls were 0/5. Final Apple→Plate is **0/50 for every model/seed**. This does not establish reliable learned control. |
| Identical benchmark can run against both models | **Met** | `benchmark.py`, `configs/mvp*.yaml`; all six matched model configurations completed the same frozen 50-reset benchmark, alongside shared controls. |
| Results stored in common format | **Met** | `result_schema.py`, common run/config/episode/summary/dependency/training-report artifacts; schema validation of both clean smoke records. |
| Backend changed through configuration only | **Met** | `config.py`, inherited backend-only YAML, `tests/test_config.py`, independently reproduced smoke swap. |

## TASK-006–023 — each acceptance criterion

Columns 1–3 follow the three acceptance bullets in each `.mc/tasks/` record.
Code filenames without a directory refer to `src/embodied_jepa/`. The common
validation command for software rows is the clean-suite command recorded below.

| Task | Criterion 1 | Criterion 2 | Criterion 3 | Evidence |
| --- | --- | --- | --- | --- |
| 006 configuration | **Met:** unknown names/paths/scales/schema/history rejected early | **Met:** inherited paths, seeds, devices and per-backend checkpoints resolved/saved | **Met:** backend-only selection | `config.py`, `registry.py`, `tests/test_config.py`, clean smoke resolved configs. |
| 007 canonical data | **Met:** image/state/action masked sequences remain within episodes | **Met:** times, joint schema, applied-action alignment and final rows validated | **Met:** deterministic session splits, train-only statistics and reproducible hashes | `data.py`, `tests/test_data.py`, [DATA_FORMAT.md](DATA_FORMAT.md), corpus/leakage manifests. |
| 008 embodiment actions | **Met:** left/right mappings, frames, units, normalization and saturation tested | **Met, simulation:** IK and explicit hand synergy; unreachable/stale rejection | **Met:** joint/workspace/target-rate/interpolation/reporting checks; joint action schema separate | `embodiment.py`, action manifest, `tests/test_embodiment.py`. Physical calibration and hard bounds on contact-induced velocity are not claimed. |
| 009 MuJoCo scene | **Met:** both arms/hands and synchronized RGB/state | **Met:** seeded resets and image goals; truth isolated | **Met:** scripted reach and separate contact/pose scoring | `simulation.py`, `goals.py`, `collection.py`; renderer tests, frozen 50-goal manifest, oracle reach 5/5. |
| 010 shared collection | **Met:** versioned data/source/license/action/split hashes | **Met:** action/state coverage, failures, timing, storage and exclusions recorded | **Met:** common batches use accepted/applied target commands with raw values preserved | `collection.py`, `manipulation_collection.py`, `audit.py`; reach/manipulation/release/corpus manifests. Command targets are distinguished from realized physical motion. |
| 011 native JEPA | **Met:** complete API and future-state-free recursion | **Met:** finite CPU/MPS training; corrected online/target collapse and shuffle diagnostics | **Met:** weights/optimizer/EMA/normalization/schema/RNG save/load and next-update equality | `models/{base,native}.py`, `tests/test_models.py`, `tests/test_training.py`, final checkpoint reports. |
| 012 CEM | **Met:** known-dynamics progress versus random/hold with bounded actions | **Met:** seeded sampling/elites/chunking/finite failure/inference-only behavior | **Met:** model-independent budget with measured runtime/host RSS | `planning.py`, `tests/test_planning.py`, common benchmark records. Feasible actuator projection is a declared limitation, below. |
| 013 MPC | **Met:** deterministic loop logs requested and applied commands | **Met:** stale/deadline/solver/rejection/runtime/stop/uncertain-execution outcomes | **Met:** one action then new observation; latent/truth boundaries preserved | `planning.py`, focused clipping and uncertain-execution tests; clean smoke traces. |
| 014 native reaching | **Met locally:** checkpoints/hashes/config/commands/fixed development resets/goals archived | **Met:** corrected within-model prediction/collapse/ablation/resource/latency reports | **Met with weak evidence:** v2 noncollapsed native reaches 1/5 versus 0/5 hold/random; failure diagnosis retained | [reach_results.md](experiments/reach_results.md), `reach-development-results.json`, v2 local reports. This is observed progress, not statistically established superiority or reliable reaching. |
| 015 LeWM | **Met:** actual external API/action prediction/goal/checkpoint integration | **Met:** CPU/MPS finite training/reload and measured cost | **Met:** shared data/planner/simulator; no future-state requirement | `models/lewm.py`, `tests/test_models.py`, final and clean training reports, [MODELS.md](MODELS.md). Final LeWM prediction does not beat persistence. |
| 016 conformance/swap | **Met:** both pass common finite/sensitivity/round-trip API checks | **Met:** only backend configuration changes shared input path | **Met:** checkpoint/default differences resolved; common data/actions/goals/seeds/budget match | `tests/test_models.py`, `tests/test_config.py`, six final configs, clean smoke. |
| 017 result records | **Met:** strict nested schema; code/data/split/action/checkpoint/budget/environment provenance | **Met:** success intervals and linked model training/resource/latency reports | **Met:** all failures retained, missing measurements explained, matched comparisons, no raw latent ranking | `benchmark.py`, `result_schema.py`, `tests/test_benchmark.py`, `tests/test_result_schema.py`; final `mvp-results-v0.json` and all 400 episode records independently checked. |
| 018 curriculum/scoring | **Met:** thresholds/dwell/timeouts/resets/task identity frozen before final evaluation | **Met:** privileged scripted feasibility and hover/held-object false-positive tests | **Met:** intermediate stages and failures retained; release corpus expands coverage without held-out pair | `task.py`, `scripted.py`, `tests/test_task.py`; final goal/protocol, 9/24 supplementary scripted successes, both collection summaries. |
| 019 generalization freeze | **Met:** explicit seen kinds/appearance versus unseen pairing, fixed goal/reset/seed manifest | **Met within audit scope:** no session/exact-frame/goal leakage; no held-out-pair training/normalization | **Met:** 50 resets × 3 seeds × 2 models plus two controls declared before final outcomes | [mvp_evaluation.md](experiments/mvp_evaluation.md), `mvp-goals-v0.json`, `mvp-leakage-audit.json`. No unseen-appearance claim or near-duplicate exclusion claim. |
| 020 full comparison | **Met:** six selected checkpoints and matched configs, all 400 attempts complete | **Met:** zero full/stage success retained with intervals, compute, replans and controls | **Met:** reproducible negative comparison and rollout references | `scripts/{evaluate_mvp,summarize_mvp}.py`, `mvp-results-v0.json`, [mvp_results.md](experiments/mvp_results.md); no learned full-task success is claimed. |
| 021 SDK2 preparation | **Met:** audited named channel/joint/time/calibration mapping | **Met, mocks:** malformed/limits/stale/disconnect/enable/stop paths | **Met:** no SDK dependency or physical execution route on Mac | `hardware.py`, `tests/test_hardware.py`, [HARDWARE.md](HARDWARE.md). Actual commissioning is TASK-026. |
| 022 future ports | **Met as specification:** transport/assets/cameras/actions/parity checklist | **Met:** missing-value templates and staged hardware procedure | **Met:** local checks separated from future host/robot tasks | [ISAAC_PORT.md](ISAAC_PORT.md), [HARDWARE.md](HARDWARE.md); TASK-025/026 remain future work. |
| 023 audit/reproduction | **Met:** fresh environment, small train/reload/swap and rendered benchmark | **Met:** all criteria mapped with measured outcomes and limitations | **Met:** license inventory, final results, future limitations and continued tracking commands | This report, `clean-reproduction.json`, `mvp-results-v0.json`, [DEPENDENCIES.md](DEPENDENCIES.md), [SETUP.md](SETUP.md). Remote merge/CI and external-maintainer approval are separate delivery checks. |

## Measured research outcomes

Development v0 selected a collapsed native representation despite 3/5 reach
successes; those results do not establish useful learned dynamics. V1 corrected
selection eligibility but used mismatched online/EMA persistence for native
diagnostics; its apparent prediction advantage was invalid. Both v1 policies
scored 0/5. V2 corrected target-space persistence and checked both spaces; both
models reached 1/5, with the other four resets stopped under controller guards.
The 95% Wilson interval for 1/5 is [0.036, 0.624], overlapping the controls'
0/5 interval [0, 0.434]. All versions and negative outcomes are retained.

Final training used the unchanged models and one frozen corpus. All six runs
completed 3,000 updates and selected eligible noncollapsed checkpoints:

| Backend | Training seed | Selected update | Prediction / own persistence MSE |
| --- | --- | --- | --- |
| Native JEPA | 0 | 2800 | 0.687104 |
| Native JEPA | 1 | 2900 | 0.396890 |
| Native JEPA | 2 | 2500 | 0.670410 |
| LeWM | 0 | 1200 | 1.884182 |
| LeWM | 1 | 2800 | 1.313574 |
| LeWM | 2 | 1400 | 1.021376 |

Ratios below one beat each model's own target-space persistence control on its
validation cohort. All native runs do so; **none of the final LeWM runs do**.
Passing noncollapse eligibility alone does not mean useful dynamics. Ratios and
raw latent errors from different representations are not rankings of task quality.
Selected target effective ranks are only 2.59–3.56, despite passing the declared
per-dimension variance guards; those guards do not establish a rich representation.
Training overlapped some clean-smoke activity, so observed training duration is
not an isolated between-model speed comparison.

The **final held-out comparison completed all 400 attempts**. Outcomes and CPU
planning times are shown per training seed; times include each recorded CEM call,
not full camera/actuation time.

| Policy | Training seed | Full success | Reach/grasp/transport/place/release | Termination | Plan p50 / p95, ms |
| --- | --- | --- | --- | --- | --- |
| Native JEPA | 0 | 0/50 | 0/0/0/0/0 | 50 joint-rate stops | 0.964 / 1.052 |
| Native JEPA | 1 | 0/50 | 0/0/0/0/0 | 50 joint-rate stops | 0.976 / 1.054 |
| Native JEPA | 2 | 0/50 | 0/0/0/0/0 | 50 joint-rate stops | 0.967 / 1.039 |
| LeWM | 0 | 0/50 | 0/0/0/0/0 | 50 joint-rate stops | 7.011 / 7.264 |
| LeWM | 1 | 0/50 | 0/0/0/0/0 | 50 joint-rate stops | 6.996 / 7.295 |
| LeWM | 2 | 0/50 | 0/0/0/0/0 | 50 joint-rate stops | 7.004 / 7.297 |
| Hold | — | 0/50 | 0/0/0/0/0 | 50 step limits | Not applicable |
| Bounded random | — | 0/50 | 0/0/0/0/0 | 50 joint-rate stops | Not applicable |

Each 0/50 result has a 95% Wilson interval approximately **[0, 0.07135]**; the
stored lower endpoint is about 7×10⁻¹⁸ from floating-point arithmetic. The same
50 physical resets repeat across training seeds, so no pooled independent-trial
interval over 150 resets is claimed. No backend outperformed hold/random on task
success. Model peak process host RSS was approximately **1.83–1.89 GB**, including
runtime dependencies; it is not model-only memory or GPU memory. All model runs
recorded 50 `right joint rate limit` rejections; there were no omitted failed
episodes. Full per-run executed-step/replan/throughput totals and rollout paths
are retained in the [result manifest](../benchmarks/manifests/mvp-results-v0.json).

These failures characterize the frozen model/controller combination. Native
validation prediction improvement did not translate to task progress. The common
actuator-feasibility mismatch below is a plausible contributor, not a causally
isolated explanation; no post-result controller changes or selected reruns were
substituted into this comparison.

## Limitations that remain after software validation

- CEM predicts canonical requested actions before state-dependent actuator
  clipping, while training records applied targets. Grasp requests can therefore
  differ substantially from executed targets. Both backends share this declared
  controller/model mismatch; candidate feasibility projection requires a new
  experiment protocol, not tuning on final outcomes.
- The scene uses fixed-pelvis G1 dynamics and simple procedural object/container
  proxies. Camera calibration, contact parameters and hand synergy are simulation
  choices. Accepted targets do not imply realized displacement; sampled velocity
  stops do not impose a mathematical bound on contact dynamics. Forbidden-contact
  classification remains null with a reason.
- The original manipulation pilot exceeded its true wall budget when the host
  suspended; it was stopped and sealed at 66 stored episodes, with the interrupted
  67th attempt and 33 unattempted episodes recorded. Later bounded workflows use
  both wall and monotonic clocks. The supplementary release corpus has 18 stored
  episodes from 24 attempts, including nine scripted successes; six invalid resets
  remain failures. These are privileged collection outcomes, not learned results.
- Source manifests/checksums are versioned; large payloads are local. Small
  OpenGL rendering variation can change regenerated pixel hashes on another
  environment. Frozen image bytes are reused and verified rather than regenerated
  during comparison.
- SDK2 is mock-only and physically disabled. Isaac remains a port specification.
  No real hardware calibration, DDS publication/CRC integration, stop response,
  physical manipulation, or Isaac parity has been validated.

## Reproduce and continue

The clean source-checkout procedure is recorded in [SETUP.md](SETUP.md) and
[clean_reproduction.md](experiments/clean_reproduction.md):

```sh
uv sync --locked --extra learning --extra sim --extra lewm --extra data --extra compatibility
.venv/bin/python scripts/fetch_assets.py
.venv/bin/python scripts/fetch_lewm.py
.venv/bin/python scripts/fetch_lerobot.py
.venv/bin/python scripts/reproduce_smoke.py --name clean-smoke
JEPA_TEST_RENDER=1 LEROBOT_SOURCE=third_party/lerobot .venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
```

Use a fresh output name/checkpoint directory; generation refuses overwrites.
The [training](experiments/mvp_final.md) and
[evaluation](experiments/mvp_evaluation.md) protocols record the larger experiment
commands, fixed resource limits and artifact locations. Continue tracking with
`mc show TASK-020`, `mc show TASK-023`, `mc task next`, `mc validate`, and `mc index`.
Only the coordinator changes task status or lands the PR.

This report incorporates independent review of training and runtime integration,
plus author verification of the reviewer's contracts/data/hardware/schema/goal
contributions. See [runtime-review.md](reviews/runtime-review.md) for exact scope.
It is not an external maintainer approval, a verified remote CI/merge result, or a
certificate of physical robot safety.
