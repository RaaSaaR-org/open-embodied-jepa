# Runtime integration review

Reviewed on 2026-09-20 against
`fac05924893255832cdee2648cf15b795b0242f0...829cf4ce0e01f03c95fcd242b8f8888837faf8cd`
(`main...829cf4c`). Source and evaluation-supervisor files were unchanged from
that head during this review. No runtime source, configuration, protocol, tracker,
Git state or experiment output was changed by this review.

**No concrete blocking defect was found in the reviewed integration scope.**
This is a software review result, not acceptance of unrun training, final task
performance, clean-environment reproduction or physical hardware.

## Scope and authorship

Reviewed `config.py`, `goals.py`, `planning.py`, `benchmark.py`, `result_schema.py`,
the six `configs/mvp*.yaml` files, `scripts/evaluate_mvp.py`, its tests, and their
interfaces to the dataset, embodiment, model and task contracts. Also read the
final evaluation protocol and checked the frozen goal artifact. The earlier
review of `data.freeze_split_assignments` found no blocking split-isolation
defect; that function is unchanged at this head.

Review of configuration, CEM/MPC, benchmark integration and evaluation supervision
is independent of their implementation author. The reviewer previously authored
contract/data/hardware components and the strict result-schema implementation,
and hardened goals plus contributed goal/config/planner tests during the earlier
review. Verification of those owned portions is an **author review**, not a claim
of independent approval. This report does not certify all roughly 12,000 added
lines in the full PR as independently reviewed.

## Checks executed

```sh
.venv/bin/pytest -q tests/test_goals.py tests/test_config.py \
  tests/test_planning.py tests/test_result_schema.py tests/test_benchmark.py \
  tests/test_evaluate_mvp.py
git diff --check main...829cf4c
```

Result: **116 passed, 2 graphics opt-in skips**, and no diff-check errors.
Before the freeze, the graphics-enabled goal/config/planner suite completed
**75 tests with no skips**, including a real reach goal and Apple→Plate goal,
loading, reset replay and manipulation reset-distribution checks. Those source
files were then frozen. This final review did not start additional model training
or a final benchmark.

Separately called `goals.load_goals` on all 50 final entries using task
`apple_to_plate`, seeds 20000–20049 and `configs/g1_sim_action.json`. It validated
the exact current source/asset/action hashes, thresholds, reset pairs/coordinates,
image paths and RGB image hashes. The manifest hash was independently recomputed:

`eef30af440d40064efecc98a84a68d4388f9f60d1151104b4859a8f9934ee38e`

Resolved all six final configurations with `ExperimentConfig.load(...,
require_checkpoint=False)`. Their common settings match exactly after excluding
backend/checkpoint mapping, training seed and the recorded planner seed. Each
uses the same frozen 50 resets, goal manifest, dataset/action paths, CPU device,
805-command ceiling, planner bounds, horizon 4, 64 candidates, 3 iterations and
8 elites. The benchmark replaces the CEM seed with the evaluation episode seed,
so actual candidate randomness is shared across training seeds and backends.
Checkpoint existence/eligibility was deliberately not inferred from this config
check; final training and the model loader supply that evidence.

## Findings resolved before this head

| Earlier defect | Resolution checked |
| --- | --- |
| Manifest could omit runtime hashes or supply unchecked reset/pair/target values | Complete required source set, strict seed/reset/pair/target validation and task-specific target restrictions now reject malformed input. |
| Image could change after initial validation or escape the manifest directory | Each image use verifies safe local path and hashes the exact bytes subsequently decoded; only bounded RGB PNG input is returned to the model. |
| Custom full-task goal reset could revert the plate to the simulator default | Generation uses the stored plate coordinates for both initial and supported goal resets. Final XY jitter follows the manipulation-training sampling order. |
| Originally requested command was lost after clipping or uncertain execution | MPC/control traces retain requested versus applied values. Focused clipping and execute-exception tests pass. |
| Benchmark result validation checked presence rather than field semantics | New validator checks nested budget/environment, hashes, seeds, finite values, outcomes/null reasons and trace/count/budget consistency; runner uses it. |
| Launch errors and interrupted JSONL could disappear from supervisor accounting | Launch errors produce explicit failed attempts; partial record errors and all missing planned seeds are retained. Incomplete artifacts cannot produce `complete=true`. |

Model inputs remain verified RGB observations plus named robot-only state under
the declared API. Frozen reset coordinates and reach target coordinates go to
reset/scoring, not latent prediction or CEM cost. Full-task inference receives no
privileged reach target. Scripted oracle use remains explicitly labeled and is
rejected as a full-task benchmark policy.

The supervisor declares all six model runs and both controls before launching,
uses a shared wall/monotonic elapsed budget, retains failed runs and reports
unattempted runs. Timeout termination can add process-cleanup latency after the
budget boundary; it does not authorize more evaluation. The final protocol does
not pool repeated training seeds as independent physical resets and makes no
unseen-appearance or physical-transfer claim.

## Limits of this review

The frozen cohort exists, but no final outcome is inferred here. The reviewed
supervisor tests cover launch failure and partial-record accounting; this review
did not deliberately terminate a live 50-episode physics benchmark. A hard kill
can leave a partly written episode, which remains explicitly unrecorded/incomplete
rather than being treated as a completed success or silently reducing the cohort.

The two actual goal checks observed one-level OpenGL intensity variation in three
channels on reset replay; robot/object reset state reproduced exactly. Frozen
goal files are reused and byte-hash verified, so evaluation does not rely on
bitwise rerendering a goal. This is not a claim of pixel-identical rendering across
machines or graphics drivers.

Final acceptance still requires the recorded training outcomes, all planned
evaluation outcomes or explicit incompleteness, and the separate clean Mac
reproduction audit. Scientific failure may be a valid completed experiment;
missing experiment evidence is not completion.

## Additional model and training review

A second agent, which authored simulation/collection rather than the models, independently inspected `models/{base,native,lewm}.py`, `training.py` and their planner boundary over `fac0592...829cf4c`. It found no blocking future-state, training/validation leakage, checkpoint or causal rollout defect. Four focused shared-API, causal-prefix and exact CPU-resume tests passed across both adapters.

It reproduced an important limitation in real MuJoCo: from an open right hand, a requested grasp target `+1` is rate-clipped to about `−0.8181818` in the first step, while CEM scores the unprojected candidate. Training correctly stores applied actions, so this is an inference/action-feasibility mismatch. The limitation is declared in the final protocol before evaluation and remains a follow-up; neither the review nor a completed benchmark establishes that the current planner is sufficient for robust dexterous control. No actuator limits or final results were changed to hide it.
