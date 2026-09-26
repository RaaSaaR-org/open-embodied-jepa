# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow authority

[AGENTS.md](AGENTS.md) is the authoritative contributor/agent workflow (branch → commit → PR → review → merge, architecture rules, research-quality rules). Read it before non-trivial changes; this file only summarizes commands and architecture.

Work is tracked in MissionControl Markdown under `.mc/` (`mc task board`, `mc task next`, `mc show TASK-023`, `mc validate`, `mc index`). `.mc/tasks/` is project management; the top-level `tasks/` directory is robot benchmark task documentation. If `mc` is unavailable, say so rather than claiming validation passed.

## Environment and commands

Python 3.12 with `uv` and the committed `uv.lock`. The core package imports only NumPy/PyYAML; torch, MuJoCo, LeRobot and LeWM live behind optional extras and lazy imports.

```sh
uv sync --locked --extra learning --extra sim --extra lewm --extra data --extra compatibility --extra pretrained
uv run --no-sync ruff check src tests scripts
uv run --no-sync ruff format --check src tests scripts
uv run --no-sync pytest
uv run --no-sync pytest tests/test_training.py::test_name   # single test
```

Always pass `--no-sync` after the initial sync: a bare `uv sync --locked` reinstalls core+dev only and drops the optional extras.

One-time pinned upstream fetches (into ignored `third_party/`, `assets/`):

```sh
uv run --no-sync python scripts/fetch_assets.py   # G1/Dex3 MuJoCo assets
uv run --no-sync python scripts/fetch_lewm.py     # pinned LeWM source
uv run --no-sync python scripts/fetch_lerobot.py  # pinned LeRobot v0.4.4
JEPA_TEST_RENDER=1 LEROBOT_SOURCE=third_party/lerobot uv run --no-sync pytest
```

`JEPA_TEST_RENDER=1` opts into graphics tests; `LEROBOT_SOURCE` points the official-reader test at the fetched checkout. Only "graphics opt-in" and "MPS unavailable" skips are tolerated by the integration CI job (`.github/workflows/integration.yml`); any other skip fails the build.

End-to-end software smoke (collect → train both backends → reload → backend-only config swap → MuJoCo MPC):

```sh
uv run --no-sync python scripts/reproduce_smoke.py --name clean-smoke
```

It refuses to overwrite an existing `outputs/<name>`; use a new `--name` for another attempt. Never overwrite prior evidence under `data/`, `checkpoints/`, `outputs/` (all git-ignored).

## Executable entry points

Modules with CLIs (`python -m embodied_jepa.<module>`):

- `collection`, `manipulation_collection` — record real MuJoCo episodes into a `DatasetStore`.
- `goals` — seal a frozen goal manifest (RGB hashes + exact reset coordinates) for evaluation.
- `training` — bounded training runner (`--dataset --backend --output --steps --horizon --seed --device --max-seconds --selection`).
- `benchmark` — closed-loop evaluation from a config (`--config --policy {model,hold,random,oracle}`).
- `audit` — dataset audit.

Experiment drivers in `scripts/`: MVP line — `assemble_mvp_data.py`, `train_mvp.py` (six-run supervisor), `evaluate_mvp.py`, `summarize_mvp.py`, `resource_probe.py`, `dependency_inventory.py`. Task-specific apple line — `collect_apple*.py`, `train_apple_sensor.py`, `evaluate_apple.py`, the `diagnose_*`/`audit_*` probes, `run_apple_wm_v4.sh`, and `bootstrap_wm_v{3,4}_contrasts.py`. Each apple protocol also has a module under `src/embodied_jepa/` (`world_model_v2.py` … `world_model_v4.py`, `object_ceiling*.py`, `privileged_rollout.py`, `hybrid_phase.py`, `trajectory_tracking.py`, `waypoint_planning.py`).

## Architecture

The single package is `src/embodied_jepa/`. The central invariant: **swapping the world model must not require changing the robot, dataset, task, planner or evaluation code** — a backend swap is a one-line config change (`configs/mvp_lewm.yaml` is just `extends: mvp_common.yaml` plus `world_model.backend`).

- `contracts.py` — versioned NumPy-only boundary types: frozen `ee_delta_grasp_v0` 14D action schema, `RobotState`/`StateSchema`, `SequenceBatch`, `Capabilities`. Arrays are validated, never coerced or clipped; value objects take read-only copies. `LatentState` is deliberately `Any` — planners must never inspect a latent.
- `registry.py` + `config.py` — `MODELS`/`EMBODIMENTS`/`PLANNERS`/`TASKS` registries hold `"module:attr"` strings so optional runtimes import lazily. All registrations live at the top of `config.py`. Configs are strict YAML with `extends:` inheritance; relative paths resolve against the file that declares them, and unknown keys are rejected.
- `models/base.py` — `VisualModel` owns preprocessing, latents, prediction, loss, goal distance and the checkpoint envelope; `models/native.py` (native JEPA) and `models/lewm.py` (adapter over pinned upstream) are the two backends. Devices are restricted to `cpu`/`mps`; each model keeps isolated RNG state (`rng_scope`) so comparing backends cannot perturb the other. `VisualModel.defaults` also carries the shared backend-agnostic research options (`state_fusion`, `readout_heads`, `cameras`, the readout-shaping weights, `action_chunk`, `predictor_step_embedding`, `multistep_tail_weight`); **all are off by default**, none has been shown to help, and enabling one needs its own preregistration (see `docs/MODELS.md`).
- `planning.py` — CEM/MPC over normalized actions only, with no backend branches. Still implemented and tested, but since TASK-054 **no longer the primary control line**, and since TASK-057 there is no primary control line — see `docs/DECISIONS.md`.
- `embodiment.py` + `simulation.py` — G1/dual-Dex3 adapter owning IK, hand synergies, frames, limits and the MuJoCo transport; `simulation.py` imports `mujoco` lazily so it is importable without it. Physical scales/limits live in `configs/g1_sim_action.json`.
- `data.py` — local LeRobot v3 storage profile (PNG-in-Parquet) plus a JEPA manifest for schemas, provenance and splits. Single-writer; opening verifies every recorded hash. T actions require T+1 observations; the final row carries `action_valid=False` and is never sampled as a transition. Sequence windows never cross episode boundaries.
- `task.py`, `goals.py`, `benchmark.py`, `result_schema.py`, `audit.py` — tasks own resets/goals/termination/scoring; simulator truth is scoring-only and must not reach model inputs or planning cost. `scripted.py` is a scripted controller baseline, not a learned result.
- `hardware.py` — Unitree SDK2 preparation, mock-only; real execution stays disabled.

Top-level directories `models/`, `planners/`, `embodiments/`, `datasets/`, `tasks/`, `simulation/`, `deployment/`, `evaluation/` are architecture documentation, not importable code.

## Research-evidence rules that bite

- Distinguish implemented software from demonstrated outcomes. Learned Apple→Plate is **0 successes**. The frozen MVP common-mode benchmark (TASK-020, `docs/experiments/mvp_results.md`: image-goal CEM on the held-out Apple→Plate pairing, the same 50 frozen resets reused across three training seeds per backend) recorded 0/150 per model for `native_jepa` and LeWM; that figure covers that benchmark only. Every *learned* task-specific apple control attempt since, through TASK-057, has ended without meeting its declared gate (failed it, stopped at an earlier offline gate or stop rule, or ran incomplete), with 0 learned successes. The four TASK-056 behaviour-cloning arms each scored 0/16 on the same 16 resets of the non-gating development cohort, reproduced unsubstituted in TASK-057; their two grasps (A2/45100, A3/45006) lifted and held the apple, but `transport` never latched and the apple ended 0.19–0.22 m from the plate. Offline screens that passed, passing controls, and scripted-collector, privileged-ceiling, oracle or scripted-expert-substitution successes are not learned-policy results. Do not describe green CI or a passing smoke run as working manipulation.
- **Two control lines have been abandoned under pre-declared clauses.** At TASK-054 (`docs/experiments/apple_world_model_v4_results.md`) all four arms failed 14 preregistered gates and CEM over this world-model cost was abandoned as the primary line in favour of behaviour cloning with the world model as a critic. At TASK-057 (`docs/experiments/apple_policy_diagnostics_v1_results.md`, Outcome X) the `apple_policy_v1.md` §7 clause fired and that behaviour-cloning line stopped on this corpus (`apple-wide-v1`) and this camera (112 px onboard): no third control formulation, and no further loss, head or output-parameterisation variant, is preregistered on them. The next task is a perception/data task, which the task owner selects; if that task does not move the closed-loop number, the §7 conclusion is that the product goal needs a data or hardware change, not another model. The perception/data tasks since are offline readability probes, not controllers: TASK-059 (O-OCC-NONE), TASK-061 (O-LOOK-RAW) and TASK-062 (O-ENC-ARCH). At TASK-062 (`docs/experiments/apple_encoder_study_v1_results.md`) the `apple_encoder_study_v1.md` §10 clause fired and the in-corpus encoder-training line is closed: training a LeWM-family encoder on `apple-wide-v1` train-split frames so that its frozen features expose the post-look apple. No further readout-point, regularisation, objective or supervision variant of the TASK-054 recipe is preregistered on this corpus without new evidence of a different kind. Capacity, input handling and the shared distribution confound are recorded as untested, not refuted. TASK-063 ended with outcome O-PT-POOLED (`docs/experiments/apple_pretrained_encoder_v1_results.md`): a frozen, externally pretrained DINOv2 ViT-S/14 reads the post-look apple at both read-out points (CLS and final patch tokens) and beats its random-init floor. This is an offline readability result only, not a world model or a controller. Its caveats, as the results doc records them: the CLS token is not detectably different from raw pixels, both random-init floors meet the bars on their own, and there is one encoder, one input size and one floor seed. Learned Apple→Plate is still 0 successes, and no control formulation is preregistered. TASK-064, preregistering the look-prefix 112 px corpus with this frozen encoder as the candidate image encoder, is in progress; any world-model or control use of the encoder needs its own preregistration. The LeWM backend, the encoder as a component and the product goal are explicitly not abandoned. Do not write new docs that present sampling-based planning or behaviour cloning as the project's current control approach.
- Raw latent MSE from different representations is not comparable across backends; compare physical success and compute. Low prediction loss alone does not establish learned dynamics — include collapse and action-sensitivity diagnostics.
- Freeze cohorts, goals, thresholds and metrics before comparison; group splits by episode/session; fit normalization on training data only; never decode test/holdout episodes during training or selection.
- Every experiment records code revision, dataset/split/action/checkpoint hashes, seeds, device, budget and artifact paths. Checkpoints enforce implementation hashes — a later source fix does not make a historical checkpoint compatible.
- Protocols and results are versioned under `docs/experiments/` and `benchmarks/manifests/`; keep failures and negative results in them.

## Conventions

- Ruff, line length 100, `py312`, lint set `E,F,I,UP,B`. Commits use conventional-style scoped subjects (`feat(simulation): ...`).
- Keep new optional dependencies behind an extra, and keep their imports lazy so the core-import CI check (`import embodied_jepa` with no `torch`/`mujoco` in `sys.modules`) keeps passing.
- Never commit datasets, checkpoints, recordings or generated runs; commit manifests and hashes instead.
