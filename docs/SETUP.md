# Setup and execution handoff

## Available now

The PRD snapshot, plan, task dependency graph, architecture/data/evaluation proposals, module directories, and example configuration files are prepared. `mc 0.1.14` was available during preparation. The shell's `python3` reported 3.9.6; this is an observation, not the selected runtime for the MVP. Observed local hardware: arm64 Mac, macOS 26.5.1, 48 GiB physical memory; `uv` is available. MuJoCo rendering, MPS execution, hardware connection, package environment, and dataset are not yet validated. The MVP uses this Mac; no remote GPU is assumed.

From the project directory:

```sh
mc validate
mc index
mc task board
mc task next
mc show TASK-001
```

The same commands can run from the parent using `mc --root open-embodied-jepa ...`. Keep task bodies and frontmatter together. After editing task files directly, run `mc validate` and `mc index`.

## First execution session

1. TASK-001: record Mac chip, MPS availability, free memory/storage, intended robot variant, camera feed, demonstration locations, and access constraints in task notes. Do not copy credentials into the repository.
2. TASK-002: pin LeWM and test canonical batch/action feasibility; inventory source, transitive packages, and weight licenses independently.
3. TASK-003: validate native MuJoCo stepping, camera rendering, and G1/Dex3 assets on this Mac. Save exact installation commands and environment export after the smoke test passes.
4. TASK-004: freeze schema v0 with coordinate frames, timing, masks, action dimensions, history, and rollout shapes.
5. TASK-005/006: add packaging, a supported Python version, reproducible dependency locks, typed config validation, registry, formatter, tests, and CI. Core installation must not import simulator or SDK2 packages.

Select mutually compatible macOS arm64 Python/MuJoCo/PyTorch versions through a smoke test. Do not require CUDA or Isaac locally. Record a verified compatibility matrix in TASK-003, then build the environment from those exact pins. Keep future Isaac, hardware, and optional research environments separate if dependencies conflict. See [platform plan](PLATFORMS.md).

## Implementation validation tiers

CPU checks cover shapes, action round trips, episode splits, configuration errors, backend conformance using small fixtures, and CEM on known dynamics. CPU/MPS checks cover real model training/reload and candidate throughput, with synchronization for timing and explicit fallback logging. Simulator checks cover G1/Dex3 execution, task scoring, and closed-loop manipulation. Physical checks require calibrated hardware and are logged separately.

Commands for training and evaluation are deliberately not advertised as working until their implementing tasks land. Each task must add reproducible commands alongside its artifact.

## Working conventions

Use `.mc/` for planning and notes; keep `tasks/` for benchmark task implementations. Record evidence before moving an MC task to `done`. Use environment variables or local untracked configuration for dataset/output paths. Keep recordings, checkpoints, simulator caches, and generated runs outside versioned source; `.gitignore` covers standard local artifact paths. Preserve dataset/split manifests and small fixtures in source control.
