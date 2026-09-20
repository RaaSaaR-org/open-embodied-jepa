# Open Embodied JEPA

One robot stack. One benchmark. Many world models.

An MVP planning workspace for interchangeable action-conditioned latent world models on Unitree G1 EDU4 with dual Dex3 hands. Start in MuJoCo on macOS and compare `native_jepa` with `leworldmodel` through the same dataset, CEM planner, task, and evaluation.

**Status:** planning and directory scaffold prepared; model training, simulation, and robot control are not implemented yet. Configuration files are design templates, not runnable experiments.

## Start here

- [PRD](PRD.md): unchanged snapshot of the original requirements.
- [MVP plan](docs/MVP_PLAN.md): scope, milestones, exit gates, and PRD traceability.
- [Architecture and contracts](docs/ARCHITECTURE.md): module boundaries, tensor conventions, and action semantics.
- [Data plan](docs/DATA_PLAN.md): acquisition, synchronization, splits, and held-out combinations.
- [Evaluation plan](docs/EVALUATION.md): common-mode comparison and evidence required for completion.
- [Mac and MuJoCo platform plan](docs/PLATFORMS.md): local runtime and future Isaac/SDK2 ports.
- [Setup and first steps](docs/SETUP.md): development workflow and environment readiness.
- [Decisions and risks](docs/DECISIONS.md): assumptions, defaults, and unresolved choices.
- [Dependency inventory](docs/DEPENDENCIES.md): upstream sources and license tracking.
- [Contributing](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md): commit, PR, review, merge, and reproducible research standards.

## Task planning

MissionControl is embedded in `.mc/`. Its task files are the source of truth for status, dependencies, and acceptance criteria. From this directory:

```sh
mc status
mc task board
mc task next
mc show TASK-001
mc task move TASK-001 in-progress
mc validate
mc index
```

Milestone tags `m0` through `m5` connect tasks to the plan. The workspace itself is the project; embedded mode does not require a separate project entity. No owners or delivery dates have been invented.

## Repository layout

The module directories follow the PRD: `models/`, `embodiments/`, `planners/`, `datasets/`, `simulation/`, `tasks/`, `benchmarks/`, `configs/`, `training/`, `evaluation/`, and `deployment/`. Each directory records its implementation responsibilities and task references. Product tasks live in `tasks/`; project-management tasks live in `.mc/tasks/`.

Original project code and documentation are licensed under [Apache-2.0](LICENSE). Optional upstream code, assets, data, and checkpoints retain their own terms; see the [dependency inventory](docs/DEPENDENCIES.md).
