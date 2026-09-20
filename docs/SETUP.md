# Development and execution

## Reproducible Mac environment

Use Python 3.12 (tested 3.12.13), `uv` (tested 0.12.3), and the committed `uv.lock`. The required core imports NumPy/YAML only; learning and simulation are optional extras. Run from the repository root:

```sh
uv sync --locked --extra learning --extra sim --extra lewm --extra data --extra compatibility
uv run --no-sync python scripts/resource_probe.py
uv run --no-sync ruff check src tests scripts
uv run --no-sync ruff format --check src tests scripts
uv run --no-sync pytest
```

Use `--no-sync` on subsequent commands to preserve installed optional extras. `uv sync --locked` alone intentionally installs only core/development dependencies. No CUDA, Isaac, DDS, or physical robot connection is required for core tests. The `lewm` extra installs minimal adapter dependencies; the pinned source fetch/probe instructions are in [LEWM_SPIKE.md](LEWM_SPIKE.md).

Measured CPU/MPS readiness, resource limits, and local artifact locations are in [RESOURCES.md](RESOURCES.md). Tested simulator assets, rendering, and viewer commands are in [MUJOCO_SPIKE.md](MUJOCO_SPIKE.md). Feasibility checks do not establish trained closed-loop manipulation performance.

To audit installed dependency metadata:

```sh
uv run --no-sync python scripts/dependency_inventory.py > outputs/feasibility/dependencies.json
```

Create the output directory first if the resource probe has not run. GitHub Actions runs core import isolation, lint, formatting, tests, and dependency inventory on Linux and macOS, plus a separate macOS optional integration job with actual model, dataset and physics dependencies. Graphics and unavailable MPS checks remain explicit hosted-runner skips. A green core job does not validate physics, MPS, or learning quality.

## MissionControl

```sh
mc task board
mc task next
mc show TASK-001
mc validate
mc index
```

Use `.mc/` for planning; `tasks/` holds benchmark documentation. Mark criteria complete only with evidence. Commit task Markdown, not generated `.mc/data/` indexes.

## Validation tiers

Core checks cover contracts, validation failures, registry/import isolation, and (as implemented) loader splits and known-dynamics planning. Model checks cover actual forward/backward, checkpoint reload, recursive prediction and collapse/action sensitivity. Simulator checks cover G1/Dex3 reset/render/action execution, scoring and closed-loop planning. Physical checks require separate calibrated hardware and are never inferred from simulation.

Record code revision, configuration, data/action/checkpoint hashes, seeds, device and environment with experiment results. Keep recordings, checkpoints, model downloads and generated runs under ignored `data/`, `checkpoints/`, `third_party/` and `outputs/`. Future Isaac and hardware runtimes must use separate optional environments and commissioning procedures.

## Runtime and development experiments

Fetch pinned sources once:

```sh
uv run --no-sync python scripts/fetch_assets.py
uv run --no-sync python scripts/fetch_lewm.py
uv run --no-sync python scripts/fetch_lerobot.py
JEPA_TEST_RENDER=1 LEROBOT_SOURCE=third_party/lerobot uv run --no-sync pytest
```

Collection, training and evaluation commands are documented in [DATA_FORMAT.md](DATA_FORMAT.md), [TRAINING.md](TRAINING.md), and the versioned [experiment protocols](experiments/reach_pilot.md). Their outputs are research evidence; current learned-control failures are retained in [reach_results.md](experiments/reach_results.md). The optional macOS integration workflow runs real CPU model/data/physics checks with pinned upstream sources; graphics and unavailable MPS tests are explicitly skipped on hosted runners.

## Clean reproduction

After installing locked extras and fetching the three pinned upstream sources above, run:

```sh
uv run --no-sync python scripts/reproduce_smoke.py --name clean-smoke
```

This performs collection → train both adapters → save/reload → backend-only configuration swap → actual MuJoCo MPC and common result validation. It records each command and exit status under `outputs/clean-smoke/report.json`. Its 100-update raw-loss selection is deliberately a software smoke, not a substitute for the preregistered noncollapse and full-task research gates. Use a new name for another attempt; never overwrite previous evidence.

For the larger fixed corpus and six-run experiment, follow [the training protocol](experiments/mvp_final.md) and [evaluation protocol](experiments/mvp_evaluation.md). Checkpoints enforce implementation hashes. Historical v0/v1/v2 experiments require their recorded source revision; a later source fix is not silently treated as checkpoint-compatible.
