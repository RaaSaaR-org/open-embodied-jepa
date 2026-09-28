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

Use `--no-sync` on subsequent commands to preserve installed optional extras. `uv sync --locked` alone intentionally installs only core/development dependencies. No CUDA, Isaac, DDS, or physical robot connection is required for core tests; CUDA is optional (see the Linux section below). The `lewm` extra installs minimal adapter dependencies; the pinned source fetch/probe instructions are in [LEWM_SPIKE.md](LEWM_SPIKE.md).

Measured CPU/MPS readiness, resource limits, and local artifact locations are in [RESOURCES.md](RESOURCES.md). Tested simulator assets, rendering, and viewer commands are in [MUJOCO_SPIKE.md](MUJOCO_SPIKE.md). Feasibility checks do not establish trained closed-loop manipulation performance.

To audit installed dependency metadata:

```sh
uv run --no-sync python scripts/dependency_inventory.py > outputs/feasibility/dependencies.json
```

Create the output directory first if the resource probe has not run. GitHub Actions runs core import isolation, lint, formatting, tests, and dependency inventory on Linux and macOS, plus a separate macOS optional integration job with actual model, dataset and physics dependencies. Graphics, unavailable-MPS and unavailable-CUDA checks remain explicit hosted-runner skips, so the CUDA tests (`tests/test_cuda_models.py`) run only on the Linux PC. A green core job does not validate physics, MPS, CUDA, or learning quality.

## Linux with CUDA (working platform since TASK-072)

On 2026-09-28 the owner made a Linux PC the project's working platform: Ubuntu 24.04.5, AMD Ryzen 7 9800X3D (16 threads), 32 GB memory, NVIDIA GeForce RTX 5080 16 GB (compute capability 12.0), driver 595.91.07, CPython 3.12.3. The locked sync installs PyTorch 2.14.0+cu130 (CUDA 13.0, cuDNN 9.24) from the same `uv.lock`; no CUDA toolkit, compiler or system package is needed. From the repository root:

```sh
uv sync --locked --extra learning --extra sim --extra lewm --extra data --extra compatibility --extra pretrained --extra jepa-wms
uv run --no-sync python scripts/fetch_assets.py
uv run --no-sync python scripts/fetch_lewm.py
uv run --no-sync python scripts/fetch_lerobot.py
uv run --no-sync python scripts/fetch_dinov2.py
uv run --no-sync python scripts/fetch_jepa_wms.py --accept-noncommercial-source  # optional, CC BY-NC source
uv run --no-sync python scripts/resource_probe.py
MUJOCO_GL=egl JEPA_TEST_RENDER=1 LEROBOT_SOURCE=third_party/lerobot uv run --no-sync pytest
MUJOCO_GL=egl uv run --no-sync python scripts/reproduce_smoke.py --name linux-smoke
uv run --no-sync python scripts/cuda_smoke.py --dataset data/linux-smoke --name linux-cuda-smoke
```

- **Headless rendering.** Over ssh there is no display: set `MUJOCO_GL=egl` for anything that renders (collection, the smoke, graphics tests). Without it MuJoCo tries GLFW/X11 and aborts.
- **Devices.** `cpu`, `mps` and `cuda` are accepted wherever a device is chosen, and `auto` prefers cuda, then mps, then cpu (`embodied_jepa.devices`). Historical protocols that froze MPS (for example `first_policy.py`, `first_policy_v2.py`, `latent_dynamics.py` and their runners) keep it; a new protocol chooses its device explicitly.
- **Determinism.** Selecting `cuda` applies one setup, in `embodied_jepa.devices.configure_determinism`: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True, warn_only=True)`, cuDNN deterministic with benchmarking off, and TF32 off. PyTorch warns at two kernels that have no deterministic CUDA variant in this mode: `native_jepa`'s `AdaptiveAvgPool2d` backward and LeWM's memory-efficient attention backward. `strict=True` turns such warnings into errors (usable when no such kernel runs, not for `native_jepa`). Same-seed reproducibility on CUDA is therefore checked, not assumed: `scripts/cuda_smoke.py` trains each backend twice with one seed, in separate processes, and requires bit-identical per-step metrics, validation metrics and weights. CUDA results are not bit-identical to CPU or MPS results.
- **Cross-platform differences.** MuJoCo physics on x86-64 and arm64 gave identical task outcomes in the bring-up, but rendered frames differ between Apple GL and NVIDIA EGL (about a third of channels by about 2/255), and a random-init DINOv2 has different low-order bits on the two CPUs. Anything that pins frame or weight hashes is therefore platform-specific; a protocol pinned on the Mac (such as `apple_first_policy_v2`) does not run unchanged on Linux.
- **Shared machine.** The GPU is shared with other services; check `nvidia-smi` for free memory before a long run, keep runs alive with `nohup` or `systemd-run --user`, and keep at least 10 GB of disk free.

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

Collection, training and evaluation commands are documented in [DATA_FORMAT.md](DATA_FORMAT.md), [TRAINING.md](TRAINING.md), and the versioned [experiment protocols](experiments/reach_pilot.md). Their outputs are research evidence; current learned-control failures are retained in [reach_results.md](experiments/reach_results.md). The optional macOS integration workflow runs real CPU model/data/physics checks with pinned upstream sources; graphics, unavailable-MPS and unavailable-CUDA tests are explicitly skipped on hosted runners.

## Clean reproduction

After installing locked extras and fetching the three pinned upstream sources above, run:

```sh
uv run --no-sync python scripts/reproduce_smoke.py --name clean-smoke
```

This performs collection → train both adapters → save/reload → backend-only configuration swap → actual MuJoCo MPC and common result validation. It records each command and exit status under `outputs/clean-smoke/report.json`. Its 100-update raw-loss selection is deliberately a software smoke, not a substitute for the preregistered noncollapse and full-task research gates. Use a new name for another attempt; never overwrite previous evidence.

For the larger fixed corpus and six-run experiment, follow [the training protocol](experiments/mvp_final.md) and [evaluation protocol](experiments/mvp_evaluation.md). Checkpoints enforce implementation hashes. Historical v0/v1/v2 experiments require their recorded source revision; a later source fix is not silently treated as checkpoint-compatible.
