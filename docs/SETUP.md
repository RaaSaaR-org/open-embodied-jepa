# Development and execution

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
- **Determinism.** Selecting `cuda` applies one setup, in `embodied_jepa.devices.configure_determinism`: `CUBLAS_WORKSPACE_CONFIG=:4096:8` (also set when `embodied_jepa.devices` is imported, before any cuBLAS use), `torch.use_deterministic_algorithms(True, warn_only=True)` by default, cuDNN deterministic with benchmarking off, and TF32 off. `native_jepa` is the only reason for warn-only: its `AdaptiveAvgPool2d` backward has no deterministic CUDA kernel, so strict mode cannot train it (the atomic kernel is deterministic in practice only where pooling bins do not overlap, as at the 64 px default, not at 112 px). LeWM's memory-efficient attention backward does have a deterministic variant, which PyTorch selects only in strict mode; a LeWM protocol should call `devices.configure_determinism("cuda", strict=True)` first, and later models and training runs keep the process strict (`tests/test_cuda_models.py` trains LeWM strictly and bit-identically). Same-seed reproducibility in warn-only mode is checked, not assumed: `scripts/cuda_smoke.py` trains each backend twice with one seed, in separate processes, and requires bit-identical per-step metrics, validation metrics and weights. CUDA results are not bit-identical to CPU or MPS results.
- **Cross-platform differences.** MuJoCo physics on x86-64 and arm64 gave identical task outcomes in the bring-up, but rendered frames differ between Apple GL and NVIDIA EGL (about a third of channels by about 2/255), and a random-init DINOv2 has different low-order bits on the two CPUs. Anything that pins frame or weight hashes is therefore platform-specific; a protocol pinned on the Mac (such as `apple_first_policy_v2`) does not run unchanged on Linux.
- **Shared machine.** The GPU is shared with other services; check `nvidia-smi` for free memory before a long run, keep runs alive with `nohup` or `systemd-run --user`, and keep at least 10 GB of disk free.

## macOS (supported; the archive platform since TASK-072)

Use Python 3.12 (tested 3.12.13), `uv` (tested 0.12.3), and the committed `uv.lock`. The required core imports NumPy/YAML only; learning and simulation are optional extras. Run from the repository root:

```sh
uv sync --locked --extra learning --extra sim --extra lewm --extra data --extra compatibility --extra pretrained
uv run --no-sync python scripts/resource_probe.py
uv run --no-sync ruff check src tests scripts
uv run --no-sync ruff format --check src tests scripts
uv run --no-sync pytest
```

Use `--no-sync` on subsequent commands to preserve installed optional extras. `uv sync --locked` alone intentionally installs only core/development dependencies. No CUDA, Isaac, DDS, or physical robot connection is required for core tests; CUDA is optional (see the Linux section above). The `pretrained` extra is not needed for the smoke run; the TASK-063+ protocols use frozen DINOv2 ViT-S/14 weights fetched once by `scripts/fetch_dinov2.py`. The `lewm` extra installs minimal adapter dependencies; the pinned source fetch/probe instructions are in [LEWM_SPIKE.md](LEWM_SPIKE.md).

Measured CPU/MPS readiness, resource limits, and local artifact locations are in [RESOURCES.md](RESOURCES.md). Tested simulator assets, rendering, and viewer commands are in [MUJOCO_SPIKE.md](MUJOCO_SPIKE.md). Feasibility checks do not establish trained closed-loop manipulation performance.

To audit installed dependency metadata:

```sh
uv run --no-sync python scripts/dependency_inventory.py > outputs/feasibility/dependencies.json
```

Create the output directory first if the resource probe has not run. GitHub Actions runs core import isolation, lint, formatting, tests, and dependency inventory on Linux and macOS, plus a separate macOS optional integration job with actual model, dataset and physics dependencies. Graphics, unavailable-MPS and unavailable-CUDA checks remain explicit hosted-runner skips, so the CUDA tests (`tests/test_cuda_models.py`) run only on the Linux PC. A green core job does not validate physics, MPS, CUDA, or learning quality.

## Worktrees, the GPU lock and run harness

Added after the 2026-10-02 audit. A worktree that uses another checkout's venv imports that checkout's `src` while it records its own revision, so each worktree gets its own venv.

```sh
scripts/new_worktree.sh ../worktrees/<name> --branch feat/<x>   # from origin/main (fetched)
scripts/new_worktree.sh ../worktrees/<name>-run --from <sha> --run
scripts/remove_worktree.sh ../worktrees/<name>
scripts/gpu_run.sh --min-free-gib 8 --max-util 30 -- uv run --no-sync python scripts/<runner>.py ...
```

- **`new_worktree.sh <dir> [--from <rev>] [--branch <name>] [--run] [--no-sync]`** checks out `origin/main` (or `<rev>`). It symlinks `third_party/` and, when the shared checkout has it, `assets/isaac/`; with `--run` it also links `data/` and `checkpoints/`. It never links `outputs/`. It then runs `uv sync --locked` with every extra and checks that `embodied_jepa` imports from the new worktree's `src`. The shared checkout is the repository's main worktree, or `SHARED_ROOT`.
- **`remove_worktree.sh <dir> [--dry-run] [--force-ignored]`** refuses when the worktree has modified or untracked files, has commits on no remote branch (run `git fetch` first), or holds its own non-empty `data/`, `checkpoints/` or `outputs/`. It also refuses the main worktree. `git worktree remove` deletes ignored files, so it also lists every other ignored file or directory that is neither a symlink nor a known cache (`.venv/`, `__pycache__/`, `*.egg-info/`, the pytest and ruff caches), such as a real `third_party/`, `wandb/` or `configs/local*.yaml`, and refuses unless `--force-ignored` is given. It keeps the branch and what the symlinks point to.
- **`gpu_run.sh [options] -- <cmd>`** is the entry point for every GPU job.
  - **Lock.** It takes the machine-wide `flock ~/.local/state/gpu/lock`, which is shared with other projects (see `~/.local/state/gpu/BOARD.md`). It fails at once when the lock is busy (exit 75), or waits with `--wait [--wait-timeout S]`.
  - **Checks.** Inside the lock it checks free VRAM (`--min-free-gib`, default 4), GPU utilisation (`--max-util`, default 30 %) and, optionally, the CPU load (`--max-load`); a failed check exits 76. It then logs the GPU's compute apps to `~/.local/state/gpu/oej-gpu_run.log` and runs the command with `GPU_RUN_LOCKED=1`.
  - **Process group.** The command runs in its own process group and without the lock's file descriptor, so only gpu_run holds the lock. gpu_run exits, and so releases the lock, only when every process in that group is gone, including a background grandchild. It then logs `end` and restores the board.
  - **Signals.** SIGTERM, SIGINT or SIGHUP sends SIGTERM to the whole group, and SIGKILL follows after `--grace` seconds (default 30).
  - **No terminal input.** The command runs in a background process group, so it must not read from the terminal: a read stops it with SIGTTIN. Give it input from a file or `</dev/null`.
  - **What it cannot cover.** A descendant that leaves the group (`setsid`, a daemon) is neither waited for nor signalled. A docker container is not a descendant at all. `docker run` without a TTY passes the TERM on to the container, but a docker client killed with SIGKILL leaves its container running on the GPU. Use `--container oej-isaac-` for Isaac runs: gpu_run then keeps the lock until no running container's name has that prefix (a plain `[A-Za-z0-9][A-Za-z0-9_-]*` prefix, compared literally, never as a docker regex), and after a stop signal it runs `docker stop` on them.
  - **Board.** It writes `BOARD.md` only when asked: `--board` sets the Holder section, and `--request TEXT` (with `--wait`) adds a Requests line while it waits and removes it afterwards. Every edit holds `BOARD.md.lock`. A gpu_run killed with SIGKILL releases the lock immediately but leaves its board lines behind. They carry the tag `[gpu_run pid N]`, so a stale line can be recognised and deleted by hand.
- **`scripts/isaac/run_isaac.sh`** no longer matches TASK-072 process names. Run it under `gpu_run.sh --container oej-isaac- -- …` or `flock`. It now fails before starting a container when the MJCF directory, `ISAAC_MODELS_DIR`, or the `assets/isaac` needed by a `/oej/usd` argument is missing.
- **`embodied_jepa.run_tools`** holds the shared pieces for new runners (`run_guards.py` is hash-pinned, so it is unchanged):
  - `assert_local_import`;
  - `install_guards`, whose `Guards.void` reports the first signal even when its raise surfaced as another exception;
  - `MemoryWatch`;
  - `budget_rule_consistent`/`check_budget`, `select_checkpoint` and `last_two_triggered`;
  - `gpu_guard(require_lock=True)`;
  - `CycledReader`, `SlotProxy` and `scale_probe`, which runs the stage's own function with `probe=True` at the real sizes.

  `tests/test_no_runner_imports.py` fails when a new `scripts/**/run_*.py` loads another script. The pinned runners' existing edges are allowlisted.
- **Preregistration checks.** A new protocol's budget rule passes `check_budget(BUDGET)`, and its memory probe goes through `scale_probe` on the stage core. Neither is applied retroactively to frozen protocols.

## MissionControl

Install the `mc` CLI once, without sudo, from the pinned release asset, checking its sha256 before
unpacking (Linux x86-64 shown; the macOS assets are on the same release):

```sh
gh release download v0.1.14 -R RaaSaaR-org/mission-control -p mc-linux-amd64.tar.gz
echo "89145d96921e086fd22c08b715c35b93ae75bf4e2d1e22adcf15cd47e56f9598  mc-linux-amd64.tar.gz" | sha256sum -c -
tar xzf mc-linux-amd64.tar.gz && install -m 755 mc ~/.local/bin/mc
mc --version   # mc 0.1.14
```

Then, from the repository root:

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

Record code revision, configuration, data/action/checkpoint hashes, seeds, device and environment with experiment results. Keep recordings, checkpoints, model downloads and generated runs under ignored `data/`, `checkpoints/`, `third_party/` and `outputs/`. Isaac runs only inside its container (`scripts/isaac/run_isaac.sh`; see [ISAAC_PORT.md](ISAAC_PORT.md) and [ARENA.md](ARENA.md)), and hardware runtimes must use separate optional environments and commissioning procedures.

## Runtime and development experiments

Fetch pinned sources once:

```sh
uv run --no-sync python scripts/fetch_assets.py
uv run --no-sync python scripts/fetch_lewm.py
uv run --no-sync python scripts/fetch_lerobot.py
uv run --no-sync python scripts/fetch_dinov2.py   # TASK-063+ protocols only
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
