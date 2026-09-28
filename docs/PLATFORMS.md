# Platform plan

## Linux working platform (TASK-072, 2026-09-28)

The owner moved day-to-day work to a Linux PC: Ubuntu 24.04.5, AMD Ryzen 7 9800X3D, 32 GB, NVIDIA GeForce RTX 5080 16 GB (compute capability 12.0) with PyTorch 2.14.0+cu130 from the locked environment. MuJoCo physics stays on the CPU and renders headless through EGL; models train on CPU or CUDA. Setup, the deterministic CUDA configuration and the smoke checks are in [SETUP.md](SETUP.md#linux-with-cuda-working-platform-since-task-072), and the measured host is in [RESOURCES.md](RESOURCES.md).

What changes and what does not:

- `cuda` is a supported device beside `cpu` and `mps` (`embodied_jepa.devices`); `auto` prefers cuda, then mps, then cpu.
- The Mac stays supported. Protocols frozen on it (MPS training, Mac-measured frame and weight pins) are not silently re-run on Linux: a Linux run of such a protocol is a new, separately versioned protocol that names every platform deviation and pins its own measurements.
- Cross-platform equality is not assumed. The bring-up found identical MuJoCo outcomes on x86-64 and arm64 (final distances within 2 µm), but rendered frames differ between Apple GL and NVIDIA EGL, and a random-init DINOv2 differs in its low-order bits.
- Same-seed CUDA runs are made reproducible on this machine by the deterministic setup and checked with `scripts/cuda_smoke.py`; CUDA and MPS results are not expected to be bit-identical.

The rest of this document is the original Mac-first plan (TASK-001 onward), kept as written.

## Mac-first plan (original)

The user confirmed local Mac-only development and MuJoCo first. Prepare Isaac Sim and physical G1/Dex3 support without making either necessary for the local MVP. The host was measured by TASK-001 and is recorded in [RESOURCES.md](RESOURCES.md): Apple M5 Pro, arm64, macOS 26.5.1, 48 GiB memory, CPython 3.12.13 with PyTorch 2.14.0, and CPU/MPS forward and backward both producing finite loss and gradients. That is a capability check, not proof that every model operation supports MPS.

| Layer | Local MVP | Future Isaac Lab / Sim | Future physical robot |
| --- | --- | --- | --- |
| Physics / execution | Native MuJoCo Python bindings, CPU physics | Separate supported simulator workstation | Unitree SDK2 transport |
| Model execution | CPU; MPS when tested; CUDA on the Linux PC (TASK-072) | Supported accelerator on target machine | Inference host chosen after latency tests |
| Robot description | Verified G1 + dual Dex3 MJCF and meshes | Corresponding USD/URDF assets | Calibrated G1 EDU4 joint/camera manifest |
| Inputs | Simulated onboard RGB and named state | Equivalent named observations | Existing camera and timestamped joint state |
| Control | Normalized EE deltas → IK/hand synergy → actuators | Same action contract, simulator-specific actuators | Same action contract, SDK2 commands |
| Evidence | Required local training/planning/benchmark | Separate parity report when available | Separate commissioning report when available |

## MuJoCo choices and feasibility gate

Use the maintained `mujoco` Python package directly. The official Python documentation describes its bundled engine and macOS viewer support. Passive viewer scripts on macOS need `mjpython`; test RGB rendering separately from interactive viewing. [MuJoCo Python documentation](https://mujoco.readthedocs.io/en/stable/python.html)

Unitree publishes MuJoCo MJCF assets and a DDS-based simulator. Its documented launcher setup includes Linux dependencies and low-level SDK2 messages. Treat its assets/control mappings as candidates, not proof that its full launcher or dual Dex3 setup works on this Mac. The local adapter should call MuJoCo directly and require no SDK2 networking. [Official Unitree MuJoCo repository](https://github.com/unitreerobotics/unitree_mujoco)

TASK-003 must verify a complete G1/Dex3 model: joint names/order, both hand articulations, actuators, mesh paths, inertias, limits, collision geometry, and camera. If a matching complete model is unavailable, compose or convert authorized assets and record the work before estimating manipulation milestones. A simplified gripper is allowed for a labeled smoke test only; it cannot satisfy the G1/Dex3 MVP gate.

Begin with a fixed/stabilized base and a single RGB view approximating the existing robot camera. Native CPU physics and rendering are independent of the device used for model inference. Benchmark one environment first. Add workers only when measured useful; no MJX/CUDA vectorization requirement.

## Training and planning on the Mac

Use a compact native model, float32 initially, short history, and bounded candidate chunks. Proposed smoke settings are horizon 4, 64 candidates, and 2 CEM iterations. Final common-mode settings start from the PRD-inspired horizon 8 / 500 candidates; freeze an affordable shared budget after measuring both models on the Mac. The smoke profile never substitutes for a labeled final evaluation.

TASK-002 checked LeWM forward/backward and action-conditioning on CPU and MPS ([LEWM_SPIKE.md](LEWM_SPIKE.md)); TASK-005 selected the tested Python/PyTorch environment now pinned in `uv.lock`. Do not assume MPS implements every upstream operation; log unsupported operations and deliberate CPU fallback. Keep timings synchronized, include host/device transfers, and identify memory metrics correctly. Reduce resolution/batch/architecture inside model configuration as needed; both models still receive the same raw observations and physical actions.

## Prepare later ports now

TASK-021 records SDK2 message/joint mappings and verifies command conversion against mock transports. TASK-022 defines the transport protocol, asset/calibration manifests, simulator parity checks, and physical commissioning runbook. Keep real hardware execution disabled by default in its future runtime configuration.

TASK-025 implements the Isaac port only when appropriate compute is available. TASK-026 performs physical commissioning only when robot access exists. Neither is a dependency of local MVP acceptance. Physics and rendering differ across engines; compare rollouts statistically and through common contracts, without claiming exact physical parity or automatic sim-to-real transfer.
