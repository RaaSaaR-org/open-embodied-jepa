# Mac-first platform plan

The user confirmed local Mac-only development and MuJoCo first. Prepare Isaac Sim and physical G1/Dex3 support without making either necessary for the local MVP. Observed host on 2026-09-20: arm64, macOS 26.5.1, 48 GiB memory. Exact chip, free disk, Python environment, and MPS runtime remain to be recorded by TASK-001.

| Layer | Local MVP | Future Isaac Lab / Sim | Future physical robot |
| --- | --- | --- | --- |
| Physics / execution | Native MuJoCo Python bindings, CPU physics | Separate supported simulator workstation | Unitree SDK2 transport |
| Model execution | CPU; MPS when tested | Supported accelerator on target machine | Inference host chosen after latency tests |
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

TASK-002 checks LeWM forward/backward and action-conditioning on CPU and MPS if available. TASK-005 selects a tested Python/PyTorch environment. Do not assume MPS implements every upstream operation; log unsupported operations and deliberate CPU fallback. Keep timings synchronized, include host/device transfers, and identify memory metrics correctly. Reduce resolution/batch/architecture inside model configuration as needed; both models still receive the same raw observations and physical actions.

## Prepare later ports now

TASK-021 records SDK2 message/joint mappings and verifies command conversion against mock transports. TASK-022 defines the transport protocol, asset/calibration manifests, simulator parity checks, and physical commissioning runbook. Keep real hardware execution disabled by default in its future runtime configuration.

TASK-025 implements the Isaac port only when appropriate compute is available. TASK-026 performs physical commissioning only when robot access exists. Neither is a dependency of local MVP acceptance. Physics and rendering differ across engines; compare rollouts statistically and through common contracts, without claiming exact physical parity or automatic sim-to-real transfer.
