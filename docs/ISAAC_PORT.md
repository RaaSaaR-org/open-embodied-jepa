# Future Isaac transport specification

TASK-022 specifies this port; TASK-025 implements and validates it on a suitable host. MuJoCo on the Mac remains the implemented simulation. No Isaac/CUDA dependency is added to the Mac core, and no Isaac runtime or parity test is claimed.

## Runtime and licensing gate

Select a released Isaac Lab/Isaac Sim pair, pin its exact revision/build and Python/driver/PyTorch versions, and run NVIDIA's compatibility checks on the target machine. As reviewed on 2026-09-20, the [official Isaac Lab full-simulator installation guidance](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/index.html) specifies supported Ubuntu/Windows hosts and NVIDIA GPU resources; that is not a native Mac/MPS execution path. Development-branch requirements change, so this document intentionally leaves the future build unselected. Kit-less physics workflows do not by themselves validate the required camera-rendered Isaac Sim benchmark.

Use a separate environment and lockfile on that host. Keep model/backend imports independent of Isaac, and lazy-load the selected simulator only after its runtime is initialized. Archive installation/version output and the first reset/step/render/close trace. Audit Isaac Lab source, Isaac Sim/runtime terms, converter code, meshes, materials and generated USD separately; an importer or source-code license does not establish the license for every asset. Preserve the pinned Unitree BSD notices and source hashes from [assets/manifest.json](../assets/manifest.json). Do not substitute a generic two-finger hand for the two articulated Dex3 hands.

## Interfaces and ownership

| Boundary | Required future behavior |
| --- | --- |
| `IsaacTransport.read()` | RGB plus named physical qpos/qvel, acquisition timestamps, clock domain and sensor validity; no scene-object truth |
| `send_joint_targets(targets, joint_names, deadline)` | Verify complete names/order, units, finite/range/rate limits and freshness, then advance the configured control interval through actual physics; return target acknowledgement and time |
| `stop(reason)` / `close()` | Defined hold/pause semantics without a zero-grasp command; cancel pending work, release renderer/simulator resources, make further use fail |
| Simulator `reset(seed, object_xy, plate_xy)` | Deterministic episode state/camera/task construction; document settling, RNG streams and backend-specific repeatability tolerance |
| Simulator task truth | Object/container/contact/scoring data in a separate evaluator-only interface |
| Embodiment `observe/state/execute` | Preserve `Observation`, cached state, `ExecutionResult`, exact state/action schemas and the frozen 14D base-frame action interpretation |

The current `G1Embodiment` uses MuJoCo Jacobians and a scratch `MjData` for IK; it is **not** already simulator-agnostic. Before the port, extract a robot kinematics provider (`forward_pose`, `jacobian`, `solve_ik`) or retain a validated MuJoCo kinematics-only model behind that provider while Isaac owns physical stepping and rendering. There must be one IK/scaling authority in the embodiment. Do not apply an Isaac delta-pose controller on top of the already scaled/solved commands. Backend-specific refactoring belongs inside the embodiment/transport, never the world model, canonical dataset, CEM/MPC or evaluator.

Map all 43 named robot joints to articulation handles and back to the declared canonical order. Do not assume an imported articulation's ordering matches MJCF qpos, actuator, SDK2 or another hand's ordering. Keep the optional object free joint outside robot state. Preserve 86D named qpos/qvel schema identity only if all field semantics/units/order remain identical; otherwise version and reject incompatible checkpoints. The left/right hand synergy is calibrated and mapped by name.

Maintain the canonical pelvis frame (+X forward, +Y left, +Z up), meters/radians, active extrinsic rotation `Rz @ Ry @ Rx @ R_current`, and absolute grasp -1/+1. Explicitly convert the selected Isaac release's quaternion ordering and camera optical axes at the adapter boundary; do not rely on a remembered default. Preserve the simulation action manifest's control interval and scale identity for common-mode parity, then measure differences rather than silently retune one backend.

## Asset conversion and controller parity

Start from the pinned, hash-verified G1/Dex3 MJCF and the same procedural tabletop/object/container definitions. Record the converter revision, options, source hashes and resulting USD hashes. The [official MJCF importer documentation](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/importer_exporter/ext_isaacsim_asset_importer_mjcf.html) notes that joint/link names can change during conversion and that actuator/drive models are not generally interchangeable. Compare the imported articulation to a name-addressed manifest before execution.

Audit fixed pelvis transform, link masses/inertias, joint axes/limits, armature/damping/friction, motor efforts, collision pairs/shapes/contact offsets, meshes and finger contact surfaces. Preserve two seven-joint Dex3 hands, the asymmetry in finger ordering, wrist-to-EE site transforms and camera pose. A visually similar USD is insufficient. Log any convex decomposition or primitive replacement, and make it part of the asset revision.

The MuJoCo runtime applies interpolated joint targets through clipped PD torques and model bias compensation. Choose and document the corresponding Isaac effort-control implementation or a deliberately different drive model. Do not label a position-teleport or unbounded high-gain drive as parity. Compare commanded versus realized motion, torque saturation, measured speed, solver failure, workspace clipping and stop/resume behavior. Contact/dynamics can differ across engines even with matching nominal parameters; report this as an experimental factor rather than altering shared thresholds until one engine succeeds.

## Camera and timing

Keep camera name `onboard_rgb`, configured resolution, intrinsics/FOV, exposure assumptions and pelvis/torso extrinsics in a versioned camera manifest. Convert rendered output explicitly to RGB uint8 HWC. Timestamp pixels at the physics state they depict, not when an asynchronous GPU readback completes. Associate state, left/right hands and camera under one episode clock with measured/render-step alignment; reject late/missing frames and render-pipeline warm-up frames rather than duplicate them silently.

Advance exactly the manifest's control interval per accepted action. Record physics substeps and rendering cadence separately; observe only after all required physics/render work is synchronized. Reset episode clocks consistently and never compare simulation seconds with wall-clock planning deadlines. Synchronize GPU timing and account for device transfers in throughput measurements. The canonical loader still requires T+1 aligned snapshots for T executed commands and rejects mismatched intervals.

## Configuration design template

This is a specification for TASK-025, **not a currently registered backend**. Nulls represent missing prerequisites and must fail construction.

```yaml
simulation:
  backend: isaac_future
  isaac_sim_build: null
  isaac_lab_revision: null
  physics_backend: null
  asset_usd_path: null
  asset_manifest_sha256: null
  camera_manifest_sha256: null
  fixed_base: true
  physics_dt_s: null
  control_dt_s: 0.05  # parity target from the current simulation action manifest
  rendering_enabled: true
  device: null
embodiment:
  backend: unitree_g1_dex3
  action_schema: ee_delta_grasp_v0
  action_manifest: configs/g1_sim_action.json
  kinematics_provider: null
  joint_name_map: null
world_model:
  backend: native_jepa  # the same adapter interface; leworldmodel is a config change
```

Keep an Isaac run's simulator identity separate from common-mode world-model comparison. Within one simulator, only the backend changes; across simulators, use a separately labeled transport-parity experiment with matching data/task/actions/planner budgets and frozen evaluation rules.

## Evidence required to admit the port

On the Mac, contracts, joint-name manifests, schema/checkpoint rejection, pure frame transforms, dataset loading, model-independent planner tests, SDK2 mocks and the specification are testable. Converted USD loading, Isaac physics/render timing, contacts, synchronization and GPU memory require the future target host.

Before completing TASK-025, archive:

- Runtime/driver/Python/device identity, dependency locks, source/license audit and asset/camera/action hashes.
- A named 43-joint/86-state correspondence report, both-hand actuation/limits, masses/inertias/collision checks and EE/camera transform comparisons.
- Deterministic reset/step/render evidence over frozen seeds, camera/state alignment, time monotonicity, finite state, close/reopen behavior, dropped-frame exclusions and bounded-resource measurements.
- Small translation/rotation and open/close command traces for both arms/hands, realized-motion/torque comparisons, and unreachable/stale/deadline/rate/stop-resume failure tests.
- The same canonical corpus feeding both real model adapters, unchanged CEM/MPC, image-goal closed-loop traces with no evaluator truth in model/planner inputs, and the common machine-readable result schema.
- A separately labeled contact/manipulation parity report with successes, failures, timeouts, sample sizes and uncertainty. Explain engine/asset differences and preserve frozen held-out pairs.

TASK-026 follows the distinct physical procedure in [HARDWARE.md](HARDWARE.md); a passing Isaac test never substitutes for hardware calibration or commissioning.
