# G1 + dual Dex3 native Mac feasibility

TASK-003 passed the local asset, physics, RGB, and bounded viewer checks on 2026-09-20. The usable model is the official Unitree G1 29-DOF with two articulated Dex3-1 hands. It is a simulation starting point; the exact physical EDU4 configuration, hand revision, camera calibration, friction, and controller gains still require commissioning.

## Source and reproducible commands

The current Unitree MuJoCo main inspected at `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d` uses rubber hands in its G1 model. The articulated hand model was deleted by [upstream commit f72e101](https://github.com/unitreerobotics/unitree_mujoco/commit/f72e101347532548fc38ad83f42e7512e84667a0). We therefore pin its parent, [ffa21a1e811a4ffd5d31d0318674c950f73eb62c](https://github.com/unitreerobotics/unitree_mujoco/tree/ffa21a1e811a4ffd5d31d0318674c950f73eb62c), dated 2025-09-10. No simplified gripper substitutes for Dex3.

[assets/manifest.json](../assets/manifest.json) records the source, exact revision, and SHA-256 of the robot, scene, meshes, and license. The pinned repository supplies a BSD-3-Clause LICENSE; no alternative per-mesh license was found. Preserve that license with any redistributed assets. Meshes remain in ignored `third_party/`; this project does not vendor them or relicense them as Apache-2.0.

From the repository root, after installing the project simulation dependencies:

```sh
.venv/bin/python scripts/fetch_assets.py
.venv/bin/python scripts/spikes/mujoco_probe.py
.venv/bin/mjpython scripts/spikes/mujoco_probe.py --viewer --output outputs/mujoco_viewer_probe
```

The fetcher refuses to overwrite an existing checkout at another revision or with local modifications, and checks every manifest hash. The probe independently verifies hashes before loading. The raw scene is `third_party/unitree_mujoco/unitree_robots/g1/scene_29dof_with_hand.xml`; its included robot is `g1_29dof_with_hand.xml`. The upstream DDS launcher, SDK2, and Isaac are unnecessary. Passive viewing on macOS uses `mjpython`, as described in the [official MuJoCo Python documentation](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).

## Measured result

Environment: arm64 macOS 26.5.1, Python 3.12.13, MuJoCo 3.13.0, NumPy 2.5.3. The plain Python run completed two 1,000-step traces at a 2 ms physics timestep with exact equality after reset, finite state, and no MuJoCo warnings. Throughput was about **30,892 steps/s**, including PD torque computation and trace copies. Thirty stationary 320×240 RGB frames were exactly equal and nonconstant; throughput was **65.0 frames/s**, including renderer setup and PNG writes. These are single-run feasibility measurements, not a statistically characterized benchmark.

The separate `mjpython` run also passed (31,161 steps/s and 69.7 frames/s), opened the native passive viewer, completed 111 syncs in a bounded 1.5-second loop, and closed normally with exit status zero. The rendered overview and onboard images were visually inspected: the humanoid and both three-finger hands are visible.

Generated artifacts are ignored: `outputs/mujoco_probe/{report.json,probe.xml,overview.png,onboard_rgb.png}` and the corresponding `outputs/mujoco_viewer_probe/` files. Reports include software versions, source/project revisions, composed XML hash, full ordered joint/actuator/body inventories, masses/inertias/collision counts, camera/site names, and timing conditions. The project revision identifies the checkout base; the generated XML hash identifies the uncommitted spike scene at execution time. Re-run after merge to associate evidence with the merged implementation revision. Generated XML uses machine-local absolute mesh paths; regenerate it on another checkout.

## Inventory and integration consequences

The raw scene loads with `nq=50`, `nv=49`, `nu=43`, 45 bodies including world, and 104 geoms including floor. The robot has 29 body joints and seven independently actuated joints per hand. Total modeled robot mass is 36.1648968 kg. There are 53 collision-enabled robot geoms, using upstream meshes and primitive finger boxes; presence is checked, contact fidelity is not established. Each of the 14 finger joints has a finite limit, an actuator, a positive body mass and principal inertia, and collision geometry.

| Field | Mapping |
| --- | --- |
| Body motor indices | 0–11 legs, 12–14 waist, 15–21 left arm, 22–28 right arm |
| Arm joint order per side | `shoulder_pitch`, `shoulder_roll`, `shoulder_yaw`, `elbow`, `wrist_roll`, `wrist_pitch`, `wrist_yaw` |
| Left hand actuator indices 29–35 | `thumb_0`, `thumb_1`, `thumb_2`, **`middle_0`, `middle_1`, `index_0`, `index_1`** |
| Right hand actuator indices 36–42 | `thumb_0`, `thumb_1`, `thumb_2`, `index_0`, `index_1`, `middle_0`, `middle_1` |
| Joint naming | Arms: `{side}_{joint}_joint`; fingers: `{side}_hand_{finger}_joint` |
| End-effector parent | `{side}_wrist_yaw_link`; palm is geometry on this body, not a separate palm body |
| Original sensor site | `imu`; no source camera or end-effector site |

**Never equate qpos order, actuator order, or SDK hand message order.** Left and right hand ordering differs. The subsequently pinned official XR Dex3 controller uses middle-before-index on the left and index-before-middle on the right, matching this MJCF; semantic grasp manifests may use another order. See `HARDWARE.md` for the exact SDK/XR source mapping. Resolve named joints and actuator transmission addresses explicitly. The source controls are torque motors; feeding target positions directly into `data.ctrl` would be incorrect. The probe uses clipped PD torques solely to hold the zero pose for this feasibility test.

The generated probe removes only the pelvis free joint (`nq=nv=43`) and adds floor/light, overview camera, torso-attached `onboard_rgb`, and `{left,right}_ee` sites. Pelvis height remains the upstream 0.793 m. The camera at torso offset `(0.08, 0, 0.35)` m and each wrist EE offset `(0.12, 0, 0)` m are explicit simulation conventions, not measured hardware extrinsics. The RGB optical axes face forward/downward and image both hands.

## Remaining model work

TASK-009 can compose the verified asset into a tabletop scene without URDF conversion or new hand meshes. It must define the table/object/plate, initial arm posture, camera visibility, and reset distribution. TASK-008 must validate IK, joint and velocity limits, motor control and mirrored grasp synergy, including failure behavior and finger contacts. This probe does not validate grasping, locomotion, physical EDU4 parity, camera intrinsics, or controller safety. Calibration and real robot commissioning remain future work; do not infer physical command limits from this smoke controller.
