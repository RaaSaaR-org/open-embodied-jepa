# Isaac Sim G1 + Dex3 bring-up spike (2026-09-28)

A development spike on the Linux PC, not a gated experiment, and not a manipulation, policy or parity result. It shows that the Unitree G1 with two Dex3 hands loads and steps in Isaac Sim with a fixed root and renders an onboard-like RGB frame. Nothing here implements the `IsaacTransport` in [ISAAC_PORT.md](ISAAC_PORT.md) or completes TASK-025. The owner approved the spike on 2026-09-28.

## Environment

| Item | Value |
| --- | --- |
| Host | Ubuntu 24.04.3, kernel 7.0.0-34, AMD Ryzen 7 9800X3D, 31 GiB RAM, RTX 5080 16 GB, driver 595.91.07 |
| Image | local `isaaclab_arena:latest`, id/digest `sha256:2588b52605d77552d4480196501c4b8b61774ef6d70d9b3d59291622d6856d9c` (built 2026-09-23 from IsaacLab-Arena `8b4a3a47fc`, 39.5 GB; not pulled for this spike) |
| Isaac Sim | `6.0.0-rc.22+release.33481.407f3ea1.gl` (a release candidate) |
| Isaac Lab | 3.0.0 (Arena submodule); Python 3.12.12; torch 2.10.0+cu128 |
| Robot USD | Arena `G1_CFG`: `…/Assets/Isaac/6.0/Isaac/Samples/Groot/Robots/g1_29dof_with_hand_rev_1_0.usd` on `omniverse-content-staging.s3-us-west-2.amazonaws.com`, fetched at run time (ETag `de519ec0ff16d32d14246389091402e2-5`, 38 195 671 bytes, Last-Modified 2026-06-02) |
| Physics | PhysX GPU pipeline, `cuda:0`, dt 0.002 s (MuJoCo's timestep), one environment |

The USD comes from NVIDIA's asset server, not from the project's pinned, hash-verified Unitree MJCF, and it is not pinned by content hash. Its terms have not been audited. ISAAC_PORT.md asks for a converted, hash-recorded asset before any parity claim.

## Commands

The container is started by `scripts/isaac/run_bringup.sh`: a fresh `docker run --rm` of the Arena image with the Arena entrypoint (host UID/GID), the spike scripts mounted read-only and only `outputs/<run-name>/` mounted writable. Isaac dependencies stay in the image; nothing is added to the `embodied_jepa` package or `uv.lock`. If the login shell predates the user's `docker` group membership, wrap the call in `sg docker -c`.

```sh
OUT_ROOT=$PWD/outputs scripts/isaac/run_bringup.sh isaac-bringup-spike-3 \
    --headless --steps 1000 --render_steps 50
uv run --no-sync python scripts/isaac/compare_joints.py \
    --isaac outputs/isaac-bringup-spike-3/run/isaac_joints.json \
    --output outputs/isaac-bringup-spike-3/joint_comparison.json
```

`bringup_g1_dex3.py` spawns a ground plane, dome light and a table box at the MuJoCo table pose, the robot with `articulation_props.fix_root_link = True` and the pelvis at MuJoCo's (0, 0, 0.793), and a 224 × 224 camera on `torso_link` at MuJoCo's `onboard_rgb` pose (pos 0.08, 0, 0.35; `xyaxes 0 -1 0 .866 0 .5`; vertical FOV 75°, OpenGL convention; quaternion converted to Isaac Lab 3's xyzw order). The robot holds its default pose through Arena's PD actuators. Outputs, all git-ignored: `outputs/isaac-bringup-spike-{1,2-lab-g1,3}/` and host-side `nvidia-smi` samples in `outputs/isaac-bringup-spike-{1,3}-gpu/`. The reference run is `isaac-bringup-spike-3`, made from commit `af8c1c2`; run 1 was the same script before ruff formatting, and run 2 loaded Isaac Lab's `G1_29DOF_CFG` (`Robots/Unitree/G1/g1.usd`).

## Results against the spike criteria

1. **Headless container with GPU access: met.** The container saw the RTX 5080. Kit started headless on Vulkan. The only problems logged were the expected missing-display and GLFW warnings.
2. **G1 + Dex3 with fixed root: met.** The articulation loaded with 43 joints and `is_fixed_base = True`. The root moved 0.0 m over 1 070 steps. All joint positions stayed finite. The largest hold error was 0.083 rad under Arena's gains; which joint that was has not been investigated. The Arena `G1_CFG` does not fix the root itself; setting `fix_root_link` in the spawn config is enough. Isaac Lab's `G1_29DOF_CFG` (`g1.usd`) also contains both Dex3 hands: 43 joints, fixed base, and the same names and limits as the Arena USD.
3. **Stepping and joint comparison: met, with Dex3 limit mismatches.** All 43 names in the MuJoCo model (`MuJoCoSimulation.joint_names`) are present in Isaac. The per-group counts match: legs 12/12, waist 3/3, arms 14/14, Dex3 14/14. Isaac orders the joints breadth-first (`left_hip_pitch, right_hip_pitch, waist_yaw, …`), not in MJCF actuator order, so an adapter must map joints by name. All 29 leg, waist and arm limits match within 1e-3 rad. **10 of the 14 Dex3 limits differ.** Only `thumb_0` and `thumb_1` match on each hand. In each case the Isaac range is a narrower range with the same sign:

   | Joints (left / right mirrored) | MuJoCo (unitree_mujoco `ffa21a1`) | Isaac USD |
   | --- | --- | --- |
   | `*_index_0`, `*_middle_0` | left [−1.8326, 0.1920], right [−0.1920, 1.8326] | left [−1.5708, 0], right [0, 1.5708] |
   | `*_index_1`, `*_middle_1` | left [−2.0944, 0], right [0, 2.0944] | left [−1.7453, 0], right [0, 1.7453] |
   | `left_thumb_2` / `right_thumb_2` | [0, 2.0944] / [−2.0944, 0] | [0, 1.7453] / [−1.7453, 0] |

   Every open and closed target of the project's grasp synergy (`configs/g1_sim_action.json`) lies inside the narrower Isaac limits, so the current synergy is not clipped. Which limit set matches the physical Dex3-1 was not checked. Effort limits were not compared: Isaac's ideal-PD actuators report the PhysX drive limit (1e9), and the MuJoCo actuators here are not force-limited. So this comparison covers names, counts and position limits only, not masses, inertias, axes, damping, collision geometry or drive models.
4. **One onboard RGB frame: met.** Saved to `outputs/isaac-bringup-spike-3/run/onboard_rgb.png` (224 × 224 × 3 uint8; mean 194.9, std 56.3). It shows the table and both arms and Dex3 hands from the torso. The framing is close to a MuJoCo `onboard_rgb` render at the same size (`outputs/isaac-bringup-spike-1/mujoco_onboard_rgb_224.png`), but the arm poses, lighting and materials differ. Isaac logged that DLSS raised the internal render resolution (the "Render resolution of (112, 112) is below minimal input resolution of 300" warning), so the frame came through the RTX/DLSS pipeline rather than being a native 224 px render. Anti-aliasing and DLSS settings would need pinning before frames are used as data.
5. **Step time and VRAM: met.** From run 3 (run 1 in parentheses):

   | Measure | Value |
   | --- | --- |
   | Physics-only step (dt 0.002, 1 env, write targets + `sim.step(render=False)` + update + CUDA sync), n = 1000 | median 2.73 ms, mean 2.79 ms, p95 2.99 ms (run 1: median 2.78 ms) |
   | Same, per 0.05 s control interval (25 substeps) | ≈ 68 ms, about 0.73× real time |
   | Step with camera render + readout, n = 50 | first 1.23 s; after the first, median 11.7 ms, p95 69.3 ms (run 1: 17.6 / 81.8 ms) |
   | App launch / `sim.reset()` | 83.6 s / 8.2 s; whole container run ≈ 2 min |
   | Isaac process GPU memory (host `nvidia-smi --query-compute-apps`, peak) | 4 666 MiB (run 1: 4 661 MiB) |
   | Device used, before launch → after render | 6 842 → 11 627 MiB, including the GR00T server's 6 626 MiB |

   For reference, a bare MuJoCo `mj_step` of the same fixed-pelvis scene on the CPU takes a median of 0.045 ms. That number excludes the PD loop and rendering, so it is not a like-for-like comparison. Isaac's advantage lies in many parallel environments, which this single-environment spike did not measure. The one-environment figures above are an upper bound on per-environment cost, not a throughput measurement.

## Blockers and caveats

No hard blocker. Things a follow-up has to handle:

- **Docker access:** the user is in the `docker` group, but the login session predates the change, so plain `docker` fails with "permission denied". `sg docker -c …` works without sudo. Logging in again fixes it for good.
- **Unpinned network assets:** the robot USD is streamed from NVIDIA's *staging* S3 bucket on every run. Mirror it and record its hashes, or convert the pinned MJCF, before any recorded experiment.
- **Release-candidate simulator:** Isaac Sim 6.0.0-rc.22 with Isaac Lab 3.0.0. Isaac Lab 3 changed APIs (warp arrays, `*_index` setters, xyzw quaternions), and a later release may break these scripts.
- **Asset parity:** the Dex3 limits differ (see above). The ordering differs. The drive model is Arena's ideal PD, not the project's clipped-PD-plus-bias-compensation. None of this is parity.

## Recommended next step

Build a name-mapped, hash-pinned G1 + Dex3 USD from the project's own pinned `g1_29dof_with_hand.xml`, using the Isaac Sim 6.0 MJCF importer. The Dex3 limits and every other audited field would then match MuJoCo by construction. Rerun `compare_joints.py` against that asset and extend it to masses, inertias and joint axes. After that, put a minimal `IsaacTransport` (`read` / `send_joint_targets` / `reset` / `close`) behind the existing contracts, following [ISAAC_PORT.md](ISAAC_PORT.md), and run the reset/step/render/timing parity checks listed there. The alternative is to keep NVIDIA's USD and version the joint-limit differences in the embodiment manifest. That is faster, but it makes cross-simulator comparisons harder to interpret.
