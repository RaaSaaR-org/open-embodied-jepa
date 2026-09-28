# Isaac Sim on the project's own G1 + Dex3 model: USD conversion, model audit and a minimal transport (2026-09-28)

A development step on the Linux PC after the [bring-up spike](ISAAC_BRINGUP_SPIKE.md). It is not a gated experiment. It is simulator infrastructure and joint-level parity only: not a manipulation, policy or contact-parity result, and it does not complete TASK-025. There is no apple or plate in the Isaac scene yet.

## What was done

1. **Conversion.** The project's pinned, hash-verified `g1_29dof_with_hand.xml` (sha256 `8b68d8f0…30cd`, `assets/manifest.json`) was converted to USD with the Isaac Sim 6.0 MJCF importer inside the local `isaaclab_arena` image (id `sha256:2588b526…6d9c`). Script: `scripts/isaac/convert_mjcf_to_usd.py`, run through `scripts/isaac/run_isaac.sh`. The USD is not committed. It lives under the ignored `assets/isaac/`.
2. **Model audit.** `scripts/isaac/audit_usd.py` loads the USD as a fixed-root Isaac Lab articulation and dumps what PhysX actually simulates. `scripts/isaac/compare_joints.py --isaac-model` compares that dump field by field with `MuJoCoSimulation`, the model every project experiment uses.
3. **Transport.** `src/embodied_jepa/isaac_transport.py` provides `IsaacTransport` (`read` / `send_joint_targets` / `stop` / `reset` / `close`). It is importable with NumPy only. All Isaac, torch and warp imports are inside the class, and it runs only inside the container. Nothing is added to `pyproject.toml` or `uv.lock`, and nothing in the package imports the module. A committed joint manifest, `configs/isaac/g1_dex3_joint_manifest_v1.json`, is generated from `MuJoCoSimulation` by `scripts/isaac/write_joint_manifest.py` and checked against it by `tests/test_isaac_transport.py`. It gives the canonical joint order (the MJCF actuator order), limits, torque ranges, the transport's PD gains, passive damping and frictionloss. Manifest v0 (`…_v0.json`) named the frictionloss field `frictionloss_not_applied`, which was a misnomer because the default mode applies it. v1 renames it to `frictionloss` and makes it required. The v0 file stays committed because the `isaac-parity-3-*` runs recorded its hash (`a50c4b10…`). Its values are identical to v1 apart from the key and the version.
4. **Parity check.** `scripts/isaac/parity_isaac.py` (container) and `scripts/isaac/parity_mujoco.py` (host) replay one fixed joint-target trajectory (`scripts/isaac/parity_targets.py`) through both transports and compare the results.

Also fixed: `scripts/isaac/run_bringup.sh` passed `''` as an argument when it was given no extra args.

## Environment

| Item | Value |
| --- | --- |
| Image | local `isaaclab_arena:latest`, `sha256:2588b52605d77552d4480196501c4b8b61774ef6d70d9b3d59291622d6856d9c` |
| Isaac Sim / Isaac Lab | `6.0.0-rc.22+release.33481.407f3ea1.gl` (a release candidate) / 3.0.0 |
| MJCF importer | `isaacsim.asset.importer.mjcf-3.2.0` (wraps `mujoco_usd_converter` 0.1.0, which compiles the MJCF with the bundled MuJoCo 3.5.0), `isaacsim.asset.transformer.rules-1.2.0` |
| Host MuJoCo (reference) | 3.13.0 (`uv.lock`) |
| Physics | PhysX GPU pipeline on `cuda:0`, dt 0.002 s, 25 substeps per 0.05 s control interval, one environment |

## 1. USD conversion and hashes

Importer options, fixed in the script: `import_scene=False` (robot only), `merge_mesh=False`, `collision_from_visuals=False`, `allow_self_collision=False` (the importer default). The importer writes a package: the main `g1_29dof_with_hand/g1_29dof_with_hand.usda`, payload layers (`base`, `robot`, `instances`, `geometries.usd`) and a `Physics` variant set (`physx`, the default; `mujoco`; `physics`).

**The conversion is not byte-reproducible.** Every layer's `doc` string embeds a random temporary directory (`/tmp/tmpXXXXXXXX/`), so every file hash changes between runs. `scripts/isaac/hash_usd.py` therefore records two hashes:

- raw sha256 per file, plus a tree hash over them, which pins the exact file set used;
- a canonical hash: each layer is exported as USDA text with that directory name replaced, and the results are hashed.

Three conversions (`assets/isaac/g1_29dof_with_hand-{conv1,conv2,ref}`) gave three different raw tree hashes and the **same canonical tree hash**. The textual diff between them is only that directory name, including in the binary `geometries.usd`. `conv1` and `conv2` were made before `hash_usd.py` existed (at `80df09d`, with uncommitted scripts), so their `conversion.json` files hold raw hashes only. Their canonical hashes were computed afterwards by running `hash_usd.py` on their `run/usd` directories, and an independent reviewer reproduced all three. Only `ref` records the canonical hash in its own `conversion.json`.

| Hash (reference conversion `g1_29dof_with_hand-ref`, commit `e2f97fe`) | Value |
| --- | --- |
| Canonical tree sha256 (reproducible) | `cd1fdb27dfe3d6c61f8100724bc62ca4dec0a28b1b7c18186f640fe3b8cc2fa8` |
| Main `.usda` canonical sha256 | `25863da17dffa0d4ac3ad3170dadfa7e125e6a96412003f073947142782d94fa` |
| Main `.usda` raw sha256 (this file set only) | `a05c53abad9827f7e691129b44a4c3d9e830bf1089b34f016fbb53a6c22c71b0` |
| Raw tree sha256 (this file set only) | `7a6b2cc40e75b63ef7206732208bb39c6816eadb72edd256605f7ae867ab80a8` |

Reproduce with `sg docker -c "scripts/isaac/run_isaac.sh convert_mjcf_to_usd.py assets/isaac/<new-name> --mjcf /oej/mjcf/g1_29dof_with_hand.xml --expected_sha256 8b68d8f06674c5c10cd2cd89764b3cfba9fabba5080b55ea67ee1dd12cf630cd"`, then compare `canonical_tree_sha256` in `run/conversion.json`. The importer warns for every actuator that the MuJoCo motor's gain and bias type is unsupported, so no PhysX drive stiffness or damping is created. That is the intended result for torque-driven motors. The USD's asset terms follow the pinned Unitree BSD-3-Clause source. The importer is NVIDIA Apache-2.0 tooling inside the image, and no NVIDIA asset is included.

## 2. Model comparison against MuJoCo

Run `outputs/isaac-mjcf-audit-5` (commit `2b346f1`, clean tree), with `compare_joints.py --isaac-model` output in `model_comparison.json`. The robot is spawned with its root fixed and no spawn offset, because the USD keeps the MJCF pelvis position `(0, 0, 0.793)` on the pelvis prim. An earlier audit run that also set a 0.793 m spawn offset put the pelvis at 1.586 m.

**Matches, within the tolerance shown:**

| Field | Result | Tolerance |
| --- | --- | --- |
| Joint names / count | 43 / 43, same set; the Isaac order is breadth-first, so joints are mapped by name | exact |
| Bodies | 44 / 44, same names; total mass 36.16490 kg on both | exact |
| Joint limits (all 43, including the 14 Dex3) | max diff 1.6e-7 rad (the USD stores float32 degrees) | 1e-4 rad |
| Joint axes (child frame) | max angle 0.0 rad | 1e-4 rad |
| Joint frames (child-in-parent transform, anchors) | max 5.9e-9 m, 1.4e-6 rad | 1e-5 m, 1e-4 rad |
| Body masses | max relative diff 5.7e-8 | 1e-5 |
| Centres of mass | max 7.3e-9 m | 1e-5 m |
| Inertia tensors (full, body frame) | max relative diff 6.8e-7 (principal moments 6.8e-7) | 1e-4 |
| Armature | 0.01 on all 43 joints on both | 1e-6 |
| Collision shapes per body | 53 / 53 (39 meshes as `convexHull`, 8 spheres, 4 cylinders, 2 boxes) | exact count |
| Forward kinematics, all 44 links, zero pose + 3 seeded random poses | max 9.3e-7 m, 2.7e-6 rad | 1e-4 m, 1e-3 rad |
| Actuator gains in the model | MuJoCo `motor` actuators (no servo); USD drives stiffness 0, damping 0 | exact |

This supersedes the spike's Dex3 limit mismatch: that mismatch belonged to NVIDIA's USD, and the converted model has MuJoCo's limits.

**Mismatches (all of them):**

| Kind | Joints | MuJoCo | Isaac, as converted | Handling in `IsaacTransport` |
| --- | --- | --- | --- | --- |
| Passive joint damping | 43 | `dof_damping` 0.05 (hands 0.01) N m s/rad | absent from the PhysX variant (only in the `mujoco` variant as `mjc:damping`) | set as a zero-stiffness PhysX drive damping, checked by read-back |
| Joint friction model | 43 | `frictionloss` 0.2 (wrist pitch/yaw 0.1, wrist roll 0.2, hands 0.1) N m, a Coulomb torque | the number is copied into PhysX's deprecated, **load-proportional** friction *coefficient*. Isaac Lab's `joint_friction_coeff` buffer reports 0, but PhysX applies it (read directly from PhysX) | legacy coefficient set to 0; PhysX static and dynamic friction **efforts** set to the frictionloss value (`joint_friction="frictionloss"`), or all zero (`"none"`), checked by read-back |
| Effort limit | 43 | actuator `ctrlrange` (88/139/50/25/5/2.45/0.7 N m) | PhysX max force 1e9 (drive `maxForce` inf) | torque clipped to `ctrlrange` in the transport and by the Isaac Lab actuator's `effort_limit` |
| Self-collision | articulation | collision geoms collide except parent–child pairs | disabled (importer default) | not changed; a known difference |

The first audit runs read Isaac Lab's friction buffer (0) and missed the legacy coefficient. It was found through its effect in the first parity run (below), and the audit now reads PhysX directly.

## 3. `IsaacTransport`

It uses the same control semantics as `MuJoCoSimulation.send_joint_targets`. At each 2 ms substep it applies linearly interpolated targets through `kp (q* − q) − kd q̇ + bias` (kp/kd 100/5, hands 4/0.2), where the bias is PhysX's own gravity plus Coriolis compensation (`get_gravity_compensation_forces` + `get_coriolis_and_centrifugal_compensation_forces`), and clips the result to `ctrlrange`. The effort goes through an Isaac Lab `IdealPDActuator` with kp = kd = 0. The transport shares the MuJoCo transport's checks: complete names in canonical order, finite values, inside the MJCF limits, and the deadline in episode time. `reset()` restores MuJoCo's initial pose (zeros, elbows 0.08 rad) and restarts the episode clock. `read()` returns canonical-order qpos/qvel, an RGB frame (96 × 96 by default; 112 × 112 in the parity runs, via `--size 112`) from an `onboard_rgb` camera on `torso_link` at the MuJoCo pose and FOV, the episode timestamp, clock domain and a validity flag. PhysX damping and friction are read back after construction and after every `reset()`, and the transport raises if they differ from what was set. After `close()`, any further call raises. Scope limits: robot and table only. `reset(object_xy=…)` raises, there is no `task_truth`, there is one environment, and a closed transport cannot be reopened in the same process.

## 4. Parity against the MuJoCo transport

Setup: the same 90 joint targets (4.5 s) go to both transports:

- hold for 10 intervals;
- both arms ramp to shoulder pitch −0.3, shoulder roll 0.25 outward, elbow +0.6 and wrist roll ±0.3 (intervals 10–50);
- both hands ramp to the closed grasp synergy (50–70);
- everything returns (70–90).

The regenerated targets on the host were bit-identical to Isaac's (`targets_regenerated_equal: true`). The MuJoCo apple and plate were made invisible and non-colliding. Runs `outputs/isaac-parity-3-{frictionloss,none}` (commit `9dde06d`, clean tree, manifest v0), with host comparison in `…/parity/parity_report.json`. After the review fixes, the MuJoCo side was rerun with contact recording (`…/parity-r2/`); the joint numbers are identical. A fresh Isaac run on the fixed transport, `outputs/isaac-parity-4-frictionloss` (commit `b1733b4`, clean tree, manifest v1), reproduced the `parity-3-frictionloss` Isaac trajectory bit for bit (max diff 0.0). **The first caveat string in the `isaac-parity-3-*/parity/parity_report.json` files, "frictionloss … not applied in Isaac", is stale and wrong for the frictionloss run.** It was hard-coded. It is now derived from the run's `joint_friction` setting, and the `parity-r2` and `parity-4` reports carry the correct string. The table reports |q_Isaac − q_MuJoCo| over all 91 reads (after reset and after each interval), in rad:

| Group | frictionloss: max / p95 / median | none: max / p95 / median | Tracking error max, MuJoCo / Isaac (frictionloss) |
| --- | --- | --- | --- |
| Legs (12, held) | 8.9e-10 / 8.4e-10 / 1.8e-10 | 3.6e-9 / 2.0e-9 / 8.5e-11 | 4.5e-10 / 8.5e-10 |
| Waist (3) | 0.0025 / 0.0016 / 1.4e-5 | 0.0039 / 0.0025 / 1.1e-4 | 0.053 / 0.054 |
| Arms (14) | 0.0091 / 0.0039 / 1.6e-4 | 0.0122 / 0.0056 / 0.0015 | 0.048 / 0.048 |
| Dex3 (14) | 0.152 / 0.019 / 7e-6 | 0.185 / 0.042 / 5.7e-5 | 0.336 / 0.216 |

- **By phase** (frictionloss): hold 1.8e-9, arms 0.0062, hands 0.152, return 0.061 rad. The final pose differs by at most 0.0014 rad (none: 0.025 rad). Applying frictionloss as PhysX friction effort brings every group closer than leaving friction out.
- **The Dex3 differences are attributed to finger–table contact. This is inferred, not established.** Only the four middle-finger joints exceed 0.01 rad in the frictionloss run (left/right `middle_0` 0.12/0.15, `middle_1` 0.04/0.03). `parity_mujoco.py` now records MuJoCo robot contacts after every interval (`mujoco_contacts_per_read`). Both `middle_1` links touch the **table top** at z 0.740 on reads 37–77. There are gaps: left 38, 39, 41, 60 and 63–69; right 38, 60, 64, 65, 67 and 69. So the trajectory, meant to stay in free space, touches the table from the late **arm** phase on, and **the arm and waist numbers from read 37 on were also taken with the fingers on the table in MuJoCo**. At the worst Dex3 reads (left `middle_0` at read 64, right `middle_0` at read 65), MuJoCo has **no** contact on those links, and `middle_1` has none at 64/65 either. The `middle_1` maxima (read 72) coincide with contact. Isaac contacts were not measured. The contact explanation therefore rests on the timing of MuJoCo's contacts and on the joints involved, not on a matched contact comparison. These numbers are not free-space controller parity. Contact parity is out of scope here.
- **The first parity run (`isaac-parity-dev-2`) found the friction mismatch.** It ran with the converter's legacy friction coefficient still active. The arms crept far behind their targets (arm difference up to 0.22 rad, Isaac arm tracking error 0.25 rad against MuJoCo's 0.048), and the hands differed by up to 0.99 rad. The transport's friction handling above came from this run, and the run is kept as evidence.
- **Determinism and state.** Two replays in one Isaac process gave bit-identical joint states (max diff 0.0). Episode time matches MuJoCo to 1e-9 s (Isaac rounds its clock to 1e-9 s; 0.05 s steps, 4.5 s at the end), is monotonic, and all states are finite.
- **Rejections.** An expired deadline returns `rejected` with the reason `command deadline expired`. An out-of-limit target, the wrong joint order and `reset(object_xy=…)` each raise `ContractError`. `read` and `send` after `close` each raise `RuntimeError`.
- **Render staleness.** A second render of the same moved state was identical to the first (max diff 0), so there is no one-frame lag in this path. Three warm-up renders are discarded at every reset.
- **Images are not comparable yet.** The framing and arm poses match (see `…/parity/mujoco_left_isaac_right_*.png`), but the mean absolute pixel difference is about 90/255. With anti-aliasing and DLSS pinned off, the Isaac frame is grainy (RTX sampling noise without a denoiser or accumulation), and the table renders near-white under the 2 500-intensity dome light. Render settings and lighting have to be chosen and pinned before Isaac frames are used as observations.

**Timing and memory.** Wall clock, one environment, CPU governor powersave. Isaac: n = 180 intervals (2 replays × 90). MuJoCo: n = 90.

| Measure | Isaac (frictionloss run) | MuJoCo (host CPU) |
| --- | --- | --- |
| `send_joint_targets`, one 0.05 s interval (25 substeps) | median 174 ms, p95 177 ms, max 189 ms (≈ 7.0 ms per substep; 0.29× real time) | median 2.09 ms, p95 2.45 ms |
| `read()` with 112 px render | median 9.7 ms, p95 69 ms, max 825 ms (the read after interval 4 of the first replay; the post-reset read is not timed. The `none`, `dev-2` and `parity-4` runs stall at the same index, 840/1 064/837 ms) | median 0.28 ms |
| App launch / transport build | 83 s / 12.6 s | — |
| Isaac process GPU memory, peak of ~1 s samples | 4 945 MiB (none run and parity-4: 5 017 MiB) | — |
| Device total, peak of samples (includes GR00T's 6 626 MiB) | 11 827 MiB | — |

Isaac's per-interval cost comes from the Python substep loop. Every 2 ms substep does several GPU→CPU readbacks (joint state, gravity and Coriolis terms) and a CPU→GPU write, so this is a correctness-first reference loop, not a throughput figure. The spike's physics-only step was 2.7 ms. Moving the PD and bias computation onto the GPU, or into an explicit Isaac Lab actuator model, is the obvious optimisation. It has not been done.

## The development-run guard

`run_isaac.sh` refuses to start while a non-shell process whose command line matches `first_policy`, `task072`, `TASK-072` or `task-072` is alive. It matched the real M2 run and blocked two of these runs. Its limits:

- It matches **names only**, so a gated task under another name is not caught.
- It checks **once, at start**. A gated run that starts while an Isaac container is already up is not detected or stopped.

It is a courtesy check, not a lock. Coordinating with the owner of any gated run is still required.

## Blockers and caveats

No hard blocker for joint-level work. What remains:

- **Release-candidate simulator** (6.0.0-rc.22) and importer 3.2.0. A later release may change the output. The canonical hash detects that.
- **Contact and collision:** self-collision off in Isaac and on in MuJoCo; contact offsets and solver differ (PhysX TGS against MuJoCo `implicitfast`); fingers were compared only during accidental table contact (MuJoCo side only; Isaac contacts not measured).
- **Friction:** the PhysX static/dynamic friction effort is the closest available match to MuJoCo `frictionloss`, but it is not the same algorithm.
- **Rendering:** not usable as observations until the render mode, sampling and lighting are pinned. The DLSS-upscaled look of the spike and the noisy look here are both unvalidated.
- **Scope:** no apple, no plate, no task truth, no IK/embodiment integration, one environment.

## Recommended next step toward apple-to-plate-v2 in Isaac

1. Pin the render settings (sampling/denoiser, lighting close to MuJoCo's directional light + headlight), then add a camera manifest and an image parity check that has a threshold.
2. Rerun the parity check with a free-space trajectory (hands above the table) to separate controller parity from contact. Then add a scripted finger–table and finger–object contact test.
3. Port the `apple-to-plate-v2` scene objects to the Isaac stage with the same geometry, masses, friction and condim-6 rolling/torsional friction intent: the apple sphere and the plate with its rim. Add `reset(object_xy, plate_xy)` and an evaluator-only `task_truth()` mirroring `MuJoCoSimulation.task_truth`.
4. Put the embodiment's IK (kept as a validated MuJoCo kinematics provider, which the audit's FK match supports) over `IsaacTransport`. Then replay e9's scripted expert in Isaac on development seeds as a separately labelled cross-simulator check before any learned policy is evaluated there.
5. Speed up the substep loop on the GPU before collecting data at scale.
