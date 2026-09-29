# Isaac Lab 3 Newton backend (MuJoCo-Warp) for the v2 scene: development spike (2026-09-29)

A development spike on the Linux PC (RTX 5080) after [ISAAC_V2_SCENE.md](ISAAC_V2_SCENE.md). It is not a gated experiment and nothing here is a learned, policy, manipulation or task result. The checks are physics parity checks: the apple is moved by teleports and fixed joint targets, and no learned policy is involved. Learned Apple→Plate is unchanged at 0 successes. Isaac stays out of the core dependencies and the core import path: every Isaac, Newton, warp and MuJoCo-Warp import is inside `IsaacTransport`, and `tests/test_isaac_transport.py` checks that the module imports without them.

**The question.** Under PhysX, the v2 apple rolls off the table and off the plate rim where MuJoCo stops it, and a finger press pins it where MuJoCo squeezes it out ([ISAAC_V2_SCENE.md §4](ISAAC_V2_SCENE.md)). Does the `isaaclab_newton` backend in the pinned image reproduce MuJoCo's contact behaviour?

**Answer.** Yes, for all five scripted cases (two straight drops, the rim drop, the roll and the press), in a development spike with one clean run each, provided the transport builds the Newton model itself. Isaac Lab's own Newton model builder does not register the MuJoCo geom attributes, so every geom would compile with condim 3 and rolling friction would silently be ignored. With the transport's own builder:

- the roll case matches MuJoCo to 0.24 mm after 3 s;
- the rim drop stays on the plate, as in MuJoCo;
- the finger press squeezes the apple out, as in MuJoCo, ending 2.8 mm from MuJoCo's final position;
- the contact-free joint parity is 1.4e-5 rad at worst, against 0.0020 rad under PhysX.

PhysX stays the default. Newton is opt-in: `IsaacTransport(physics="newton")`, or `--physics newton` on the run scripts.

## 1. What Newton supports in this image

The facts below come from the image's installed sources (`isaaclab_arena:latest`, `sha256:2588b52605d7…`). Paths are relative to `/workspaces/isaaclab_arena/submodules/IsaacLab/source/` (Isaac Lab) or `/isaac-sim/kit/python/lib/python3.12/site-packages/` (Newton, MuJoCo-Warp). Anything not read directly from source is labelled *inferred*.

| Topic | Finding | Source |
| --- | --- | --- |
| Versions | Isaac Lab 3.0.0, `isaaclab_newton` 0.5.9, `newton` 1.1.0.dev0, `mujoco_warp` 3.5.0.2, `mujoco` 3.5.0, `warp` 1.12.0, Isaac Sim 6.0.0-rc.22. The host runs MuJoCo 3.13.0 | `isaaclab_newton/config/extension.toml`; `*.dist-info`; `/isaac-sim/VERSION` |
| Backend selection | `SimulationCfg.physics = NewtonCfg(...)`. `Articulation` and `RigidObject` are factories that return the backend's class | `isaaclab/sim/simulation_cfg.py`; `isaaclab/assets/articulation/articulation.py` (`FactoryBase`) |
| Solvers | `MJWarpSolverCfg` (the default), `XPBDSolverCfg`, `FeatherstoneSolverCfg` | `isaaclab_newton/physics/newton_manager_cfg.py` |
| Solver used here | Newton's `SolverMuJoCo` compiles a MuJoCo model (`mj_model`) from the Newton model and steps it with MuJoCo-Warp. With `use_mujoco_contacts=True` (the default), MuJoCo-Warp runs its own collision detection | `newton/_src/solvers/mujoco/solver_mujoco.py` (`run_collision_detection`) |
| MuJoCo options exposed | `integrator` (euler, rk4, implicitfast), `cone`, `impratio`, `solver` (cg, newton), `iterations`, `ls_iterations`, `njmax`, `nconmax`, `ccd_iterations` | `newton_manager_cfg.py` (`MJWarpSolverCfg`) |
| Rolling and torsional friction | The geom friction is `[mu, mu_torsional, mu_rolling]` from the shape material | `solver_mujoco.py`, geom export (`geom_params["friction"]`) |
| condim | Taken from the custom attribute `mujoco:condim` (USD `mjc:condim`), but only if `SolverMuJoCo.register_custom_attributes` was called on the builder. Otherwise it is left out of the geom and MuJoCo's default, 3, applies (*inferred* from MjSpec's default) | `solver_mujoco.py` (`get_custom_attribute("condim")`) |
| Isaac Lab's builder | `NewtonManager.instantiate_builder_from_stage` calls `add_usd` with the Newton and PhysX schema resolvers and does **not** register the MuJoCo custom attributes. Only the cloner does (`newton_replicate.py`). So a stage built the default way gets condim 3 on every geom, and rolling and torsional friction have no effect | `isaaclab_newton/physics/newton_manager.py`; `isaaclab_newton/cloner/newton_replicate.py` |
| USD friction attributes | `newton:torsionalFriction` and `newton:rollingFriction` on a material. Unauthored values use the builder defaults 0.005 and 0.0001, which are MuJoCo's defaults. Isaac Lab does not pass the `mjc:` schema resolver, so `mjc:rollingfriction` is not read | `newton/_src/usd/schemas.py`; `newton/_src/utils/import_usd.py`; `newton_manager.py` |
| solref | Set from the shape's contact stiffness and damping: `convert_solref(ke, kd, 1, 1)`, with timeconst = 2/kd and dampratio = kd/2·√(1/ke). Newton's defaults (ke 2500, kd 100) give MuJoCo's default (0.02, 1) | `solvers/mujoco/kernels.py` (`convert_solref`); `sim/builder.py` (`ShapeConfig`) |
| solimp | Custom attribute `mujoco:geom_solimp`. The default is MuJoCo's (0.9, 0.95, 0.001, 0.5, 2) | `solver_mujoco.py` |
| Solver tolerance | MuJoCo-Warp raises `opt.tolerance` to at least 1e-6 for float32 | `mujoco_warp/_src/io.py` (`opt.tolerance = max(opt.tolerance, 1e-6)`) |
| Joints | `joint_friction` becomes MuJoCo `frictionloss`, and the custom attribute `dof_passive_damping` becomes MuJoCo joint `damping`, so both use MuJoCo's own model. Isaac Lab's `write_joint_damping_to_sim` writes the drive `kd`, which becomes a MuJoCo actuator gain, not passive damping | `solver_mujoco.py` (joint export); `isaaclab_newton/assets/articulation/articulation.py` |
| Kinematic bodies | A kinematic free body is exported with a free joint and armature 1e10 (`_KINEMATIC_ARMATURE`). A world-fixed root becomes a mocap body | `solver_mujoco.py` |
| Camera and render | The RTX camera works. `NewtonManager` writes Newton body poses to Fabric (`sync_transforms_to_usd`) at render time, but only when a step has marked them dirty. The transport marks them dirty itself after resets and teleports. A separate Warp renderer exists; it is not used | `newton_manager.py`; `isaaclab_newton/renderers/` |
| Same USD | Yes. `IsaacTransport` loads the same converted USD (`g1_29dof_with_hand-ref`, canonical tree sha256 `cd1fdb27…2fa8`, recorded in every run) with the same scene and camera manifests | this spike's runs |

## 2. The Newton backend in `IsaacTransport`

Code: `src/embodied_jepa/isaac_transport.py` (`physics="newton"`) and `src/embodied_jepa/isaac_scene.py` (`newton_solver_options`, `newton_contact_params`, `solref_to_newton_ke_kd`, `shape_role`, `UNMATCHED_IN_NEWTON`). All of it is Isaac-free and unit-tested in `tests/test_isaac_scene.py` and `tests/test_isaac_transport.py`, except the transport's Newton methods.

- **Solver.** `NewtonCfg(num_substeps=1, use_cuda_graph=True, solver_cfg=MJWarpSolverCfg(...))`:
  - from the scene manifest's `mujoco_option`: integrator `implicitfast`, cone `pyramidal`, impratio 1;
  - from `NEWTON_SOLVER`: solver `newton`, 100 iterations and 50 line-search iterations (the host's), njmax 1000, nconmax 200;
  - dt 0.002 s, and 25 steps per 0.05 s interval, as before.
- **Model.** Before `sim.reset`, the transport makes its own `ModelBuilder`, calls `SolverMuJoCo.register_custom_attributes`, runs `add_usd` on the stage with Isaac Lab's two resolvers, and hands the builder to `NewtonManager.set_builder`. It then writes the following:
  - per shape role (floor, table, apple, plate base, plate rim, robot, found by prim path), the MuJoCo friction triple, condim, solref (as ke/kd), solimp and margin, taken from the scene manifest; the robot uses MuJoCo's geom defaults, which the G1 MJCF uses;
  - per robot joint: passive damping, frictionloss (or 0 with `joint_friction="none"`), zero drive gains, and a limit `ke` of 0, which leaves MuJoCo's default limit solref.
- **Scene.** The floor and table are static colliders (MuJoCo world geoms), not kinematic bodies. The plate is kinematic, as under PhysX.
- **Read-back.** After `sim.reset`, the compiled MuJoCo-Warp model is read back into `newton_model`: options, and per geom, joint and body. It is recorded in every run. Damping and frictionloss are re-checked after every reset, and the transport raises if they differ.
- **Control.** The control law is unchanged: `kp (q* − q) − kd qd + bias`, clipped to `ctrlrange` and applied as joint effort.
  - The bias is MuJoCo-Warp's `qfrc_bias`. Like `MuJoCoSimulation`'s, it comes from the forward pass of the previous step.
  - Right after a reset, the bias is `qfrc_bias` from a CPU `mj_forward` of the compiled model at the reset pose; `MuJoCoSimulation.reset` also runs `mj_forward`.
- **Reset.** Reset zeroes MuJoCo-Warp's history-carrying data (`qacc_warmstart`, `qacc`, `qfrc_applied`, `xfrc_applied`, `act`, `ctrl`), as `mj_resetData` does. Without this, a second replay in the same process differed from the first by up to 0.0065 rad (`isaac-newton-parity-1-free`, commit `c5f0136`). With it, three replays agree to 4e-6 rad (`isaac-newton-parity-dev-2`). Which of the zeroed arrays mattered was not isolated; the warm start is the likely one (*inferred*). The list is manual (`NEWTON_HISTORY_FIELDS`) because `mujoco_warp.reset_data` would also overwrite the pose the reset just wrote; a fake-solver unit test in `tests/test_isaac_transport.py` pins it.
- **Contacts.** Contacts are read from MuJoCo-Warp's contact list: body pairs at every physics step, and at the last step of each interval `mj_contactForce` magnitudes summed per pair. Labels follow the host's `parity_mujoco.scene_contacts`, and robot self-contacts are flagged.
- **Recording.** Every run record carries `physics_backend`. `scripted_checks_mujoco.py` and `parity_mujoco.py` report it. Records made before this spike have no such key and are PhysX runs.
- **Provenance.** A `usd_canonical_tree_sha256` of `None` in any run record means that no `conversion.json` was found above the USD, so the USD's provenance is **missing**. It does not mean "no hash needed" (`isaac_scene.usd_canonical_hash`).

## 3. Runs

Image `isaaclab_arena:latest` `sha256:2588b52605d77552d4480196501c4b8b61774ef6d70d9b3d59291622d6856d9c`, the same USD, scene manifest `e698b44d…01af3e72`, camera manifest `866c596a…9423133` and joint manifest v1 as [ISAAC_V2_SCENE.md](ISAAC_V2_SCENE.md). Host MuJoCo 3.13.0. Every run is under `outputs/` in the worktree (git-ignored), made with `--physics newton` via `scripts/isaac/run_isaac.sh`.

| Run | Code | What |
| --- | --- | --- |
| `isaac-newton-scripted-dev-1` … `-dev-4` | uncommitted development code | first scripted runs. dev-1 ended silently after two cases (a traceback lost at `app.close()`, then an `IndexError` in contact labelling, fixed). dev-2 hung (see §6) |
| `isaac-newton-scripted-1`, `isaac-newton-parity-1-free` | `c5f0136`, clean tree | before the reset fix: superseded, kept as evidence of the replay defect |
| `isaac-newton-parity-dev-2` | `c5f0136` plus the uncommitted reset fix (then committed unchanged as `df9b363`) | three replays, repeatability check of the fix |
| `isaac-newton-scripted-2` | `df9b363`, clean tree | hung before the model was built. Stopped, no record (§6) |
| **`isaac-newton-parity-2-free`** | **`df9b363`, clean tree** | **free-space joint parity and image parity (reported below)** |
| **`isaac-newton-scripted-3`** | **`77b53d8`, clean tree** | **the scripted checks (reported below)**. `77b53d8` differs from `df9b363` only by the diagnostic stack dump in the run scripts |

Review follow-ups of #107 came after these runs and were **not** rerun in Isaac: the history field list became the constant `NEWTON_HISTORY_FIELDS` (same six fields), `scripted_checks_isaac.py` records `cases_requested`, `scripted_checks_mujoco.py` refuses a run with missing cases unless `--allow_partial` (then flags it `partial`), and `audit_newton_model.py` gained the body flag and existence check; the host-side scripts were re-applied to the recorded runs.

Reproduce (host, then container, then host):

```sh
U=/oej/usd/g1_29dof_with_hand-ref/run/usd/g1_29dof_with_hand/g1_29dof_with_hand.usda
sg docker -c "scripts/isaac/run_isaac.sh scripted_checks_isaac.py outputs/<new> --usd $U \
  --manifest /oej/configs/isaac/g1_dex3_joint_manifest_v1.json \
  --scene /oej/configs/isaac/apple_to_plate_v2_scene_v1.json --physics newton"
uv run --no-sync python scripts/isaac/scripted_checks_mujoco.py --isaac outputs/<new>/run --output outputs/<new>/compare
uv run --no-sync python scripts/isaac/audit_newton_model.py --isaac outputs/<new>/run --output outputs/<new>/newton_audit
sg docker -c "scripts/isaac/run_isaac.sh parity_isaac.py outputs/<new2> --usd $U \
  --manifest /oej/configs/isaac/g1_dex3_joint_manifest_v1.json \
  --action_manifest /oej/configs/g1_sim_action.json \
  --scene /oej/configs/isaac/apple_to_plate_v2_scene_v1.json \
  --camera /oej/configs/isaac/onboard_camera_v1.json --trajectory free_space_v1 --physics newton"
uv run --no-sync python scripts/isaac/parity_mujoco.py --isaac outputs/<new2>/run --output outputs/<new2>/parity
```

## 4. Results

### Parameter audit (`audit_newton_model.py`, `isaac-newton-scripted-3` and `isaac-newton-parity-2-free`, identical)

The compiled MuJoCo-Warp model was compared with the host MuJoCo v2 model:

- **Solver options.** timestep, integrator, cone, solver, iterations, line-search iterations, `ls_tolerance`, impratio and gravity match. Tolerance differs: 1e-6 against 1e-8.
- **Contact parameters.** For every role (floor, table, apple, plate base, 16 rim capsules, 53 colliding robot geoms), the counts, geom types, friction triple, condim, solref, solimp and margin all match within float32 rounding, except the floor's geom type (plane in MuJoCo, box in Newton; §5 item 6). The apple is condim 6, friction (1, 0.01, 0.001).
- **Joints.** For all 43 canonical joints, damping, armature, frictionloss, range, limit solref and limit solimp match to at most 3.4e-7 (the range, float32).
- **Bodies.** All 46 bodies exist on both sides. Every body mass matches to within 4.3e-7 kg (bar 1e-6 kg), and principal inertias to within a relative 1.37e-6 (bar 1e-5, about 100 float32 ulps, stated in the script): PASS. The body flag and the existence check were added after the runs (review of #107) and applied to the same records.
- **Not compared.** Geom sizes and poses, priority, solmix, gap, CCD settings and mesh hull vertices; geoms are compared per role as sets of values, not geom by geom (the report lists these under `not_compared`).
- **Sizes.** nv is 55 against 49, because the plate is a 6-dof kinematic body. nu is 0 against 43, because efforts are applied directly. nexclude is 950 against 0.
- **Self-collision.** Robot self-collision is off in Newton (by contype/conaffinity) and on in the host.

### Scripted checks (`isaac-newton-scripted-3`; same metrics as #105)

The PhysX column restates `isaac-v2-scripted-1` from [ISAAC_V2_SCENE.md](ISAAC_V2_SCENE.md).

| Case | MuJoCo | Newton (MuJoCo-Warp) | Newton − MuJoCo | PhysX (#105) |
| --- | --- | --- | --- | --- |
| Drop 10 cm onto the table | rests at z 0.766831, settles 0.35 s | rests at z 0.766831, settles 0.35 s | xy 0.00 mm, z −0.0004 mm, settle 0.00 s. Max mid-fall position gap 1.96 mm (read 2) | z +0.17 mm, settles 0.20 s earlier |
| Drop 10 cm onto the plate centre | z 0.778831, 0.35 s | z 0.778831, 0.35 s | xy 0.00 mm, z −0.0003 mm, settle 0.00 s. Max mid-fall gap 1.96 mm | z +0.17 mm, 0.20 s earlier |
| Drop 5 cm off the plate centre | stays on the plate. Ends 3.4 cm from the centre, still creeping at 6.7 mm/s at 3 s | **stays on the plate**. Ends 0.72 cm from the centre, still creeping at 9.1 mm/s | same outcome, different path. Final xy 41.1 mm apart, max gap 72.5 mm, path 0.209 against 0.139 m. Neither leaves the 5 cm circle | rolls off the plate and wedges against the robot hip |
| Roll on the table, 0.2 m/s with matching spin | 0.2 → 0.1357 → 0.0852 → 0.0528 m/s at 0/1/2/3 s. 0.3431 m travelled | 0.2 → 0.1356 → 0.0851 → 0.0526 m/s. 0.3429 m | final xy 0.24 mm, max gap 0.39 mm | does not slow; leaves the table at ~2.8 s |
| Right middle finger pressed onto the resting apple | contact from read 47 (`middle_1` max 5.83 N, `middle_0` from read 54, 6.97 N). Apple squeezed out: net xy displacement 6.7 cm (xy path length 7.1 cm) | contact from read 47 (`middle_1` 6.33 N, `middle_0` from read 55, 6.93 N). **Squeezed out**: net xy displacement 6.4 cm (xy path length 6.8 cm) | final xy 2.8 mm. Joint difference up to 0.0028 rad | pinned (4 mm), joint difference up to 0.031 rad |

- The plate did not move in any Newton case (max displacement 0).
- **Drops.** The mid-fall gap of 1.96 mm at read 2 is about one physics step of fall at that speed. The cause is not identified; the rest positions and settle times agree.
- **Off-centre drop.** It lands on the rim and keeps rolling around inside the plate in both engines, a sensitive, chaotic contact sequence. Only the outcome, staying on the plate, is shared. The final positions and paths are not.
- **Contact forces.** They are sampled once per interval. As in #105 they are not a stiffness comparison. The off-centre drop's peak sampled force (26.5 N against 11.0 N) belongs to different contact sequences.

### Contact-free joint parity (`isaac-newton-parity-2-free`, `free_space_v1`, 91 reads)

The PhysX column restates `isaac-v2-parity-1-free`.

| Group | Newton − MuJoCo max | p95 | median | Tracking-error max, MuJoCo / Newton | PhysX − MuJoCo max (#105) |
| --- | --- | --- | --- | --- | --- |
| Legs (12, held) | 5.7e-10 | 4.5e-10 | 1.1e-10 | 5.5e-10 / 1.1e-10 | 8.2e-10 |
| Waist (3) | 2.4e-6 | 1.6e-6 | 3.1e-10 | 0.0085 / 0.0085 | 3.1e-4 |
| Arms (14) | 3.7e-6 | 5.3e-7 | 2.7e-8 | 0.064 / 0.064 | 0.0020 |
| Dex3 (14) | 1.4e-5 | 2.5e-6 | 4.5e-9 | 0.122 / 0.122 | 5.2e-4 |

- **By phase.** hold 1.8e-9, arms 3.2e-6, hands 2.7e-6, return 1.4e-5 rad. The worst joint is `left_hand_thumb_1_joint` at read 79. The final pose differs by 2.3e-6 rad.
- **Contacts.** Neither simulator recorded any robot contact on any read, including every Newton substep. The apple rests on the table in both, with positions within 0.078 mm.
- **Repeatability.** Two replays in one process agree to 2.9e-6 rad, not bit-identically; PhysX was bit-identical. The same replay in two processes (`parity-2-free` against `parity-dev-2`) agrees to 3.3e-6 rad. MuJoCo-Warp on the GPU is therefore repeatable to a few 1e-6 rad here, not exactly.
- **Rejections.** Expired deadline, out-of-limit target, wrong order, apple off the table and apple on the plate are all rejected as under PhysX. Use after `close()` fails.

### Image parity (same declared metric and bars as #105)

The renderer is the pinned RTX path tracer. Physics does not change it, so photometry is the same as under PhysX.

| Read | MAD | PSNR dB | SSIM | IoU robot / table / apple / plate | Centroid px apple / plate |
| --- | --- | --- | --- | --- | --- |
| 0 (calibration pose) | 26.2 | 15.9 | 0.747 | 0.998 / 0.999 / 0.952 / 0.984 | 0.10 / 0.10 |
| 30 | 28.1 | 15.6 | 0.721 | 0.995 / 0.996 / 1.000 / 0.975 | 0.00 / 0.15 |
| 50 | 28.2 | 16.0 | 0.733 | 0.998 / 0.999 / 1.000 / 0.985 | 0.00 / 0.11 |
| 70 | 29.4 | 15.8 | 0.724 | 0.999 / 0.999 / 1.000 / 0.985 | 0.00 / 0.11 |
| 90 | 26.3 | 15.9 | 0.746 | 0.996 / 0.998 / 1.000 / 0.984 | 0.00 / 0.07 |

- **Verdict.** Geometry: PASS. Photometry: FAIL (MAD 26–29 against ≤ 15; SSIM 0.72–0.75 against ≥ 0.80). This is the same verdict as PhysX.
- **Render freshness.** It passes: a read after reset equals the first reset frame (max diff 0) and differs from the last pre-reset frame (mean 12.9); two renders of a moved state are equal.
- **Frame repeatability.** Frames across the two replays differ by at most 2/255 (reads 70 and 90), following the 3e-6 rad joint differences. Under PhysX they were identical.

### Timing and memory (`isaac-newton-parity-2-free`)

| Measure | Newton | PhysX (#105) |
| --- | --- | --- |
| `send_joint_targets` per 0.05 s interval (25 steps) | median 25.4 ms, p95 26.4 ms | median 173 ms |
| First interval of the parity run (includes the CUDA-graph capture) | 81 s | — |
| `read()` with the 112 px path-traced render | median 19.1 ms, p95 26.7 ms | median 20.6 ms |
| App launch / transport build | 84 s / 44 s | 83 s / 6.9 s |
| Isaac process GPU memory (peak of ~1 s samples) | 2 899 MiB (parity, rendering) | 5 029 MiB (parity) |

In the scripted run (no rendering), the solver initialisation took 13 s and the CUDA-graph capture 82 s (`CUDA graph took: 82.36 s` in `scripted-3/log.txt`), a separate measurement from the parity run's 81 s first interval.

## 5. Parameters Newton does not match (`isaac_scene.UNMATCHED_IN_NEWTON`)

1. **Solver tolerance.** MuJoCo-Warp raises it to at least 1e-6 for float32; the host uses 1e-8. This is unmatchable in this image.
2. **Precision.** Newton runs float32 on the GPU; the host runs float64 on the CPU. This is unmatchable, and it is the likely source of the ~1e-5 rad joint and ~1e-6 rad repeatability differences (*inferred*).
3. **Engine versions.** The container has MuJoCo-Warp 3.5.0.2 and MuJoCo 3.5.0; the host has MuJoCo 3.13.0. Collision detection is MuJoCo-Warp's GPU implementation.
4. **Robot self-collision.** Off in Newton (contype/conaffinity plus 950 exclude pairs), on in the host MJCF. This is the same choice as the PhysX runs. It is probably configurable (*inferred*), but was not changed here.
5. **Plate.** In Newton the plate is a kinematic free body (6 dofs, armature 1e10), written at reset; in MuJoCo it is a static body. It did not move in any case.
6. **Floor.** In both Isaac backends the floor is a box with its top at z = 0; in MuJoCo it is an infinite plane.
7. **Robot mesh hulls.** MuJoCo-Warp builds hulls from the converted USD meshes, the host from the MJCF meshes. The counts and types match; the hulls were not compared vertex by vertex.
8. **Actuation.** The host uses 43 gain-1 motor actuators; Newton applies the same clipped torque as `qfrc_applied`. These are equivalent for motors under implicitfast (*inferred*).
9. **Bias right after reset.** Newton uses a CPU `mj_forward` of the compiled model; afterwards it uses MuJoCo-Warp's `qfrc_bias`, with the host's one-step staleness.
10. **CCD settings.** The host uses `ccd_iterations` 35 (MuJoCo's default); `MJWarpSolverCfg.ccd_iterations` was left at its default and neither set nor read back, and the apple's sphere-box and sphere-cylinder contacts go through convex collision. Not verified to be equal.
11. **Collision groups.** Newton's contype/conaffinity colouring puts floor and table in one group (8) and the plate base and rims in another (16), so those pairs never collide; in the host they are all static geoms that MuJoCo does not collide either. Harmless here (static or kinematic bodies), but the masks differ.

## 6. Defects, caveats and open issues

- **Isaac Lab's default Newton builder drops condim.** Without the transport's own builder, Newton would compile the apple with condim 3 and the rolling and torsional friction would have no effect. This was found from source, not run: every Newton run here used the transport's builder, so there is no measured default-builder baseline.
- **Intermittent start-up hang.** 2 of the 10 Newton container runs (`scripted-dev-2` and `scripted-2`) hung before the Newton model was built: one CPU core stayed busy for more than 20 minutes and nothing was logged. Both were stopped. The container blocks ptrace, so py-spy could not attach. `77b53d8` adds a `faulthandler` stack dump every 5 minutes to both run scripts, and the one run since (`scripted-3`) did not hang. The cause is unknown. Whether PhysX runs can hang the same way was not tested.
- **Lost tracebacks.** An exception in the run scripts was hidden because `app.close()` in `finally` ends the process first. The scripts now print the traceback before closing.
- **Not bit-deterministic.** Replays agree to about 3e-6 rad, not exactly. Anything that needs exact replay (for example the image repeat check) must use a tolerance under Newton.
- **One run each.** Each reported number comes from one clean run of one scripted case set and one trajectory. The off-centre drop is sensitive to small differences.
- **Not a task result.** None of this is a learned or task result. It establishes that, in this image, Newton/MuJoCo-Warp reproduces MuJoCo's v2 contact behaviour in these five scripted cases and the free-space trajectory.

## 7. Recommendation

- **For the tested cases (five scripted apple cases and one free-space trajectory, one clean run each), use Newton/MuJoCo-Warp rather than PhysX when an Isaac run involves the v2 apple, and keep PhysX as the default.** Other contact situations (grasps, lifts, the e9 expert) are untested. For the tested cases the fallbacks listed in #105 are not needed:
  - (b) an explicit rolling-resistance torque;
  - (c) angular damping;
  - (d) accepting an engine factor.
- **Next steps:**
  1. Replay the TASK-070 expert e9 on development seeds as a separately labelled cross-simulator check, with the embodiment's IK (MuJoCo kinematics provider) over `IsaacTransport(physics="newton")`.
  2. Decide whether to enable robot self-collision.
  3. Watch the start-up hang with the stack dumps before relying on Newton in unattended runs.
- **Images.** The photometric gap is a renderer issue and is unchanged by the physics backend.
