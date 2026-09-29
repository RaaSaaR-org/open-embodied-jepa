# Isaac Sim: pinned onboard rendering, contact-free joint parity and the v2 apple and plate (2026-09-29)

A development step on the Linux PC (RTX 5080) after [ISAAC_MJCF_TRANSPORT.md](ISAAC_MJCF_TRANSPORT.md). It is not a gated experiment. It is simulator infrastructure and physics parity only. It is not a manipulation, policy or task result. The scripted checks below move the apple with fixed joint targets and teleports. They involve no learned policy, and scripted behaviour in Isaac is not a learned result. Learned Apple→Plate is unchanged by this work. Isaac stays out of the core dependencies and the core import path: `isaac_scene.py` imports NumPy only, `isaac_transport.py` imports Isaac only inside the class, and nothing in the package imports either module.

All four items of the brief were delivered: (1) pinned rendering, a camera manifest and an image-parity check; (2) contact-free joint parity with contacts recorded in both simulators; (3) the v2 table, apple and plate in Isaac, with `reset(object_xy, plate_xy)` and an evaluator-only `task_truth()`; (4) scripted drop, roll and finger-press checks.

## Runs and environment

Image, Isaac Sim/Lab, importer, USD (`g1_29dof_with_hand-ref`, canonical tree sha256 `cd1fdb27…2fa8`) and joint manifest v1 are those of [ISAAC_MJCF_TRANSPORT.md](ISAAC_MJCF_TRANSPORT.md). The runs below record only the USD **path**, not its hash; the hash comes from that directory's `conversion.json`. From the review fixes onward, `parity_isaac.py` and `scripted_checks_isaac.py` also record `usd_canonical_tree_sha256`. A `usd_canonical_tree_sha256` of `None` in a run record means that no `conversion.json` was found above the USD: the USD's provenance is missing, and the run can be tied to a converted USD only by its recorded path. PhysX GPU pipeline, dt 0.002 s, 25 substeps per 0.05 s interval, one environment. Host MuJoCo 3.13.0.

| Run (`outputs/`, git-ignored) | Code | What |
| --- | --- | --- |
| `isaac-v2-render-probe-{1..4}` | uncommitted development code | render-setting probes (reset frame only) |
| `isaac-v2-parity-dev-1`, `isaac-v2-scripted-dev-1` | uncommitted development code | first runs; identical numbers to the runs below |
| `isaac-v2-parity-1-free` | `45e4d55`, clean tree | free-space trajectory with apple and plate; image parity |
| `isaac-v2-parity-1-tablecontact` | `45e4d55`, clean tree | the 2026-09-28 trajectory, now with contacts measured in Isaac |
| `isaac-v2-scripted-1` | `45e4d55`, clean tree | scripted apple checks |

Committed manifests (canonical JSON sha256):

- `configs/isaac/apple_to_plate_v2_scene_v1.json`: `e698b44d60a52793be734e375d71b5e74f6ac41ec5eca74a3d20ed2b01af3e72`. The runs recorded `82471775be1b7b41a47f370d1456f5dc41f6b9d39fdb764102bb9a0cfc4848d0`, the file at `45e4d55`. The review fixes changed only two descriptive fields, `isaac_physx.solver` and `unmatched_in_physx`; every physical value is identical. To recheck a scripted run made at `45e4d55`, pass `--scene <(git show 45e4d55:configs/isaac/apple_to_plate_v2_scene_v1.json)` to `scripted_checks_mujoco.py`.
- `configs/isaac/onboard_camera_v1.json`: `866c596a2d3d6e979adf55a354bf6cd1501b2d898c034c5ca47a5438f9423133`

Both are written by `scripts/isaac/write_scene_manifests.py` from `MuJoCoSimulation` (switched to v2 by `apply_v2_scene`) plus the constants in `isaac_scene.py`. `tests/test_isaac_scene.py` checks that they still match.

Reproduce (host, then container, then host):

```sh
U=/oej/usd/g1_29dof_with_hand-ref/run/usd/g1_29dof_with_hand/g1_29dof_with_hand.usda
sg docker -c "scripts/isaac/run_isaac.sh parity_isaac.py outputs/<new> --usd $U \
  --manifest /oej/configs/isaac/g1_dex3_joint_manifest_v1.json \
  --action_manifest /oej/configs/g1_sim_action.json \
  --scene /oej/configs/isaac/apple_to_plate_v2_scene_v1.json \
  --camera /oej/configs/isaac/onboard_camera_v1.json --trajectory free_space_v1"
uv run --no-sync python scripts/isaac/parity_mujoco.py --isaac outputs/<new>/run --output outputs/<new>/parity
sg docker -c "scripts/isaac/run_isaac.sh scripted_checks_isaac.py outputs/<new2> --usd $U \
  --manifest /oej/configs/isaac/g1_dex3_joint_manifest_v1.json \
  --scene /oej/configs/isaac/apple_to_plate_v2_scene_v1.json"
uv run --no-sync python scripts/isaac/scripted_checks_mujoco.py --isaac outputs/<new2>/run --output outputs/<new2>/compare
```

## 1. Rendering, camera manifest and image parity

### A stale-frame defect in the earlier transport

Isaac Lab 3 renders at most once per physics step: `ensure_isaac_rtx_render_update` skips Kit's `app.update()` when the step count has not changed, and `SimulationContext.render()` no longer renders. So in the 2026-09-28 transport:

- a read right after `reset()` returned the frame of the last physics step **before** the reset;
- a second render of the same state returned the same buffer without rendering; and
- the "three warm-up renders" did nothing.

The earlier check "a second render of the same moved state was identical, so there is no one-frame lag" therefore tested nothing. The first probe showed the defect: 18 renderer and lighting variants gave byte-identical frames. `IsaacTransport._pump_render` now syncs physics to Fabric, pumps `app.update()` with physics stepping paused, and marks the step as rendered.

In `isaac-v2-parity-1-free`:

- a read after `reset()` equals the first replay's reset frame (max diff 0);
- it differs from the last pre-reset frame (mean 12.9, max 149 /255); and
- a second render of a moved state equals the first (max diff 0). This shows determinism, not freshness: it is the check that tested nothing before the fix.

Only the second check discriminates freshness. `parity_isaac.py` now raises unless it holds and the first holds (`isaac_scene.check_render_freshness`, unit-tested).

The joint-level numbers of 2026-09-28 are unaffected. Only its frames and staleness claim are.

### Choosing the render settings (development, one frame)

The settings were chosen in probes 1–4 on **one** frame: the seed-0 reset scene (robot reset pose, v2 apple and plate at their defaults). Choosing settings on the frame that is then scored would be circular, so read 0 of a parity run, which has the same robot pose, is reported separately. The held-out frames are reads 30, 50, 70 and 90. Probe findings, measured on that frame:

| Renderer | Change between two renders of the same state (mean /255) | Note |
| --- | --- | --- |
| Real-time path tracing (RT2), AA off | 10.4–19 | the grain seen on 2026-09-28 |
| RT2 + FXAA | 12.7–18.9 | |
| RT2 + DLAA, 8 updates per frame | 1.6–2.3 | still changes between renders; DLSS warns that 112 px is below its 300 px minimum |
| Path tracing, 64 spp in one frame, OptiX denoiser | **0** | chosen |

The earlier near-white frames came from the 2 500-intensity dome light, which gave a mean pixel value of 220. The pinned lighting mirrors MuJoCo's:

- a distant light pointing down with hard shadows, intensity 300 (MuJoCo: directional light `0 0 -1`, diffuse 0.7);
- a shadowless distant "headlight" parented to the camera, intensity 300 × 0.4/0.7 (MuJoCo headlight diffuse 0.4).

The intensity was the lowest-MAD point of the grid 150/200/300/450 × sRGB gamma on/off × tonemapper 6/1. The path tracer ignores RTX's ambient light, so MuJoCo's headlight ambient (0.1) has no equivalent. All renderer settings are set from the manifest and read back at construction; a mismatch raises. The floor is a 2 cm box with its top at z = 0 in MuJoCo's colour. The earlier grid ground plane was fetched from NVIDIA's asset server; the box needs no network asset.

### Camera manifest

`onboard_camera_v1.json` records:

- the camera itself: name `onboard_rgb`, parent `torso_link`, position `0.08 0 0.35`, quaternion (wxyz, from MuJoCo's compiled model), vertical fovy 75°, 112 × 112, near/far clip = MuJoCo's `znear`/`zfar` × extent (0.0157 / 78.6 m), and output RGB uint8 HWC;
- MuJoCo's lights and headlight, and its colour pipeline (fixed-function, no tonemap or gamma);
- the pinned Isaac render settings; and
- the image-parity metric, declared before the first image-parity run.

### Declared metric and result

The reference is MuJoCo's `onboard_rgb` for the same joint state in the v2 scene. Metrics:

- MAD over pixels and channels, PSNR, and SSIM of luma (7 × 7 uniform window);
- per-class mask IoU (robot, table, apple, plate, floor/background), using MuJoCo's segmentation render and Isaac's `instance_id_segmentation_fast`;
- the distance between apple and plate mask centroids.

The bars were declared in the manifest before the run:

- geometry: robot and table IoU ≥ 0.90, apple and plate IoU ≥ 0.80, centroids ≤ 1 px;
- photometry: MAD ≤ 15 and SSIM ≥ 0.80.

Both must hold on every held-out read.

`isaac-v2-parity-1-free` (`scripts/isaac/image_parity.py`):

| Read | MAD | PSNR dB | SSIM | IoU robot / table / apple / plate | Centroid px apple / plate |
| --- | --- | --- | --- | --- | --- |
| 0 (calibration pose) | 26.2 | 15.9 | 0.747 | 0.998 / 0.999 / 0.952 / 0.984 | 0.10 / 0.10 |
| 30 | 28.1 | 15.6 | 0.721 | 0.993 / 0.995 / 1.000 / 0.975 | 0.00 / 0.15 |
| 50 | 28.2 | 16.0 | 0.733 | 0.997 / 0.999 / 1.000 / 0.990 | 0.00 / 0.08 |
| 70 | 29.3 | 15.8 | 0.724 | 0.992 / 0.996 / 1.000 / 0.985 | 0.00 / 0.11 |
| 90 | 26.4 | 15.9 | 0.745 | 0.997 / 0.998 / 0.974 / 0.979 | 0.09 / 0.04 |

**Geometry: PASS. Photometry: FAIL** (MAD 26–29 against ≤ 15; SSIM 0.72–0.75 against ≥ 0.80).

- **What passes.** The camera pose, intrinsics and scene geometry agree to within about one pixel of mask boundary.
- **What fails.** Shading differs. The Isaac frames are paler and softer, with fainter shadows, because the path tracer, denoiser, tonemapper and sRGB output are not MuJoCo's fixed-function colours.
- **Determinism.** Frames repeat exactly within a process: the second replay's frames at all five reads equal the first's (max diff 0). They also repeat across two processes on the same host, driver and image. `parity-dev-1` and `parity-1-free` are separate container runs with byte-identical frame PNGs and identical joint traces, although `dev-1` ran uncommitted code. Determinism across drivers or hosts was not tested.
- **Consequence.** Isaac frames are geometrically consistent observations, but they are **not** a drop-in replacement for MuJoCo frames. A model trained on MuJoCo pixels should be expected to see a distribution shift. Side-by-side images and class maps are in `outputs/isaac-v2-parity-1-free/parity/`.

## 2. Joint parity without contact

The new trajectory variant is `free_space_v1` in `scripts/isaac/parity_targets.py`. It has the same phases as before, but the arms are raised to shoulder pitch −0.8 instead of −0.3. During development, every robot collision geom stayed at least 5 cm from the table over all 90 intervals (`mujoco.mj_geomDistance`). The apple and plate are present at the seed-0 layout.

Contacts are recorded in both simulators:

- MuJoCo: body pairs after each interval, with `mj_contactForce` magnitudes;
- Isaac: PhysX tensor contact views (`create_rigid_contact_view`, per-pair force matrices), at the last physics step and at any substep of the interval.

**Neither simulator recorded any robot contact on any read**, including any Isaac substep, and MuJoCo recorded no robot self-contact. |q_Isaac − q_MuJoCo| over 91 reads, in rad (`isaac-v2-parity-1-free`, frictionloss mode):

| Group | Max | p95 | Median | Tracking-error max, MuJoCo / Isaac |
| --- | --- | --- | --- | --- |
| Legs (12, held) | 8.2e-10 | 7.9e-10 | 2.2e-10 | 5.5e-10 / 8.5e-10 |
| Waist (3) | 3.1e-4 | 1.0e-4 | 7.5e-7 | 0.0085 / 0.0084 |
| Arms (14) | 0.0020 | 0.0013 | 5.1e-5 | 0.064 / 0.064 |
| Dex3 (14) | 5.2e-4 | 1.2e-4 | 5.5e-6 | 0.122 / 0.122 |

- **By phase:** hold 1.8e-9, arms 3.5e-4, hands 0.0020, return 0.0020 rad. The worst joint is `left_wrist_roll_joint` at read 71. The final pose differs by 3.4e-4 rad.
- **Repeatability:** two replays in one process are bit-identical, episode time matches, and no joint exceeds 0.01 rad.
- **The Dex3 figure.** Without contact, Dex3 agreement goes from 0.152 rad (the 2026-09-28 run) to 5e-4 rad.

`isaac-v2-parity-1-tablecontact` reruns the 2026-09-28 trajectory (`table_contact_v0`) in the v2 scene, now with Isaac contacts measured. The contacts were:

| Contact | MuJoCo reads | Isaac reads (last physics step) |
| --- | --- | --- |
| Middle fingers on the table | 37–90, with gaps at 38, 67, 68, 78 and 80 | 37–90, gap at 80 only |
| `left_hand_middle_1_link` on the table | 37–77, 32 reads | 37–79, 43 reads |
| `right_hand_middle_1_link` on the table | 37–79, 35 reads | 37–90, 48 reads |
| `right_hand_middle_0_link` on the apple | reads 50 and 56 | reads 51–58 (7 reads) |
| `right_hand_middle_1_link` on the plate | 81–90 | 81–90 |

The same body pairs occur in both simulators, on largely but not exactly the same reads. The apple was pushed 6.6 cm (MuJoCo) and 7.6 cm (Isaac). Because the apple ended in different places, this run's image geometry **fails** its declared bars: apple IoU 0.74 at read 70 and 0.56 at read 90, and apple centroid 1.85 px at read 90. Photometry also fails (MAD 24–28, SSIM 0.70–0.75).

The differences in that run: Dex3 max 0.21 rad at read 68 (`left_hand_middle_0`), arms 0.0073, waist 0.0059. These are larger than on 2026-09-28 (0.152, 0.0091, 0.0025). One untested hypothesis is that this scene uses friction 1 in PhysX (see §3), where the earlier default was 0.5, and contains the apple and plate.

The worst Dex3 read (68) is one where Isaac records finger–table contact and MuJoCo records none. So the largest gap coincides with a contact-state mismatch.

Together with the free-space result, this **strongly supports**, but does not confirm, what the earlier document could only infer: the Dex3 differences come from contact, not from free-space tracking. It is not a controlled ablation: `free_space_v1` is a different trajectory (shoulder pitch −0.8, not −0.3).

## 3. The v2 table, apple and plate in Isaac

`IsaacTransport(…, scene_manifest=…, objects=True)` spawns the following from the scene manifest:

- the floor and table, as kinematic boxes so that contact views can name them;
- the apple, a dynamic sphere;
- the plate, a kinematic body made of a cylinder base and 16 rim capsules taken from MuJoCo's compiled `plate_rim_*` geoms.

**Reset.** `reset(seed, object_xy=, plate_xy=, object_on_container=)` places the objects with `isaac_scene.reset_layout`. The tests show that function reproduces `MuJoCoSimulation.reset` (same seeded default apple jitter, plate default, tabletop bounds, overlap and fit errors) on 16 cases. Wrong resets raise `ContractError` in Isaac too (see the rejections in the parity run).

**Truth.** `task_truth()` returns `MuJoCoSimulation.task_truth`'s dict: object/plate position, velocity, `hand_contact` (apple against any `hand_`/`wrist_` link, from the contact view), and `lifted`/`placed`/`dropped`. `contacts()` returns the PhysX contact pairs. **Both are evaluator-only:**

- `read()` contains no object state;
- `reset()` returns only the timestamp and clock (unlike `MuJoCoSimulation.reset`, which returns truth);
- `set_object_state()` is documented as a scripted-check harness method, not a controller API.

**Read-back.** PhysX holds apple mass 0.0800 kg, diagonal inertia 2.3328e-5 kg m² (MuJoCo's solid-sphere value), and material static/dynamic/restitution 1.0 / 1.0 / 0.0.

| Parameter | MuJoCo v2 | Isaac / PhysX | Matched? |
| --- | --- | --- | --- |
| Apple sphere r, mass, inertia | 0.027 m, 0.08 kg, 2.3328e-5 | same (read back) | yes |
| Table box, plate cylinder, 16 rim capsules | MJCF geometry | same dimensions and poses | yes, except that the plate base is a **convex hull** of the cylinder (Isaac Lab disables PhysX custom cylinder geometry); rims are native capsules |
| Sliding friction | 1 (apple, plate base, table, rims), combined per component by max | static = dynamic = 1, combine mode max, on every scene material and the scene default | nominal; MuJoCo has one coefficient, PhysX two |
| Torsional friction | 0.01 m (condim 6) | torsional patch radius = min radius = 0.01 m on the apple | nominal only; different algorithm, and the per-pair combination of patch radii is assumed max, not verified |
| **Rolling friction** | 0.001 m (condim 6) | **none**: PhysX rigid bodies have no rolling friction | **no**; not substituted, because angular damping would be viscous rather than Coulomb and would also act in free flight |
| Contact softness | solref 0.02 s / 1, solimp 0.9 0.95 0.001 (about 0.2 mm penetration at rest) | rigid contacts, rest offset 0, contact offset auto | no |
| Restitution | none (critically damped) | 0 | nominal |
| Friction cone | pyramidal | PhysX patch friction | no |
| Body damping, sleeping | none | linear/angular damping 0 (PhysX default angular 0.05), sleep threshold 0 | yes |
| Robot geoms | friction (1, 0.005, 0.0001), condim 3, self-collision on | scene default material (friction 1), self-collision off | partly |
| Integrator | implicitfast | PhysX TGS (apple: 16 position / 1 velocity iterations, as authored) | no |
| PhysX-only settings | none | apple `maxDepenetrationVelocity` 1.0 m/s, contact offset auto, CCD and speculative CCD off, `stabilizationThreshold` 1e-5, `cfmScale` 0.025 (read back) | no counterpart |
| Floor | infinite plane at z = 0 | 4 m × 4 m × 2 cm box, top at z = 0 | nominal |
| Robot link collision shapes | MJCF meshes (convex) and primitives | the converter's `convexHull` of the same meshes, plus the primitives | counts match (§2 of ISAAC_MJCF_TRANSPORT.md); hulls not compared vertex by vertex |

## 4. Scripted checks (no learned policy)

`scripts/isaac/scripted_cases.py` defines the cases. Both sides start from their transport's `reset` and log once per 0.05 s interval. Settle time is the first time from which the apple speed stays ≤ 0.001 m/s (the `apple_at_rest_v0` speed bar), at 0.05 s resolution. Results from `isaac-v2-scripted-1`:

| Case | MuJoCo | Isaac | Difference |
| --- | --- | --- | --- |
| Drop 10 cm onto the table | rests at z 0.76683, settles at 0.35 s | rests at z 0.76700, settles at 0.15 s | xy 0.00 mm, z +0.17 mm (MuJoCo's soft-contact penetration), settles 0.20 s earlier |
| Drop 10 cm onto the plate centre | z 0.77883, 0.35 s | z 0.77900, 0.15 s | xy 0.00 mm, z +0.17 mm, 0.20 s earlier |
| Drop 5 cm off the plate centre (onto base and rim) | stays on the plate: ends 3.4 cm from the centre, still creeping at 0.0067 m/s after 3 s | rolls off the plate and across the table toward the robot, and ends wedged between the table edge and `right_hip_roll_link`: at rest from 1.55 s at z 0.661, in contact with both, never with the floor | qualitatively different |
| Roll on the table, 0.2 m/s with matching spin | slows to 0.053 m/s after 3 s (0.343 m travelled) | does not slow; rolls off the far edge onto the floor at about 2.8 s | the missing rolling friction |
| Right middle finger pressed onto the resting apple | finger contact from read 47 (max 5.8 N, plus `middle_0` 7.0 N); the apple is squeezed sideways 6.5 cm | finger contact from read 46 (max 35.9 N; table reaction up to 37.8 N); the apple is pinned and moves 4 mm | the outcome differs (pinned against squeezed out); joint difference up to 0.031 rad |

- **Drops.** Straight drops agree to 0.17 mm in rest height and 0 mm in xy. Isaac settles 0.2 s sooner because its rigid contacts do not oscillate the way MuJoCo's soft contacts do.
- **Roll and rim.** Anything involving rolling or the rim differs qualitatively. This is the gap v2 was created to close: MuJoCo's condim-6 rolling friction keeps the apple on the plate, and PhysX has no equivalent.
- **Press.** The finger press differs in outcome. The sampled peak forces (5.8 N against 35.9 N) belong to different outcomes, a pinned apple against one squeezed out, and are sampled once per interval. They are not a contact-stiffness ratio. The finger's convex hull, rigid versus soft contact, and the friction model are all candidate causes. This run does not separate them.
- **Contact forces.** Contact forces sampled once per interval are not comparable at impact: MuJoCo 7.4 N against Isaac 0.78 N at the first contact read. The timing of the samples differs, so these are not reported as a difference.

**Implication for running `apple-to-plate-v2` in Isaac.** The TASK-070 expert's at-rest success depends on the apple not rolling away. The roll and rim cases predict that, with PhysX as configured, the apple will roll off the plate where MuJoCo keeps it. An Isaac replay of e9 would therefore measure this engine difference, not the expert's behaviour.

## Timing and memory

| Measure | Value |
| --- | --- |
| Isaac `send_joint_targets` per 0.05 s interval (25 substeps) | median 173 ms, p95 175 ms, max 178 ms (n = 180) |
| Isaac `read()` with the path-traced 112 px render | median 20.6 ms, p95 81 ms, max 846 ms (the one early stall, as on 2026-09-28) |
| MuJoCo, same | 2.04 ms per step; 0.41 ms per read |
| App launch / transport build | 83 s / 6.9 s |
| Isaac process GPU memory, peak of ~1 s samples | 5 029 MiB (parity), 2 839 MiB (scripted, no rendering) |
| Device total after the transport build | 11 848 MiB, including GR00T's 6 626 MiB |

## Blockers and caveats

**No hard blocker for items 1–4.** The blocking issue for the next step is physical, not software: **PhysX cannot express v2's rolling friction.** Options, not chosen here:

- **(a)** Run Isaac Lab 3's Newton physics backend. The image's Isaac Lab ships `isaaclab_newton`, and Newton's MuJoCo-Warp solver is expected to follow MuJoCo contact semantics, including condim 6. It is untested here and may not support this articulation, camera or contact API.
- **(b)** Add a scripted rolling-resistance torque on the apple each substep, `−μ_r N ω̂`, from the contact normal force. This is an explicit model, reported as a deviation.
- **(c)** Approximate with angular damping calibrated on the roll case. This is viscous, acts in free flight, and would need a declared calibration.
- **(d)** Accept the difference and report cross-simulator results as a separately labelled engine factor.

Caveats:

- **Renderer.** The image choice rests on one frame and one grid, photometric parity fails its declared bars, and the path tracer costs about 20 ms per 112 px frame. Output frames are identical within a process and across two processes on the same host, driver and image. Determinism across drivers or hosts was not tested. The apple image-geometry PASS in the free-space run rests on 17–42 apple pixels of a static apple, so it tests projection, not object dynamics.
- **Contacts.** Isaac contacts come from tensor contact views against named scene bodies, so robot self-contacts are not observed (self-collision is off anyway). MuJoCo contacts are sampled after each interval; Isaac's at the last step and at any substep.
- **Settle time.** It has 0.05 s resolution.
- **Simulator version.** Isaac Sim is still release candidate 6.0.0-rc.22.

## Next step

1. Decide the rolling-friction route (a–d above). *Update 2026-09-29: option (a) was spiked in [ISAAC_NEWTON_SPIKE.md](ISAAC_NEWTON_SPIKE.md); Newton/MuJoCo-Warp reproduces MuJoCo's roll, rim-drop and press outcomes, and is available as `IsaacTransport(physics="newton")`.* Option (a) is worth a one-day spike, because it would also remove the soft/rigid contact and friction-cone mismatches.
2. With that route, rerun the scripted roll, rim and press checks, then put the embodiment's IK (MuJoCo kinematics provider) over `IsaacTransport`. Replay e9 on development seeds as a separately labelled cross-simulator check.
3. If Isaac frames are to be used as observations, decide whether shading must match (for example an unlit/albedo-plus-shadow path, or a colour transform fitted on training frames only) or whether the distribution shift is accepted and measured.
