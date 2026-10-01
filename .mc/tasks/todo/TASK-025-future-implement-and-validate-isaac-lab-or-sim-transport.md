---
id: TASK-025
aliases:
- TASK-025
title: 'Future: implement and validate Isaac Lab or Sim transport'
slug: future-implement-and-validate-isaac-lab-or-sim-transport
status: in-progress
priority: 4
owner: ''
projects: []
customers: []
tags:
- future_platform
- optional
sprint: ''
depends_on:
- "[[TASK-022]]"
due_date: ''
created: 2026-09-20
updated: 2026-10-02
---

# Future: implement and validate Isaac Lab or Sim transport

## Description

Execute this task only when suitable Isaac compute and assets are available. It is not a prerequisite for local MuJoCo MVP acceptance.

## Acceptance Criteria

- [ ] Pin a supported Isaac environment and corresponding G1/Dex3 assets, then implement the prepared transport interface.
- [ ] Run reset/observation/action/timing parity checks and a separately labeled cross-simulator benchmark.
- [ ] Reuse models, canonical data, CEM/MPC and evaluator interfaces without simulator-specific branches outside adapters.

## Notes

- Milestone: future_platform.
- Execution status and dependencies live in MissionControl frontmatter.
- Record commands, artifacts, and validation evidence here before marking done.
- See `docs/MVP_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/PLATFORMS.md`.

## 2026-09-20 full-task request audit

No supported Isaac compute/runtime is available on the current Mac; runtime parity and cross-simulator results require that external resource. Prepared contracts/runbooks remain available; acceptance is not complete. Continue all locally executable apple-MVP work independently.

## 2026-09-28 Isaac Sim bring-up spike (development, not gated)

The owner approved a development spike on the Linux PC (RTX 5080). It used the local `isaaclab_arena` image (Isaac Sim 6.0.0-rc.22, Isaac Lab 3.0.0, image `sha256:2588b526…`) and NVIDIA's G1 + Dex3 USD. The spike met all five of its criteria: headless GPU container; 43-joint articulation with the root fixed (0.0 m drift); named joint comparison against the MuJoCo model; one onboard-pose RGB frame; step time and VRAM. The 29 leg, waist and arm limits match. 10 of the 14 Dex3 limits in the USD are strict subsets of the MuJoCo ranges, and the joint order differs from MJCF, so joints must be mapped by name. Physics-only stepping took a median 2.73 ms per 0.002 s step with one environment, and the Isaac process's GPU memory peaked at 4 666 MiB (peak of host `nvidia-smi` samples about 1 s apart, a lower bound). Details, commands and caveats (unpinned staging-S3 asset, release-candidate simulator, no drive-model parity) are in `docs/ISAAC_BRINGUP_SPIKE.md`; scripts are in `scripts/isaac/`. This does not meet any acceptance criterion above: nothing is pinned by hash, no transport is implemented and no parity check has run. Status stays `backlog`.

## 2026-09-28 Converted MJCF, model audit and minimal transport (development, not gated)

The pinned `g1_29dof_with_hand.xml` was converted to USD with the Isaac Sim 6.0 MJCF importer (`isaacsim.asset.importer.mjcf-3.2.0`, `mujoco_usd_converter` 0.1.0) in the same `isaaclab_arena` image. The importer embeds a random temp-dir name, so raw file hashes differ per run. The canonical tree hash `cd1fdb27…2fa8` was identical across three conversions. The USD is not committed (`assets/isaac/`, ignored). A field-by-field audit against `MuJoCoSimulation` matched names, limits, axes, joint frames, masses, COMs, inertias, armature, collision-shape counts and forward kinematics at 4 poses (max 9.3e-7 m). All 43 joints mismatch on passive damping (absent), friction model (frictionloss copied into PhysX's load-proportional coefficient) and effort limit (unbounded), and self-collision is off in Isaac. A minimal robot-only `IsaacTransport` (`src/embodied_jepa/isaac_transport.py`, lazy imports, container only, joint manifest v1) compensates for the damping, friction and effort-limit mismatches. Against the MuJoCo transport with identical targets, the maximum differences were: legs < 1e-8 rad; waist 0.0025; arms 0.0091; Dex3 0.152 rad. The Dex3 figure is attributed to finger–table contact. That is inferred, not established: in MuJoCo the middle fingers touch the table on reads 37–77, which also covers part of the arm phase. The worst Dex3 reads have no MuJoCo contact on their link, and Isaac contacts were not measured. Replays were deterministic. It runs at 174 ms per 0.05 s interval (MuJoCo 2.1 ms). Rendered frames are not yet usable (noisy, different lighting). Details and the recommended next steps are in `docs/ISAAC_MJCF_TRANSPORT.md`. This is simulator infrastructure and joint-level parity only, with no manipulation, contact or policy result, so no acceptance criterion is met. `mc` is not installed on this host, so `mc validate` was not run.

## 2026-09-29 Pinned rendering, contact-free joint parity and the v2 apple and plate (development, not gated)

Simulator infrastructure and physics parity only. No learned policy was run; scripted checks in Isaac are not learned or task results. Details, commands and caveats are in `docs/ISAAC_V2_SCENE.md`. All runs are at commit `45e4d55` (clean tree): `outputs/isaac-v2-parity-1-{free,tablecontact}` and `outputs/isaac-v2-scripted-1`. Render probes 1–4 ran on uncommitted development code.

- **Render defect found and fixed.** Isaac Lab 3 renders at most once per physics step. The 2026-09-28 transport's post-reset frames were therefore stale, and its second-render staleness check tested nothing.
- **Rendering pinned.** The RTX path tracer at 64 spp in one frame with the OptiX denoiser, plus MuJoCo-like top and head lights, is recorded in the camera manifest `configs/isaac/onboard_camera_v1.json`. Frames are identical within a process and across two processes on the same host, driver and image. They were not tested across drivers or hosts. The settings were chosen on one reset frame.
- **Image parity.** On the declared metric over held-out reads 30/50/70/90, geometry PASSES (robot IoU ≥ 0.992, table ≥ 0.995, apple ≥ 0.974, plate ≥ 0.975, centroids ≤ 0.15 px). Photometry FAILS (MAD 26–29 against ≤ 15; SSIM 0.72–0.75 against ≥ 0.80).
- **Contact-free joint parity.** The `free_space_v1` trajectory records no robot contact in either simulator. Maximum |Δq| is 0.0020 rad for the arms, 5.2e-4 for Dex3, 3.1e-4 for the waist and < 1e-9 for the legs. The 2026-09-28 trajectory, rerun with Isaac contacts measured, has the same finger–table, finger–apple and finger–plate body pairs in both simulators, on largely but not exactly the same reads. For example, MuJoCo has no table contact on reads 38, 67, 68, 78 and 80, and Isaac misses only read 80. The worst Dex3 read (68) has contact in Isaac only. Together with the free-space result, this strongly supports, but does not confirm, that the earlier Dex3 gap came from contact: the free-space trajectory is a different trajectory, not a controlled ablation.
- **v2 objects in Isaac.** The scene manifest `configs/isaac/apple_to_plate_v2_scene_v1.json` adds the apple, the plate and its rim, with `reset(object_xy, plate_xy)` following `MuJoCoSimulation.reset`'s rules and an evaluator-only `task_truth()`/`contacts()`. PhysX cannot express rolling friction, contact softness or the analytic cylinder. Torsional friction is nominal only; its per-pair combination in PhysX is assumed to be max, not verified.
- **Scripted checks.** Straight drops onto the table or plate centre agree to 0.17 mm in rest height; Isaac settles 0.2 s sooner. The rolling case, the rim case and the finger press differ qualitatively: in Isaac the apple rolls off the table and off the plate, and the pressed apple is pinned rather than squeezed out.

The next step is to decide the rolling-friction route; see the options in the doc. No acceptance criterion is met, and status stays `backlog`. `mc` is not installed on this host, so `mc validate` and `mc index` were not run.

## 2026-09-29 to 2026-10-02 Newton backend, e9 replay, Arena scene and GR00T reference (development, not gated)

Development cross-simulator work on the Linux PC (RTX 5080). None of it is a gated run, a
preregistered comparison or a project-learned result; e9 is a privileged scripted expert, and
GR00T is NVIDIA's policy, not ours. MuJoCo stays the reference simulator, and no acceptance
criterion above is met yet. Status set to `in-progress` because work is active (#123 open).

- **#107 (`75a481e`), Newton backend.** Isaac Lab 3's opt-in Newton (MuJoCo-Warp) backend, with
  the transport building the Newton model itself, reproduces MuJoCo's contact behaviour on all
  five scripted cases (roll within 0.24 mm, rim drop stays, press squeezes out); contact-free joint
  parity 1.4e-5 rad. PhysX stays the default. `docs/ISAAC_NEWTON_SPIKE.md`.
- **#110 (`57f3ae4`), e9 replay in Isaac/Newton.** On 16 development seeds × 2 plate errors, e9's
  actions match MuJoCo (grasp on the same step, the apple lands on the plate every time), but the
  at-rest outcome does not match seed by seed: 24–26/32 per Isaac cell against 31/32 in MuJoCo, a
  gap about as large as Isaac's own run-to-run variation. `docs/ISAAC_E9_REPLAY.md`.
- **#112 (`8e96d5e`), start-up hang fix.** `PXR_WORK_THREAD_LIMIT=1` for Newton start-up: 0 of 24
  starts hung. Upstream newton#4390 is still open, so the workaround stays.
- **#119 (`29eac7b`), Arena scene.** The GR00T tutorial's `galileo_g1_static_pick_and_place` scene
  runs headless in the local Arena container exactly as the tutorial configures it; scene facts
  (50-D absolute joint actions, floating WBC-balanced base, one 640 × 480 head camera, a loose
  contact-and-speed success rule) are in `docs/ARENA.md` §1–§6.
- **#120 (`0d9f16c`), Arena cross-sim checks.** The arm kinematics agree (wrist within 0.8 mm);
  the Dex3 fingertips differ by up to 3.1 cm. e9 left the apple at rest on 0/16 development seeds
  in Arena (16/16 in MuJoCo): it never grasps, because the arms stall without a gravity offset
  and, with one, the floating base steps back during the close. `docs/ARENA.md` §7.
- **#121 (`a1b67f4`), GR00T reference baseline.** GR00T N1.7, client-only against the owner's
  server: 16/30 and 10/30 under Arena's rule in two processes, 0/30 under the strict
  `apple_at_rest_arena_v0`, whose check used PhysX's reported velocity (stale for a resting apple,
  inferred); a post-hoc position-based diagnostic gives 6/30. An external reference only.
  `docs/ARENA.md` §8.
- **#122 (`c16fb04`), white plate.** Not Isaac work, but the same plate question: an opt-in white
  plate for the MuJoCo v2 scene does not change TASK-075's readouts.
  `docs/experiments/apple_white_plate_dev.md`.
- **#123 (open, `feat/arena-e9-adapted`), e9-arena.** An Arena-adapted e9 and a shelf-press probe,
  with its declaration (`docs/ARENA.md` §9.0–9.2) committed before any Arena run of e9-arena. Tuning
  run 1: T2 (`shelf_servo`) reached 4/8 `apple_at_rest_v0` on seeds 50200–50207. T3, T4 and the
  fresh-seed evaluation wait for the GPU. Not reviewed or merged yet.

`mc validate` passed with mc 0.1.14 (installed 2026-10-02 from the v0.1.14 release asset, sha256
checked).
