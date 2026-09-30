# The TASK-070 expert e9 in Isaac Lab over Newton (MuJoCo-Warp): a cross-simulator development check (2026-09-30)

This is a development step on the Linux PC (RTX 5080), following [ISAAC_NEWTON_SPIKE.md](ISAAC_NEWTON_SPIKE.md) §7, next step 1.

**Scope.**
- It is not a gated run and not a preregistered comparison. No bar was declared before the runs.
- e9 is TASK-070's privileged scripted expert ([apple_to_plate_v2_expert.md](experiments/apple_to_plate_v2_expert.md)). It reads simulator truth by design. Nothing here is a learned, policy or task result, and a scripted expert's success in Isaac is not a learned result.
- **Learned Apple→Plate is still 0 successes.**
- Only development seeds were simulated: TASK-070's development seeds 50200–50215, which TASK-070 had already used in MuJoCo, with its development plate-error directions (`default_rng(6850)`). No TASK-070 gated seed (50600–50631) and no TASK-073 seed (53000–53999) was simulated; `isaac_e9.check_seeds` refuses them.
- The v1 task, the v2 task, the TASK-070 gate and its result are unchanged.

**The question.** e9 leaves the apple at rest on the plate on 32/32 gated seeds in MuJoCo on `apple-to-plate-v2`. Does it behave the same when Isaac Lab steps the physics with the opt-in Newton (MuJoCo-Warp) backend?

**Answer, with the caveats of §6.**
- **The expert's actions match.** In all 128 Isaac attempts, the grasp starts on the same control step as in MuJoCo. The lift, release and plate landing are within a few steps of MuJoCo. The apple is carried, released and lands on the plate every time.
- **The at-rest outcome does not match seed by seed.**
  - At rest in MuJoCo: 31/32.
  - At rest in Isaac: 24–26/32 in each of four Isaac cells (closed and open loop, two runs each).
  - Every Isaac failure is the same kind of failure: the apple rests supported and still on the plate, with no hand contact, but its centre is 4.0–4.6 cm from the plate centre, just outside the 4 cm radius.
- **The mismatch is about as large as Isaac's own run-to-run variation.**
  - Two Isaac runs of the same attempts disagree on the at-rest outcome in 10 of 32 attempts in each mode. Their final apple positions differ by a median of 1.9–2.4 cm.
  - This is about the same as the Isaac–MuJoCo gap: a median of 1.4–2.3 cm per cell.
  - MuJoCo itself, with 1 mrad of noise on the joint targets, changes 9 of 64 outcomes. Its final positions then move by a median of 1.5 cm.
- **Why this happens (inferred).** The final at-rest verdict of e9 is sensitive: in MuJoCo the apple ends 2.27–4.08 cm from the plate centre (q10–q90 about 2.4–3.7 cm), near the 4 cm bar. Isaac's run-to-run spread is enough to push a quarter of the attempts over the bar.
- **What it does not show.** Whether Newton's at-rest rate is really lower than MuJoCo's is not settled by these runs. Isaac is 24–26/32 per cell, against 31/32 for MuJoCo and 26–31/32 per cell for noisy MuJoCo. The attempts are the same 32 in every cell, so the cells are not independent samples.

## 1. Method

Code:
- `src/embodied_jepa/isaac_e9.py` (NumPy only at import);
- `scripts/isaac/e9_replay.py` (host side);
- `scripts/isaac/e9_server_isaac.py` (container side);
- `scripts/isaac/startup_probe.sh`;
- tests in `tests/test_isaac_e9.py` and `tests/test_isaac_transport.py`.

- **MuJoCo reference.** `e9_replay.py mujoco` runs TASK-070's own harness, `resting_expert.run_attempt`, unchanged, on `v2.make_v2_robot()`. The harness is: wide-jitter reset, the look, e9 (`RestingPlaceExpert(release_pitch_rad=0.45, release_dx=0.015)`) and the 60-step settle, followed by `apple_at_rest_v0`.
  - Attempts: 16 seeds × plate error 0 and 1.0 cm = 32.
  - A `StepRecorder` logs every joint-target command the embodiment sends: 793 per attempt, 8 of them the look. It also logs the joint state after each command and any robot–robot contact.
- **Isaac, closed loop.** The same harness runs on `isaac_e9.MirrorSimulation`, which is a `MuJoCoSimulation` whose physics is remote.
  - Each joint-target command goes over a Unix socket to an `IsaacTransport(physics="newton", objects=True, render=False)` in the container.
  - The reply carries Isaac's state: joints, apple pose and velocity, plate, Isaac's own contact list and its apple–hand flag. The mirror writes this state into its local `MjData`.
  - So everything that reads `sim.data` reads Isaac's state: the embodiment's IK, `ee_pose`, e9's phase logic, the scorer and the at-rest check.
  - `task_truth()` uses Isaac's apple–hand contact. The mirror's own geometric flag agreed with it on ≥ 99.75 % of steps in every attempt.
  - The privileged expert reads exactly what it reads in MuJoCo, now from Isaac.
- **Isaac, open loop.** MuJoCo's recorded joint-target arrays are replayed unchanged after the same reset, and scored the same way (`isaac_e9.replay_open_loop`).
- **mj_step layout.** After `mj_step`, MuJoCo's `MjData` has the new `qpos` but body and site poses and contacts of the state *before* the last physics step, and TASK-070's harness reads them that way.
  - `IsaacTransport` now records the state at the start of each interval's last physics step (`last_substep_start`). The mirror computes kinematics and contacts there and then writes the new `qpos`/`qvel` without a forward pass.
  - An earlier version ran a plain `mj_forward` instead. Its host plumbing check (below) differed from MuJoCo by up to 3e-3 rad.
- **Initial pose.** `G1Embodiment.reset` opens the Dex3 hands by writing `qpos` after `MuJoCoSimulation.reset`. `IsaacTransport.reset(joint_positions=...)` (new, checked by `isaac_transport.initial_joint_pose`) starts Isaac from that same pose. The mirror sends it lazily on the first read.
- **Plumbing check (sham).** `e9_replay.py isaac --sham` runs the same client against `isaac_e9.MuJoCoEndpoint`, a host MuJoCo v2 scene behind the same protocol.
  - On all 32 attempts the mirror reproduced the plain MuJoCo run exactly, closed and open loop: identical joint trajectories (difference 0), identical apple trajectories and identical verdicts (`e9-sham-1`).
  - `test_mirror_over_mujoco_reproduces_a_plain_mujoco_e9_attempt` pins this for one attempt.
  - MuJoCo replaying its own commands open loop is also exact on all 32 (joint difference 0).
- **Calibrations** (host MuJoCo only; `e9_replay.py sensitivity`, open loop):
  - the apple's reset xy moved by 0.1 mm in 4 directions (128 replays);
  - seeded Gaussian noise of 1 mrad and 5 mrad on every non-leg joint target after the look (64 replays each).
- **Isaac repeatability.** The same 32 closed- and open-loop attempts were run again in a second container process (`e9_replay.py repeat`).

### Runs

All runs are under `outputs/` in the worktree (git-ignored). The image is `isaaclab_arena:latest` `sha256:2588b52605d7…`. The other inputs are those of [ISAAC_NEWTON_SPIKE.md](ISAAC_NEWTON_SPIKE.md):
- USD canonical tree `cd1fdb27…2fa8`;
- scene manifest `e698b44d…`;
- joint manifest v1 (`manifest_sha256` `c6c3ccfc…`);
- the Newton solver settings.

Host MuJoCo is 3.13.0.

| Run | Revision | What | report sha256 |
| --- | --- | --- | --- |
| `e9-mujoco-ref-1` | `6a7cc26`, clean | MuJoCo reference, 32 attempts, and self-replay | `d406de57` |
| `e9-sham-1` | `6a7cc26`, clean | mirror against a host MuJoCo endpoint | `b108a7ab` |
| `isaac-e9-server-1` + `isaac-e9-newton-1` | `6a7cc26`, clean | Isaac/Newton run 1, closed and open loop | `ccbd94a3` |
| `isaac-e9-server-2` | `6a7cc26`, clean | **hung at start-up** (§4). Stopped after 16 min; no attempts | — |
| `isaac-e9-server-3` + `isaac-e9-newton-3` | `56f09cb` (see note) | Isaac/Newton run 3, the repeat | `eeec5fa5` |
| `e9-mujoco-sens-0.1mm-1` | `18b87e4`, clean | MuJoCo, apple reset +0.1 mm | `2467e8c1` |
| `e9-mujoco-sens-noise0.001-1`, `…0.005-1` | `dfa8e46`, clean | MuJoCo, 1 and 5 mrad target noise | `0e6188d5`, `c99ecbf3` |
| `isaac-e9-repeat-1-3` | `cbb12cb` | run 1 against run 3 | `309aa656` |
| `isaac-e9-startup-{1..8}` | `cbb12cb`, clean | start-up probes (§4) | `isaac-e9-startup-summary.tsv` |

Note on run 3:
- Its client and server started at `56f09cb` with a clean tree.
- Its report records `tracked_tree_dirty: true` because the report took its revision at the end of the run, after the `repeat` subcommand had been edited in. That edit was not loaded by the running process.
- `6a7cc26`…`56f09cb` differ only in host-side subcommands that the runs do not use: `sensitivity` and `startup_probe.sh`.
- The reports now take the revision at the start (`cbb12cb`).

Reproduce (host, container, host):

```sh
uv run --no-sync python scripts/isaac/e9_replay.py mujoco --output outputs/<ref>
U=/oej/usd/g1_29dof_with_hand-ref/run/usd/g1_29dof_with_hand/g1_29dof_with_hand.usda
sg docker -c "scripts/isaac/run_isaac.sh e9_server_isaac.py outputs/<srv> --usd $U \
  --manifest /oej/configs/isaac/g1_dex3_joint_manifest_v1.json \
  --scene /oej/configs/isaac/apple_to_plate_v2_scene_v1.json --physics newton"
MUJOCO_GL=egl uv run --no-sync python scripts/isaac/e9_replay.py isaac --reference outputs/<ref> \
  --socket outputs/<srv>/run/e9.sock --output outputs/<isaac>
uv run --no-sync python scripts/isaac/e9_replay.py compare --reference outputs/<ref> \
  --isaac outputs/<isaac> --output outputs/<isaac>/compare
```

## 2. Results

### At rest (`apple_at_rest_v0`) and final distance

"Final" is the distance of the apple centre from the plate centre at the end, in cm, as q10 / q50 / q90.

| Cell | at rest, plate exact (of 16) | at rest, 1.0 cm (of 16) | final, exact | final, 1.0 cm |
| --- | --- | --- | --- | --- |
| MuJoCo | 16 | 15 | 2.38 / 3.33 / 3.61 | 2.71 / 3.13 / 3.67 |
| Isaac closed loop, run 1 | 14 | 12 | 2.92 / 3.48 / 3.94 | 2.76 / 3.30 / 4.20 |
| Isaac closed loop, run 3 | 12 | 12 | 2.79 / 3.65 / 4.17 | 2.63 / 3.61 / 4.21 |
| Isaac open loop, run 1 | 12 | 13 | 3.13 / 3.55 / 4.27 | 3.04 / 3.56 / 4.22 |
| Isaac open loop, run 3 | 13 | 12 | 2.62 / 3.08 / 4.18 | 2.97 / 3.56 / 4.26 |
| MuJoCo + 1 mrad target noise (2 draws per attempt, of 32) | 31 | 26 | 2.55 / 3.29 / 3.77 (both levels) | 2.74 / 3.43 / 4.34 |

- **Every Isaac attempt completed**: 128 of 128, with no guard stop, rejection or error. The latched scorer (`AppleToPlateTask`, reported beside the at-rest check) succeeded in all 64 closed-loop Isaac attempts, as in all 32 MuJoCo attempts. The open-loop replay does not run the latched scorer.
- **Every at-rest failure fails on "inside" only**, in Isaac and in MuJoCo's one failure (1.0 cm, 50206: 4.08 cm). The apple ends supported on the plate, still (≤ 0.001 m/s) and out of hand contact, 4.0–4.6 cm from the centre. The apple never left the plate.
- **The MuJoCo reference is consistent with TASK-070.** It gives 16/16 and 15/16 on these seeds. TASK-070's development block 50200–50231 gave 32/32 and 30/32 on the Mac. The Mac per-attempt records are not on this machine, so a seed-by-seed check was not possible.

### Per attempt

Each cell gives at rest (yes / no), the final distance in cm, and in parentheses the final apple xy distance from MuJoCo's final apple position, in cm.

| plate error, seed | MuJoCo | closed, run 1 | closed, run 3 | open, run 1 | open, run 3 |
|---|---|---|---|---|---|
| 0.0 cm, 50200 | yes 3.34 | yes 3.89 (6.1) | yes 2.60 (2.6) | yes 3.65 (2.4) | yes 2.90 (3.1) |
| 0.0 cm, 50201 | yes 3.42 | yes 3.14 (1.6) | yes 3.65 (1.3) | yes 3.12 (1.2) | yes 3.69 (3.7) |
| 0.0 cm, 50202 | yes 2.39 | yes 3.45 (2.4) | yes 3.81 (1.5) | yes 3.72 (1.4) | yes 3.57 (1.7) |
| 0.0 cm, 50203 | yes 2.27 | yes 3.84 (1.8) | yes 3.76 (1.5) | yes 3.59 (1.4) | yes 3.11 (1.7) |
| 0.0 cm, 50204 | yes 3.69 | yes 2.91 (2.7) | yes 2.95 (3.3) | **no** 4.44 (7.5) | yes 3.74 (3.8) |
| 0.0 cm, 50205 | yes 3.53 | yes 3.09 (0.4) | yes 3.91 (3.3) | yes 3.15 (1.8) | yes 3.04 (1.0) |
| 0.0 cm, 50206 | yes 3.34 | **no** 4.08 (2.4) | yes 2.65 (1.6) | yes 1.88 (2.9) | yes 2.83 (1.3) |
| 0.0 cm, 50207 | yes 3.06 | yes 2.54 (0.6) | yes 3.41 (2.1) | yes 3.24 (0.9) | **no** 4.41 (3.9) |
| 0.0 cm, 50208 | yes 3.20 | yes 3.34 (0.7) | yes 2.94 (3.6) | yes 3.50 (3.4) | **no** 4.35 (7.5) |
| 0.0 cm, 50209 | yes 3.17 | **no** 3.99 (3.2) | **no** 4.08 (5.4) | yes 3.45 (0.4) | yes 2.39 (1.5) |
| 0.0 cm, 50210 | yes 3.27 | yes 2.92 (1.2) | **no** 4.24 (2.5) | **no** 4.08 (2.2) | yes 2.53 (2.4) |
| 0.0 cm, 50211 | yes 3.91 | yes 3.25 (2.0) | yes 3.22 (0.7) | yes 3.48 (1.8) | yes 2.93 (1.0) |
| 0.0 cm, 50212 | yes 3.51 | yes 3.76 (2.1) | yes 3.66 (0.6) | **no** 4.58 (2.2) | yes 2.73 (1.0) |
| 0.0 cm, 50213 | yes 3.32 | yes 3.84 (0.9) | yes 3.32 (0.3) | yes 3.88 (2.1) | yes 2.70 (1.4) |
| 0.0 cm, 50214 | yes 3.46 | yes 3.64 (2.4) | **no** 4.40 (7.7) | **no** 4.11 (1.4) | **no** 4.00 (2.4) |
| 0.0 cm, 50215 | yes 2.37 | yes 3.52 (2.2) | **no** 4.10 (2.2) | yes 3.43 (2.0) | yes 3.57 (3.5) |
| 1.0 cm, 50200 | yes 3.42 | **no** 4.20 (1.4) | **no** 4.05 (6.0) | **no** 4.60 (5.1) | yes 3.57 (1.9) |
| 1.0 cm, 50201 | yes 3.36 | yes 3.45 (1.4) | **no** 4.08 (2.2) | yes 3.22 (0.3) | yes 3.34 (2.8) |
| 1.0 cm, 50202 | yes 2.83 | **no** 4.22 (1.8) | yes 2.91 (0.2) | yes 3.17 (0.7) | yes 3.35 (1.0) |
| 1.0 cm, 50203 | yes 2.90 | yes 3.61 (0.7) | **no** 4.35 (1.4) | yes 3.09 (1.7) | yes 2.96 (2.5) |
| 1.0 cm, 50204 | yes 3.24 | yes 3.95 (3.0) | **no** 4.44 (7.6) | yes 3.09 (0.4) | yes 3.54 (2.4) |
| 1.0 cm, 50205 | yes 2.94 | yes 2.86 (1.1) | yes 2.66 (1.0) | yes 3.67 (4.8) | **no** 4.25 (4.1) |
| 1.0 cm, 50206 | **no** 4.08 | yes 2.14 (2.9) | yes 3.40 (1.8) | yes 3.71 (3.3) | yes 3.08 (1.7) |
| 1.0 cm, 50207 | yes 3.18 | yes 2.65 (0.6) | yes 2.83 (2.2) | **no** 4.37 (1.3) | **no** 4.12 (1.9) |
| 1.0 cm, 50208 | yes 2.37 | yes 3.93 (3.5) | yes 3.28 (1.8) | yes 3.44 (2.1) | **no** 4.53 (4.5) |
| 1.0 cm, 50209 | yes 3.31 | yes 3.04 (0.7) | yes 3.56 (3.1) | yes 3.92 (6.6) | yes 3.70 (2.0) |
| 1.0 cm, 50210 | yes 3.05 | yes 3.03 (1.5) | yes 3.65 (1.1) | yes 2.94 (0.6) | yes 3.34 (2.9) |
| 1.0 cm, 50211 | yes 3.49 | yes 3.09 (1.8) | yes 3.86 (1.8) | yes 3.89 (2.6) | yes 3.85 (2.4) |
| 1.0 cm, 50212 | yes 3.09 | yes 2.98 (1.8) | yes 2.20 (1.8) | yes 3.79 (2.4) | yes 2.54 (0.7) |
| 1.0 cm, 50213 | yes 3.85 | **no** 4.13 (0.7) | yes 3.67 (0.2) | yes 2.99 (1.6) | yes 3.77 (3.3) |
| 1.0 cm, 50214 | yes 2.60 | yes 3.14 (0.7) | yes 2.60 (1.8) | yes 3.13 (1.0) | **no** 4.27 (2.2) |
| 1.0 cm, 50215 | yes 2.98 | **no** 4.20 (4.3) | yes 3.91 (1.1) | **no** 4.06 (1.5) | yes 2.98 (0.2) |

- **The failures are mostly not tied to seeds.** Of the 31 attempts at rest in MuJoCo, 20 fail in at least one of the four Isaac cells. Four attempt-and-mode pairs fail in both Isaac runs:
  - 0.0 cm, 50209, closed loop;
  - 0.0 cm, 50214, open loop;
  - 1.0 cm, 50200, closed loop;
  - 1.0 cm, 50207, open loop.

  Two attempts (0.0 cm, 50214 and 1.0 cm, 50200) fail in three of the four cells.
- **The one MuJoCo failure** (1.0 cm, 50206) is at rest in all four Isaac cells.

### Grasp, lift and release timing

The steps are control steps (0.05 s) after the look. The events come from `isaac_e9.events`:
- **grasp**: first apple–hand contact;
- **lift**: apple 1 cm above its start height;
- **lifted**: apple above 0.82 m;
- **release**: first step after the last hand contact following the opening;
- **landing**: first plate contact after the release;
- **settle**: the apple's speed stays ≤ 0.001 m/s from here on.

The table gives Isaac − MuJoCo, per attempt, as min / median / max over the 32 attempts.

| Event | MuJoCo (median step) | closed, run 1 | closed, run 3 | open, run 1 | open, run 3 |
| --- | --- | --- | --- | --- | --- |
| grasp | 221 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| lift | 245 | −4 / −1 / +1 | −4 / −1 / +1 | −2 / −1 / +1 | −3 / −1 / +1 |
| lifted | 266 | +1 / +3 / +5 | +1 / +3 / +4 | −1 / −0.5 / 0 | −1 / 0 / 0 |
| release | 605 | −2 / 0 / +1 | −1 / 0 / +1 | −2 / 0 / +1 | −1 / 0 / +1 |
| landing | 609 | −1 / 0 / 0 | −1 / 0 / 0 | −1 / 0 / 0 | −1 / 0 / 0 |
| settle | 719 | −47 / −12.5 / +34 | −47 / −6.5 / +22 | −54 / −10.5 / +31 | −42 / −6.5 / +24 |

- **Hand contact.** Isaac's apple–hand contact was taken from its own contact list; MuJoCo's from its contact list.
- **The expert's actions match to within a few steps; the apple's roll after landing does not.** The roll after landing is where the attempts separate.
- **Closed-loop lifted step.** In the closed loop the apple reaches 0.82 m about 3 steps later than in MuJoCo. In open-loop replay it does not. The closed-loop IK sees Isaac's slightly different arm state and commands a slightly different lift (*inferred*).

### Joint trajectory divergence (|q_Isaac − q_MuJoCo|, rad, over all 793 commands)

| Group | closed loop: max / median of per-attempt max | open loop: max / median of per-attempt max |
| --- | --- | --- |
| Legs (held) | 6e-7 / 3e-7 | 4e-7 / 3e-7 |
| Waist | 0.009 / 0.007 | 0.022 / 0.015 |
| Arms | 0.118 / 0.065–0.070 | 0.017 / 0.013–0.014 |
| Dex3 | 0.43 / 0.29–0.30 | 0.43 / 0.28–0.30 |

The ranges combine runs 1 and 3.

- **Dex3.** The worst joint is `right_hand_thumb_1_joint` in every attempt.
  - It passes 0.01 rad at command 150–152, while the hand descends with no apple contact. It passes 0.1 rad at commands 253–257, while the hand closes around the apple.
  - In one open-loop attempt (0.0 cm, 50200) the thumb peaks at 0.155 rad apart at command 260, while it is being driven by apple contact. It is 0.018 rad apart while holding, and 0.000 rad after the hand has opened.
  - The left hand, which never touches anything, agrees to 1.7e-5 rad.
- **Arms.** In closed loop the arm differences (≤ 0.12 rad) are the IK responding to the different hand and apple state. In open loop the arms follow the same targets and differ by ≤ 0.017 rad, compared with 3.7e-6 rad in the contact-free free-space parity of the spike.
- **So the divergence is contact-driven,** as the spike's table-contact trajectory suggested for PhysX.

### Calibration: how sensitive is the final outcome?

| Comparison | at-rest outcome changed | final apple xy gap, cm (q50 / q90 / max) | joint difference max, rad |
| --- | --- | --- | --- |
| Isaac run 1 vs run 3, closed loop | 10 of 32 | 2.38 / 5.18 / 8.12 | 0.42 |
| Isaac run 1 vs run 3, open loop | 10 of 32 | 1.86 / 6.38 / 8.11 | 0.40 |
| Isaac vs MuJoCo, closed loop (runs 1 and 3; per-cell medians) | 7 and 9 of 32 | 1.43–2.14 (per-cell medians) | 0.43 |
| Isaac vs MuJoCo, open loop (runs 1 and 3; per-cell medians) | 8 and 8 of 32 | 1.63–2.30 (per-cell medians) | 0.43 |
| MuJoCo, apple reset +0.1 mm | 0 of 128 | 4e-5 / 1.5e-4 / 2.6e-4 | — |
| MuJoCo, 1 mrad target noise | 9 of 64 | 1.45 / 2.81 / 4.16 | 0.069 |
| MuJoCo, 5 mrad target noise | 12 of 64 | 1.55 / 3.19 / 5.12 | 0.52 |

- **Isaac/Newton is not repeatable once there is contact.** Two runs of the same open-loop commands agree to 1e-5 rad for the first ~220 commands in most attempts. They separate at the grasp (median first step over 1e-3 rad: 229; over 1e-2 rad: 244). The final apple positions then differ by centimetres.
- **Consistent with the spike.** The spike found GPU MuJoCo-Warp repeatable to ~3e-6 rad without contact. Contact amplifies that.
- **Isaac vs MuJoCo is about as large as Isaac vs Isaac,** and about as large as MuJoCo with 1 mrad of target noise.
- **A small reset difference does not matter.** A 0.1 mm apple reset difference is erased by the grasp.

## 3. What this means

- **For using Newton for e9.**
  - Newton reproduces e9's manipulation sequence: reach, grasp, carry, release and landing, with the same timing to a few steps.
  - It does not reproduce e9's per-seed at-rest verdicts, and it cannot: it does not reproduce its own verdicts across two processes.
  - Any Isaac at-rest figure for e9 therefore has to be read as a rate over attempts and repeats, not as a per-seed replica of MuJoCo.
  - These runs cannot say whether Newton's rate is lower. Every Isaac cell is at 24–26/32, against MuJoCo's 31/32 and noisy MuJoCo's 26–31/32 per cell. More seeds and repeats would be needed.
- **For the benchmark itself (*inferred*).** e9's at-rest success depends on the apple ending inside 4 cm after rolling toward the rim; it typically ends 2.4–3.7 cm out. Small physics differences move a quarter of attempts across the bar. This is a property of the expert and the task, and it would affect any simulator or hardware transfer. It is recorded here and not acted on.
- **Not shown.** This does not show whether Isaac is a suitable data source for learned policies. Images were not rendered, and the photometric mismatch of [ISAAC_V2_SCENE.md](ISAAC_V2_SCENE.md) §1 is unchanged.

## 4. The start-up hang

**Recurrence.** In this task, 1 of the 3 full server starts hung (`isaac-e9-server-2`), and 1 of the 8 start-up probes hung (probe 4).

**Where it hangs.**
- `e9_server_isaac.py` dumps all Python stacks every 120 s during start-up.
- All 8 dumps of `server-2`, over 16 minutes, show the main thread at the same line:

```
newton/_src/utils/import_usd.py, line 317, in parse_usd
    ret_dict = UsdPhysics.LoadUsdPhysicsFromRange(stage, [root_path], excludePaths=...)
newton/_src/sim/builder.py, line 2403, in add_usd
embodied_jepa/isaac_transport.py, line 657, in _install_newton_builder
```

**What the process was doing.**
- The main thread (the process's own PID) used a full core in user space, with no wait channel, and its RSS stayed at 2.33 GB.
- No other thread was busy (`/proc/<pid>/task/*/stat`).
- This fits a livelock or infinite loop inside the C++ `UsdPhysics.LoadUsdPhysicsFromRange` parse of the live Kit stage. It does not fit a lock wait or a crash.
- A native stack could not be taken: `ptrace_scope` is 1 and the process is not a child of the shell.
- The spike's hung run `scripted-2` also stopped logging right after Warp initialised, before `Finalize builder`, which is the same place.

**Not the kernel compilation.**
- A healthy start spends about 83 s compiling Warp kernels and capturing the CUDA graph (`CUDA graph took: 83 s`), because the per-run `home_cache` starts empty.
- Its 120 s dump shows `warp/_src/build.py compile_lto_dot` (`isaac-e9-server-3`). That dump is not a hang: `server-3` was listening 130 s after start, as `server-1` was.

**Cause: unknown.**
- The same USD, code and image hang only sometimes.
- One candidate is a race between the parse and Kit's background threads touching the same stage (*inferred, untested*).

Start-up probes (`startup_probe.sh`, `--startup_only`, limit 420 s, one container at a time):

| Probe | Outcome | Seconds to "transport built" |
| --- | --- | --- |
| 1 | built | 128.1 |
| 2 | built | 132.0 |
| 3 | built | 151.6 |
| 4 | **hung**: stopped at 420 s; all 3 stack dumps at `import_usd.py:317` | — |
| 5 | built | 126.6 |
| 6 | built | 126.0 |
| 7 | built | 124.9 |
| 8 | built | 125.5 |

- **In this task:** 2 hangs in 11 starts (3 servers, 8 probes). Both hangs were at the same line.
- **With the spike:** 4 in 21 Newton starts, about 1 in 5.
- **Where:** every hang recorded with a stack dump is at the same line. The spike's two hangs were at the same log position.
- **After a successful build** there were no hangs: the 3 probe steps took 0.2–0.3 s, and the two full servers ran 50 752 steps each.

**Mitigations** (not implemented; owner or next task to choose):
- **(a) A start-up watchdog with retry.** It is safe because the hang happens before any episode. The probe script already detects the hang (no "transport built" within the limit) and stops only its own container.
- **(b) Parse a detached copy.** Run `add_usd` on a detached, flattened copy of the stage (`stage.Flatten()` into an anonymous in-memory stage) instead of the live Kit stage. This tests the race hypothesis, and it needs about 20+ probes per arm to be informative at this hang rate.
- **(c) Report upstream** to Newton / Isaac Lab, with the stack.

## 5. Robot self-collision: evidence and options (not decided)

**Current state.**
- The converted USD authors `newton:selfCollisionEnabled = 0` and `physxArticulation:enabledSelfCollisions = 0` (`payloads/Physics/*.usda`). This comes from `convert_mjcf_to_usd.py` `allow_self_collision: False`, the importer default.
- Newton's `add_usd` resolves self-collision from these attributes, so both Isaac backends run without self-collision.
- The host MJCF has self-collision on, minus MuJoCo's default parent–child exclusion.

**Evidence from this task.**
- **MuJoCo, e9.** Across all 32 e9 attempts (25 376 control steps), MuJoCo recorded **0 robot–robot contacts** (`StepRecorder`, from the contact list after every command). The grasp, including the thumb closing against the apple, never brings two robot geoms into contact.
- **Isaac state under MuJoCo's rules.** Replaying Isaac's joint states through MuJoCo's collision detection (the mirror, with self-collision on) found **0 robot–robot contacts** in all 128 Isaac attempts. Isaac's missing self-collision therefore never allowed an interpenetration that MuJoCo would have resolved.
- **Isaac's own contact list** reported no self-contacts either, as expected with self-collision off.
- **The spike's free-space trajectory** had no MuJoCo self-contacts ([ISAAC_NEWTON_SPIKE.md](ISAAC_NEWTON_SPIKE.md) §4).

**Options.**
- **(A) Keep it off.** No change. It is harmless for every trajectory measured so far (e9, the free-space parity trajectory), by the evidence above. The e9 comparison tooling now flags any MuJoCo self-contact per attempt, so a future trajectory that needs self-collision would show up.
- **(B) Turn it on in Newton only.** The transport would override the resolved attribute when it builds the Newton model, with MuJoCo's parent–child filter and the MJCF's contype/conaffinity. This matches the host, but it needs a new parameter audit: the exclude list, and Newton's collision-group colouring against MuJoCo's bitmasks. PhysX would stay off.
- **(C) Reconvert the USD with `allow_self_collision: True`.** This turns it on for both backends. It changes the USD hash and every recorded Isaac provenance, and PhysX's filtering then also needs an audit.

**Recommendation: (A) now.** Revisit with (B) if a trajectory of interest shows MuJoCo self-contacts: for example a two-handed task, or a learned policy's rollouts, which are less tidy than a scripted expert. The recommendation rests on 32 scripted e9 attempts and one free-space trajectory. It does not cover other motions.

## 6. Caveats

- **Development only.** Nothing was declared in advance, and the counts are descriptive. There are 16 seeds × 2 plate levels, and the same 32 attempts are used in every cell, so the cells are not independent. No interval or test is claimed.
- **Two Isaac runs.** Run-to-run variation in Isaac is large (§2), so two runs give only a rough picture of Isaac's own outcome distribution.
- **The mirror is a harness choice.** Closed loop in Isaac means the MuJoCo-side IK and expert logic read Isaac's state through a MuJoCo `MjData`.
  - The kinematics are MuJoCo's (the host MJCF). Newton's compiled model matches it on joints and bodies ([ISAAC_NEWTON_SPIKE.md](ISAAC_NEWTON_SPIKE.md) §4), but geom hulls were not compared vertex by vertex.
  - The contact counts in the events table (plate base or rim) are the mirror's geometric contacts on Isaac's state. Apple–hand contact is Isaac's own.
- **Float32 transfer.** Isaac's joint and apple state is float32 on the GPU, and the MuJoCo reference is float64 on the CPU.
- **Open loop is not closed loop.** The open-loop replay feeds MuJoCo's commands to Isaac, so it tests physics under identical commands. It is not how e9 would run in Isaac.
- **No rendering.** The server runs with `render=False`, and the expert does not read pixels. Photometry is unchanged and still fails its bar.
- **Versions.** Newton's MuJoCo-Warp is 3.5.0.2 / MuJoCo 3.5.0 in the image; the host is 3.13.0. The known unmatched parameters of [ISAAC_NEWTON_SPIKE.md](ISAAC_NEWTON_SPIKE.md) §5 apply, including solver tolerance 1e-6 against 1e-8 and float32.
- **Timing and memory.**
  - Isaac `step` per 0.05 s interval: median 26.7 ms, p95 35.7 ms (n = 50 752, run 1).
  - A full e9 attempt: about 22 s in Isaac, against 3.5 s in MuJoCo.
  - Start-up is about 130 s, including about 83 s of Warp kernel compilation into an empty per-run cache.
  - Peak device memory sampled during run 1 was 8 511 MiB, including GR00T's 6 626 MiB.

## 7. Owner decisions needed

1. **Self-collision.** Accept (A), keep it off with the per-attempt MuJoCo self-contact flag, or ask for (B) or (C) (§5).
2. **Start-up hang.** Choose a mitigation (§4): (a) a watchdog with retry, (b) a detached-stage parse experiment, (c) an upstream report, or a combination. Until then, unattended Newton runs need a start-up timeout.
3. **How to use Isaac for at-rest outcomes.** If Isaac/Newton is to report e9-style at-rest outcomes, do they need repeats per attempt (Isaac is not repeatable under contact) and a larger seed set? Or should the per-seed Isaac–MuJoCo comparison be limited to the manipulation events, which do match? This affects any later Isaac benchmark design, not TASK-070 or TASK-073.
4. **Margin of the at-rest verdict** (*inferred*). Whether the at-rest verdict's sensitivity near the 4 cm bar (e9 ends 2.4–3.7 cm out, q10–q90 in MuJoCo) matters for the v2 expert's demonstrations is a separate question. Nothing is changed here.

Learned Apple→Plate is still 0 successes.
