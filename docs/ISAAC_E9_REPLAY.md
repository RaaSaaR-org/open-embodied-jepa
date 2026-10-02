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
  - Every Isaac failure is the same kind of failure: the apple rests supported and still on the plate, with no hand contact, but its centre is just outside the 4 cm radius. It ends 3.99–4.60 cm from the plate centre, and every failure exceeds 4 cm within the 20-step at-rest window. Two failures end just below 4 cm (3.994 and 3.999 cm), and their windows reach 4.006 and 4.009 cm.
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
- `scripts/isaac/startup_probe.sh` and `scripts/isaac/serve_with_watchdog.sh` (§4);
- tests in `tests/test_isaac_e9.py`, `tests/test_isaac_transport.py` and `tests/test_isaac_watchdog.py`.

- **MuJoCo reference.** `e9_replay.py mujoco` runs TASK-070's own harness, `resting_expert.run_attempt`, unchanged, on `v2.make_v2_robot()`. The harness is: wide-jitter reset, the look, e9 (`RestingPlaceExpert(release_pitch_rad=0.45, release_dx=0.015)`) and the 60-step settle, followed by `apple_at_rest_v0`.
  - Attempts: 16 seeds × plate error 0 and 1.0 cm = 32.
  - A `StepRecorder` logs every joint-target command the embodiment sends: 793 per attempt, 8 of them the look. It also logs the joint state after each command and any robot–robot contact.
- **Isaac, closed loop.** The same harness runs on `isaac_e9.MirrorSimulation`, which is a `MuJoCoSimulation` whose physics is remote.
  - Each joint-target command goes over a Unix socket to an `IsaacTransport(physics="newton", objects=True, render=False)` in the container.
  - The reply carries Isaac's state: joints, apple pose and velocity, plate, Isaac's own contact list and its apple–hand flag. The mirror writes this state into its local `MjData`.
  - So everything that reads `sim.data` reads Isaac's state: the embodiment's IK, `ee_pose`, e9's phase logic, the scorer and the at-rest check.
  - `task_truth()` uses Isaac's apple–hand contact. The mirror's own geometric flag agreed with it on ≥ 99.7 % of steps in every attempt (minimum 99.748 %). §6 lists the one-substep offset between the two.
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

All runs are under `outputs/` in the worktree `isaac-e9` (git-ignored; archived, see docs/STORAGE.md). The image is `isaaclab_arena:latest` `sha256:2588b52605d7…`. The other inputs are those of [ISAAC_NEWTON_SPIKE.md](ISAAC_NEWTON_SPIKE.md):
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
| `isaac-e9-newton-{1,3}/compare-selfcontact` | `dfa6707` plus the review changes (uncommitted when run; committed unchanged in this PR) | runs 1 and 3 re-compared with the per-run self-contact summary (§5) | `4c05cb86`, `87022ef1` |
| `isaac-e9-watchdog-smoke-1` | `dfa6707` plus the review changes (as above) | one start through the watchdog, `--startup_only` (§4) | `isaac-e9-watchdog-smoke-1-watchdog.log` |
| `isaac-usd-threads-1-s1` | `9482ba5`, clean | **excluded**: one start that exited before building, on a bug in the new logging (`Path(pxr.__file__)` with `pxr.__file__` None), fixed in `906b7e3` | `isaac-usd-threads-1-s1-t1/log.txt` |
| `isaac-usd-threads-2-s1` | `906b7e3`, clean | **excluded** from the counts: one start with `PXR_WORK_THREAD_LIMIT=1` passed from outside: Kit reset it to 16 (§4); built, no hang | `isaac-usd-threads-2-s1-t1/log.txt` |
| `isaac-usd-threads-wprobe/` | probe scripts copied into that directory (not in git) | **excluded**: two throwaway Kit containers testing the work limit (§4) | `wprobe-3-unpinned.txt`, `wprobe-4-pinned.txt` |
| `isaac-usd-threads-3-s{1..24}` | `e27e78f`, clean | 24 start-up-only starts with the thread-limit pin (§4) | `isaac-usd-threads-3-summary.tsv` `f80bf2c3` |
| `isaac-usd-threads-4-ctrl-s{1..4}` | `e27e78f`, clean | 4 unpinned start-up-only starts, timing control (§4) | `isaac-usd-threads-4-ctrl-summary.tsv` `d19fec5d` |
| `isaac-usd-threads-5-replay-pin1-b`, `isaac-usd-threads-6-replay-nopin` | `e27e78f`, clean | 2-seed closed-loop step-time check, pinned and unpinned (§4) | `report.json` |

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
| MuJoCo + 1 mrad target noise (2 draws per attempt, of 32) | 31 | 26 | 2.55 / 3.29 / 3.77 | 2.74 / 3.43 / 4.34 |

For the noisy-MuJoCo row, the final distance pooled over both plate levels (64 replays) is 2.67 / 3.36 / 3.99 cm.

- **Every Isaac attempt completed**: 128 of 128, with no guard stop, rejection or error. The latched scorer (`AppleToPlateTask`, reported beside the at-rest check) succeeded in all 64 closed-loop Isaac attempts, as in all 32 MuJoCo attempts. The open-loop replay does not run the latched scorer.
- **Every at-rest failure fails on "inside" only**, in Isaac and in MuJoCo's one failure (1.0 cm, 50206: 4.08 cm). The apple ends supported on the plate, still (≤ 0.001 m/s) and out of hand contact. It ends 3.99–4.60 cm from the centre, and every failure exceeds 4 cm within the 20-step at-rest window. Two Isaac failures end just below 4 cm: closed run 1, 0.0 cm, 50209 at 3.994 cm (window maximum 4.006 cm), and open run 3, 0.0 cm, 50214 at 3.999 cm (window maximum 4.009 cm). The apple never left the plate.
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
| lifted | 265.5 | +1 / +3 / +5 | +1 / +3 / +4 | −1 / −0.5 / 0 | −1 / 0 / 0 |
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

The ranges combine runs 1 and 3. Command indices count all 793 commands, including the 8 of the look, so the grasp (step 221 after the look) is command 229.

- **Dex3.** The worst joint is `right_hand_thumb_1_joint` in every attempt.
  - **When the Dex3 group first passes 0.01 rad.** In most attempts this happens at commands 220–222, as the hand closes on the apple. In 2 of 32 attempts per closed-loop cell and 6–7 of 32 per open-loop cell, it happens earlier, at commands 150–152. That earlier crossing is on `right_hand_thumb_2_joint`, while the hand descends with no apple–hand contact in either simulator. There the finger is displaced from its unchanged target in both simulators, by slightly different amounts.
  - **The worst joint.** `right_hand_thumb_1_joint` itself passes 0.01 rad at commands 220–224 and 0.1 rad at commands 253–265, while the hand closes around the apple. In run 3 it never passes 0.1 rad in 2 closed-loop attempts and 1 open-loop attempt.
  - **One open-loop attempt (0.0 cm, 50200).** The thumb difference peaks at 0.364 rad at command 263 in run 1 (0.381 rad at command 263 in run 3), while apple contact drives the thumb. It is 0.155 rad at command 260. The median is 0.017 rad while holding (commands 400–579), and it is 0.000 rad after the hand has opened.
  - The left hand, which never touches anything, agrees to 1.7e-5 rad in that attempt and to at most 2.1e-5 rad over all attempts.
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

- **Isaac/Newton is not repeatable once there is contact, and it is not exactly repeatable before contact either.** Open loop, run 1 against run 3 (the largest difference over all 43 joints at each command):
  - **Before the grasp.** 18 of 32 attempts stay within 1e-5 rad over commands 0–219 (17 of 32 through command 220). Two attempts (0.0 and 1.0 cm, 50202) pass 1e-5 rad at command 8. Twelve more pass it at commands 117–150, while the robot touches nothing.
  - **Mostly at the grasp.** The first command over 1e-3 rad has a median of 229, which is the first apple–hand contact; the median over 1e-2 rad is 244. In closed loop the medians are 242 and 260.
  - **In 5 of the 64 attempt-and-mode pairs, about 1e-3 rad earlier.** These pairs pass 1e-3 rad at commands 117–146, on `right_wrist_roll_joint`, before any hand contact. Isaac's only contact then is apple–table. The pairs are open 0.0 cm/50201 (command 146), open 0.0 cm/50210 (117), open 1.0 cm/50210 (117), closed 0.0 cm/50203 (117) and closed 1.0 cm/50203 (117).
  - **The other 59 pairs** pass 1e-3 rad at commands 223–262, from 6 commands before the first apple–hand contact (as the hand closes) onwards.
  - After that, the final apple positions differ by centimetres.
- **Mostly consistent with the spike.** The spike found GPU MuJoCo-Warp repeatable to ~3e-6 rad on a trajectory with no contact at all. Here the apple rests on the table from the start, and runs can already differ by 1e-5 rad (in 14 of 32 open-loop attempts) and occasionally by 1e-3 rad (the 5 pairs above) before any robot contact. Robot contact then amplifies the differences to centimetres. Why the right wrist roll separates early without robot contact is not known. One candidate is the GPU solver's float32, non-deterministic reductions over a system that includes the apple–table contact (*inferred, untested*).
- **Isaac vs MuJoCo is about as large as Isaac vs Isaac,** and about as large as MuJoCo with 1 mrad of target noise.
- **A small reset difference does not matter.** A 0.1 mm apple reset difference is erased by the grasp.

## 3. What this means

- **For using Newton for e9.**
  - Newton reproduces e9's manipulation sequence: reach, grasp, carry, release and landing, with the same timing to a few steps.
  - It does not reproduce e9's per-seed at-rest verdicts, and it cannot: it does not reproduce its own verdicts across two processes.
  - These runs cannot say whether Newton's rate is lower. Every Isaac cell is at 24–26/32, against MuJoCo's 31/32 and noisy MuJoCo's 26–31/32 per cell. More seeds and repeats would be needed.
- **Ruling: how Isaac results may be used** (2026-09-30, decided under owner delegation; it applies to every later Isaac comparison, not to TASK-070 or TASK-073):
  - **Per seed, Isaac is compared with MuJoCo only on event timing:** grasp, lift, release and landing (§2, "Grasp, lift and release timing"). No per-seed at-rest verdict, final distance or success is compared between the simulators.
  - **An Isaac at-rest success rate must come from repeated runs over more seeds than these 16, and must be reported with its uncertainty.** A single run per seed is not enough.
  - The reason is that neither simulator is repeatable at this bar. Newton on the GPU is not repeatable once there is contact: 10 of 32 at-rest verdicts flip between two runs in each mode. MuJoCo is similarly sensitive: 1 mrad of target noise flips 9 of 64.
- **For the benchmark itself (*inferred*).** e9's at-rest success depends on the apple ending inside 4 cm after rolling toward the rim; it typically ends 2.4–3.7 cm out. Small physics differences move a quarter of attempts across the bar. This is a property of the expert and the task, and it would affect any simulator or hardware transfer. It is recorded here and not acted on.
- **Not shown.** This does not show whether Isaac is a suitable data source for learned policies. Images were not rendered, and the photometric mismatch of [ISAAC_V2_SCENE.md](ISAAC_V2_SCENE.md) §1 is unchanged.

## 4. The start-up hang

**Recurrence.** In this task, 1 of the 3 full server starts hung (`isaac-e9-server-2`), 1 of the 8 start-up probes hung (probe 4), and the one start through the watchdog (`isaac-e9-watchdog-smoke-1`) did not hang.

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
- Update (2026-09-30): the leading candidate is now the known OpenUSD thread-safety bug in this function. Pinning USD's work pool to one thread gave 0 hangs in 24 starts (below).

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

- **In this task:** 2 hangs in 12 starts (3 servers, 8 probes, 1 watchdog start), so 10 healthy. Both hangs were at the same line.
- **With the spike:** 4 in 22 Newton starts, about 1 in 5.
- **Where:** every hang recorded with a stack dump is at the same line. The spike's two hangs were at the same log position.
- **After a successful build** there were no hangs: the 3 probe steps took 0.2–0.3 s, and the two full servers ran 50 752 steps each.

**Ruling** (2026-09-30, decided under owner delegation): (a) and (c) now; (b) is not run.
- **(a) A start-up watchdog with retry: implemented.**
  - What it does. `scripts/isaac/serve_with_watchdog.sh <out-prefix> <limit-s> <tries> [server args]` starts `e9_server_isaac.py` through `run_isaac.sh`, as try `<out-prefix>-t<k>`.
  - Hang detection. If the try has not logged `transport built` within the limit, the watchdog stops only that try's own container (`oej-isaac-<try>`). It then stops the launcher process it started itself, if that is still running, and starts the next try.
  - Limits and records. The number of tries is bounded. Every try, hang and retry is written to `<out-prefix>-watchdog.log`. The socket of the try that built is written to `<out-prefix>-ready`, for `e9_replay.py isaac --socket`.
  - Other failures. A start that exits before building is a different failure, so it is not retried. If stopping the hung try's own container fails, the watchdog logs the failure and the stop command's output, starts nothing more, and exits 6. The hung container may still hold GPU memory, so it needs a person.
  - The limit. **Now 480 s**, because the thread-limit pin (below) makes healthy starts take 172–228 s. *(Historical, before the pin: the recommendation was 300 s, about twice the slowest of the 10 healthy starts of this task, 124.9–151.6 s.)* The recommended number of tries is 3: at about 1 hang in 5 starts, three hangs in a row would happen about 1 time in 100, if starts are independent (*inferred*).
  - Why a restart is safe. The hang happens before any episode.
  - Tests. `tests/test_isaac_watchdog.py` checks, with a fake launcher: retry after a hang; giving up after the bounded number of tries (exit 4); no retry after a crash; no retry after a failed container stop (exit 6); and that only the hung try's container is stopped.
  - Real check. One real start went through the watchdog (`isaac-e9-watchdog-smoke-1`, `--startup_only`). It built at 125.9 s by the server's clock, which the watchdog saw at 128 s. It did not hang, so the retry path has been exercised only with the fake launcher.
  - Still manual. `startup_probe.sh` still records hangs without retrying, because counting them is its purpose.
- **(b) Parse a detached copy: not run.** The idea is to run `add_usd` on a detached, flattened copy of the stage (`stage.Flatten()` into an anonymous in-memory stage) instead of the live Kit stage. This would test the race hypothesis. At this hang rate it needs about 20 or more probes per arm to be informative. It stays open as an option.
- **(c) Report upstream: drafted, not posted.** The draft is [isaac/newton_usd_hang_report.md](isaac/newton_usd_hang_report.md). It includes versions, repro steps, frequency and stack excerpts. The project owner posts it after checking.

### The OpenUSD thread limit (`PXR_WORK_THREAD_LIMIT=1`), 2026-09-30

**Why.** Newton documents a known OpenUSD thread-safety bug in this same function: `UsdPhysics.LoadUsdPhysicsFromRange` can crash when many mesh colliders sit under one rigid body ([newton#1743](https://github.com/newton-physics/newton/issues/1743), [#2216](https://github.com/newton-physics/newton/issues/2216); Newton's `docs/concepts/usd_parsing.rst`, "Limitations"). The documented workaround is `PXR_WORK_THREAD_LIMIT=1`. The fix is OpenUSD commit `ed857d77` (PR 4002), first in v26.05-rc1; Newton's current docs say "fixed in OpenUSD 26.08".

**This image does not have the fix.**
- In the running Kit process, `pxr.Usd.GetVersion()` is 0.25.11, loaded from `omni.usd.libs-1.0.1` (the image also carries `usd_core` 25.8 and `usd_exchange` 3.0.0 in site-packages).
- `ed857d77` is not in stock v25.11 (GitHub compare: the commit is 597 commits ahead of the v25.11 tag), and `UsdPhysics.LoadStageFromPrimRange`, the name OpenUSD 26.05 introduced with the fix, does not exist. Whether NVIDIA's build carries a backport was not checked.

**Setting the variable from outside does not work in Kit.**
- USD reads `PXR_WORK_THREAD_LIMIT` once, when its env settings initialise, and the variable then overrides `pxr.Work.SetConcurrencyLimit`.
- Kit overwrites the variable before USD reads it, unconditionally: `SimulationApp` sets min(cores, `limit_cpu_threads`), and the `omni.usd.config` extension sets `"16"` (its comment: OMPE-59303).
- Measured: with `-e PXR_WORK_THREAD_LIMIT=1` on `docker run`, the process reported `PXR_WORK_THREAD_LIMIT=16` and `Work.GetConcurrencyLimit() == 16` after the app launched (`isaac-usd-threads-2-s1`), and a throwaway probe container showed that `Work.SetConcurrencyLimit` cannot change it afterwards (below). **So every earlier start ran USD's work pool with 16 threads** (measured in this image; inferred for the spike's starts, which used the same image and code path).
- **Throwaway probes** (excluded from all counts; outputs copied to `outputs/isaac-usd-threads-wprobe/`, image as above). A minimal script launches the headless app through `AppLauncher` and prints the variable and `Work.GetConcurrencyLimit()`, then calls `Work.SetConcurrencyLimit(1)`, `(4)` and `(0)`.
  - **Unpinned** (`wprobe-3-unpinned.txt`, 10:16:48–10:16:50Z log stamps; `docker run -e PXR_WORK_THREAD_LIMIT=1`): the variable went from 1 to 16, the limit was 16, and it stayed 16 after each `SetConcurrencyLimit` call. Its script was later edited in place into the pinned one, so `wprobe_unpinned_reconstructed.py` is the pinned script with the pin block removed (reconstructed, not the file as run).
  - **Pinned** (`wprobe-4-pinned.txt`, script `wprobe_pinned.py`, 10:17:14–10:17:16Z log stamps): with the variable pinned in `os.environ` before the app starts, the limit was 1 and stayed 1 after each `SetConcurrencyLimit` call.
  - Two earlier attempts printed nothing useful: one had no output captured, and one (`wprobe-2.txt`) stopped at the image's user set-up (`chown: cannot access '/home/huhn'`).
- What works: `e9_server_isaac.py` now pins the variable in `os.environ` before the app starts, so later writes of that one key keep the pinned value. **The limit in effect is verified; the mechanism is inferred.** Which Kit write the pin replaces was not logged: in the pinned probe, the only intercepted write printed was the probe's own, perhaps because Kit redirects Python output during start-up. The account of how Kit sets the variable comes from reading Kit's Python sources. The server now logs each replaced write to fd 2, and it rejects a negative `--pxr_work_thread_limit` or a non-integer inherited value. The server logs `pxr.Work.GetConcurrencyLimit()` after the app launches and again right before `add_usd`; it was 1 in all 24 pinned starts below.

**Series** (`isaac-usd-threads-3-s{1..24}`, revision `e27e78f`, clean; these runs are under `outputs/` of the `fix/isaac-usd-thread-limit` worktree): 24 starts through `serve_with_watchdog.sh` (limit 600 s, 1 try each), `--startup_only`, pin 1. No seed was simulated. The series ran one container at a time, with one exception near its start. The throwaway probe containers above ran from about 10:16:40Z to just after 10:17:17Z, while the excluded start `isaac-usd-threads-2-s1` (10:15:29–10:17:36Z) was running, and ended shortly before series start 1 (10:17:52Z). Start 1 may have overlapped the last probe container's shutdown.

| Arm | Starts | Hung | Work limit | `transport built`, s (min / median / max) | `add_usd`, s (min / median / max) |
| --- | --- | --- | --- | --- | --- |
| Earlier starts (spike + this task, §4 above) | 22 | 4 | 16 (inferred: same image and code path) | 124.9 / – / 151.6 (the 10 healthy starts with a timing) | not measured |
| **Pinned to 1** (`-3-s1..24`) | **24** | **0** | 1 | 171.6 / 174.1 / 227.5 | 11.27 / 11.64 / 16.33 |
| Unpinned control, timing only (`-4-ctrl-s1..4`) | 4 | 0 | 16 | 125.1 / 125.7 / 126.1 | 9.10 / 9.36 / 9.39 |

- **Hangs: 0 of 24 against 4 of 22.** One-sided Fisher exact test p = 0.045 (the two-sided p is also 0.045). With the pin, the hang rate is at most 12 % (one-sided 95 % Clopper–Pearson bound, 0 of 24; the two-sided 95 % interval is 0–14 %); the base rate is 18 % (4 of 22, 95 % interval 5–40 %). This is suggestive, not proof: the base-rate starts are historical, not a concurrent randomised arm, and a hang rate of 5–10 % with the pin is not excluded.
- **The mechanism fits, but is inferred.** Newton's issue describes a crash from a data race on a `std::vector` in the multithreaded collider parse; what we saw is a main thread spinning at 100 % in the same function. A corrupted container can loop as well as crash. We have no native stack to confirm it.
- **Start-up is about 48 s slower with the pin** (median 174.1 s against 125.7 s for the 4 unpinned controls on the same host, same day). The parse itself is only about 2.3 s slower (11.6 s against 9.4 s). The rest is in solver set-up and kernel compilation: `Initialize solver` 19.9 s against 13.4 s, and `CUDA graph took` about 112 s against 82 s. The likely reason is that the USD work limit also caps the process's shared TBB pool, which other start-up work uses (*inferred*).
- Starts 3 and 4 (223.6 s and 227.5 s) were slower; the cause was not recorded (the load-average log starts at 10:29:51Z, after start 3). Starts 5–24 took 171.6–176.1 s.

**Stepping is slower too** (non-gated development replay, closed loop, seeds 50200 and 50201 at plate error 0, one server each, both at `e27e78f`):

| Server | Work limit | Server step time, ms (median / p95, 1 586 steps) | Seconds per attempt | At rest (50200, 50201) | Final distance, cm |
| --- | --- | --- | --- | --- | --- |
| `isaac-usd-threads-5-replay-pin1` (`-srv-t1`, report `-pin1-b`) | 1 | 32.0 / 33.5 | 27.6, 27.1 | yes, no | 3.54, 4.16 |
| `isaac-usd-threads-6-replay-nopin` | 16 | 27.2 / 27.8 | 23.3, 23.1 | yes, yes | 3.88, 2.34 |
| Runs 1 and 3 (§2, all 50 752 steps) | 16 | 26.7 / 35.7 and 26.5 / 33.6 | about 22.5 | yes, yes (both runs) | 3.89, 3.14 and 2.60, 3.65 |

- The pin makes each step about 18 % slower (32.0 against 27.2 ms median), so a full 32-attempt closed- plus open-loop run takes about 4.5 minutes longer (64 attempts × about 4.2 s).
- The one verdict difference (50201 not at rest with the pin, at 4.16 cm) is within Isaac's own run-to-run variation (10 of 32 at-rest verdicts flip between runs 1 and 3, §2); two attempts cannot show an effect of the pin on the physics.
- The first pinned replay attempt failed to connect: the absolute socket path was longer than the 108-byte AF_UNIX limit. The same server was then used with a relative socket path (`-pin1-b`).

**Decision (development).**
- **The Newton path now defaults to the pin (`PXR_WORK_THREAD_LIMIT=1`).** It removed the hang in 24 of 24 starts, and a start that cannot hang is worth more than the 48 s it costs: an unpinned hang costs the watchdog limit plus a new start.
- **The watchdog stays as the backstop**, because 0 of 24 does not prove the hang is gone. With the pin, healthy starts take 172–228 s, so a 300 s limit leaves too little margin on a loaded host; use **480 s**.
- `--pxr_work_thread_limit 0` restores Kit's 16 threads (for timing-sensitive work or to reproduce the base rate); `--pxr_work_thread_limit N` pins N. A value passed in the environment (`run_isaac.sh` passes `PXR_WORK_THREAD_LIMIT` through) is pinned as given. PhysX starts are not pinned by default.
- Remove the pin once the image's USD contains the fix, after a new start-up series without it. The fix commit is in OpenUSD v26.05 (and 26.08), and Newton's current `usd_parsing.rst` names 26.08 as the fixed version. A practical check is `hasattr(pxr.UsdPhysics, "LoadStageFromPrimRange")` (the 26.05 rename), which the server logs.

## 5. Robot self-collision: ruling, evidence and options

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
- **(A) Keep it off.** No change. It is harmless for every trajectory measured so far (e9, the free-space parity trajectory), by the evidence above. The e9 tooling flags any self-contact per run, so a future trajectory that needs self-collision would show up.
- **(B) Turn it on in Newton only.** The transport would override the resolved attribute when it builds the Newton model, with MuJoCo's parent–child filter and the MJCF's contype/conaffinity. This matches the host, but it needs a new parameter audit: the exclude list, and Newton's collision-group colouring against MuJoCo's bitmasks. PhysX would stay off.
- **(C) Reconvert the USD with `allow_self_collision: True`.** This turns it on for both backends. It changes the USD hash and every recorded Isaac provenance, and PhysX's filtering then also needs an audit.

**Ruling (2026-09-30, decided under owner delegation): (A). Self-collision stays off in both Isaac backends, and every run reports any robot self-contact.**
- **Why off.**
  - It is harmless for every trajectory measured: 0 self-contacts in 32 MuJoCo e9 attempts, in 128 Isaac attempts under MuJoCo's collision rules, and on the spike's free-space trajectory.
  - (B) and (C) each need a new parameter audit. (C) would also change the USD hash and every recorded Isaac provenance.
- **Why flagged.** With self-collision off, Isaac's own contact list cannot report a self-contact. So the check that matters is the mirror's: MuJoCo collision detection, with self-collision on as in the host MJCF, applied to Isaac's poses. A self-contact found there means Isaac let two links interpenetrate where MuJoCo would have pushed them apart.
- **The flag.**
  - `isaac_e9.self_contact_summary` counts, per source, the steps and attempts with a robot self-contact. The sources are MuJoCo's own state, the mirror on Isaac's state, and Isaac's own contact list.
  - `e9_replay.py` writes the summary into every `mujoco` and `isaac` report and every `compare` result (`self_contact`), and prints `SELF-CONTACT FLAG` on stderr if any count is non-zero.
  - Re-comparing runs 1 and 3 with it (`compare-selfcontact`) gives 0 steps in every source for 32 attempts each: MuJoCo, closed-loop mirror, closed-loop Isaac, open-loop mirror and open-loop Isaac. The flag is false.
- **When to revisit.** Revisit with (B) if a run raises the flag, or if a trajectory of interest shows MuJoCo self-contacts, for example a two-handed task or a learned policy's rollouts, which are less tidy than a scripted expert. The evidence rests on 32 scripted e9 attempts and one free-space trajectory. It does not cover other motions.

## 6. Caveats

- **Development only.** Nothing was declared in advance, and the counts are descriptive. There are 16 seeds × 2 plate levels, and the same 32 attempts are used in every cell, so the cells are not independent. No interval or test is claimed.
- **Two Isaac runs.** Run-to-run variation in Isaac is large (§2), so two runs give only a rough picture of Isaac's own outcome distribution.
- **The mirror is a harness choice.** Closed loop in Isaac means the MuJoCo-side IK and expert logic read Isaac's state through a MuJoCo `MjData`.
  - The kinematics are MuJoCo's (the host MJCF). Newton's compiled model matches it on joints and bodies ([ISAAC_NEWTON_SPIKE.md](ISAAC_NEWTON_SPIKE.md) §4), but geom hulls were not compared vertex by vertex.
  - The contact counts in the events table (plate base or rim) are the mirror's geometric contacts on Isaac's state. Apple–hand contact is Isaac's own.
- **Float32 transfer.** Isaac's joint and apple state is float32 on the GPU, and the MuJoCo reference is float64 on the CPU.
- **Known limitations of the mirror** (from the independent review; none changes a result here):
  - **One-substep offset in hand contact.** Isaac's apple–hand flag comes from `task_truth()` after the interval's last physics substep. The mirror's own contacts, which include the plate base or rim counts, the landing and `rim_contact_after_open`, come from the state at the *start* of that substep, which is MuJoCo's `mj_step` layout. The two can disagree on the one substep where contact starts or ends. This fits the ≥ 99.7 % agreement (minimum 99.748 %).
  - **Velocities are written before the kinematics pass.** `MirrorSimulation._apply` writes the new `qvel` and then runs `mj_forward` at the pre-step pose. So derived velocity quantities in `MjData` (`cvel` and the like) combine the new velocities with the old pose. Nothing on the e9 path reads them: the harness reads `qpos`, `qvel`, body and site poses, and contacts. They must not be relied on if the mirror is reused for anything that reads them.
  - **The `placed` rule is a copy.** The mirror's `task_truth` evaluates `MuJoCoSimulation.task_truth`'s `placed` rule with Isaac's hand flag (`isaac_e9.placed_rule`). The rule is inline in `simulation.py`, and that file is byte-pinned by the benchmark manifests' source hashes. It is therefore not factored out there, and a copy was the simple option. `test_placed_rule_matches_mujoco_task_truth` pins the copy to the original on placed and not-placed states, so a change to the base rule fails the test instead of drifting silently.
- **Open loop is not closed loop.** The open-loop replay feeds MuJoCo's commands to Isaac, so it tests physics under identical commands. It is not how e9 would run in Isaac.
- **No rendering.** The server runs with `render=False`, and the expert does not read pixels. Photometry is unchanged and still fails its bar.
- **Versions.** Newton's MuJoCo-Warp is 3.5.0.2 / MuJoCo 3.5.0 in the image; the host is 3.13.0. The known unmatched parameters of [ISAAC_NEWTON_SPIKE.md](ISAAC_NEWTON_SPIKE.md) §5 apply, including solver tolerance 1e-6 against 1e-8 and float32.
- **Timing and memory.**
  - Isaac `step` per 0.05 s interval: median 26.7 ms, p95 35.7 ms (n = 50 752, run 1).
  - A full e9 attempt: about 22 s in Isaac, against 3.5 s in MuJoCo.
  - Start-up is about 130 s, including about 83 s of Warp kernel compilation into an empty per-run cache.
  - Peak device memory sampled during run 1 was 8 511 MiB, including GR00T's 6 626 MiB.

## 7. Rulings (2026-09-30, decided under owner delegation) and the open question

1. **Self-collision: (A).** It stays off, and each run reports any robot self-contact. The count is 0 today (§5).
2. **Start-up hang: (a) and (c).** A start-up watchdog with a bounded retry is implemented, and an upstream report is drafted but not posted (§4). (b) is not run.
3. **Use of Isaac results.** Per seed, Isaac is compared with MuJoCo only on event timing: grasp, lift, release and landing. An Isaac at-rest success rate must come from repeated runs over more seeds, with its uncertainty reported (§3).
4. **Still open: the margin of the at-rest verdict** (*inferred*). The at-rest verdict is sensitive near the 4 cm bar: in MuJoCo, e9 ends 2.4–3.7 cm out (q10–q90). Whether that matters for the v2 expert's demonstrations is a separate question. Nothing is changed here.

Learned Apple→Plate is still 0 successes.
