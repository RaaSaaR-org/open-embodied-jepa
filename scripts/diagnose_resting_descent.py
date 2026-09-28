"""TASK-068 descent diagnosis: why could TASK-067's place redesign not lower the palm?

Diagnostic only, with no gate. Scripted and privileged: no learned policy, no corpus read, no
test split. Three parts, all written to one ``report.json``:

1. **Replay of the TASK-067 place redesign** (``PlaceThenOpen`` from
   ``scripts/probe_first_policy_place.py``, unchanged) on the first ``--count`` TASK-068
   development seeds, plate exact, after the look. During its ``lower_closed`` phase every step
   records: the policy's command; the backtracking factor the embodiment's projection accepted;
   why the full (factor-1) command was refused, by re-running the embodiment's own acceptance
   check (``G1Embodiment._prepare_side``) on a scratch copy; whether the accepted IK solution
   equals the measured arm joints (a hold); the right arm's smallest margin to a joint limit;
   and hand contacts other than with the apple.
2. **Reach map**: the lowest palm-down height the embodiment's own IK (``solve_ik``, the
   collector's palm-down rotation) reaches on a grid of palm xy, with fixed random restarts; and
   the right shoulder position. Position-only reach is sampled by forward kinematics.
3. **Rolling check**: an apple placed on the plate with a small rolling velocity, the robot
   holding still; its speed is logged for 10 s (the apple is a condim-3 sphere).

    uv run --no-sync python scripts/diagnose_resting_descent.py \\
        --output outputs/task068-descent-diagnosis/run-1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import first_policy_runtime as rt  # noqa: E402
from embodied_jepa import resting_expert as rx  # noqa: E402
from embodied_jepa.contracts import ContractError  # noqa: E402

IK_RESTART_SEED = 6831
IK_RESTARTS = 20
IK_ITERATIONS = 300  # for the reach map only; the replay uses the manifest's 80
FK_SAMPLES = 150_000
REACH_X = tuple(round(0.40 + 0.01 * i, 2) for i in range(13))
REACH_Y = (-0.11, -0.09, -0.07)
REACH_Z = tuple(round(0.30 - 0.01 * i, 2) for i in range(31))
ROLL_SPEEDS = (0.0, 0.002, 0.005, 0.01)
FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0)


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# ----- part 1 -----------------------------------------------------------------------------------
def replay(robot, bounds, seed: int, reset: dict, place_module) -> dict:
    lower, upper = bounds
    truth, _scorer, counter, _obs, _facts = rt.reset_and_look(robot, seed, reset)
    counter.remove()
    policy = place_module.PlaceThenOpen(truth)
    model = robot.model
    ids = robot.arm_ids["right"]
    qadr = model.jnt_qposadr[ids]
    limits = model.jnt_range[ids]
    target = policy.policy.phases[5].target_base.copy()
    rows = []
    while not policy.done and policy.phase_index <= 5:
        robot.observe()
        phase = policy.phase_index
        command = np.clip(np.asarray(policy.action(robot), np.float32), lower, upper)
        command = command.astype(np.float32)
        record = phase == 5
        if record:
            data = robot._snapshot_kinematics()
            trial, targets = command.copy(), robot.sim.targets.copy()
            try:
                robot._prepare_side(trial, targets, 1, "right", data=data)
                refusal = "accepted"
            except ContractError as error:
                refusal = str(error)
            measured = data.qpos[qadr].copy()
        projection = robot.project_candidates(command[None, None, None])
        applied = projection.actions[0, 0, 0]
        if record:
            requested = command[6:12]
            factor = next(
                (f for f in FACTORS if np.allclose(applied[6:12], requested * f, atol=1e-7)),
                None,
            )
            hold_targets = robot.sim.targets.copy()
            robot._prepare_side(applied.copy(), hold_targets, 1, "right", data=data)
            arm = np.array([robot.actuator_for_joint[int(i)] for i in ids])
            dq = float(np.abs(hold_targets[arm] - measured).max())
            position, _ = robot.ee_pose("right")
            margin = float(np.min(np.minimum(measured - limits[:, 0], limits[:, 1] - measured)))
        result = robot.execute(applied)
        policy.advance(result)
        if record:
            other_hand = sum(
                1
                for c in robot.sim.data.contact[: robot.sim.data.ncon]
                if _hand_not_apple(robot, c)
            )
            rows.append(
                {
                    "palm_z": float(position[2]),
                    "palm_x": float(position[0]),
                    "command_xyz": command[6:9].round(3).tolist(),
                    "factor": factor,
                    "full_command": refusal,
                    "accepted_ik_equals_measured": dq < 1e-9,
                    "accepted_step_norm_mm": float(np.linalg.norm(applied[6:9]) * 15.0),
                    "joint_limit_margin_rad": margin,
                    "hand_contacts_not_apple": other_hand,
                }
            )
    robot.stop("diagnosis")
    z = np.array([r["palm_z"] for r in rows])
    factors = [r["factor"] for r in rows]
    refusals = [r["full_command"] for r in rows]
    shoulder = rx_shoulder(robot)
    return {
        "seed": seed,
        "steps": len(rows),
        "place_target_base": target.round(4).tolist(),
        "place_target_distance_from_shoulder_m": float(np.linalg.norm(target - shoulder)),
        "palm_z_first": float(z[0]),
        "palm_z_min": float(z.min()),
        "descent_mm_per_step_last20": float(-np.diff(z[-21:]).mean() * 1000),
        "factor_counts": {str(f): factors.count(f) for f in sorted(set(factors), key=str)},
        "full_command_counts": {k: refusals.count(k) for k in sorted(set(refusals))},
        "accepted_hold_steps": int(sum(r["accepted_ik_equals_measured"] for r in rows)),
        "max_accepted_step_norm_mm_when_hold": max(
            (r["accepted_step_norm_mm"] for r in rows if r["accepted_ik_equals_measured"]),
            default=None,
        ),
        "min_joint_limit_margin_rad": float(min(r["joint_limit_margin_rad"] for r in rows)),
        "hand_contact_steps_not_apple": int(sum(r["hand_contacts_not_apple"] > 0 for r in rows)),
        "rows": rows,
    }


def _hand_not_apple(robot, contact) -> bool:
    model = robot.model
    names = []
    for geom in (int(contact.geom1), int(contact.geom2)):
        names.append((model.geom(geom).name, model.body(int(model.geom_bodyid[geom])).name))
    hand = ["hand_" in b or "wrist_" in b for _, b in names]
    apple = [g == "apple_geom" for g, _ in names]
    return any(hand) and not any(apple)


def rx_shoulder(robot) -> np.ndarray:
    data = robot.sim.data
    pelvis = data.body("pelvis")
    rotation = pelvis.xmat.reshape(3, 3)
    return rotation.T @ (data.body("right_shoulder_pitch_link").xpos - pelvis.xpos)


# ----- part 2 -----------------------------------------------------------------------------------
def reach_map(robot) -> dict:
    from embodied_jepa.embodiment import rotation_delta

    robot.reset(0)
    model = robot.model
    ids = robot.arm_ids["right"]
    limits = model.jnt_range[ids]
    qadr = model.jnt_qposadr[ids]
    data = robot.mj.MjData(model)
    rng = np.random.default_rng(IK_RESTART_SEED)
    down = rotation_delta([-np.pi / 2, 0, 0])  # the collector's palm-down rotation
    saved = robot.manifest["ik_iterations"]
    robot.manifest["ik_iterations"] = IK_ITERATIONS

    def reachable(position) -> bool:
        for _ in range(IK_RESTARTS):
            data.qpos[:] = robot.sim.data.qpos
            data.qpos[qadr] = rng.uniform(limits[:, 0], limits[:, 1])
            robot._forward_kinematics(data)
            if robot.solve_ik("right", position, down, data=data) is not None:
                return True
        return False

    shoulder = rx_shoulder(robot)
    lowest = {}
    try:
        for y in REACH_Y:
            for x in REACH_X:
                found = None
                for z in REACH_Z:
                    if reachable(np.array([x, y, z])):
                        found = z
                    elif found is not None:
                        break
                lowest[f"{x:.2f},{y:.2f}"] = found
    finally:
        robot.manifest["ik_iterations"] = saved
    # position-only reach, sampled by forward kinematics over the joint box
    distances = np.empty(FK_SAMPLES)
    for i in range(FK_SAMPLES):
        data.qpos[qadr] = rng.uniform(limits[:, 0], limits[:, 1])
        robot.mj.mj_kinematics(model, data)
        position, _ = robot.ee_pose("right", data=data)
        distances[i] = np.linalg.norm(position - shoulder)
    return {
        "shoulder_base": shoulder.round(4).tolist(),
        "lowest_palm_down_z_by_xy": lowest,
        "position_only_max_sampled_distance_m": float(distances.max()),
        "position_only_q999_distance_m": float(np.quantile(distances, 0.999)),
        "ik_restarts": IK_RESTARTS,
        "ik_iterations": IK_ITERATIONS,
        "restart_seed": IK_RESTART_SEED,
        "fk_samples": FK_SAMPLES,
    }


# ----- part 3 -----------------------------------------------------------------------------------
def rolling_check(robot) -> list[dict]:
    sim = robot.sim
    out = []
    for speed in ROLL_SPEEDS:
        robot.reset(0, plate_xy=[0.49, -0.09], object_on_container=True)
        joint = sim.data.joint("apple_free")
        joint.qvel[:] = 0
        joint.qvel[0] = speed
        joint.qvel[4] = speed / sim.object_support_height  # rolling without slip about +y
        start = sim.data.body("apple").xpos.copy()
        speeds = []
        for _ in range(200):
            sim.send_joint_targets(sim.targets.copy(), joint_names=sim.joint_names, deadline=1e9)
            speeds.append(float(np.linalg.norm(joint.qvel[:3])))
        out.append(
            {
                "initial_speed_m_s": speed,
                "speed_at_steps_10_50_100_200": [speeds[i] for i in (9, 49, 99, 199)],
                "min_speed_over_200_steps": float(min(speeds)),
                "final_xy_from_start_cm": float(
                    np.linalg.norm(sim.data.body("apple").xpos[:2] - start[:2]) * 100
                ),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    seeds = rx.DEV_SEEDS[: args.count]
    rx.check_seeds(seeds)
    started = time.monotonic()
    place = load("_probe_place", "scripts/probe_first_policy_place.py")
    wide_reset = load("_evaluate_apple", "scripts/evaluate_apple.py").wide_reset
    robot = rt.make_robot()
    bounds = rt.configured_bounds()
    attempts = []
    for seed in seeds:
        r = wide_reset(seed)
        reset = {"object_xy": list(r["object_xy"]), "plate_xy": list(r["plate_xy"])}
        attempts.append(replay(robot, bounds, seed, reset, place))
    report = {
        "diagnosis": "TASK-068 descent diagnosis (no gate; scripted and privileged)",
        "seeds": list(seeds),
        "seed_note": "TASK-068 development seeds, plate exact; no plate error",
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "replay": attempts,
        "reach": reach_map(robot),
        "rolling": rolling_check(robot),
    }
    report["seconds"] = time.monotonic() - started
    args.output.mkdir(parents=True)
    payload = json.dumps(report, indent=1, sort_keys=True) + "\n"
    (args.output / "report.json").write_text(payload)
    short = {k: v for k, v in report.items() if k not in ("replay",)}
    short["replay"] = [{k: v for k, v in a.items() if k != "rows"} for a in attempts]
    print(json.dumps(short, indent=1))
    print("report sha256", hashlib.sha256(payload.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
