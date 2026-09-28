"""TASK-069 reach measurements (development only, no gate; kinematics only, no episode).

Two questions, answered with the embodiment's own kinematic model on a fixed scene:

1. **(b2) The set-down region.** For which palm xy can the right palm reach the set-down height
   with the collector's palm-down rotation? The set-down height is the resting apple's centre
   plus the carried palm-over-apple offset (0.0444 m, TASK-067 logs) plus a clearance. It uses
   ``G1Embodiment.solve_ik`` (the 7 arm joints) with fixed random restarts.
2. **(b1) The waist.** How much closer does a position-only IK get to the TASK-068 place targets
   and to the v1 plate region if the three waist joints (yaw, roll, pitch; actuated in the MJCF
   and held at their reset targets by the transport) are added to the 7 arm joints, within
   their joint limits? Kinematics only: balance, torque, and the torso camera moving with the
   waist are not modelled.

    uv run --no-sync python scripts/measure_v2_reach.py --output outputs/task069-reach/run-1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import first_policy_runtime as rt  # noqa: E402
from embodied_jepa.embodiment import rotation_delta  # noqa: E402

RESTART_SEED = 6841
RESTARTS = 20
ITERATIONS = 300
HOLD_OFFSET_M = 0.0444  # palm over apple while carried (TASK-067 landing diagnosis logs)
CLEARANCES_M = (0.0, 0.005, 0.01)
GRID_X = tuple(round(0.26 + 0.02 * i, 2) for i in range(13))  # 0.26 .. 0.50
GRID_Y = tuple(round(-0.30 + 0.02 * i, 2) for i in range(18))  # -0.30 .. 0.04
WAIST = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
# The TASK-068 descent diagnosis's eight place targets (palm, base frame), and the v1 plate
# region's corners and centre as palm set-down targets (plate xy minus 1.5 cm in x).
TASK068_TARGETS = (
    (0.4772, -0.0726, 0.0354),
    (0.4551, -0.0780, 0.0354),
    (0.4785, -0.1043, 0.0354),
    (0.4681, -0.0724, 0.0354),
    (0.4616, -0.1086, 0.0354),
    (0.4591, -0.1031, 0.0354),
    (0.4921, -0.0889, 0.0354),
    (0.4599, -0.0983, 0.0354),
)


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def rest_z_base(robot) -> float:
    truth = robot.sim.task_truth()
    base = np.asarray(truth["base_position_world"])
    return float(truth["container_surface_z"] + truth["object_support_height"] - base[2])


def setdown_region(robot, rng) -> dict:
    model = robot.model
    ids = robot.arm_ids["right"]
    limits = model.jnt_range[ids]
    qadr = model.jnt_qposadr[ids]
    data = robot.mj.MjData(model)
    down = rotation_delta([-np.pi / 2, 0, 0])
    saved = robot.manifest["ik_iterations"]
    robot.manifest["ik_iterations"] = ITERATIONS
    rest = rest_z_base(robot)
    out = {}
    try:
        for clearance in CLEARANCES_M:
            z = rest + HOLD_OFFSET_M + clearance
            grid = {}
            for y in GRID_Y:
                for x in GRID_X:
                    ok = False
                    for _ in range(RESTARTS):
                        data.qpos[:] = robot.sim.data.qpos
                        data.qpos[qadr] = rng.uniform(limits[:, 0], limits[:, 1])
                        robot._forward_kinematics(data)
                        if (
                            robot.solve_ik("right", np.array([x, y, z]), down, data=data)
                            is not None
                        ):
                            ok = True
                            break
                    grid[f"{x:.2f},{y:.2f}"] = ok
            out[f"{clearance:.3f}"] = {"palm_z_base": z, "reachable": grid}
    finally:
        robot.manifest["ik_iterations"] = saved
    return {"rest_z_base": rest, "hold_offset_m": HOLD_OFFSET_M, "by_clearance": out}


def position_residual(robot, rng, targets, joint_names) -> dict:
    model = robot.model
    ids = np.array([model.joint(n).id for n in joint_names])
    limits = model.jnt_range[ids]
    qadr, vadr = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    data = robot.mj.MjData(model)
    site = model.site("right_ee").id
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    pelvis_id = model.body("pelvis").id
    out = {}
    for target in targets:
        best = (np.inf, None)
        for _ in range(RESTARTS):
            data.qpos[:] = robot.sim.data.qpos
            data.qpos[qadr] = rng.uniform(limits[:, 0], limits[:, 1])
            for _ in range(ITERATIONS):
                robot._forward_kinematics(data)
                goal = data.xpos[pelvis_id] + data.xmat[pelvis_id].reshape(3, 3) @ np.asarray(
                    target
                )
                error = goal - data.site_xpos[site]
                robot.mj.mj_jacSite(model, data, jacp, jacr, site)
                jac = jacp[:, vadr]
                step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(3), error)
                data.qpos[qadr] = np.clip(
                    data.qpos[qadr] + np.clip(step, -0.1, 0.1), limits[:, 0], limits[:, 1]
                )
            robot._forward_kinematics(data)
            goal = data.xpos[pelvis_id] + data.xmat[pelvis_id].reshape(3, 3) @ np.asarray(target)
            residual = float(np.linalg.norm(goal - data.site_xpos[site]))
            if residual < best[0]:
                best = (residual, data.qpos[qadr].copy())
        out[",".join(f"{v:.4f}" for v in target)] = {
            "residual_m": best[0],
            "joints": dict(zip(joint_names, np.round(best[1], 3).tolist(), strict=True)),
        }
    return out


def palm_down_residual(robot, rng, targets, joint_names) -> dict:
    """As ``position_residual``, but for the full palm-down pose (the collector's rotation),
    weighted as ``G1Embodiment.solve_ik`` weights it (rotation error x 0.3). Also reports how far
    the LEFT palm moves from its reset pose at the solution, since the waist carries both arms."""
    model = robot.model
    ids = np.array([model.joint(n).id for n in joint_names])
    limits = model.jnt_range[ids]
    qadr, vadr = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    data = robot.mj.MjData(model)
    site = model.site("right_ee").id
    left = model.site("left_ee").id
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    pelvis_id = model.body("pelvis").id
    down = rotation_delta([-np.pi / 2, 0, 0])
    data.qpos[:] = robot.sim.data.qpos
    robot._forward_kinematics(data)
    left_reset = data.site_xpos[left].copy()
    out = {}
    for target in targets:
        best = (np.inf, None, None, None)
        for _ in range(RESTARTS):
            data.qpos[:] = robot.sim.data.qpos
            data.qpos[qadr] = rng.uniform(limits[:, 0], limits[:, 1])
            for _ in range(ITERATIONS):
                robot._forward_kinematics(data)
                base_rot = data.xmat[pelvis_id].reshape(3, 3)
                goal = data.xpos[pelvis_id] + base_rot @ np.asarray(target)
                dp = goal - data.site_xpos[site]
                quat = np.empty(4)
                current = data.site_xmat[site].reshape(3, 3)
                robot.mj.mju_mat2Quat(quat, ((base_rot @ down) @ current.T).ravel())
                if quat[0] < 0:
                    quat *= -1
                dr = np.empty(3)
                robot.mj.mju_quat2Vel(dr, quat, 1.0)
                robot.mj.mj_jacSite(model, data, jacp, jacr, site)
                jac = np.vstack((jacp[:, vadr], 0.3 * jacr[:, vadr]))
                error = np.concatenate((dp, 0.3 * dr))
                step = jac.T @ np.linalg.solve(jac @ jac.T + 0.02**2 * np.eye(6), error)
                data.qpos[qadr] = np.clip(
                    data.qpos[qadr] + np.clip(step, -0.12, 0.12), limits[:, 0], limits[:, 1]
                )
            robot._forward_kinematics(data)
            base_rot = data.xmat[pelvis_id].reshape(3, 3)
            goal = data.xpos[pelvis_id] + base_rot @ np.asarray(target)
            position = float(np.linalg.norm(goal - data.site_xpos[site]))
            quat = np.empty(4)
            current = data.site_xmat[site].reshape(3, 3)
            robot.mj.mju_mat2Quat(quat, ((base_rot @ down) @ current.T).ravel())
            angle = float(2 * np.arccos(min(1.0, abs(quat[0]))))
            score = position + 0.3 * angle
            if score < best[0]:
                shift = float(np.linalg.norm(data.site_xpos[left] - left_reset))
                best = (score, (position, angle), data.qpos[qadr].copy(), shift)
        out[",".join(f"{v:.4f}" for v in target)] = {
            "position_residual_m": best[1][0],
            "rotation_residual_rad": best[1][1],
            "left_palm_shift_m": best[3],
            "joints": dict(zip(joint_names, np.round(best[2], 3).tolist(), strict=True)),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    started = time.monotonic()
    robot = rt.make_robot()
    robot.reset(0)
    rng = np.random.default_rng(RESTART_SEED)
    region = setdown_region(robot, rng)
    arm = tuple(
        f"right_{n}_joint"
        for n in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    )
    rest = region["rest_z_base"]
    palm_z = rest + HOLD_OFFSET_M + 0.005
    v1_region = tuple(
        (x - 0.015, y, palm_z) for x in (0.47, 0.49, 0.51) for y in (-0.11, -0.09, -0.07)
    )
    targets = TASK068_TARGETS + v1_region
    waist = {
        "arm_only": position_residual(robot, rng, targets, arm),
        "arm_plus_waist": position_residual(robot, rng, targets, WAIST + arm),
    }
    palm_down = {
        "arm_only": palm_down_residual(robot, rng, targets, arm),
        "arm_plus_waist": palm_down_residual(robot, rng, targets, WAIST + arm),
    }
    report = {
        "task": "TASK-069 reach measurements (development only, kinematics only)",
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "restart_seed": RESTART_SEED,
        "restarts": RESTARTS,
        "iterations": ITERATIONS,
        "waist_joint_ranges_rad": {
            n: robot.model.jnt_range[robot.model.joint(n).id].tolist() for n in WAIST
        },
        "setdown_region": region,
        "position_only_residuals": waist,
        "palm_down_residuals": palm_down,
        "v1_region_palm_targets": [list(t) for t in v1_region],
        "seconds": time.monotonic() - started,
    }
    args.output.mkdir(parents=True)
    payload = json.dumps(report, indent=1, sort_keys=True) + "\n"
    (args.output / "report.json").write_text(payload)
    print("report sha256", hashlib.sha256(payload.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
