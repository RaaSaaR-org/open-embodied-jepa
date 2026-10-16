"""Development only (SO-101 S1, TASK-097; not gated, no claim): a privileged scripted reach,
grasp and lift for the simulated SO-101.

It reads the cube pose from simulator truth, solves damped least-squares IK on a scratch copy of
the model for a top-down grasp (approach axis down, jaw-closing axis on a cube face normal), and
drives the arm there with ``joint_delta_so101_v0`` actions through ``SO101Embodiment.execute``.
It exists to show that the scene is reachable and that the gripper can close on and raise the
cube. It is not a controller, a collector or a result: it reads simulator truth, and scripted
successes are never learned results. S2 (TASK-098) declares its own scripted follower.

NumPy only at import; MuJoCo comes in through the simulation object.
"""

from __future__ import annotations

import numpy as np

DOWN = np.array([0.0, 0.0, -1.0])
# gripperframe lies at the fingertips; the fixed jaw's tip is 2.0 cm along the site's -z (the
# jaw-closing axis), measured from the MJCF jaw-tip geoms. A 3 cm cube touching the fixed jaw
# has its centre at about -0.5 cm, so the site goes 0.5 cm along +z from the cube centre.
JAW_CENTRE_OFFSET_M = 0.005


def solve_ik(sim, q0, position, closing_axis, *, iterations=300, damping=1e-3):
    """Arm joints placing gripperframe at ``position`` with its approach axis down."""
    mj, model = sim.mj, sim.model
    data = mj.MjData(model)
    data.qpos[:] = sim.data.qpos
    q = np.asarray(q0, dtype=float).copy()
    site = model.site("gripperframe").id
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    cols = sim.vadr[:5]
    lower, upper = sim.limits[:5, 0], sim.limits[:5, 1]
    for _ in range(iterations):
        data.qpos[sim.qadr[:5]] = q
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        rotation = data.site_xmat[site].reshape(3, 3)
        approach, closing = rotation[:, 0], rotation[:, 2]
        # Either sign of the closing axis grasps the same face pair.
        target_closing = closing_axis if closing @ closing_axis >= 0 else -closing_axis
        error = np.concatenate(
            (
                position - data.site_xpos[site],
                0.05 * (np.cross(approach, DOWN) + np.cross(closing, target_closing)),
            )
        )
        if np.linalg.norm(error[:3]) < 5e-4 and np.linalg.norm(error[3:]) < 5e-4:
            break
        mj.mj_jacSite(model, data, jacp, jacr, site)
        jacobian = np.vstack((jacp[:, cols], 0.05 * jacr[:, cols]))
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(6), error)
        q = np.clip(q + step, lower, upper)
    data.qpos[sim.qadr[:5]] = q
    mj.mj_kinematics(model, data)
    return q, float(np.linalg.norm(position - data.site_xpos[site]))


def drive(robot, q_goal, gripper, steps, log):
    """Joint-delta actions toward ``q_goal`` (arm) with an absolute gripper command."""
    for _ in range(steps):
        robot.observe()
        delta = (q_goal - robot.sim.targets[:5]) / robot.scales.astype(float)
        action = np.concatenate((np.clip(delta, -1, 1), [gripper])).astype(np.float32)
        result = robot.execute(action)
        log.append(result.status)
        if result.status == "stopped":
            raise RuntimeError(result.reason)


def scripted_lift(robot, seed, *, on_phase=None):
    """Reset ``robot`` with ``seed`` and run the four phases; returns a per-phase record.

    ``on_phase(name)`` is called after each phase (for example to save camera frames).
    """
    sim = robot.sim
    truth = robot.reset(seed)
    cube = truth["cube_position"]
    w, x, y, z = truth["cube_quaternion"]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    face = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    # Prefer the face normal most perpendicular to the base-to-cube ray (less wrist roll).
    ray = np.array([cube[0], cube[1], 0.0]) / np.linalg.norm(cube[:2])
    other = np.array([-face[1], face[0], 0.0])
    closing = face if abs(face @ ray) < abs(other @ ray) else other
    q = robot.home[:5]
    statuses, record = [], {"seed": seed, "cube_xy": cube[:2].round(4).tolist()}
    phases = (
        ("pregrasp", 0.07, 1.0, 40),
        ("descend", 0.0, 1.0, 30),
        ("close", 0.0, -1.0, 20),
        ("lift", 0.08, -1.0, 40),
    )
    for name, height, gripper, steps in phases:
        # gripperframe is at the fingertips; offset so the cube centre is between the jaws.
        point = cube + np.array([0, 0, height])
        q_goal, _ = solve_ik(sim, q, point, closing)
        centred, residual = solve_ik(
            sim, q_goal, point + _closing_axis(sim, q_goal) * JAW_CENTRE_OFFSET_M, closing
        )
        q = centred
        drive(robot, q, gripper, steps, statuses)
        t = sim.task_truth()
        record[name] = {
            "ik_residual_m": round(residual, 5),
            "site_minus_cube_m": (t["gripper_site_position"] - t["cube_position"])
            .round(4)
            .tolist(),
            "cube_z_m": round(float(t["cube_position"][2]), 4),
            "jaw_contacts": [t["fixed_jaw_contact"], t["moving_jaw_contact"]],
            "gripper_rad": round(float(sim.data.qpos[sim.qadr[5]]), 3),
        }
        if on_phase is not None:
            on_phase(name)
    record["status_counts"] = {s: statuses.count(s) for s in sorted(set(statuses))}
    record["lifted_cm"] = round(100 * (float(sim.task_truth()["cube_position"][2]) - 0.015), 2)
    return record


def _closing_axis(sim, q):
    data = sim.mj.MjData(sim.model)
    data.qpos[:] = sim.data.qpos
    data.qpos[sim.qadr[:5]] = q
    sim.mj.mj_kinematics(sim.model, data)
    return data.site("gripperframe").xmat.reshape(3, 3)[:, 2].copy()
