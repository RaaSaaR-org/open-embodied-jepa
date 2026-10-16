"""Development only (SO-101 S2, TASK-098; not gated, no claim): privileged scripted followers
for ``so101_cube_lift_v0`` and ``so101_cube_place_v0``.

They extend S1's scripted lift (``so101_scripted.py``: damped least-squares IK on a scratch
copy of the model, a top-down grasp with the jaw closing across a cube face, joint-delta actions
through ``SO101Embodiment.execute``) with Cartesian waypoints on the descent and the transport,
a hold, and for place a transport, a lowered release, a retreat and a hold.

They read simulator truth (the cube pose at the reset, and for place the cube's offset in the
hand after the lift), so they are **privileged scripted references**: they measure what the
scene and the scorers allow, they set what later bars can be, and their successes are never
learned results.

``grasp_offset_m`` is a rigid error of the grasp aim, ``(closing, along, up)`` in the frame of
the zero-offset grasp pose: along the jaw-closing axis (the fingertip site's +z, from the fixed
jaw toward the moving jaw; horizontal), along the site's +y (horizontal) and world +z. It
is applied to every grasp-phase IK target (pre-grasp, descent, close, lift), as a calibration
error of the gripper would be. ``place_offset_m`` is a world-xy error of the place aim. These
are the K-style placement-tolerance doses of the S2 envelope (TASK-094's K follower, here with
a jaw gripper).

NumPy only at import; MuJoCo comes in through the simulation object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from embodied_jepa.so101_scripted import JAW_CENTRE_OFFSET_M, _closing_axis, solve_ik
from embodied_jepa.so101_tasks import REST_Z

UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class FollowerConfig:
    """Declared follower schedule (control steps at 20 Hz). Fixed on development seeds
    1000-1015 before the reported measurement; never tuned on the reported seeds."""

    version: str = "so101_follower_v1"
    # v1: the fixed jaw clears the cube face by 5 mm on the descent (v0, at 925153f: 0 mm, the
    # S1 alignment, which left the fixed jaw tangent to the cube); grasp offsets in the jaw frame.
    fixed_jaw_clearance_m: float = 0.005
    pregrasp_height_m: float = 0.06
    pregrasp_steps: int = 40
    descend_waypoints: int = 6
    descend_steps: int = 30
    close_steps: int = 20
    lift_height_m: float = 0.08
    lift_steps: int = 30
    transport_waypoints: int = 6
    transport_steps: int = 50
    lower_steps: int = 30
    release_clearance_m: float = 0.004  # cube bottom 4 mm above the table at release
    release_gripper: float = 0.2  # normalized: about 60 % open
    release_steps: int = 15
    release_backoff_m: float = 0.008  # move the fixed jaw this far off the cube's face ...
    backoff_steps: int = 10  # ... before rising
    retreat_height_m: float = 0.07
    retreat_waypoints: int = 7
    retreat_steps: int = 35


FOLLOWER_V0 = FollowerConfig()


class EpisodeDone(Exception):
    """The task terminated (success, lost cube or timeout) during a phase."""


class _Runner:
    def __init__(self, robot, task):
        self.robot, self.task, self.sim = robot, task, robot.sim
        self.statuses: dict[str, int] = {}
        self.score = None
        self.q = robot.home[:5].copy()

    def step(self, action):
        self.robot.observe()
        result = self.robot.execute(np.asarray(action, dtype=np.float32))
        self.statuses[result.status] = self.statuses.get(result.status, 0) + 1
        if result.status == "stopped":
            raise RuntimeError(result.reason)
        self.score = self.task.evaluate()
        if self.score["done"]:
            raise EpisodeDone

    def drive(self, q_goal, gripper, steps):
        for _ in range(steps):
            delta = (q_goal - self.sim.targets[:5]) / self.robot.scales.astype(float)
            self.step(np.concatenate((np.clip(delta, -1, 1), [gripper])))

    def hold(self):
        while True:
            self.step(self.robot.hold_action())

    def ik(self, site_point, closing):
        """Arm joints putting the fingertip site at ``site_point``, approach down."""
        q, residual = solve_ik(self.sim, self.q, site_point, closing)
        self.q = q
        return q, residual

    def through(self, start, end, waypoints, gripper, steps, closing):
        """Straight-line Cartesian waypoints from ``start`` to ``end`` (fingertip site)."""
        residual = 0.0
        per = max(1, steps // waypoints)
        for k in range(1, waypoints + 1):
            q, residual = self.ik(start + (end - start) * k / waypoints, closing)
            self.drive(q, gripper, per)
        return residual


def _site_rotation(run, site_point, closing):
    """The fingertip site's orientation at the IK solution for ``site_point``."""
    q, _ = solve_ik(run.sim, run.q, site_point, closing)
    data = run.sim.mj.MjData(run.sim.model)
    data.qpos[:] = run.sim.data.qpos
    data.qpos[run.sim.qadr[:5]] = q
    run.sim.mj.mj_kinematics(run.sim.model, data)
    return data.site("gripperframe").xmat.reshape(3, 3).copy()


def _grasp_frame(cube_position, cube_quaternion):
    """The jaw-closing axis (a cube face normal, the one most across the base-to-cube ray, as
    in S1) and the horizontal axis perpendicular to it."""
    w, x, y, z = cube_quaternion
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    face = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    other = np.array([-face[1], face[0], 0.0])
    ray = np.array([cube_position[0], cube_position[1], 0.0])
    ray /= np.linalg.norm(ray)
    closing = face if abs(face @ ray) < abs(other @ ray) else other
    along = np.cross(UP, closing)
    return closing, along


def _pick(run, cfg, grasp_offset):
    """Pre-grasp, descend, close and lift; returns (closing axis, lifted site point, record).

    The fixed jaw's inner face is 2.0 cm along the fingertip site's -z axis. The site goes
    ``JAW_CENTRE_OFFSET_M - fixed_jaw_clearance_m`` along +z from the cube centre, which leaves
    ``fixed_jaw_clearance_m`` between the fixed jaw and the cube face (v0: none); the closing
    moving jaw, which opens toward +z, then pushes the cube onto the fixed jaw. The grasp
    offset is applied in the frame of that zero-offset grasp pose: ``closing`` along the site's
    +z (from the fixed jaw toward the moving jaw, so a positive offset moves the fixed jaw toward
    the cube), ``along`` along the site's +y (horizontal) and ``up`` along world +z.
    """
    truth = run.sim.task_truth()
    cube = truth["cube_position"]
    closing, _ = _grasp_frame(cube, truth["cube_quaternion"])
    record = {}
    shift = JAW_CENTRE_OFFSET_M - cfg.fixed_jaw_clearance_m

    def aim(height, offset):
        point = cube + np.array([0.0, 0.0, height])
        q, _ = solve_ik(run.sim, run.q, point, closing)
        return point + _closing_axis(run.sim, q) * shift + offset

    rotation = _site_rotation(run, aim(0.0, np.zeros(3)), closing)
    g = np.asarray(grasp_offset, dtype=float)
    offset = g[0] * rotation[:, 2] + g[1] * rotation[:, 1] + g[2] * UP
    record["grasp_offset_world_m"] = offset.round(5).tolist()
    pre, grasp = aim(cfg.pregrasp_height_m, offset), aim(0.0, offset)
    q, record["pregrasp_ik_residual_m"] = run.ik(pre, closing)
    run.drive(q, 1.0, cfg.pregrasp_steps)
    record["grasp_ik_residual_m"] = run.through(
        pre, grasp, cfg.descend_waypoints, 1.0, cfg.descend_steps, closing
    )
    run.drive(run.q, -1.0, cfg.close_steps)
    lifted = grasp + np.array([0.0, 0.0, cfg.lift_height_m])
    run.through(grasp, lifted, 3, -1.0, cfg.lift_steps, closing)
    return closing, lifted, record


def _finish(run, record, reason_if_running="follower_ended"):
    score = run.score
    record.update(
        {
            "success": bool(score and score["success"]),
            "termination_reason": score["termination_reason"] if score else reason_if_running,
            "score": score,
            "status_counts": dict(sorted(run.statuses.items())),
        }
    )
    return record


def follow_lift(robot, task, seed, *, grasp_offset_m=(0.0, 0.0, 0.0), config=FOLLOWER_V0):
    """Reset ``task`` for ``seed`` and run the lift follower until the task terminates."""
    task.reset(seed)
    run = _Runner(robot, task)
    record = {
        "seed": seed,
        "task": task.name,
        "reset": task.reset_spec,
        "grasp_offset_m": list(map(float, grasp_offset_m)),
        "follower": asdict(config),
    }
    try:
        _, _, picked = _pick(run, config, grasp_offset_m)
        record.update(picked)
        run.hold()
    except EpisodeDone:
        pass
    return _finish(run, record)


def follow_place(
    robot,
    task,
    seed,
    *,
    grasp_offset_m=(0.0, 0.0, 0.0),
    place_offset_m=(0.0, 0.0),
    config=FOLLOWER_V0,
):
    """Reset ``task`` for ``seed`` and run the place follower until the task terminates.

    After the lift it reads the cube's offset from the fingertip site in the site frame
    (simulator truth), turns the wrist so the jaw closes across the base-to-target ray, and aims
    the site so that the cube centre lands over ``target + place_offset_m``, its bottom
    ``release_clearance_m`` above the table. After opening it backs the fixed jaw off the cube
    and rises.
    """
    task.reset(seed)
    run = _Runner(robot, task)
    target = np.asarray(task.reset_spec["target_xy"], dtype=float)
    aim_xy = target + np.asarray(place_offset_m, dtype=float)
    record = {
        "seed": seed,
        "task": task.name,
        "reset": task.reset_spec,
        "grasp_offset_m": list(map(float, grasp_offset_m)),
        "place_offset_m": list(map(float, place_offset_m)),
        "follower": asdict(config),
    }
    try:
        _, lifted, picked = _pick(run, config, grasp_offset_m)
        record.update(picked)
        truth = run.sim.task_truth()
        # The cube's offset from the fingertip site, in the site frame (privileged truth).
        site_rotation = run.sim.data.site("gripperframe").xmat.reshape(3, 3).copy()
        in_hand = site_rotation.T @ (truth["cube_position"] - truth["gripper_site_position"])
        record["in_hand_offset_site_frame_m"] = in_hand.round(4).tolist()
        # Any wrist yaw holds a cube; place with the jaw closing across the base-to-target ray.
        ray = np.array([aim_xy[0], aim_xy[1], 0.0]) / np.linalg.norm(aim_xy)
        closing = np.cross(UP, ray)
        cube_goal = np.array([*aim_xy, REST_Z + config.release_clearance_m])
        rotation = _site_rotation(run, cube_goal, closing)
        down = cube_goal - rotation @ in_hand
        above = down + np.array([0.0, 0.0, lifted[2] - down[2]])
        c = config
        run.through(lifted, above, c.transport_waypoints, -1.0, c.transport_steps, closing)
        record["place_ik_residual_m"] = run.through(
            above, down, config.descend_waypoints, -1.0, config.lower_steps, closing
        )
        run.drive(run.q, config.release_gripper, config.release_steps)
        # Back the fixed jaw (on the site's -z side of the cube) away before rising.
        jaw_axis = run.sim.data.site("gripperframe").xmat.reshape(3, 3)[:, 2].copy()
        clear = down - config.release_backoff_m * jaw_axis
        run.through(down, clear, 2, config.release_gripper, config.backoff_steps, closing)
        up = clear + np.array([0.0, 0.0, config.retreat_height_m])
        run.through(clear, up, c.retreat_waypoints, c.release_gripper, c.retreat_steps, closing)
        run.hold()
    except EpisodeDone:
        pass
    return _finish(run, record)


def _pose_check(robot, q, gripper, cube_position, cube_quaternion, *, cube_penetration_m):
    """Collisions at an arm pose on a scratch copy: robot-robot or robot-table contacts, and
    gripper penetration of the cube deeper than ``cube_penetration_m``."""
    sim = robot.sim
    mj, model = sim.mj, sim.model
    data = mj.MjData(model)
    data.qpos[:] = sim.data.qpos
    data.qpos[sim.qadr[:5]] = q
    data.qpos[sim.qadr[5]] = gripper
    cube_q = int(model.joint("cube_free").qposadr[0])
    data.qpos[cube_q : cube_q + 3] = cube_position
    data.qpos[cube_q + 3 : cube_q + 7] = cube_quaternion
    mj.mj_forward(model, data)
    scene = {model.geom(name).id for name in ("table", "floor")}
    cube = model.geom("cube_geom").id
    bad = 0
    for contact in data.contact[: data.ncon]:
        pair = {int(contact.geom1), int(contact.geom2)}
        if cube in pair:
            bad += int(not (pair - {cube}) <= scene and contact.dist < -cube_penetration_m)
        elif not pair <= scene:
            bad += 1
    return bad


def reach_map(robot, *, xs, ys, yaws, heights, mode, config=FOLLOWER_V0):
    """Kinematic reach of the followers' poses (development, privileged; no physics stepping).

    ``mode="grasp"``: for each cube xy and yaw, the follower's top-down grasp poses at each
    height above the cube centre (its closing-axis rule and jaw centring), with the jaw open.
    ``mode="place"``: for each target xy, the place pose (jaw closing across the base-to-target
    ray) at each height above the release height, with the cube moved away. A cell is ``ok``
    when every pose's IK residual is below 1 mm and no pose has a robot-robot or robot-table
    contact (nor, for grasp, a jaw more than 1 mm inside the cube).
    """
    if mode not in ("grasp", "place"):
        raise ValueError("mode must be 'grasp' or 'place'")
    sim = robot.sim
    rows = []
    for x in xs:
        for y in ys:
            worst, collisions = 0.0, 0
            for yaw in yaws if mode == "grasp" else (0.0,):
                if mode == "grasp":
                    cube = np.array([x, y, REST_Z])
                    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
                    closing, _ = _grasp_frame(cube, quat)
                    base, gripper = cube, float(sim.limits[5, 1])
                else:
                    cube = np.array([0.0, 0.30, REST_Z])
                    quat = np.array([1.0, 0.0, 0.0, 0.0])
                    ray = np.array([x, y, 0.0]) / np.hypot(x, y)
                    closing = np.cross(UP, ray)
                    base, gripper = np.array([x, y, REST_Z + config.release_clearance_m]), 0.0
                for height in heights:
                    point = base + np.array([0.0, 0.0, height])
                    q, _ = solve_ik(sim, robot.home[:5], point, closing)
                    if mode == "grasp":
                        point = point + _closing_axis(sim, q) * JAW_CENTRE_OFFSET_M
                    q, residual = solve_ik(sim, q, point, closing)
                    worst = max(worst, residual)
                    collisions += int(
                        _pose_check(robot, q, gripper, cube, quat, cube_penetration_m=1e-3) > 0
                    )
            rows.append(
                {
                    "x": round(float(x), 4),
                    "y": round(float(y), 4),
                    "worst_ik_residual_m": worst,
                    "poses_in_collision": collisions,
                    "ok": worst < 1e-3 and collisions == 0,
                }
            )
    return rows
