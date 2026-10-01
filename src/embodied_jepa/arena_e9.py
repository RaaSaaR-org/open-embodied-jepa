"""Cross-simulator development check of e9 in Isaac Lab-Arena's G1 scene (and its kinematics).

Development only (TASK-025 Arena track; docs/ARENA.md §7): not a gated run, not a learned
result and not a policy result. e9 is TASK-070's privileged scripted expert
(``docs/experiments/apple_to_plate_v2_expert.md``); it reads simulator truth by design.
Learned Apple->Plate is still 0 successes.

The adapter is the mirror pattern of the Isaac e9 replay (``isaac_e9.MirrorSimulation``,
docs/ISAAC_E9_REPLAY.md §1): the unchanged harness (``resting_expert.run_attempt``: the
embodiment's IK, the look, e9, the scorer, ``apple_at_rest_v0``) runs on a local MuJoCo model
whose state is Arena's. ``ArenaEndpoint`` sits between that mirror and the Arena server
(``scripts/isaac/arena_e9_server.py``) and owns the three differences:

* **joint order and action**: e9's 43 joint targets (MJCF actuator order) are mapped by name
  into Arena's 50-D ``g1_wbc_agile_joint`` action (``arena_transport.wbc_action``) with
  ``navigate = 0`` and ``base_height = 0.75``; the AGILE WBC uses only the 28 arm, wrist and
  Dex3 targets;
* **rate**: one 20 Hz command is held for 2 or 3 Arena steps of 0.02 s, alternately
  (``arena_steps``), which is exact over every pair of commands (0.1 s);
* **floating pelvis**: Arena's state is written into the mirror *relative to Arena's live
  pelvis* (``FrameMap``): the mirror's fixed pelvis at (0, 0, 0.793) stands for Arena's pelvis
  at every step, so FK, IK and e9's palm targets are pelvis-relative, as the brief asks.

NumPy only at import; MuJoCo is imported lazily by the simulator classes.
"""

from __future__ import annotations

import numpy as np

from embodied_jepa import arena_transport as at
from embodied_jepa import isaac_e9 as ie
from embodied_jepa.contracts import ContractError

VERSION = "arena_e9_crosssim_v1"
OUR_PELVIS_POS = (0.0, 0.0, 0.793)  # MuJoCoSimulation: the fixed pelvis
OUR_TABLE_TOP_Z = 0.74  # simulation._scene: table box centre 0.70, half-height 0.04
OUR_PLATE_BASE_HALF = 0.006  # the plate body sits at the base cylinder's centre (0.746)
ARENA_SHELF_TOP_Z = -0.030  # Arena source SHELF_SURFACE_Z (env-local)
ARENA_AIRGAP = 0.005  # Arena source SHELF_AIRGAP
ARENA_APPLE_ORIGIN_ABOVE_BOTTOM = 0.0171  # Arena source _USD_ORIGIN_ABOVE_BOTTOM_M
ARENA_DT = 0.02  # 50 Hz (sim dt 0.005, decimation 4)
COMMAND_DT = 0.05  # 20 Hz
COMMAND_STEPS = (2, 3)  # Arena steps per command, alternately: 0.04 s + 0.06 s = 2 x 0.05 s
HAND_FORCE_CONTACT_N = 0.0  # any reported apple-hand force counts as contact


def arena_steps(command_index: int) -> int:
    """Arena steps that hold the ``command_index``-th 20 Hz command (2, 3, 2, 3, ...)."""
    if int(command_index) < 0:
        raise ContractError("command index must be non-negative")
    return COMMAND_STEPS[int(command_index) % 2]


# ----- frames -------------------------------------------------------------------------------
def matrix_to_quat_xyzw(r) -> np.ndarray:
    """Rotation matrix to a unit quaternion (x, y, z, w) with w >= 0."""
    r = np.asarray(r, float)
    t = np.trace(r)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, s / 4]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4, (r[1, 0] - r[0, 1]) / s]
    q = np.asarray(q, float)
    q /= np.linalg.norm(q)
    return -q if q[3] < 0 else q


def rotation_angle(r) -> float:
    """Angle (rad) of a rotation matrix."""
    c = (np.trace(np.asarray(r, float)) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def yaw_of(r) -> float:
    r = np.asarray(r, float)
    return float(np.arctan2(r[1, 0], r[0, 0]))


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class FrameMap:
    """Arena env-local frame -> our MuJoCo world frame through the pelvis.

    ``x_ours = R_ours (R_arena^T (x_arena - p_arena)) + p_ours`` with the pelvis poses of both
    simulators; ours is upright at ``OUR_PELVIS_POS``. Every quantity written into the mirror
    is thereby relative to Arena's pelvis, as our fixed-pelvis kinematics assume."""

    def __init__(self, arena_pelvis_pose, our_pelvis_pos=OUR_PELVIS_POS):
        pose = np.asarray(arena_pelvis_pose, float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ContractError("pelvis pose must be 7 finite values (xyz, xyzw)")
        self.p_arena = pose[:3]
        self.r_arena = at.quat_xyzw_to_matrix(pose[3:])
        self.p_ours = np.asarray(our_pelvis_pos, float)

    def point(self, p) -> np.ndarray:
        return self.r_arena.T @ (np.asarray(p, float) - self.p_arena) + self.p_ours

    def vector(self, v) -> np.ndarray:
        return self.r_arena.T @ np.asarray(v, float)

    def rotation(self, r) -> np.ndarray:
        return self.r_arena.T @ np.asarray(r, float)

    def inverse_point(self, p) -> np.ndarray:
        return self.r_arena @ (np.asarray(p, float) - self.p_ours) + self.p_arena


def layout_in_arena(
    object_xy,
    plate_xy,
    *,
    arena_pelvis_pose,
    apple_com_offset_xy=(0.0, 0.0),
    shelf_top_z: float = ARENA_SHELF_TOP_Z,
) -> dict:
    """Arena env-local spawn positions for our v2 reset (apple and plate xy in our world).

    Our xy are relative to our pelvis; they are placed at the same offsets from Arena's settled
    pelvis, rotated by its yaw only (the shelf is horizontal). z is Arena's own flush spawn
    height on the shelf: the apple's USD origin ``ARENA_APPLE_ORIGIN_ABOVE_BOTTOM`` above the
    shelf plus the airgap, the plate (origin at its bottom) at the airgap. The apple's root is
    shifted by ``-apple_com_offset_xy`` (its centre-of-mass offset in xy) so that its centre of
    mass, which the mirror reads, lands on our xy."""
    pose = np.asarray(arena_pelvis_pose, float)
    yaw = yaw_of(at.quat_xyzw_to_matrix(pose[3:]))
    rz = rot_z(yaw)[:2, :2]
    out = {}
    for name, xy in (("apple", object_xy), ("plate", plate_xy)):
        xy = np.asarray(xy, float) - np.asarray(OUR_PELVIS_POS[:2])
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ContractError(f"{name} xy must be 2 finite values")
        out[name] = pose[:2] + rz @ xy
    apple = out["apple"] - np.asarray(apple_com_offset_xy, float)
    return {
        "apple": [*apple.tolist(), shelf_top_z + ARENA_AIRGAP + ARENA_APPLE_ORIGIN_ABOVE_BOTTOM],
        "plate": [*out["plate"].tolist(), shelf_top_z + ARENA_AIRGAP],
        "yaw_rad": yaw,
    }


def shelf_height_match(*, arena_pelvis_z: float, shelf_top_z: float = ARENA_SHELF_TOP_Z):
    """How far (m) the robot must stand higher so the shelf top is as far below its pelvis as
    our table top is below ours (5.3 cm); positive: raise the robot."""
    ours = OUR_PELVIS_POS[2] - OUR_TABLE_TOP_Z
    return float(ours - (arena_pelvis_z - shelf_top_z))


# ----- joints -------------------------------------------------------------------------------
def reorder(values, source_names, target_names) -> np.ndarray:
    """Values keyed by ``source_names`` order, returned in ``target_names`` order (by name)."""
    source = list(source_names)
    if sorted(source) != sorted(target_names) or len(set(source)) != len(source):
        raise ContractError("joint name sets differ")
    index = {n: i for i, n in enumerate(source)}
    values = np.asarray(values, float)
    return np.array([values[index[n]] for n in target_names])


def command_action(targets, mirror_names, arena_names) -> np.ndarray:
    """One 50-D Arena action from the mirror's 43 joint targets (MJCF actuator order).

    Mapped by name; navigate 0, base height ``at.STANDING_HEIGHT_M`` (0.75), torso RPY 0. The
    WBC uses the arm, wrist and Dex3 targets only (legs and waist come from AGILE / PD)."""
    targets = np.asarray(targets, float)
    if targets.shape != (len(mirror_names),) or not np.isfinite(targets).all():
        raise ContractError("targets must be finite and match the mirror's joints")
    return at.wbc_action(dict(zip(mirror_names, targets.tolist(), strict=True)), arena_names)


# ----- Arena state -> mirror state ----------------------------------------------------------
def mirror_state(raw: dict, *, arena_names, mirror_names, time: float) -> dict:
    """An ``isaac_e9.MirrorSimulation`` state message from one Arena raw state.

    Positions, rotations and velocities pass through ``FrameMap`` (Arena's live pelvis).
    The apple is its centre of mass. The plate body of our model sits at its base cylinder's
    centre, Arena's plate origin at its bottom, hence ``OUR_PLATE_BASE_HALF``. ``hand_contact``
    is Arena's own apple-hand contact sensor when present (else None: the mirror's geometric
    flag is used). ``contacts`` is empty (Arena reports no contact list here); ``pre`` is None
    (no ``mj_step`` layout: the mirror runs a plain forward pass of the new state)."""
    frame = FrameMap(raw["pelvis_pose"])
    com = np.asarray(raw["apple_com_pose"], float)
    link = np.asarray(raw["apple_pose"], float)
    rotation = frame.rotation(at.quat_xyzw_to_matrix(link[3:]))
    vel = np.asarray(raw["apple_com_vel"], float)
    plate = frame.point(np.asarray(raw["plate_pose"], float)[:3]) + [0, 0, OUR_PLATE_BASE_HALF]
    hand = raw.get("apple_hand_force_n")
    return {
        "time": float(time),
        "q": reorder(raw["q"], arena_names, mirror_names).tolist(),
        "qd": reorder(raw["qd"], arena_names, mirror_names).tolist(),
        "apple_pos": frame.point(com[:3]).tolist(),
        "apple_quat_xyzw": matrix_to_quat_xyzw(rotation).tolist(),
        "apple_lin_w": frame.vector(vel[:3]).tolist(),
        "apple_ang_w": frame.vector(vel[3:6]).tolist(),
        "plate_pos": plate.tolist(),
        "hand_contact": None if hand is None else bool(hand > HAND_FORCE_CONTACT_N),
        "contacts": [],
        "pre": None,
    }


# ----- the mirror ---------------------------------------------------------------------------
class ArenaMirrorSimulation(ie.MirrorSimulation):
    """``isaac_e9.MirrorSimulation`` for Arena: the remote layout is Arena's settled scene.

    Arena places the objects on its own shelf and lets them settle, so the remote apple and
    plate differ from the local reset by millimetres; the difference is recorded in
    ``layout_error_m`` instead of raising. Without Arena's apple-hand sensor (``hand_contact``
    None) the mirror's geometric flag stands in."""

    layout_error_m: dict | None = None

    def _apply(self, state: dict) -> None:
        if state.get("hand_contact") is None:
            state = {**state, "hand_contact": False, "_geometric_hand": True}
        super()._apply(state)
        if state.get("_geometric_hand"):
            self._remote["hand_contact"] = ie.apple_hand_contact(self)

    def _sync(self) -> None:
        if self._pending is None:
            return
        request, self._pending = self._pending, None
        local_apple = self.data.body("apple").xpos.copy()
        local_plate = self.model.body("plate").pos.copy()
        state = self.endpoint.reset(
            **request, joint_positions=self.data.qpos[self.qadr].astype(float).tolist()
        )
        self._apply(state)
        self.layout_error_m = {
            "apple": (self._remote["apple_pos"] - local_apple).tolist(),
            "plate": (self._remote["plate_pos"] - local_plate).tolist(),
        }
        self.targets[:] = self.data.qpos[self.qadr]


def make_arena_mirror_robot(endpoint, *, render=False):
    """``G1Embodiment`` over a v2 ``ArenaMirrorSimulation`` (as ``isaac_e9.make_mirror_robot``)."""
    from embodied_jepa import apple_to_plate_v2 as v2
    from embodied_jepa import first_policy as fp
    from embodied_jepa.embodiment import G1Embodiment

    sim = ArenaMirrorSimulation(
        endpoint,
        render=render,
        object_kind="apple",
        container_kind="plate",
        width=fp.IMAGE_SIZE,
        height=fp.IMAGE_SIZE,
    )
    scene = v2.apply_v2_scene(sim.model)  # geometry for contact detection only
    return G1Embodiment(sim), scene


class ArenaEndpoint:
    """The mirror's endpoint over an Arena server connection (``call(cmd, **kw) -> dict``).

    ``reset``: the server resets the env, holds our initial joint pose for ``settle_steps``,
    and reports the settled pelvis; the objects are then placed at our layout relative to it
    (``layout_in_arena``) and settle for ``place_steps``. ``step``: one 20 Hz command as 2 or 3
    Arena steps. Every Arena step's raw record (success term, forces, apple, plate, pelvis) is
    kept in ``records`` for the Arena-side verdicts; ``records`` restarts at each reset."""

    def __init__(self, call, mirror_names, *, settle_steps=75, place_steps=50):
        self.call = call
        hello = call("hello")
        self.arena_names = list(hello["joint_names"])
        self.mirror_names = list(mirror_names)
        if sorted(self.arena_names) != sorted(self.mirror_names):
            raise ContractError("Arena and mirror joint names differ")
        self.hello = hello
        self.settle_steps = int(settle_steps)
        self.place_steps = int(place_steps)
        self.commands = 0
        self.time = 0.0
        self.records: list[dict] = []
        self.reset_info: dict = {}
        self._hold = None

    def _state(self, raw) -> dict:
        return mirror_state(
            raw, arena_names=self.arena_names, mirror_names=self.mirror_names, time=self.time
        )

    def reset(self, *, seed, object_xy, plate_xy, object_on_container, joint_positions):
        if object_on_container:
            raise ContractError("the Arena layout places the apple on the shelf")
        hold = np.asarray(joint_positions, float)
        action = command_action(hold, self.mirror_names, self.arena_names)
        by_name = dict(zip(self.mirror_names, hold.tolist(), strict=True))  # recorded
        settled = self.call(
            "settle", action=action.tolist(), steps=self.settle_steps, joint_positions=by_name
        )
        raw = settled["raw"]
        offset = np.asarray(raw["apple_com_pose"][:2]) - np.asarray(raw["apple_pose"][:2])
        layout = layout_in_arena(
            object_xy, plate_xy, arena_pelvis_pose=raw["pelvis_pose"], apple_com_offset_xy=offset
        )
        placed = self.call(
            "place",
            apple=layout["apple"],
            plate=layout["plate"],
            action=action.tolist(),
            steps=self.place_steps,
            joint_positions=by_name,
        )
        self.commands, self.time, self.records = 0, 0.0, []
        self._hold = action
        self.reset_info = {
            "seed": int(seed),
            "settled_pelvis_pose": list(map(float, raw["pelvis_pose"])),
            "layout": layout,
            "apple_com_offset_xy": offset.tolist(),
            "placed_raw": _scalars(placed["raw"]),
            "place_records": placed.get("records", []),
        }
        return self._state(placed["raw"])

    def step(self, targets, deadline):
        if not np.isfinite(deadline) or deadline < self.time:
            return {
                "ack": {"status": "rejected", "reason": "command deadline expired"},
                "state": self._state(self.call("read")["raw"]),
            }
        n = arena_steps(self.commands)
        action = command_action(targets, self.mirror_names, self.arena_names)
        reply = self.call("step", action=action.tolist(), steps=n)
        self.commands += 1
        self.time = round(self.time + n * ARENA_DT, 9)
        for r in reply["records"]:
            self.records.append({**r, "command": self.commands - 1})
        return {"ack": {"status": "applied"}, "state": self._state(reply["raw"])}

    def stop(self, reason):
        return self._state(self.call("read")["raw"])

    def close(self):
        self.call("close")


def _scalars(raw: dict) -> dict:
    return {
        k: (np.asarray(v, float).tolist() if not isinstance(v, int | float) else float(v))
        for k, v in raw.items()
        if k != "body_pose"
    }


# ----- Arena-side verdicts ------------------------------------------------------------------
def arena_verdicts(records, *, rest_height_m: float, window_steps: int = 20) -> dict:
    """Both verdicts on Arena's own state, from the per-Arena-step records of one attempt.

    * **Arena contact success** (``object_on_destination``): the success term was True at any
      Arena step (Arena would have terminated there). Its first step, and whether the apple
      touched the hand then, are reported.
    * **``apple_at_rest_v0`` on Arena's world state**: over the last ``window_steps`` *commands*
      (one record per command: its last Arena step), the apple's centre of mass within 4 cm of
      the plate origin in xy, within 1.2 cm of ``plate_z + rest_height_m`` in z (the resting
      height calibrated by a drop on the plate in the same run), speed <= 0.001 m/s, and no
      apple-hand force. World (env-local) frame, not pelvis-relative."""
    from embodied_jepa.at_rest import apple_at_rest

    if not records:
        raise ContractError("no Arena records")
    success = np.array([bool(r["success"]) for r in records])
    first = int(np.flatnonzero(success)[0]) if success.any() else None
    by_command: dict[int, dict] = {}
    for r in records:
        by_command[int(r["command"])] = r  # the command's last Arena step
    last = [by_command[k] for k in sorted(by_command)]
    obj = np.array([r["apple_com_pose"][:3] for r in last], float)
    plate = np.array([r["plate_pose"][:3] for r in last], float)
    vel = np.array([r["apple_com_vel"][:3] for r in last], float)
    hand = np.array(
        [float(r.get("apple_hand_force_n") or 0.0) > HAND_FORCE_CONTACT_N for r in last]
    )
    expected_z = plate[:, 2] + float(rest_height_m)
    flat_plate = plate.copy()
    flat_plate[:, 2] = 0.0
    at_rest = apple_at_rest(
        obj - np.c_[np.zeros((len(obj), 2)), expected_z],  # height error measured from 0
        flat_plate,
        vel,
        hand,
        expected_z=0.0,
    )
    return {
        "arena_success": bool(success.any()),
        "arena_first_success_step": first,
        "arena_first_success_command": None if first is None else int(records[first]["command"]),
        "arena_success_with_hand_contact": None
        if first is None
        else bool(float(records[first].get("apple_hand_force_n") or 0.0) > 0.0),
        "arena_success_inputs": None if first is None else records[first].get("success_inputs"),
        "at_rest_arena": bool(at_rest["at_rest"]),
        "at_rest_arena_detail": at_rest,
        "final_distance_arena_cm": float(np.linalg.norm(obj[-1, :2] - plate[-1, :2]) * 100),
        "dropped": bool(any(r.get("object_dropped") for r in records)),
        "window_steps": window_steps,
    }


# ----- kinematic check (step 1) -------------------------------------------------------------
ARM_SWEEPS = (  # (joint suffix, delta for the left arm; the right arm mirrors roll and yaw)
    ("shoulder_pitch_joint", 0.4),
    ("shoulder_roll_joint", 0.4),
    ("shoulder_yaw_joint", -0.4),
    ("elbow_joint", 0.8),
    ("wrist_roll_joint", 0.6),
    ("wrist_pitch_joint", 0.6),
    ("wrist_yaw_joint", 0.6),
)
MIRRORED = ("roll", "yaw")
FINGERS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")


def _ramp(k: int, start: int, length: int) -> float:
    x = min(max((k - start) / length, 0.0), 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


def kinematic_segments(default_q: dict, arena_limits: dict, closed: dict) -> list[dict]:
    """Free-space target poses for the kinematic check, as deltas from Arena's default pose.

    One segment per arm joint (both arms, mirrored), a combined arm pose, the Dex3 closed
    synergy (``closed``: joint name -> rad), and one segment per finger joint at 70 % of
    Arena's range (the intersection with the MJCF's is Arena's). Each target is clipped to
    Arena's limits."""
    segments = []

    def clip(q):
        return {n: float(np.clip(v, *arena_limits[n])) for n, v in q.items()}

    for suffix, delta in ARM_SWEEPS:
        q = dict(default_q)
        for side, sign in (("left", 1.0), ("right", -1.0)):
            s = sign if any(m in suffix for m in MIRRORED) else 1.0
            q[f"{side}_{suffix}"] += s * delta
        segments.append({"name": f"arm_{suffix.removesuffix('_joint')}", "q": clip(q)})
    q = dict(default_q)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        q[f"{side}_shoulder_pitch_joint"] += -0.3
        q[f"{side}_shoulder_roll_joint"] += sign * 0.6
        q[f"{side}_elbow_joint"] += 1.0
        q[f"{side}_wrist_pitch_joint"] += 0.4
        q[f"{side}_wrist_yaw_joint"] += sign * 0.3
    segments.append({"name": "arm_combined", "q": clip(q)})
    segments.append({"name": "hands_closed_synergy", "q": clip({**default_q, **closed})})
    for finger in FINGERS:
        q = dict(default_q)
        for side in ("left", "right"):
            name = f"{side}_hand_{finger}_joint"
            lo, hi = arena_limits[name]
            q[name] = lo + 0.7 * (hi - lo) if abs(hi) >= abs(lo) else hi + 0.7 * (lo - hi)
        segments.append({"name": f"finger_{finger}", "q": clip(q)})
    return segments


def kinematic_trajectory(default_q: dict, segments, *, settle=50, ramp=25, hold=25):
    """Per-Arena-step targets (name -> rad) and segment labels: settle at the default pose,
    then for each segment ramp out, hold, ramp back, hold."""
    names = list(default_q)
    base = np.array([default_q[n] for n in names])
    rows, labels = [], []
    for _ in range(settle):
        rows.append(base.copy())
        labels.append("settle")
    for seg in segments:
        goal = np.array([seg["q"][n] for n in names])
        span = 2 * ramp + 2 * hold
        for k in range(span):
            a = _ramp(k, 0, ramp) - _ramp(k, ramp + hold, ramp)
            rows.append(base + a * (goal - base))
            labels.append(seg["name"] + (":hold" if ramp <= k < ramp + hold else ""))
    return names, np.array(rows), labels


def relative_pose(body_pose, pelvis_pose) -> tuple[np.ndarray, np.ndarray]:
    """A body's position and rotation in the pelvis frame (poses: xyz + xyzw)."""
    rp = at.quat_xyzw_to_matrix(np.asarray(pelvis_pose, float)[3:])
    rb = at.quat_xyzw_to_matrix(np.asarray(body_pose, float)[3:])
    p = rp.T @ (np.asarray(body_pose, float)[:3] - np.asarray(pelvis_pose, float)[:3])
    return p, rp.T @ rb


def mujoco_fk(model, data, qadr, mirror_names, q_by_name: dict, bodies) -> dict:
    """Pelvis-relative position and rotation of ``bodies`` in our MJCF at the given joints.

    ``data`` is scratch (overwritten); ``qadr`` maps ``mirror_names`` to qpos addresses."""
    import mujoco

    for name, adr in zip(mirror_names, qadr, strict=True):
        data.qpos[adr] = q_by_name[name]
    mujoco.mj_kinematics(model, data)
    pelvis = data.body("pelvis")
    rp = pelvis.xmat.reshape(3, 3)
    out = {}
    for b in bodies:
        body = data.body(b)
        out[b] = (rp.T @ (body.xpos - pelvis.xpos), rp.T @ body.xmat.reshape(3, 3))
    return out


def pose_difference(a, b) -> tuple[float, float]:
    """(position difference in m, rotation difference in rad) between two (p, R) poses."""
    return float(np.linalg.norm(a[0] - b[0])), rotation_angle(a[1].T @ b[1])


# ----- plumbing check -----------------------------------------------------------------------
class MuJoCoArenaSham:
    """The Arena server protocol over a host MuJoCo v2 scene, in a displaced "Arena" frame.

    A plumbing check of ``ArenaEndpoint`` + ``ArenaMirrorSimulation`` (frames, joint order,
    layout): the scene's pelvis is reported at ``pelvis_pose`` (an Arena-like settled pose with
    a yaw), joints in reversed MJCF order, the plate origin at its bottom. Each command steps
    MuJoCo once (one 0.05 s control interval) however many Arena steps it asks for, so the
    mirror must reproduce a plain MuJoCo run up to the float32 rounding of Arena's action."""

    def __init__(self, pelvis_pose=(0.262, 0.074, -0.045, 0.0, 0.0, 0.0499792, 0.9987503)):
        self.endpoint = ie.MuJoCoEndpoint()
        self.sim = self.endpoint.sim
        self.mirror_names = list(self.sim.joint_names)
        self.joint_names = list(reversed(self.mirror_names))
        self.pelvis_pose = np.asarray(pelvis_pose, float)
        self.frame = FrameMap(self.pelvis_pose)

    def _to_arena(self, p):
        return self.frame.inverse_point(p)

    def _targets(self, action):
        return reorder(
            np.asarray(action, float)[: at.G1_NUM_JOINTS], self.joint_names, self.mirror_names
        )

    def raw(self) -> dict:
        sim, d = self.sim, self.sim.data
        joint = d.joint("apple_free")
        w, x, y, z = joint.qpos[3:7]
        r_apple = self.frame.r_arena @ at.quat_xyzw_to_matrix([x, y, z, w])
        apple = self._to_arena(joint.qpos[:3])
        q_apple = matrix_to_quat_xyzw(r_apple)
        r_world = at.quat_xyzw_to_matrix([x, y, z, w])
        plate = self._to_arena(sim.model.body("plate").pos - [0, 0, OUR_PLATE_BASE_HALF])
        return {
            "q": reorder(d.qpos[sim.qadr], self.mirror_names, self.joint_names),
            "qd": reorder(d.qvel[sim.vadr], self.mirror_names, self.joint_names),
            "pelvis_pose": self.pelvis_pose.copy(),
            "apple_pose": np.r_[apple, q_apple],
            "apple_com_pose": np.r_[apple, q_apple],
            "apple_com_vel": np.r_[
                self.frame.r_arena @ joint.qvel[:3],
                self.frame.r_arena @ (r_world @ joint.qvel[3:6]),
            ],
            "plate_pose": np.r_[plate, 0.0, 0.0, 0.0, 1.0],
            "apple_plate_force_n": 0.0,
            "apple_hand_force_n": 1.0 if sim.task_truth()["hand_contact"] else 0.0,
        }

    def _record(self):
        raw = self.raw()
        return {
            "success": False,
            "object_dropped": False,
            "success_inputs": None,
            **{
                k: np.asarray(raw[k]).tolist()
                for k in ("pelvis_pose", "apple_com_pose", "apple_com_vel", "plate_pose")
            },
            "apple_plate_force_n": 0.0,
            "apple_hand_force_n": raw["apple_hand_force_n"],
        }

    def call(self, cmd, **kw) -> dict:
        sim = self.sim
        if cmd == "hello":
            return {"joint_names": self.joint_names, "physics": "mujoco (sham)"}
        if cmd == "settle":
            self._hold = self._targets(kw["action"])
            return {"raw": self.raw()}
        if cmd == "place":
            ours = []
            for key in ("apple", "plate"):
                p = self.frame.point([kw[key][0], kw[key][1], self.pelvis_pose[2]])
                ours.append(p[:2].tolist())
            sim.reset(0, object_xy=ours[0], plate_xy=ours[1])
            q = kw["joint_positions"]  # float64; the action is float32-rounded
            sim.data.qpos[sim.qadr] = [q[n] for n in self.mirror_names]
            sim.mj.mj_forward(sim.model, sim.data)
            sim.targets[:] = sim.data.qpos[sim.qadr]
            return {"raw": self.raw(), "records": [self._record()]}
        if cmd == "step":
            sim.send_joint_targets(
                self._targets(kw["action"]),
                joint_names=sim.joint_names,
                deadline=float(sim.data.time + sim.control_dt),
            )
            rec = self._record()
            return {"raw": self.raw(), "records": [rec] * int(kw["steps"])}
        if cmd == "read":
            return {"raw": self.raw()}
        if cmd in ("info", "close"):
            return {}
        raise ContractError(f"unknown command {cmd!r}")
