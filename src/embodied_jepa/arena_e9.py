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
# Arena's IdealPDActuator stiffness (N m/rad) for the upper body (read back in arena-probe-2;
# isaaclab_arena g1 embodiment cfg). Used only by the opt-in gravity offset.
ARENA_UPPER_KP = {
    "shoulder_pitch": 100.0,
    "shoulder_roll": 100.0,
    "shoulder_yaw": 40.0,
    "elbow": 40.0,
    "wrist_roll": 20.0,
    "wrist_pitch": 20.0,
    "wrist_yaw": 20.0,
    "hand_": 4.0,
}


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
def mirror_state(
    raw: dict, *, arena_names, mirror_names, time: float, zero_leg_velocity: bool = False
) -> dict:
    """An ``isaac_e9.MirrorSimulation`` state message from one Arena raw state.

    Positions, rotations and velocities pass through ``FrameMap`` (Arena's live pelvis).
    The apple is its centre of mass. The plate body of our model sits at its base cylinder's
    centre, Arena's plate origin at its bottom, hence ``OUR_PLATE_BASE_HALF``. ``hand_contact``
    is Arena's own apple-hand contact sensor when present (else None: the mirror's geometric
    flag is used). ``contacts`` is empty (Arena reports no contact list here); ``pre`` is None
    (no ``mj_step`` layout: the mirror runs a plain forward pass of the new state).
    ``zero_leg_velocity`` (e9-arena only, docs/ARENA.md §8) writes the WBC's leg joint
    velocities as 0: with the pelvis fixed in the mirror they enter neither FK, IK nor the arm's
    bias, only the embodiment's 5 rad/s guard, which was written for our fixed-pelvis robot."""
    frame = FrameMap(raw["pelvis_pose"])
    qd = reorder(raw["qd"], arena_names, mirror_names)
    if zero_leg_velocity:
        qd = np.where([is_leg_joint(n) for n in mirror_names], 0.0, qd)
    com = np.asarray(raw["apple_com_pose"], float)
    link = np.asarray(raw["apple_pose"], float)
    rotation = frame.rotation(at.quat_xyzw_to_matrix(link[3:]))
    vel = np.asarray(raw["apple_com_vel"], float)
    plate = frame.point(np.asarray(raw["plate_pose"], float)[:3]) + [0, 0, OUR_PLATE_BASE_HALF]
    hand = raw.get("apple_hand_force_n")
    return {
        "time": float(time),
        "gravity": frame.vector([0.0, 0.0, -9.81]).tolist(),
        "q": reorder(raw["q"], arena_names, mirror_names).tolist(),
        "qd": qd.tolist(),
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
        if state.get("gravity") is not None:  # Arena's gravity in its pelvis frame
            self.model.opt.gravity[:] = state["gravity"]
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


def upper_body_kp(names) -> np.ndarray:
    """Arena's PD stiffness per joint (0 for legs and waist, which the WBC drives)."""
    out = np.zeros(len(names))
    for i, n in enumerate(names):
        for key, kp in ARENA_UPPER_KP.items():
            if key in n:
                out[i] = kp
    return out


def gravity_offset(sim) -> np.ndarray:
    """Position-target offsets that give Arena's PD our actuator's bias torque (arm and hand).

    Our MuJoCo actuator is PD plus bias compensation (``MuJoCoSimulation``: torque =
    kp (target - q) - kd qd + qfrc_bias); Arena's ``IdealPDActuator`` has no bias term, so the
    same targets sag. Adding ``qfrc_bias / kp_arena`` to the target gives Arena's PD the same
    bias torque. ``qfrc_bias`` is gravity **plus** Coriolis and centrifugal terms at Arena's
    measured joint velocities, as in our actuator; gravity dominates at e9's slow speeds. It is
    our MJCF's, at Arena's joint state, with gravity in Arena's pelvis frame (``mirror_state``'s
    ``gravity``) and the pelvis treated as fixed; legs and waist get no offset."""
    kp = upper_body_kp(sim.joint_names)
    bias = np.asarray(sim.data.qfrc_bias[sim.vadr], float)
    return np.divide(bias, kp, out=np.zeros_like(bias), where=kp > 0)


def make_arena_mirror_robot(endpoint, *, render=False, gravity_offset_targets=False):
    """``G1Embodiment`` over a v2 ``ArenaMirrorSimulation`` (as ``isaac_e9.make_mirror_robot``).

    ``gravity_offset_targets`` adds ``gravity_offset`` to every command sent to Arena (the
    mirror and e9 still see and record the unshifted targets)."""
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
    if gravity_offset_targets:
        endpoint.feedforward = lambda: gravity_offset(sim)
    return G1Embodiment(sim), scene


class ArenaEndpoint:
    """The mirror's endpoint over an Arena server connection (``call(cmd, **kw) -> dict``).

    ``reset``: the server resets the env, holds our initial joint pose for ``settle_steps``,
    and reports the settled pelvis; the objects are then placed at our layout relative to it
    (``layout_in_arena``) and settle for ``place_steps``. ``step``: one 20 Hz command as 2 or 3
    Arena steps. Every Arena step's raw record (success term, forces, apple, plate, pelvis) is
    kept in ``records`` for the Arena-side verdicts; ``records`` restarts at each reset."""

    def __init__(
        self, call, mirror_names, *, settle_steps=75, place_steps=50, zero_leg_velocity=False
    ):
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
        self.feedforward = None  # optional: () -> offsets added to each command's targets
        self.offsets: list[np.ndarray] = []
        self.zero_leg_velocity = bool(zero_leg_velocity)
        self.last_raw: dict | None = None  # Arena's latest raw state (e9-arena's live truth)
        self.leg_speed_max = 0.0  # the largest raw leg joint speed since the reset (rad/s)

    def live(self) -> dict:
        """Arena's latest raw state plus ``_command``, the index of the next command (privileged
        truth for e9-arena and the probe; never a policy input)."""
        if self.last_raw is None:
            raise ContractError("no Arena state yet")
        return {**self.last_raw, "_command": self.commands}

    def _state(self, raw) -> dict:
        self.last_raw = raw
        legs = [i for i, n in enumerate(self.arena_names) if is_leg_joint(n)]
        self.leg_speed_max = max(
            self.leg_speed_max, float(np.abs(np.asarray(raw["qd"], float)[legs]).max())
        )
        return mirror_state(
            raw,
            arena_names=self.arena_names,
            mirror_names=self.mirror_names,
            time=self.time,
            zero_leg_velocity=self.zero_leg_velocity,
        )

    def reset(self, *, seed, object_xy, plate_xy, object_on_container, joint_positions):
        if object_on_container:
            raise ContractError("the Arena layout places the apple on the shelf")
        hold = np.asarray(joint_positions, float)
        # With the gravity offset, the initial pose is held against gravity too (as our MuJoCo
        # actuator does); without it Arena's arms sag onto the apple's reset position.
        hold_offset = (
            np.asarray(self.feedforward(), float) if self.feedforward is not None else 0.0 * hold
        )
        action = command_action(hold + hold_offset, self.mirror_names, self.arena_names)
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
        self.commands, self.time, self.records, self.offsets = 0, 0.0, [], []
        self.leg_speed_max = 0.0
        self._hold = action
        prc = placed["raw"]
        target_com = np.asarray(layout["apple"][:2]) + offset
        self.reset_info = {
            "placement_error_xy_m": float(
                np.linalg.norm(np.asarray(prc["apple_com_pose"][:2]) - target_com)
            ),
            "placement_apple_speed_m_s": float(np.linalg.norm(prc["apple_com_vel"][:3])),
            "placement_apple_hand_force_n": prc.get("apple_hand_force_n"),
            "hold_offset_max_rad": float(np.abs(hold_offset).max()),
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
        targets = np.asarray(targets, float)
        if self.feedforward is not None:
            offset = np.asarray(self.feedforward(), float)
            self.offsets.append(offset)
            targets = targets + offset
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
        k: (
            v
            if v is None or isinstance(v, str)
            else float(v)
            if isinstance(v, int | float)
            else np.asarray(v, float).tolist()
        )
        for k, v in raw.items()
        if k != "body_pose"
    }


# ----- Arena-side verdicts ------------------------------------------------------------------
def arena_verdicts(records, *, rest_height_m: float, thresholds=None) -> dict:
    """Both verdicts on Arena's own state, from the per-Arena-step records of one attempt.

    * **Arena contact success** (``object_on_destination``): the success term was True at any
      Arena step (Arena would have terminated there). Its first step, and whether the apple
      touched the hand then, are reported.
    * **``apple_at_rest_v0`` on Arena's world state**: over the last ``window_steps`` *commands*
      of ``thresholds`` (default ``AtRestThresholds``: 20; one record per command: its last
      Arena step), the apple's centre of mass within 4 cm of
      the plate origin in xy, within 1.2 cm of ``plate_z + rest_height_m`` in z (the resting
      height calibrated by a drop on the plate in the same run), speed <= 0.001 m/s, and no
      apple-hand force. World (env-local) frame, not pelvis-relative. The speed is the
      centre of mass's displacement over each command interval (2 or 3 Arena steps) divided by
      its duration: PhysX reports a non-zero velocity for an apple whose pose does not change
      (about 5-10 mm/s at rest on the plate in the calibration drops), so the reported
      velocity is used only for a second, labelled reading (``at_rest_arena_reported_vel``)."""
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
    commands = sorted(by_command)
    dt = np.array([arena_steps(k) * ARENA_DT for k in commands])
    fd = np.zeros_like(obj)
    fd[1:] = np.diff(obj, axis=0) / dt[1:, None]
    hand = np.array(
        [float(r.get("apple_hand_force_n") or 0.0) > HAND_FORCE_CONTACT_N for r in last]
    )
    expected_z = plate[:, 2] + float(rest_height_m)
    flat_plate = plate.copy()
    flat_plate[:, 2] = 0.0
    shifted = obj - np.c_[np.zeros((len(obj), 2)), expected_z]  # height error from 0
    at_rest = apple_at_rest(shifted, flat_plate, fd, hand, expected_z=0.0, thresholds=thresholds)
    reported = apple_at_rest(shifted, flat_plate, vel, hand, expected_z=0.0, thresholds=thresholds)
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
        "at_rest_arena_reported_vel": bool(reported["at_rest"]),
        "at_rest_arena_reported_vel_detail": reported,
        "final_distance_arena_cm": float(np.linalg.norm(obj[-1, :2] - plate[-1, :2]) * 100),
        "dropped": bool(any(r.get("object_dropped") for r in records)),
        "window_steps": int(at_rest["window_steps"]),
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


# ----- e9-arena: the Arena-adapted e9 (docs/ARENA.md §8) -------------------------------------
# Development only, declared in docs/ARENA.md §8 before any Arena run of it. It is a privileged
# scripted expert, **not e9**: always label it "e9-arena". Its targets are recomputed every
# command from Arena's live state (pelvis, apple, plate and, for the shelf-clearance close, the
# right hand's lowest collision point), which is privileged truth, as e9's reset truth is.
E9_ARENA = "e9-arena"
E9_ARENA_VERSION = "e9_arena_v1"
LEG_JOINT_KEYS = ("hip_", "knee_", "ankle_")  # the WBC's leg joints (AGILE drives them)
E9_PICK_DX = -0.015  # e9: the collector's -0.03 plus palm_x_offset 0.015 (pelvis x)
E9_ORIENT_DZ = 0.13
E9_LIFT_DZ = 0.21
# Measured in Arena by the shelf-press probe (docs/ARENA.md §8.1, arena-shelf-probe-1)
ARENA_FINGER_DROP_M = 0.1182  # palm site to the lowest right-hand collision point, open
ARENA_APPLE_COM_ABOVE_SHELF_M = 0.0263  # resting apple centre above the shelf top


def declared_variant(name: str, clearance_m: float, close_mode: str, close_ramp=None) -> dict:
    """An e9-arena tuning variant of docs/ARENA.md §8.2 (h_stop from the measured geometry)."""
    return {
        "name": name,
        "stop_height_m": stop_height(
            finger_drop_m=ARENA_FINGER_DROP_M,
            apple_com_above_shelf_m=ARENA_APPLE_COM_ABOVE_SHELF_M,
            clearance_m=clearance_m,
        ),
        "shelf_clearance_m": float(clearance_m),
        "close_mode": close_mode,
        "close_ramp": close_ramp,
    }


E9_REACH = {"reach_radius_m": 0.485, "release_z_floor_m": 0.10, "release_z_ceiling_m": 0.26}
PHASE_NAMES = (
    "orient",
    "descend",
    "close",
    "lift",
    "transfer",
    "lower",
    "steady",
    "open",
    "clear",
    "retreat",
)


def is_leg_joint(name: str) -> bool:
    return any(k in name for k in LEG_JOINT_KEYS)


def heading(pelvis_pose) -> np.ndarray:
    """The pelvis's forward (x) axis projected on the horizontal plane, unit length."""
    r = at.quat_xyzw_to_matrix(np.asarray(pelvis_pose, float)[3:])
    f = np.array([r[0, 0], r[1, 0], 0.0])
    n = np.linalg.norm(f)
    if n < 1e-6:
        raise ContractError("the pelvis's x axis is vertical")
    return f / n


def world_to_base(p_world, pelvis_pose) -> np.ndarray:
    """An Arena env-local point in e9's base frame (our pelvis frame: the mirror minus its base)."""
    return FrameMap(pelvis_pose).point(p_world) - np.asarray(OUR_PELVIS_POS)


def base_to_world(p_base, pelvis_pose) -> np.ndarray:
    return FrameMap(pelvis_pose).inverse_point(np.asarray(p_base, float) + OUR_PELVIS_POS)


def anchored_target(anchor_w, pelvis_pose, *, dx: float, dz: float) -> np.ndarray:
    """e9's palm target ``anchor + dx`` along the pelvis heading ``+ dz`` up, in the base frame.

    The offsets are applied in the world (``dz`` along gravity, ``dx`` along the horizontal
    heading), then the point is mapped through Arena's live pelvis. With an upright, fixed pelvis
    this is exactly e9's ``obj + [dx, 0, dz]``."""
    p = np.asarray(anchor_w, float) + dx * heading(pelvis_pose) + [0.0, 0.0, dz]
    return world_to_base(p, pelvis_pose)


def stop_height(*, finger_drop_m: float, apple_com_above_shelf_m: float, clearance_m: float):
    """e9-arena's descent stop: the palm (``*_ee``) height above the apple centre at which Arena's
    lowest open-hand collision point is ``clearance_m`` above the shelf top.

    ``finger_drop_m`` is how far that lowest point hangs below the palm site in e9's palm-down,
    open-hand pose, measured in Arena (the shelf-press probe); ``apple_com_above_shelf_m`` is
    Arena's resting apple centre above the shelf top."""
    for name, v in (
        ("finger_drop_m", finger_drop_m),
        ("apple_com_above_shelf_m", apple_com_above_shelf_m),
        ("clearance_m", clearance_m),
    ):
        if not np.isfinite(v) or v < 0:
            raise ContractError(f"{name} must be finite and non-negative")
    return float(finger_drop_m + clearance_m - apple_com_above_shelf_m)


class ArenaAdaptedE9:
    """e9-arena: TASK-070's e9 with the changes declared in docs/ARENA.md §8, nothing else.

    It wraps e9 (``resting_expert.RestingPlaceExpert`` with ``isaac_e9.E9``: same phases, same
    command counts, same grasp and opening schedule, same clips, rotations and release rule) and
    rewrites only the current phase's palm target before each command:

    1. **descent stop**: descend and close aim at ``stop_height_m`` above the apple centre
       instead of e9's 0.052 m (which e9 never reaches in MuJoCo: its thumb lands on the table);
    2. **live targets** (privileged): every command, the targets are recomputed from Arena's
       live pelvis, apple and plate (``live()``: the server's last raw state). Orient and
       descend follow the live apple; at the first close command the apple's world position is
       frozen as the grasp anchor, which close and lift use; transfer to retreat use e9's
       release rule on the live plate. Offsets are applied in the world (``anchored_target``);
    3. (the gravity offset is the endpoint's, unchanged);
    4. **close** (``close_mode``): ``"hold"`` keeps the descent stop height; ``"shelf_servo"``
       moves the palm so that Arena's lowest right-hand collision point stays
       ``shelf_clearance_m`` above the shelf while the fingers close (MuJoCo's hand follows the
       table down during e9's close; this follows it without pressing). ``close_ramp`` (per
       command, optional) ramps the grasp scalar instead of e9's single step."""

    def __init__(
        self,
        initial_truth,
        *,
        live,
        stop_height_m: float,
        close_mode: str = "hold",
        shelf_clearance_m: float = 0.01,
        shelf_top_z: float = ARENA_SHELF_TOP_Z,
        close_ramp: float | None = None,
    ):
        from embodied_jepa import resting_expert as rx

        if close_mode not in ("hold", "shelf_servo"):
            raise ContractError("close_mode must be 'hold' or 'shelf_servo'")
        if close_ramp is not None and not 0 < close_ramp <= 2:
            raise ContractError("close_ramp must lie in (0, 2]")
        if not np.isfinite(stop_height_m):
            raise ContractError("stop height must be finite")
        self.e9_truth = initial_truth
        self.e9 = rx.RestingPlaceExpert(initial_truth, **ie.E9)
        names = tuple(p.name for p in self.e9.phases)
        if names != PHASE_NAMES:
            raise ContractError(f"unexpected e9 phases {names}")
        self.live = live
        self.stop_height_m = float(stop_height_m)
        self.close_mode = close_mode
        self.shelf_clearance_m = float(shelf_clearance_m)
        self.shelf_top_z = float(shelf_top_z)
        self.close_ramp = None if close_ramp is None else float(close_ramp)
        self.anchor_w: np.ndarray | None = None
        self.log: list[dict] = []

    # e9's interface (run_attempt reads these)
    @property
    def phases(self):
        return self.e9.phases

    @property
    def phase_index(self):
        return self.e9.phase_index

    @property
    def done(self):
        return self.e9.done

    @property
    def max_steps(self):
        return self.e9.max_steps

    @property
    def step_count(self):
        return self.e9.step_count

    @property
    def release_target(self):
        return self.e9.release_target

    def target(self, name: str, raw: dict, ee_base) -> np.ndarray:
        """The palm target (base frame) of phase ``name`` on Arena's live state ``raw``."""
        from embodied_jepa import resting_expert as rx

        pelvis = raw["pelvis_pose"]
        apple = np.asarray(raw["apple_com_pose"], float)[:3]
        if name == "orient":
            return anchored_target(apple, pelvis, dx=E9_PICK_DX, dz=E9_ORIENT_DZ)
        if name == "descend":
            return anchored_target(apple, pelvis, dx=E9_PICK_DX, dz=self.stop_height_m)
        if self.anchor_w is None:  # the first close command freezes the grasp anchor
            self.anchor_w = apple.copy()
        if name == "close":
            hold = anchored_target(self.anchor_w, pelvis, dx=E9_PICK_DX, dz=self.stop_height_m)
            if self.close_mode == "hold":
                return hold
            low = raw.get("hand_min_z")
            if low is None:
                raise ContractError("shelf_servo needs the server's hand geometry")
            ee_w = base_to_world(ee_base, pelvis)
            z = ee_w[2] + (self.shelf_top_z + self.shelf_clearance_m - float(low))
            p = self.anchor_w + E9_PICK_DX * heading(pelvis)
            return world_to_base([p[0], p[1], z], pelvis)
        lift = anchored_target(self.anchor_w, pelvis, dx=E9_PICK_DX, dz=E9_LIFT_DZ)
        if name == "lift":
            return lift
        plate = world_to_base(np.asarray(raw["plate_pose"], float)[:3], pelvis)
        release = rx.RestingPlaceExpert.release_pose(
            plate[:2] + [ie.E9["release_dx"], 0.0],
            reach_radius_m=E9_REACH["reach_radius_m"],
            floor=E9_REACH["release_z_floor_m"],
            ceiling=E9_REACH["release_z_ceiling_m"],
        )
        if name == "transfer":
            return np.array([release[0], release[1], max(lift[2], release[2])])
        if name in ("lower", "steady", "open", "clear"):
            return release  # e9's open_dx_m is 0
        if name == "retreat":
            return release + [0.0, 0.0, 0.08]
        raise ContractError(f"unknown phase {name!r}")

    def action(self, robot):
        from dataclasses import replace

        i = self.e9.phase_index
        phase = self.e9.phases[i]
        raw = self.live()
        ee_base, _ = robot.ee_pose("right")
        target = self.target(phase.name, raw, ee_base)
        phases = list(self.e9.phases)
        phases[i] = replace(phase, target_base=np.asarray(target, float))
        self.e9.phases = tuple(phases)
        action = self.e9.action(robot)
        if self.close_ramp is not None and phase.name == "close":
            action[13] = min(float(action[13]), self.e9.accepted_grasp + self.close_ramp)
        self.log.append(
            {
                "phase": phase.name,
                "command": raw.get("_command"),
                "target_base": np.asarray(target).tolist(),
            }
        )
        return action

    def advance(self, result):
        self.e9.advance(result)


# ----- the shelf-press probe (docs/ARENA.md §8.1) ---------------------------------------------
PROBE_PHASES = (  # (name, commands, grasp): e9's orient, descend and close counts, then a hold
    ("orient", 130, -1.0),
    ("descend", 80, -1.0),
    ("close", 45, 1.0),
    ("hold", 100, 1.0),
    ("retreat", 60, -1.0),
)
PROBE_MODES = ("press", "hover", "high")
E9_DESCEND_DZ = 0.052  # e9's descend and close target above the apple centre


class ShelfPressProbe:
    """Empty-hand probe: does the WBC step when the right hand presses the shelf?

    e9's palm-down approach to ``site_w`` (Arena env-local: where the apple centre would be;
    the apple is placed out of reach), then e9's descend and close command counts and a
    100-command closed hold:

    * ``press``: e9's own descend/close target (0.052 m above the apple centre), which puts the
      hand on the shelf as e9 did in Arena;
    * ``hover``: the palm is servoed so that Arena's lowest right-hand collision point stays
      ``clearance_m`` above the shelf top (live ``hand_min_z``);
    * ``high``: the palm stays at e9's orient height (0.13 m above the site).

    Targets are world-anchored and mapped through the live pelvis each command (as e9-arena).
    ``log`` keeps, per command, the palm's world height, the hand's lowest point and the pelvis."""

    def __init__(
        self,
        initial_truth,
        *,
        live,
        site_w,
        mode: str,
        clearance_m: float = 0.01,
        shelf_top_z: float = ARENA_SHELF_TOP_Z,
    ):
        from embodied_jepa.scripted import OracleManipulationPolicy, OraclePhase

        if mode not in PROBE_MODES:
            raise ContractError(f"mode must be one of {PROBE_MODES}")
        self.p = OracleManipulationPolicy(initial_truth)
        self.p.phases = tuple(OraclePhase(n, np.zeros(3), g, c) for n, c, g in PROBE_PHASES)
        self.live = live
        self.site_w = np.asarray(site_w, float)
        if self.site_w.shape != (3,) or not np.isfinite(self.site_w).all():
            raise ContractError("site must be 3 finite values (Arena env-local)")
        self.mode = mode
        self.clearance_m = float(clearance_m)
        self.shelf_top_z = float(shelf_top_z)
        self.log: list[dict] = []

    phases = property(lambda self: self.p.phases)
    phase_index = property(lambda self: self.p.phase_index)
    done = property(lambda self: self.p.done)
    max_steps = property(lambda self: self.p.max_steps)
    step_count = property(lambda self: self.p.step_count)

    def target(self, name: str, raw: dict, ee_base) -> np.ndarray:
        pelvis = raw["pelvis_pose"]
        high = anchored_target(self.site_w, pelvis, dx=E9_PICK_DX, dz=E9_ORIENT_DZ)
        if name in ("orient", "retreat") or self.mode == "high":
            return high
        if self.mode == "press":
            return anchored_target(self.site_w, pelvis, dx=E9_PICK_DX, dz=E9_DESCEND_DZ)
        low = raw.get("hand_min_z")
        if low is None:
            raise ContractError("hover needs the server's hand geometry")
        ee_w = base_to_world(ee_base, pelvis)
        z = ee_w[2] + (self.shelf_top_z + self.clearance_m - float(low))
        p = self.site_w + E9_PICK_DX * heading(pelvis)
        return world_to_base([p[0], p[1], z], pelvis)

    def action(self, robot):
        from dataclasses import replace

        i = self.p.phase_index
        phase = self.p.phases[i]
        raw = self.live()
        ee_base, _ = robot.ee_pose("right")
        target = self.target(phase.name, raw, ee_base)
        phases = list(self.p.phases)
        phases[i] = replace(phase, target_base=np.asarray(target, float))
        self.p.phases = tuple(phases)
        self.log.append(
            {
                "phase": phase.name,
                "command": raw.get("_command"),
                "ee_world": base_to_world(ee_base, raw["pelvis_pose"]).tolist(),
                "hand_min_z": raw.get("hand_min_z"),
                "hand_min_link": raw.get("hand_min_link"),
                "pelvis_pose": list(map(float, raw["pelvis_pose"])),
                "target_base": np.asarray(target).tolist(),
            }
        )
        return self.p.action(robot)

    def advance(self, result):
        self.p.advance(result)
