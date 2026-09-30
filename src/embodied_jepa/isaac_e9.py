"""Cross-simulator development check of the TASK-070 expert e9: MuJoCo against Isaac/Newton.

Development only (TASK-025 Isaac track): not a gated run, not a learned result, and not a
policy result. e9 is the privileged scripted expert of ``docs/experiments/
apple_to_plate_v2_expert.md``; it reads simulator truth by design. Learned Apple->Plate is
still 0 successes. Report: ``docs/ISAAC_E9_REPLAY.md``.

Two ways of running e9 in Isaac, both driven from the host:

* **closed loop**: the unchanged harness (``resting_expert.run_attempt``, the embodiment's IK,
  the look, the scorer and ``apple_at_rest_v0``) runs on a ``MirrorSimulation``. Its physics
  is remote (an ``IsaacTransport`` in the container, reached through an *endpoint*); after
  every command the Isaac state (robot joints and velocities, apple pose and velocity, plate,
  Isaac's own apple-hand contact) is written into a local MuJoCo ``MjData`` with
  ``mj_step``'s layout (``MirrorSimulation._apply``): kinematics and collision are computed
  at the state at the start of the interval's last physics step (the transport's
  ``last_substep_start``), and the new ``qpos``/``qvel`` are then written without a forward
  pass. Everything that reads ``sim.data`` (IK, ``ee_pose``, contact counts) reads Isaac's
  state; only the apple-hand flag of ``task_truth`` is Isaac's own contact, every other
  contact is MuJoCo collision detection on Isaac's poses;
* **open loop**: the joint-target arrays that e9 sent in MuJoCo are replayed unchanged.

``MuJoCoEndpoint`` is the same endpoint backed by a host MuJoCo v2 scene; with it the mirror
must reproduce a plain MuJoCo run exactly (a plumbing check, also used by the tests).

NumPy only at import; MuJoCo is imported lazily by the simulator classes.
"""

from __future__ import annotations

import numpy as np

from embodied_jepa import apple_to_plate_v2 as v2
from embodied_jepa.contracts import ContractError
from embodied_jepa.simulation import MuJoCoSimulation

VERSION = "isaac_e9_replay_v1"
# TASK-070 design e9 (scripts/develop_v2_expert.py DESIGNS["e9"]), frozen as v2.GATE_EXPERT.
E9 = {"release_pitch_rad": 0.45, "release_dx": 0.015}
# The first 16 TASK-070 development seeds (already simulated in MuJoCo by TASK-070 e1-e9).
SEEDS = tuple(v2.DEV_SEEDS[:16])
LEVELS_CM = (0.0, 1.0)
STATE_KEYS = (
    "time",
    "q",
    "qd",
    "apple_pos",
    "apple_quat_xyzw",
    "apple_lin_w",
    "apple_ang_w",
    "plate_pos",
    "hand_contact",
    "contacts",
)
ISAAC_PAIR_TYPES = ("apple_hand", "apple_plate", "apple_table", "hand_plate", "robot_self")
JOINT_GROUPS = ("legs", "waist", "arms", "hands")
SPEED_AT_REST = 0.001  # m/s, the apple_at_rest_v0 speed bar


def check_seeds(seeds) -> tuple[int, ...]:
    """Development seeds only: TASK-070's development range, never its gate or TASK-073's."""
    from embodied_jepa import wm_critic_v2 as wc

    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ContractError("seeds must be non-empty and distinct")
    if not set(seeds) <= set(v2.DEV_SEEDS):
        raise ContractError("e9 replay seeds must be TASK-070 development seeds (50200-50299)")
    if set(seeds) & set(v2.GATE_SEEDS):
        raise ContractError("e9 replay seeds overlap the TASK-070 gate")
    low, high = wc.TASK_BLOCK
    if any(low <= s <= high for s in seeds):
        raise ContractError("e9 replay seeds overlap the TASK-073 block")
    v2.check_seeds(seeds)
    return seeds


# ----- state messages ----------------------------------------------------------------------------
def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if not np.isfinite(n) or n < 1e-9:
        raise ContractError("quaternion must be finite and non-zero")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def validate_state(state: dict, joints: int) -> dict:
    """A remote state message: every key, finite values, the declared shapes."""
    if not isinstance(state, dict) or any(k not in state for k in STATE_KEYS):
        raise ContractError("state message is missing keys")
    shapes = {
        "q": (joints,),
        "qd": (joints,),
        "apple_pos": (3,),
        "apple_quat_xyzw": (4,),
        "apple_lin_w": (3,),
        "apple_ang_w": (3,),
        "plate_pos": (3,),
    }
    out = dict(state)
    for key, shape in shapes.items():
        value = np.asarray(state[key], dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ContractError(f"state {key} must be {shape} finite values")
        out[key] = value
    out["time"] = float(state["time"])
    if not np.isfinite(out["time"]) or out["time"] < 0:
        raise ContractError("state time must be finite and non-negative")
    out["hand_contact"] = bool(state["hand_contact"])
    pre = state.get("pre")
    if pre is not None:
        out["pre"] = {}
        for key, shape in (("q", (joints,)), ("apple_pos", (3,)), ("apple_quat_xyzw", (4,))):
            value = np.asarray(pre.get(key), dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ContractError(f"state pre.{key} must be {shape} finite values")
            out["pre"][key] = value
    else:
        out["pre"] = None
    return out


def isaac_pair_counts(contacts) -> list[int]:
    """Isaac's own contact list (body labels) as flags per pair type (``ISAAC_PAIR_TYPES``).

    Isaac labels the plate as one body, so base and rim are not separated (unlike
    ``resting_expert.contact_counts``); robot self-contacts are counted only if reported."""
    from embodied_jepa.isaac_scene import is_hand_body

    counts = dict.fromkeys(ISAAC_PAIR_TYPES, 0)
    scene = {"apple", "plate", "table", "floor"}
    for c in contacts:
        a, b = c["bodies"]
        pair = {a, b}
        hand = is_hand_body(a) or is_hand_body(b)
        if "apple" in pair and hand:
            counts["apple_hand"] += 1
        elif pair == {"apple", "plate"}:
            counts["apple_plate"] += 1
        elif pair == {"apple", "table"}:
            counts["apple_table"] += 1
        elif "plate" in pair and hand:
            counts["hand_plate"] += 1
        elif not pair & scene:
            counts["robot_self"] += 1
    return [counts[k] for k in ISAAC_PAIR_TYPES]


# ----- simulators (MuJoCo imported lazily through MuJoCoSimulation) ------------------------------
def robot_self_contacts(sim) -> list[tuple[str, str, float]]:
    """Robot-robot contacts in ``sim.data``'s contact list: (body, body, distance)."""
    m, d = sim.model, sim.data
    root = m.body("pelvis").id
    out = []
    for c in d.contact[: d.ncon]:
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if m.body_rootid[b1] == root and m.body_rootid[b2] == root:
            names = sorted((m.body(b1).name, m.body(b2).name))
            out.append((names[0], names[1], float(c.dist)))
    return out


def _body_or_world_geom(model, geom: int) -> str:
    """A contact label like Isaac's: the body name, or the geom name (floor, table) on world."""
    body = model.body(int(model.geom_bodyid[geom])).name
    return model.geom(geom).name if body == "world" else body


def apple_hand_contact(sim) -> bool:
    """``MuJoCoSimulation.task_truth``'s hand-contact rule on ``sim.data``'s contact list."""
    m, d = sim.model, sim.data
    apple = m.geom("apple_geom").id
    for c in d.contact[: d.ncon]:
        if apple in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == apple else c.geom1
            name = m.body(int(m.geom_bodyid[other])).name
            if "hand_" in name or "wrist_" in name:
                return True
    return False


def placed_rule(truth: dict, *, hand_contact: bool) -> bool:
    """``MuJoCoSimulation.task_truth``'s ``placed`` rule, evaluated with ``hand_contact``.

    A copy, not a call: the rule is inline in ``MuJoCoSimulation.task_truth`` and
    ``simulation.py`` is byte-pinned by the benchmark manifests (``src/embodied_jepa/
    simulation.py`` hashes), so it is not factored out there. The mirror needs it with Isaac's
    apple-hand flag in place of its own geometric one.
    ``test_placed_rule_matches_mujoco_task_truth`` pins this copy to the original."""
    apple = np.asarray(truth["object_position"], float)
    plate = np.asarray(truth["plate_position"], float)
    return bool(
        np.linalg.norm(apple[:2] - plate[:2]) < 0.04
        and abs(apple[2] - (truth["container_surface_z"] + truth["object_support_height"])) < 0.012
        and np.linalg.norm(truth["object_velocity"]) < 0.1
        and not hand_contact
    )


SELF_CONTACT_FIELDS = {
    "self_contact_steps": "mirror_self_contact_steps",
    "remote_self_contact_steps": "isaac_self_contact_steps",
}


def _steps(summary: dict, field: str) -> int | None:
    for name in (field, SELF_CONTACT_FIELDS[field]):
        if summary.get(name) is not None:
            return int(summary[name])
    return None


def self_contact_summary(rows, sources) -> dict:
    """Robot self-contact seen in a run or comparison: steps and attempts per source; a flag.

    ``rows`` are per-attempt dicts with a ``key`` and, per source in ``sources`` (``mujoco``,
    ``closed``, ``open``), a summary carrying ``self_contact_steps`` (MuJoCo collision detection
    on the state in ``sim.data``: MuJoCo's own state, or Isaac's in the mirror) and/or
    ``remote_self_contact_steps`` (Isaac's own contact list). ``e9_replay.py compare`` names
    them ``mirror_self_contact_steps`` and ``isaac_self_contact_steps``; both spellings are
    read (``SELF_CONTACT_FIELDS``). Both Isaac backends run without
    robot self-collision (ruling in ``docs/ISAAC_E9_REPLAY.md`` §5), so Isaac's own list cannot
    show one; the mirror's MuJoCo check, with self-collision on as in the host MJCF, is the
    check that matters. ``flag`` is True if any source saw a self-contact: that run needs
    review, because Isaac let links interpenetrate that MuJoCo would have pushed apart."""
    out: dict = {}
    for source in sources:
        entries = [
            (r["key"], r[source])
            for r in rows
            if isinstance(r.get(source), dict) and r[source].get("ok", True)
        ]
        for field in SELF_CONTACT_FIELDS:
            values = [(key, v) for key, s in entries if (v := _steps(s, field)) is not None]
            if values:
                out[f"{source}_{field}"] = {
                    "steps": sum(v for _, v in values),
                    "attempts_with_contact": sorted(k for k, v in values if v > 0),
                    "attempts_checked": len(values),
                }
    out["flag"] = any(v["steps"] > 0 for v in out.values())
    return out


class MirrorSimulation(MuJoCoSimulation):
    """A ``MuJoCoSimulation`` whose physics runs behind ``endpoint``; ``data`` mirrors it.

    The endpoint implements ``reset(seed, object_xy, plate_xy, object_on_container,
    joint_positions) -> state``, ``step(targets, deadline) -> {"ack", "state"}`` and
    ``stop(reason) -> state`` (state: ``STATE_KEYS``). The local model is only kinematics,
    geometry for contact detection and (optionally) rendering; ``mj_step`` never runs on it.

    ``reset`` computes the layout locally (``MuJoCoSimulation.reset``); the remote reset is sent
    lazily on the next read, step, stop or truth call, so ``G1Embodiment.reset``'s open hands,
    written into ``data.qpos`` after ``sim.reset``, are part of the remote initial pose.
    ``task_truth`` reads the mirrored apple and plate and the remote simulator's own apple-hand
    contact; the mirror's geometric one is kept as ``mirror_hand_contact``.
    """

    def __init__(self, endpoint, *, render=True, **kwargs):
        self.endpoint = endpoint
        self._pending = None
        self._remote = None
        self.mirror_hand_contact = None
        super().__init__(render=render, **kwargs)
        self._blank = np.zeros((self.height, self.width, 3), np.uint8)
        joint = self.model.joint("apple_free")
        self._apple_q = int(joint.qposadr[0])
        self._apple_v = int(joint.dofadr[0])

    # ---- remote state
    def _write_pose(self, q, apple_pos, apple_quat_xyzw) -> None:
        d = self.data
        d.qpos[self.qadr] = q
        x, y, z, w = apple_quat_xyzw
        d.qpos[self._apple_q : self._apple_q + 3] = apple_pos
        d.qpos[self._apple_q + 3 : self._apple_q + 7] = [w, x, y, z]

    def _apply(self, state: dict) -> None:
        """Write the remote state into ``data`` with ``mj_step``'s layout.

        After ``mj_step`` a MuJoCo ``MjData`` has the new ``qpos``/``qvel`` but body and site
        poses and contacts of the state before the last physics step. With ``pre`` (the state
        at the start of the remote interval's last physics step) the mirror reproduces that:
        forward kinematics and collision at ``pre``, then the new ``qpos``/``qvel`` written
        without a forward pass. Without ``pre`` (after a reset or stop) it is a plain
        ``mj_forward`` of the new state, as ``MuJoCoSimulation.reset`` does."""
        s = validate_state(state, len(self.joint_names))
        d = self.data
        self.model.body("plate").pos[:] = s["plate_pos"]
        d.time = s["time"]
        rotation = quat_xyzw_to_matrix(s["apple_quat_xyzw"])
        d.qvel[self.vadr] = s["qd"]
        d.qvel[self._apple_v : self._apple_v + 3] = s["apple_lin_w"]
        d.qvel[self._apple_v + 3 : self._apple_v + 6] = rotation.T @ s["apple_ang_w"]
        if s["pre"] is not None:
            pre = s["pre"]
            self._write_pose(pre["q"], pre["apple_pos"], pre["apple_quat_xyzw"])
            self.mj.mj_forward(self.model, d)
            self._write_pose(s["q"], s["apple_pos"], s["apple_quat_xyzw"])
        else:
            self._write_pose(s["q"], s["apple_pos"], s["apple_quat_xyzw"])
            self.mj.mj_forward(self.model, d)
        self._remote = s

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
        if np.abs(self._remote["apple_pos"] - local_apple).max() > 1e-4 or (
            np.abs(self._remote["plate_pos"] - local_plate).max() > 1e-4
        ):
            raise ContractError("remote reset layout differs from the local one")
        self.targets[:] = self.data.qpos[self.qadr]

    @property
    def remote_state(self) -> dict | None:
        return self._remote

    # ---- transport API
    def reset(
        self,
        seed=0,
        *,
        object_xy=None,
        plate_xy=None,
        object_kind=None,
        container_kind=None,
        object_on_container=False,
    ):
        self._pending, self._remote = None, None
        truth = super().reset(
            seed,
            object_xy=object_xy,
            plate_xy=plate_xy,
            object_kind=object_kind,
            container_kind=container_kind,
            object_on_container=object_on_container,
        )
        apple = self.data.body("apple").xpos
        plate = self.model.body("plate").pos
        self._pending = {
            "seed": int(seed),
            "object_xy": [float(apple[0]), float(apple[1])],
            "plate_xy": [float(plate[0]), float(plate[1])],
            "object_on_container": bool(object_on_container),
        }
        return truth

    def render(self, camera="onboard_rgb"):
        self._sync()
        if not self.render_enabled:
            return self._blank.copy()  # the e9 harness never reads pixels
        return super().render(camera)

    def read(self):
        self._sync()
        return super().read()

    def send_joint_targets(self, targets, *, joint_names, deadline):
        self._require_open()
        self._sync()
        targets = np.asarray(targets)
        if (
            tuple(joint_names) != self.joint_names
            or targets.shape != self.targets.shape
            or not np.isfinite(targets).all()
        ):
            raise ContractError("joint targets must match named robot actuator order and be finite")
        if not np.isfinite(deadline) or deadline < self.data.time:
            self.stop("command deadline expired")
            return {
                "status": "rejected",
                "timestamp": float(self.data.time),
                "reason": self.stopped_reason,
            }
        limits = self.model.jnt_range[self.joint_ids]
        if np.any(targets < limits[:, 0]) or np.any(targets > limits[:, 1]):
            raise ContractError("joint targets exceed MJCF limits")
        reply = self.endpoint.step(targets.astype(float).tolist(), float(deadline))
        self._apply(reply["state"])
        ack = reply["ack"]
        if ack["status"] != "applied":
            self.stopped_reason = str(ack.get("reason", "remote rejected"))
            return {"status": ack["status"], "timestamp": self._remote["time"], **ack}
        self.targets[:] = targets
        self.stopped_reason = ""
        return {"status": "applied", "timestamp": self._remote["time"], "targets": targets.copy()}

    def stop(self, reason):
        self._sync()
        super().stop(reason)
        if self._remote is not None:
            # A stop does not step the remote physics; like MuJoCoSimulation.stop, it leaves
            # ``data`` as it is (the returned state is only checked).
            validate_state(self.endpoint.stop(str(reason)), len(self.joint_names))

    def task_truth(self):
        self._sync()
        truth = super().task_truth()
        self.mirror_hand_contact = bool(truth["hand_contact"])
        if self._remote is None:
            return truth
        hand = bool(self._remote["hand_contact"])
        truth["hand_contact"] = hand
        truth["placed"] = placed_rule(truth, hand_contact=hand)
        return truth


class MuJoCoEndpoint:
    """The endpoint protocol over a host ``MuJoCoSimulation`` switched to v2 (plumbing check)."""

    def __init__(self):
        self.sim = MuJoCoSimulation(object_kind="apple", container_kind="plate", render=False)
        self.scene = v2.apply_v2_scene(self.sim.model)

    def state(self) -> dict:
        sim = self.sim
        d = sim.data
        joint = d.joint("apple_free")
        w, x, y, z = joint.qpos[3:7]
        rotation = quat_xyzw_to_matrix([x, y, z, w])
        contacts = []
        m = sim.model
        seen = set()
        for c in d.contact[: d.ncon]:
            key = tuple(sorted(_body_or_world_geom(m, int(g)) for g in (c.geom1, c.geom2)))
            if key[0] != key[1] and key not in seen:
                seen.add(key)
                contacts.append({"bodies": list(key), "force_n": 0.0})
        return {
            "time": float(d.time),
            "q": d.qpos[sim.qadr].tolist(),
            "qd": d.qvel[sim.vadr].tolist(),
            "apple_pos": joint.qpos[:3].tolist(),
            "apple_quat_xyzw": [float(x), float(y), float(z), float(w)],
            "apple_lin_w": joint.qvel[:3].tolist(),
            "apple_ang_w": (rotation @ joint.qvel[3:6]).tolist(),
            "plate_pos": m.body("plate").pos.tolist(),
            "hand_contact": bool(sim.task_truth()["hand_contact"]),
            "contacts": contacts,
        }

    def reset(self, *, seed, object_xy, plate_xy, object_on_container, joint_positions):
        sim = self.sim
        sim.reset(
            seed,
            object_xy=object_xy,
            plate_xy=plate_xy,
            object_on_container=object_on_container,
        )
        sim.data.qpos[sim.qadr] = joint_positions
        sim.mj.mj_forward(sim.model, sim.data)
        sim.targets[:] = sim.data.qpos[sim.qadr]
        return self.state()

    def step(self, targets, deadline):
        sim, mj = self.sim, self.sim.mj
        joint = sim.data.joint("apple_free")
        pre = {}

        class _Recording:  # records the state before each mj_step (the last one is kept)
            def __getattr__(self, name):
                return getattr(mj, name)

            def mj_step(self, model, data):
                w, x, y, z = joint.qpos[3:7]
                pre.update(
                    q=data.qpos[sim.qadr].tolist(),
                    apple_pos=joint.qpos[:3].tolist(),
                    apple_quat_xyzw=[float(x), float(y), float(z), float(w)],
                )
                mj.mj_step(model, data)

        sim.mj = _Recording()
        try:
            ack = sim.send_joint_targets(
                np.asarray(targets), joint_names=sim.joint_names, deadline=deadline
            )
        finally:
            sim.mj = mj
        ack = {k: v for k, v in ack.items() if k != "targets"}
        state = self.state()
        state["pre"] = pre or None
        return {"ack": ack, "state": state}

    def stop(self, reason):
        self.sim.stop(reason)
        return self.state()

    def close(self):
        self.sim.close()


def make_mirror_robot(endpoint, *, render=False):
    """``G1Embodiment`` over a v2 ``MirrorSimulation`` (112 px, as ``make_v2_robot``)."""
    from embodied_jepa import first_policy as fp
    from embodied_jepa.embodiment import G1Embodiment

    sim = MirrorSimulation(
        endpoint,
        render=render,
        object_kind="apple",
        container_kind="plate",
        width=fp.IMAGE_SIZE,
        height=fp.IMAGE_SIZE,
    )
    scene = v2.apply_v2_scene(sim.model)  # geometry for contact detection only
    return G1Embodiment(sim), scene


class StepRecorder:
    """Per executed command: time, targets, joints, apple, self-contacts, remote contacts.

    Installs itself on ``sim.send_joint_targets`` (instance attribute) and on
    ``robot.reset``, which it uses to capture the initial joint pose."""

    KEYS = (
        "time",
        "targets",
        "q",
        "qd",
        "apple_pos",
        "self_count",
        "self_min_dist",
        "isaac_counts",
        "mirror_hand",
        "remote_hand",
    )

    def __init__(self, robot):
        self.robot = robot
        self.sim = sim = robot.sim
        self.initial_q = None
        self.self_pairs: dict[str, int] = {}
        self.clear()
        send, reset = sim.send_joint_targets, robot.reset

        def recorded_send(targets, **kwargs):
            ack = send(targets, **kwargs)
            if ack["status"] == "applied":
                self._record(targets)
            return ack

        def recorded_reset(*args, **kwargs):
            truth = reset(*args, **kwargs)
            self.clear()
            self.initial_q = sim.data.qpos[sim.qadr].copy()
            return truth

        sim.send_joint_targets = recorded_send
        robot.reset = recorded_reset

    def clear(self) -> None:
        self.rows = {k: [] for k in self.KEYS}
        self.self_pairs = {}

    def _record(self, targets) -> None:
        sim, d = self.sim, self.sim.data
        pairs = robot_self_contacts(sim)
        remote = getattr(sim, "remote_state", None)
        self.rows["time"].append(float(d.time))
        self.rows["targets"].append(np.asarray(targets, float).copy())
        self.rows["q"].append(d.qpos[sim.qadr].copy())
        self.rows["qd"].append(d.qvel[sim.vadr].copy())
        self.rows["apple_pos"].append(d.body("apple").xpos.copy())
        self.rows["self_count"].append(len(pairs))
        self.rows["self_min_dist"].append(min((p[2] for p in pairs), default=np.nan))
        self.rows["isaac_counts"].append(
            isaac_pair_counts(remote["contacts"]) if remote else [0] * len(ISAAC_PAIR_TYPES)
        )
        local = apple_hand_contact(sim)  # geometric, on sim.data (the mirror: Isaac's state)
        self.rows["mirror_hand"].append(local)
        self.rows["remote_hand"].append(bool(remote["hand_contact"]) if remote else local)
        for a, b, _dist in pairs:
            key = f"{a}|{b}"
            self.self_pairs[key] = self.self_pairs.get(key, 0) + 1

    def arrays(self) -> dict:
        out = {f"rec_{k}": np.asarray(v) for k, v in self.rows.items()}
        out["rec_initial_q"] = np.asarray(self.initial_q)
        return out


# ----- open-loop replay --------------------------------------------------------------------------
def replay_open_loop(robot, *, seed, reset, initial_q, targets, look_steps, thresholds=None):
    """Replay ``targets`` after the same reset; score the tail with ``apple_at_rest_v0``.

    ``robot`` is a ``G1Embodiment`` (normally over a ``MirrorSimulation``) with a
    ``StepRecorder`` installed. The at-rest check records every step after the look, as
    ``run_attempt`` does. Returns ``(summary, arrays)``."""
    from embodied_jepa import resting_expert as rx
    from embodied_jepa.at_rest import AppleAtRestCheck

    robot.reset(int(seed), object_xy=reset["object_xy"], plate_xy=reset["plate_xy"])
    sim = robot.sim
    truth = sim.task_truth()
    q0 = sim.data.qpos[sim.qadr]
    if not np.allclose(q0, np.asarray(initial_q), atol=1e-6):
        raise ContractError("the replay's initial pose differs from the recorded one")
    check = AppleAtRestCheck(robot, thresholds)
    rows = {k: [] for k in ("apple_pos", "apple_lin", "counts", "distance")}
    stop_reason = "complete"
    for k, target in enumerate(np.asarray(targets, float)):
        ack = sim.send_joint_targets(
            target, joint_names=sim.joint_names, deadline=float(sim.data.time + sim.control_dt)
        )
        if ack["status"] != "applied":
            stop_reason = f"rejected at command {k}: {ack.get('reason')}"
            break
        if k < look_steps:
            continue
        check.record()
        now = sim.task_truth()
        rows["apple_pos"].append(np.asarray(now["object_position"], float))
        rows["apple_lin"].append(np.asarray(now["object_velocity"], float))
        rows["counts"].append(rx.contact_counts(sim))
        rows["distance"].append(
            float(np.linalg.norm(now["object_position"][:2] - now["plate_position"][:2]))
        )
    robot.stop(stop_reason)
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    complete = stop_reason == "complete"
    verdict = check.verdict() if complete and len(arrays["distance"]) else None
    summary = {
        "seed": int(seed),
        "stop_reason": stop_reason,
        "complete": complete,
        "steps": len(arrays["distance"]),
        "at_rest": bool(verdict["at_rest"]) if verdict else False,
        "at_rest_detail": verdict,
        "final_distance_cm": float(arrays["distance"][-1] * 100)
        if len(arrays["distance"])
        else None,
        "rest_z": float(truth["container_surface_z"] + truth["object_support_height"]),
    }
    return summary, arrays


# ----- comparison (NumPy only) -------------------------------------------------------------------
def joint_groups(names) -> dict[str, np.ndarray]:
    names = list(names)
    groups = {g: [] for g in JOINT_GROUPS}
    for i, n in enumerate(names):
        if "hand_" in n:
            groups["hands"].append(i)
        elif "waist" in n:
            groups["waist"].append(i)
        elif any(k in n for k in ("shoulder", "elbow", "wrist")):
            groups["arms"].append(i)
        elif any(k in n for k in ("hip", "knee", "ankle")):
            groups["legs"].append(i)
        else:
            raise ContractError(f"joint {n!r} has no group")
    return {g: np.asarray(v, dtype=int) for g, v in groups.items()}


def _first(mask) -> int | None:
    hits = np.flatnonzero(np.asarray(mask, bool))
    return int(hits[0]) if hits.size else None


def settle_step(lin_vel, bar: float = SPEED_AT_REST) -> int | None:
    """First step from which the speed stays at or below ``bar`` to the end (None: never)."""
    speed = np.linalg.norm(np.asarray(lin_vel, float), axis=1)
    above = np.flatnonzero(speed > bar)
    if above.size == 0:
        return 0 if speed.size else None
    return None if above[-1] == speed.size - 1 else int(above[-1] + 1)


def events(*, hand, apple_pos, apple_lin, plate_contact, start_z, open_step=None) -> dict:
    """Grasp, lift, release, landing and settle steps of one attempt (steps after the look).

    ``hand`` and ``plate_contact`` are per-step booleans (apple-hand contact; apple on the plate
    base or plate). Grasp: first hand contact. Lift: the apple 1 cm above its starting height
    ``start_z`` (``lift_step``) and at the scorer's lifted height 0.82 m (``lifted_step``).
    Release: the first step after the last hand contact that follows ``open_step`` (the expert
    opening).
    Landing: first plate contact after that release. None: the event did not happen."""
    hand = np.asarray(hand, bool)
    pos = np.asarray(apple_pos, float)
    plate_contact = np.asarray(plate_contact, bool)
    out = {
        "grasp_step": _first(hand),
        "lift_step": _first(pos[:, 2] > start_z + 0.01),
        "lifted_step": _first(pos[:, 2] > 0.82),
        "open_step": open_step,
        "release_step": None,
        "landing_step": None,
        "settle_step": settle_step(apple_lin),
    }
    start = 0 if open_step is None else int(open_step)
    after = np.flatnonzero(hand[start:])
    if hand[start:].size:
        release = start + (int(after[-1]) + 1 if after.size else 0)
        if release < hand.size:
            out["release_step"] = release
            landing = _first(plate_contact[release:])
            out["landing_step"] = None if landing is None else release + landing
    return out


def divergence(q_a, q_b, names, *, thresholds=(0.01, 0.1)) -> dict:
    """|q_a - q_b| per joint group over the common steps: max, p95, median, first step over
    each threshold, and the worst joint. The two runs may differ in length."""
    a, b = np.asarray(q_a, float), np.asarray(q_b, float)
    n = min(len(a), len(b))
    diff = np.abs(a[:n] - b[:n])
    names = list(names)
    out = {"common_steps": n, "length_a": len(a), "length_b": len(b), "groups": {}}
    for group, idx in joint_groups(names).items():
        g = diff[:, idx] if n else np.zeros((0, len(idx)))
        per_step = g.max(axis=1) if n else np.zeros(0)
        row = {
            "max": float(g.max()) if g.size else 0.0,
            "p95": float(np.quantile(per_step, 0.95)) if n else 0.0,
            "median": float(np.median(per_step)) if n else 0.0,
        }
        for t in thresholds:
            row[f"first_over_{t:g}"] = _first(per_step > t)
        if g.size:
            k = int(np.argmax(g.max(axis=0)))
            row["worst_joint"] = names[int(idx[k])]
        out["groups"][group] = row
    return out
