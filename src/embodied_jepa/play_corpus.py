"""TASK-085 play corpus (Phase 1 of ``docs/JEPA_ZERO_SHOT_PLAN.md``): scene, play policy, episode.

A fixed-pelvis G1 tabletop with four movable objects (apple, cube, banana, can) and the static
plate, played by the right arm and right Dex3 hand only (the left arm is held, its hand open).
The play policy is a **privileged scripted mixture** (it reads simulator object truth to choose
where to reach); it is a data collector, never a learned result. Successes or grasps in this
corpus are not project-learned results.

NumPy only at import; MuJoCo is imported lazily by :mod:`embodied_jepa.simulation`.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embodied_jepa.contracts import ContractError
from embodied_jepa.embodiment import rotation_delta

SCENE_VERSION = "play_scene_v0"
POLICY_VERSION = "play_policy_v0"
TASK_NAME = "play_v0"

# ----- scene ----------------------------------------------------------------------------------
TABLE_TOP_Z = 0.74
PELVIS_Z = 0.793
# name -> MJCF geom attributes, resting centre height above the table top, xy collision radius.
OBJECTS = {
    "apple": {"geom": "apple_geom", "rest": 0.027, "radius": 0.027},
    "cube": {
        "geom": "play_cube_geom",
        "shape": {"type": "box", "size": ".023 .023 .023", "rgba": ".1 .75 .2 1"},
        "mass": ".06",
        "rest": 0.023,
        "radius": 0.033,
    },
    "banana": {
        "geom": "play_banana_geom",
        "shape": {
            "type": "capsule",
            "size": ".018 .035",
            "quat": ".70710678 0 .70710678 0",
            "rgba": ".95 .8 .06 1",
        },
        "mass": ".05",
        "rest": 0.018,
        "radius": 0.053,
    },
    "can": {
        "geom": "play_can_geom",
        "shape": {"type": "cylinder", "size": ".024 .03", "rgba": ".55 .2 .75 1"},
        "mass": ".07",
        "rest": 0.03,
        "radius": 0.024,
    },
}
OBJECT_NAMES = tuple(OBJECTS)
PLATE_RADIUS = 0.071
# Off-table parking (world xy, on the floor behind the robot, out of the onboard view).
PARKING = {name: (-0.55, -0.45 + 0.3 * i) for i, name in enumerate(OBJECT_NAMES)}
CONTACT_FRICTION = (1.0, 0.01, 0.001)  # the v1 scene's declared apple friction (v2 makes it act)
CONTACT_CONDIM = 6

# Reset regions (world xy).  The object region contains the Phase 3 test object region.
OBJECT_REGION = ((0.26, 0.54), (-0.34, 0.00))
PLATE_REGION = ((0.30, 0.52), (-0.30, -0.02))

# ----- the Phase 3 test workspace used by the coverage gate (protocol §6) ----------------------
TEST_OBJECT_REGION = ((0.30, 0.46), (-0.28, -0.04))  # world xy, 4 cm cells: 4 x 6
TEST_HAND_REGION = ((0.24, 0.44), (-0.30, -0.02), (0.06, 0.26))  # pelvis xyz, 4 cm: 5 x 7 x 5
WANDER_REGION = ((0.20, 0.52), (-0.36, -0.02), (0.00, 0.28))  # pelvis xyz, wander targets
CELL = 0.04


def _play_simulation_class():
    from embodied_jepa.simulation import MuJoCoSimulation

    class _PlaySimulation(MuJoCoSimulation):
        def __init__(self, **kwargs):
            self._layout = None
            super().__init__(object_kind="apple", container_kind="plate", **kwargs)
            self._apply_contacts()
            self.object_body = {
                name: self.model.body("apple" if name == "apple" else f"play_{name}").id
                for name in OBJECT_NAMES
            }
            self.object_geom = {name: self.model.geom(OBJECTS[name]["geom"]).id for name in OBJECTS}
            self.object_qadr = {
                name: int(
                    self.model.joint(
                        "apple_free" if name == "apple" else f"play_{name}_free"
                    ).qposadr[0]
                )
                for name in OBJECT_NAMES
            }
            self.object_vadr = {
                name: int(
                    self.model.joint(
                        "apple_free" if name == "apple" else f"play_{name}_free"
                    ).dofadr[0]
                )
                for name in OBJECT_NAMES
            }
            pelvis = self.model.body("pelvis").id
            self.robot_geom = np.array(
                [int(self.model.body_rootid[b]) == pelvis for b in self.model.geom_bodyid]
            )
            names = [self.model.body(int(b)).name for b in self.model.geom_bodyid]
            self.hand_geom = np.array(
                ["right_hand_" in n or n == "right_wrist_yaw_link" for n in names]
            )
            self.finger_body = {}
            for g, n in enumerate(names):
                if self.hand_geom[g]:
                    kind = (
                        "thumb"
                        if "thumb" in n
                        else ("index" if "index" in n else ("middle" if "middle" in n else "palm"))
                    )
                    self.finger_body[g] = kind
            self.reset()

        def _scene(self):
            root = ET.fromstring(super()._scene())
            world = root.find("worldbody")
            for name, spec in OBJECTS.items():
                if name == "apple":
                    continue
                body = ET.SubElement(world, "body", name=f"play_{name}", pos="-0.55 0 .1")
                ET.SubElement(body, "freejoint", name=f"play_{name}_free")
                ET.SubElement(
                    body,
                    "geom",
                    name=spec["geom"],
                    mass=spec["mass"],
                    friction=" ".join(map(str, CONTACT_FRICTION)),
                    condim=str(CONTACT_CONDIM),
                    **spec["shape"],
                )
            return ET.tostring(root, encoding="unicode")

        def _apply_contacts(self):
            from embodied_jepa.apple_to_plate_v2 import apply_v2_scene

            self.v2_record = apply_v2_scene(self.model)

        def reset(self, seed=0, *, layout=None, **_ignored):
            self._require_open()
            if not hasattr(self, "object_qadr"):  # during the base constructor
                return super().reset(seed)
            if layout is None:
                layout = sample_layout(np.random.default_rng(seed))
            check_layout(layout)
            self.mj.mj_resetData(self.model, self.data)
            for name in OBJECT_NAMES:
                q = self.object_qadr[name]
                xy = layout["objects"].get(name)
                if xy is None:
                    px, py = PARKING[name]
                    self.data.qpos[q : q + 3] = [px, py, OBJECTS[name]["rest"]]
                else:
                    self.data.qpos[q : q + 3] = [xy[0], xy[1], TABLE_TOP_Z + OBJECTS[name]["rest"]]
                yaw = float(layout.get("yaw", {}).get(name, 0.0))
                self.data.qpos[q + 3 : q + 7] = _yaw_quat(yaw, name)
            self.model.body("plate").pos[:] = [*layout["plate"], 0.746]
            for side in ("left", "right"):
                self.data.joint(f"{side}_elbow_joint").qpos[0] = 0.08
            self.mj.mj_forward(self.model, self.data)
            self.targets[:] = self.data.qpos[self.qadr]
            self.stopped_reason = ""
            self._layout = layout
            return self.task_truth()

        def play_truth(self):
            """Collector/label-only object truth; never a model input."""
            contacts = {name: set() for name in OBJECT_NAMES}
            robot_touch = dict.fromkeys(OBJECT_NAMES, False)
            hand_touch = dict.fromkeys(OBJECT_NAMES, False)
            geom_to_object = {g: n for n, g in self.object_geom.items()}
            for c in self.data.contact[: self.data.ncon]:
                for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    name = geom_to_object.get(int(a))
                    if name is None:
                        continue
                    if self.robot_geom[b]:
                        robot_touch[name] = True
                    if self.hand_geom[b]:
                        hand_touch[name] = True
                        contacts[name].add(self.finger_body.get(int(b), "palm"))
            out = {
                "position": np.array([self.data.xpos[self.object_body[n]] for n in OBJECT_NAMES]),
                "quat": np.array([self.data.xquat[self.object_body[n]] for n in OBJECT_NAMES]),
                "velocity": np.array(
                    [
                        self.data.qvel[self.object_vadr[n] : self.object_vadr[n] + 3]
                        for n in OBJECT_NAMES
                    ]
                ),
                "robot_contact": np.array([robot_touch[n] for n in OBJECT_NAMES]),
                "hand_contact": np.array([hand_touch[n] for n in OBJECT_NAMES]),
                "grasp_contact": np.array(
                    [
                        "thumb" in contacts[n] and bool(contacts[n] & {"index", "middle"})
                        for n in OBJECT_NAMES
                    ]
                ),
                "plate_position": self.data.body("plate").xpos.copy(),
            }
            return out

    return _PlaySimulation


def _yaw_quat(yaw, name):
    base = np.array([1.0, 0, 0, 0])
    if name == "banana":
        base = np.array([0.70710678, 0, 0.70710678, 0])
    q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = base
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def make_play_robot(*, image_size=112):
    """The play robot: 112 px onboard camera, renderer warmed up (first frame discarded)."""
    from embodied_jepa.embodiment import G1Embodiment

    sim = _play_simulation_class()(width=image_size, height=image_size)
    robot = G1Embodiment(sim)
    sim.render()
    return robot


# ----- layouts ----------------------------------------------------------------------------------
def sample_layout(rng):
    """Plate anywhere in its region; 1-4 objects on the table, the rest parked; no overlaps."""
    for _ in range(1000):
        plate = [rng.uniform(*PLATE_REGION[0]), rng.uniform(*PLATE_REGION[1])]
        count = int(rng.integers(1, 5))
        chosen = sorted(rng.choice(len(OBJECT_NAMES), size=count, replace=False).tolist())
        objects, placed = {}, [(np.array(plate), PLATE_RADIUS)]
        ok = True
        for index in chosen:
            name = OBJECT_NAMES[index]
            radius = OBJECTS[name]["radius"]
            for _ in range(200):
                xy = np.array([rng.uniform(*OBJECT_REGION[0]), rng.uniform(*OBJECT_REGION[1])])
                if all(np.linalg.norm(xy - p) >= radius + r + 0.01 for p, r in placed):
                    break
            else:
                ok = False
                break
            placed.append((xy, radius))
            objects[name] = xy.tolist()
        if ok:
            yaw = {name: float(rng.uniform(-np.pi, np.pi)) for name in objects}
            return {"plate": plate, "objects": objects, "yaw": yaw}
    raise ContractError("could not sample a layout")


def check_layout(layout):
    plate = np.asarray(layout["plate"], float)
    if plate.shape != (2,) or not np.isfinite(plate).all():
        raise ContractError("plate xy must be finite")
    for name, xy in layout["objects"].items():
        if name not in OBJECTS:
            raise ContractError(f"unknown object {name}")
        xy = np.asarray(xy, float)
        if not (0.15 <= xy[0] <= 0.70 and -0.40 <= xy[1] <= 0.40):
            raise ContractError(f"{name} must start on the table")


# ----- the play policy ----------------------------------------------------------------------------
TOP_DOWN = rotation_delta([-np.pi / 2, 0, 0])
MODES = ("scripted", "perturbed", "random")
MODE_WEIGHTS = (0.40, 0.35, 0.25)
SKILLS = ("pick_place", "push", "poke", "wander")
SKILL_WEIGHTS = (0.55, 0.25, 0.10, 0.10)
FAILURES = ("none", "miss", "early_close", "drop", "no_close", "abort")
FAILURE_WEIGHTS = (0.65, 0.10, 0.07, 0.08, 0.05, 0.05)
NOISE_SIGMA = {"scripted": 0.10, "perturbed": 0.25, "random": 0.45}
GRASP_OFFSET = np.array([-0.015, 0.0, 0.052])  # e9's palm target relative to the object centre
GRASP_RAMP = 0.2  # max change of the requested right grasp per command
AIM_SIGMA = 0.008  # m, xy aim noise of a pick (z: half of it)
YAW_RANGE = 0.2  # rad, per-skill palm yaw about the top-down grasp
HOVER_DZ = 0.08  # m, hover height above the grasp point
CLOSURE = (0.85, 1.0)  # requested grasp closure range
ROT_CLIP = 0.8  # normalized rotation command clip (e9: 0.5)


@dataclass
class Waypoint:
    target: np.ndarray  # pelvis-frame right EE position
    grasp: float
    steps: int
    yaw: float = 0.0


def to_pelvis(xyz_world):
    p = np.asarray(xyz_world, float).copy()
    p[2] -= PELVIS_Z
    return p


class PlayPolicy:
    """Per-episode mixture of noisy scripted skills, perturbation bursts and random wandering.

    ``act(robot, truth)`` returns a normalized 14-D request; the left arm is zero and its hand
    open. Failures are injected deliberately (FAILURES)."""

    def __init__(self, rng, *, mode=None):
        self.rng = rng
        self.mode = mode or str(rng.choice(MODES, p=MODE_WEIGHTS))
        self.sigma = NOISE_SIGMA[self.mode]
        self.noise = np.zeros(6)
        self.queue: list[Waypoint] = []
        self.skill = "none"
        self.failure = "none"
        self.skill_log: list[tuple[str, str]] = []
        self.burst = 0
        self.wander_rot = np.zeros(3)
        self.grasp = -1.0

    # -- skill planning ----------------------------------------------------------------------
    def _plan(self, truth, ee):
        rng = self.rng
        on_table = [
            i
            for i, _ in enumerate(OBJECT_NAMES)
            if truth["position"][i][2] > TABLE_TOP_Z - 0.01 and truth["position"][i][0] > 0.1
        ]
        if self.mode == "random" or not on_table:
            skill = "wander"
        else:
            skill = str(rng.choice(SKILLS, p=SKILL_WEIGHTS))
        failure = "none"
        if skill == "pick_place":
            failure = str(rng.choice(FAILURES, p=FAILURE_WEIGHTS))
        self.skill, self.failure = skill, failure
        self.skill_log.append((skill, failure))
        yaw = float(rng.uniform(-YAW_RANGE, YAW_RANGE))
        if skill == "wander":
            self.queue = [
                Waypoint(
                    self._random_hand_point(),
                    float(rng.choice([-1.0, rng.uniform(-1, 1)])),
                    int(rng.integers(15, 45)),
                    float(rng.uniform(-0.5, 0.5)),
                )
                for _ in range(int(rng.integers(2, 6)))
            ]
            return
        index = int(rng.choice(on_table))
        obj = to_pelvis(truth["position"][index])
        aim = obj + GRASP_OFFSET + np.r_[rng.normal(0, AIM_SIGMA, 2), rng.normal(0, AIM_SIGMA / 2)]
        if failure == "miss":
            angle = rng.uniform(0, 2 * np.pi)
            aim[:2] += rng.uniform(0.04, 0.08) * np.array([np.cos(angle), np.sin(angle)])
        if skill == "poke":
            self.queue = [
                Waypoint(aim + [0, 0, 0.12], -1.0, int(rng.integers(20, 40)), yaw),
                Waypoint(
                    aim + [0, 0, rng.uniform(-0.01, 0.02)],
                    float(rng.uniform(-1, 0)),
                    int(rng.integers(15, 30)),
                    yaw,
                ),
                Waypoint(aim + [0, 0, 0.12], -1.0, int(rng.integers(10, 20)), yaw),
            ]
            return
        if skill == "push":
            angle = rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            low = obj[2] + rng.uniform(0.02, 0.05)
            start = obj - direction * 0.09
            end = obj + direction * rng.uniform(0.05, 0.15)
            grasp = float(rng.uniform(-1, 0.3))
            self.queue = [
                Waypoint(np.r_[start[:2], low + 0.10], grasp, int(rng.integers(20, 40)), yaw),
                Waypoint(np.r_[start[:2], low], grasp, int(rng.integers(12, 25)), yaw),
                Waypoint(np.r_[end[:2], low], grasp, int(rng.integers(20, 40)), yaw),
                Waypoint(np.r_[end[:2], low + 0.10], -1.0, int(rng.integers(10, 20)), yaw),
            ]
            return
        # pick_place: e9's pick recipe (hover, descend, close, lift) with noise and failures
        closure = float(rng.uniform(*CLOSURE))
        lift = float(rng.uniform(0.10, 0.20))
        if rng.random() < 0.4:
            dest = to_pelvis(truth["plate_position"])[:2]
        else:
            dest = np.array([rng.uniform(*OBJECT_REGION[0]), rng.uniform(*OBJECT_REGION[1])])
        dest = dest + GRASP_OFFSET[:2] + rng.normal(0, 0.01, 2)
        from embodied_jepa.resting_expert import RestingPlaceExpert

        reach = RestingPlaceExpert.release_pose(dest, reach_radius_m=0.485, floor=0.0, ceiling=0.26)
        dest = reach[:2]
        place_z = max(aim[2] + float(rng.uniform(0.0, 0.05)), reach[2])
        close = closure if failure != "no_close" else -1.0
        hover = aim + [0, 0, HOVER_DZ]
        q = [Waypoint(hover, -1.0, int(rng.integers(50, 80)), yaw)]
        if failure == "early_close":
            q.append(Waypoint(hover, closure, int(rng.integers(15, 25)), yaw))
        q += [
            Waypoint(
                aim, close if failure == "early_close" else -1.0, int(rng.integers(40, 60)), yaw
            ),
            Waypoint(aim, close, int(rng.integers(25, 40)), yaw),
        ]
        if failure == "abort":
            self.queue = q + [Waypoint(self._random_hand_point(), -1.0, 30, yaw)]
            return
        q.append(Waypoint(aim + [0, 0, lift], close, int(rng.integers(40, 70)), yaw))
        transfer = np.r_[dest, max(aim[2] + lift, place_z + 0.03)]
        if failure == "drop":
            mid = (aim + [0, 0, lift] + transfer) / 2
            q += [
                Waypoint(mid, close, int(rng.integers(15, 30)), yaw),
                Waypoint(mid, -1.0, 20, yaw),
            ]
            self.queue = q
            return
        q += [
            Waypoint(transfer, close, int(rng.integers(40, 70)), yaw),
            Waypoint(np.r_[dest, place_z], close, int(rng.integers(25, 40)), yaw),
            Waypoint(np.r_[dest, place_z], -1.0, int(rng.integers(20, 30)), yaw),
            Waypoint(np.r_[dest, place_z + 0.10], -1.0, int(rng.integers(20, 30)), yaw),
        ]
        self.queue = q

    def _random_hand_point(self):
        return np.array([self.rng.uniform(lo, hi) for lo, hi in WANDER_REGION])

    # -- acting -------------------------------------------------------------------------------
    def act(self, robot, truth):
        rng = self.rng
        position, rotation = robot.ee_pose("right")
        if not self.queue:
            self._plan(truth, position)
        wp = self.queue[0]
        wp.steps -= 1
        if wp.steps <= 0:
            self.queue.pop(0)
        scale = robot.manifest["translation_per_step_m"]
        action = np.zeros(14, np.float32)
        if self.skill == "wander":
            action[6:9] = np.clip(0.5 * (wp.target - position) / scale, -0.6, 0.6)
        else:  # e9's tracking law: full gain, clipped at 0.4
            action[6:9] = np.clip((wp.target - position) / scale, -0.4, 0.4)
        if self.mode == "random":
            self.wander_rot = 0.95 * self.wander_rot + rng.normal(0, 0.05, 3)
            desired = rotation_delta(np.clip(self.wander_rot, -0.6, 0.6)) @ TOP_DOWN
        else:
            desired = rotation_delta([0, 0, wp.yaw]) @ TOP_DOWN
        error = 0.5 * sum(np.cross(rotation[:, i], desired[:, i]) for i in range(3))
        action[9:12] = np.clip(error / robot.manifest["rotation_per_step_rad"], -ROT_CLIP, ROT_CLIP)
        # Ornstein-Uhlenbeck action noise, plus perturbation bursts in 'perturbed'.
        self.noise = 0.85 * self.noise + rng.normal(0, self.sigma * np.sqrt(1 - 0.85**2), 6)
        action[6:12] += self.noise * np.array([1, 1, 1, 0.5, 0.5, 0.5])
        if self.mode == "perturbed":
            if self.burst == 0 and rng.random() < 1 / 60:
                self.burst = int(rng.integers(8, 20))
                self.burst_dir = rng.normal(0, 0.7, 6)
            if self.burst > 0:
                self.burst -= 1
                action[6:12] += self.burst_dir * np.array([1, 1, 1, 0.4, 0.4, 0.4])
        action[6:12] = np.clip(action[6:12], -1, 1)
        action[12] = -1.0
        self.grasp = float(np.clip(wp.grasp, self.grasp - GRASP_RAMP, self.grasp + GRASP_RAMP))
        action[13] = np.float32(self.grasp)
        return action


# ----- seeds, salts, sizes (protocol §5, §7, §8) -----------------------------------------------
POLICY_SALT = 8701
SPLIT_SALT = 8702
DEV_SEEDS = range(85000, 86000)
DEBUG_SEEDS = range(86000, 86100)
CORPUS_FIRST_SEED = 850000
SHARD_SIZE = 100
SHARDS = 32
SETTLE_COMMANDS = 10
EPISODE_COMMANDS = 400
MIN_STORED_COMMANDS = 50
VAL_PER_SHARD = 5
TEST_PER_SHARD = 5
DISK_FLOOR_GIB = 12.0
FPS = 20
CAMERA = "onboard_rgb"


def shard_seeds(shard: int, *, first=CORPUS_FIRST_SEED, size=SHARD_SIZE) -> list[int]:
    return list(range(first + size * shard, first + size * (shard + 1)))


def split_rank(seed: int, *, salt: int = SPLIT_SALT) -> str:
    return hashlib.sha256(f"{salt}:{seed}".encode()).hexdigest()


def split_assignment(seeds, *, n_val=VAL_PER_SHARD, n_test=TEST_PER_SHARD) -> dict:
    """Rank by sha256("8702:<seed>"): the lowest n_val val, the next n_test test, the rest train."""
    ranked = sorted(seeds, key=split_rank)
    if len(ranked) < n_val + n_test + 1:
        raise ContractError("a shard needs more stored episodes than val + test")
    return {
        "val": sorted(ranked[:n_val]),
        "test": sorted(ranked[n_val : n_val + n_test]),
        "train": sorted(ranked[n_val + n_test :]),
    }


def episode_id(seed: int) -> str:
    return f"play-{seed}"


# ----- one episode ----------------------------------------------------------------------------
def _palms(robot):
    parts = []
    for side in ("right", "left"):
        position, rotation = robot.ee_pose(side)
        parts.append(np.r_[position, rotation.ravel()])
    return np.concatenate(parts)


def run_episode(robot, seed: int, *, salt: int = POLICY_SALT, commands: int = EPISODE_COMMANDS):
    """Collect one play episode (protocol §5). Returns arrays and facts; stores nothing.

    Every request goes through ``project_candidates`` (the embodiment's backtracking of the arm
    delta through 1, 1/2, ..., 1/64, 0) and is then executed; the stored action is the executed
    one. An infeasible or raising projection (the measured-velocity stop) or a rejected command
    ends the episode early; the frame observed before it is dropped, so every stored action was
    executed and the last frame is the state after the last executed command."""
    rng = np.random.default_rng([salt, seed])
    layout = sample_layout(rng)
    robot.reset(seed, layout=layout)
    sim = robot.sim
    hold = np.zeros(14, np.float32)
    hold[12:] = -1.0
    for _ in range(SETTLE_COMMANDS):
        robot.observe()
        result = robot.execute(hold)
        if result.status not in ("applied", "clipped"):
            raise ContractError(f"settle command rejected: {result.reason}")
    policy = PlayPolicy(rng)
    rows = {k: [] for k in ("frame", "state", "time", "palms", "targets", "truth", "skill")}
    requested, applied, physical, reduced = [], [], [], []
    stop_reason = ""

    def record(observation, skill):
        rows["frame"].append(observation.images[CAMERA][0])
        rows["state"].append(observation.robot_state[0])
        rows["time"].append(float(observation.timestamps[0]))
        rows["palms"].append(_palms(robot))
        rows["targets"].append(sim.targets.copy())
        rows["truth"].append(sim.play_truth())
        rows["skill"].append(skill)

    for _ in range(commands):
        observation = robot.observe()
        request = policy.act(robot, sim.play_truth())
        record(observation, len(policy.skill_log) - 1)
        try:
            projection = robot.project_candidates(request[None, None, None])
        except ContractError as error:
            speed = np.abs(sim.data.qvel[sim.vadr])
            joint = sim.joint_names[int(speed.argmax())]
            stop_reason = f"projection: {error} [{joint} {speed.max():.2f}]"
            break
        if not bool(projection.feasible[0, 0]):
            stop_reason = "projection infeasible"
            break
        command = projection.actions[0, 0, 0].copy()
        result = robot.execute(command)
        if result.status not in ("applied", "clipped"):
            stop_reason = f"execute {result.status}: {result.reason}"
            break
        requested.append(request)
        applied.append(result.applied_action)
        physical.append(robot.denormalize_action(result.applied_action))
        reduced.append(not np.array_equal(result.applied_action, request))
    if len(rows["frame"]) > len(applied):
        for values in rows.values():
            del values[len(applied) :]
    record(robot.observe(), len(policy.skill_log) - 1)
    truths = rows.pop("truth")
    empty = np.zeros((0, 14), np.float32)
    return {
        "seed": int(seed),
        "layout": layout,
        "mode": policy.mode,
        "skill_log": [list(x) for x in policy.skill_log],
        "stop_reason": stop_reason,
        "frames": np.stack(rows["frame"]),
        "states": np.stack(rows["state"]).astype(np.float32),
        "timestamps": np.asarray(rows["time"], np.float64),
        "actions": np.stack(applied).astype(np.float32) if applied else empty,
        "requested": np.stack(requested).astype(np.float32) if requested else empty,
        "physical": np.stack(physical).astype(np.float32) if physical else empty,
        "reduced": np.asarray(reduced, bool),
        "palms": np.stack(rows["palms"]).astype(np.float32),
        "targets": np.stack(rows["targets"]).astype(np.float32),
        "skill_index": np.asarray(rows["skill"], np.int16),
        "object_position": np.stack([t["position"] for t in truths]).astype(np.float32),
        "object_quat": np.stack([t["quat"] for t in truths]).astype(np.float32),
        "object_velocity": np.stack([t["velocity"] for t in truths]).astype(np.float32),
        "robot_contact": np.stack([t["robot_contact"] for t in truths]),
        "hand_contact": np.stack([t["hand_contact"] for t in truths]),
        "grasp_contact": np.stack([t["grasp_contact"] for t in truths]),
        "plate_position": truths[0]["plate_position"].astype(np.float32),
    }


SIDECAR_KEYS = (
    "requested",
    "reduced",
    "palms",
    "targets",
    "skill_index",
    "object_position",
    "object_quat",
    "object_velocity",
    "robot_contact",
    "hand_contact",
    "grasp_contact",
    "plate_position",
)


def on_table_mask(layout) -> np.ndarray:
    return np.array([name in layout["objects"] for name in OBJECT_NAMES])


def episode_facts(ep) -> dict:
    """Per-episode facts for the gate (protocol §10) from sidecar arrays and the layout."""
    pos = np.asarray(ep["object_position"], float)
    on_table = on_table_mask(ep["layout"])
    start = pos[0]
    displacement = np.linalg.norm(pos[:, :, :2] - start[None, :, :2], axis=-1).max(axis=0)
    rise = (pos[:, :, 2] - start[None, :, 2]).max(axis=0)
    touched = np.asarray(ep["robot_contact"]).any(axis=0) & on_table
    grasped = np.asarray(ep["grasp_contact"]).any(axis=0) & on_table
    moved = (displacement >= 0.02) & touched
    lifted = (rise >= 0.03) & grasped
    return {
        "commands": int(len(ep["palms"]) - 1),
        "objects_on_table": int(on_table.sum()),
        "touched": [bool(x) for x in touched],
        "moved": [bool(x) for x in moved],
        "lifted": [bool(x) for x in lifted],
        "grasped": [bool(x) for x in grasped],
        "moved_any": bool(moved.any()),
        "lifted_any": bool(lifted.any()),
        "grasp_contact_any": bool(grasped.any()),
    }


# ----- storage (protocol §6) ----------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_episode(store, robot, ep) -> dict:
    """Write the sidecar, then the episode; returns the episode's metadata."""
    from embodied_jepa.data import Episode

    eid = episode_id(ep["seed"])
    sidecar = Path(store.root) / "play" / f"{eid}.npz"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    if sidecar.exists():
        raise FileExistsError(f"sidecar exists: {sidecar}")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{k: np.asarray(ep[k]) for k in SIDECAR_KEYS})
    sidecar.write_bytes(buffer.getvalue())
    facts = episode_facts(ep)
    metadata = {
        "seed": ep["seed"],
        "scene": SCENE_VERSION,
        "policy": POLICY_VERSION,
        "layout": ep["layout"],
        "mode": ep["mode"],
        "skill_log": ep["skill_log"],
        "stop_reason": ep["stop_reason"],
        "facts": facts,
        "sidecar": f"play/{eid}.npz",
        "sidecar_sha256": _sha256(sidecar),
    }
    episode = Episode(
        episode_id=eid,
        session_id=eid,
        task=TASK_NAME,
        object_id="play",
        container_id="plate",
        observations={CAMERA: ep["frames"]},
        robot_states=ep["states"],
        state_mask=np.ones(ep["states"].shape, dtype=bool),
        actions=ep["actions"],
        timestamps=ep["timestamps"],
        state_schema=robot.state_schema,
        terminated=False,
        truncated=True,
        raw_actions=ep["physical"],
        metadata=metadata,
    )
    store.write_episode(episode)
    return metadata


def free_gib(path="/") -> float:
    return shutil.disk_usage(path).free / 2**30


# ----- the gate (protocol §9, §10) ----------------------------------------------------------------
def _cells(points, region, cell=CELL):
    lo = np.array([r[0] for r in region])
    hi = np.array([r[1] for r in region])
    shape = tuple(int(n) for n in np.round((hi - lo) / cell))
    points = np.asarray(points, float).reshape(-1, len(region))
    inside = np.all((points >= lo) & (points < hi), axis=1)
    index = np.minimum(np.floor((points[inside] - lo) / cell).astype(int), np.array(shape) - 1)
    return shape, [tuple(i) for i in index]


def gate_counts(episodes) -> dict:
    """Accumulate the gate quantities over episodes (dicts with layout and sidecar arrays)."""
    hand_shape, _ = _cells(np.zeros((0, 3)), TEST_HAND_REGION)
    obj_shape, _ = _cells(np.zeros((0, 2)), TEST_OBJECT_REGION)
    hand = np.zeros(hand_shape, np.int64)
    starts = np.zeros(obj_shape, np.int64)
    contacts = np.zeros(obj_shape, np.int64)
    moved = lifted = grasped = 0
    commands = 0
    for ep in episodes:
        _, idx = _cells(np.asarray(ep["palms"])[:, :3], TEST_HAND_REGION)
        for i in idx:
            hand[i] += 1
        facts = episode_facts(ep)
        commands += facts["commands"]
        moved += facts["moved_any"]
        lifted += facts["lifted_any"]
        grasped += facts["grasp_contact_any"]
        on_table = on_table_mask(ep["layout"])
        start = np.asarray(ep["object_position"])[0, :, :2]
        start_cells, contact_cells = set(), set()
        for k in np.flatnonzero(on_table):
            _, idx = _cells(start[k], TEST_OBJECT_REGION)
            if idx:
                start_cells.add(idx[0])
                if facts["touched"][k]:
                    contact_cells.add(idx[0])
        for c in start_cells:
            starts[c] += 1
        for c in contact_cells:
            contacts[c] += 1
    return {
        "episodes": len(episodes) if hasattr(episodes, "__len__") else None,
        "commands": commands,
        "hand": hand,
        "starts": starts,
        "contacts": contacts,
        "moved": moved,
        "lifted": lifted,
        "grasped": grasped,
    }


def gate_row(train: dict, *, total_commands: int, void: str = "") -> dict:
    """Protocol §10 on the train-split counts; first match P1-VOID, P1-PASS, P1-FAIL."""
    n = train["episodes"]
    hours = total_commands / FPS / 3600
    hand_frac = float((train["hand"] >= 200).mean())
    start_ok = bool((train["starts"] >= 20).all())
    contact_frac = float((train["contacts"] >= 10).mean())
    move_frac = train["moved"] / n if n else 0.0
    gates = {
        "G-SIZE": {"hours": hours, "pass": hours >= 10.0},
        "G-HAND": {"fraction_cells_ge_200": hand_frac, "pass": hand_frac >= 0.95},
        "G-OBJ": {
            "all_cells_ge_20_starts": start_ok,
            "min_starts": int(train["starts"].min()),
            "fraction_cells_ge_10_contact_episodes": contact_frac,
            "pass": start_ok and contact_frac >= 0.90,
        },
        "G-MOVE": {"fraction": move_frac, "pass": move_frac >= 0.20},
    }
    if void:
        row = "P1-VOID"
    elif all(g["pass"] for g in gates.values()):
        row = "P1-PASS"
    else:
        row = "P1-FAIL"
    return {
        "row": row,
        "void_reason": void,
        "failing": [k for k, g in gates.items() if not g["pass"]],
        "gates": gates,
    }


def load_sidecar(store_root: Path, row: dict) -> dict:
    path = Path(store_root) / row["metadata"]["sidecar"]
    if _sha256(path) != row["metadata"]["sidecar_sha256"]:
        raise ContractError(f"sidecar hash mismatch: {path}")
    with np.load(path) as data:
        out = {k: data[k] for k in data.files}
    out["layout"] = row["metadata"]["layout"]
    return out


def corpus_manifest_path(root) -> Path:
    return Path(root) / "corpus.json"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
