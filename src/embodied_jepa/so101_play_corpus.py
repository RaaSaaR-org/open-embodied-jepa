"""SO-101 S3 (TASK-099; development data, not gated, no claim): the scripted play collector
``so101_play_policy_v0`` and its corpus storage.

Modelled on TASK-085's ``play_corpus.py`` and TASK-090's ``play_corpus_grasp.py`` (neither is
edited): a per-episode mixture of noisy scripted skills on the S1 tabletop scene (one 3 cm cube,
one flat 8 cm target marker), with perturbations injected on purpose. Skills:

- ``pick_place``: pre-grasp, descend, close, lift, an optional hold, transport, lower, release,
  back off and retreat; the destination is the target marker or a random table point.
- ``pick_lift``: pick, lift and hold, then put the cube down where it is or drop it.
- ``push``: a low, partly closed sweep through the cube.
- ``poke``: descend onto or beside the cube with a random jaw opening and rise again.
- ``wander``: random joint-space or Cartesian targets with a random jaw command.

Injected failures of a pick (``FAILURES``): ``miss`` (aim 2-4 cm off the cube), ``offset`` (a
rigid grasp-aim error beyond S2's measured envelope), ``early_close``, ``no_close``, ``drop``
(open during transport) and ``abort`` (leave after the descent). After a lift the collector
checks (privileged) whether the cube rose with a gripper contact and, if not, may **regrasp**:
open, rise and pick the same cube again. Perturbations: Ornstein-Uhlenbeck noise on the arm
channels (per-mode scale), bursts of random joint deltas in ``perturbed`` mode, random detours
on approach and transport, random tracking gains and speed caps, and gripper-command noise.

The collector reads simulator truth (cube pose, contacts) to choose where to reach and whether
to regrasp: it is a **privileged scripted** collector, and its grasps, lifts and placements are
never learned results.

The stored action is exactly the normalized action ``SO101Embodiment.execute`` applied
(``result.applied_action``: a joint-limit or gripper-rate clip is already in it), and the
physical column is its denormalization; the requested action is kept in the sidecar. A
``stopped`` execution ends the episode early; the frame observed before it is dropped, so every
stored action was executed and the last frame is the state after the last executed command.

NumPy only at import; MuJoCo comes in lazily through ``so101_simulation.py``.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from embodied_jepa.action_schemas import JOINT_DELTA_SO101_V0
from embodied_jepa.contracts import ContractError

TASK = "TASK-099"
DOC = "docs/experiments/so101_play_corpus_v1.md"
SCENE_VERSION = "so101_tabletop_v0"  # S1's scene, unchanged
POLICY_VERSION = "so101_play_policy_v0"
TASK_NAME = "so101_play_v0"
CAMERAS = ("wrist_rgb", "overview_rgb")
IMAGE_SIZE = 224  # 16 x 16 DINOv2 ViT-S/14 patches; S1/S2 inspected frames at this size
FPS = 20  # the embodiment's 0.05 s control step

# ----- scene constants (so101_simulation.py; repeated so this module needs no MuJoCo) ----------
CUBE_HALF_SIZE_M = 0.015
REST_Z = CUBE_HALF_SIZE_M
UP = np.array([0.0, 0.0, 1.0])

# ----- resets (world metres; development, wider than S2's task boxes, which they contain) ------
# S2's declared boxes: cube x 0.18-0.23, |y| <= 0.10; target x 0.17-0.21, |y| <= 0.12.
CUBE_REGION = ((0.16, 0.25), (-0.12, 0.12))
TARGET_REGION = ((0.16, 0.22), (-0.14, 0.14))
MIN_CUBE_TARGET_M = 0.08
HOME_JITTER_RAD = 0.06  # arm joints, uniform +-
HOME_GRIPPER_RAD = (0.2, 1.2)
# Random table destinations of a place (reachable band in front of the base).
PLACE_REGION = ((0.15, 0.24), (-0.14, 0.14))
# Cartesian wander targets of the fingertip site.
WANDER_REGION = ((0.10, 0.30), (-0.20, 0.20), (0.04, 0.22))
WANDER_JOINT_SPAN = np.array([1.0, 0.7, 0.7, 0.9, 2.0])  # uniform +- about home, then clipped
WANDER_MIN_SITE_Z = 0.03

# ----- the policy ------------------------------------------------------------------------------
MODES = ("scripted", "perturbed", "random")
MODE_WEIGHTS = (0.45, 0.40, 0.15)
NOISE_SIGMA = {"scripted": 0.05, "perturbed": 0.15, "random": 0.25}
OU_THETA = 0.85
BURST_RATE = 1 / 50  # perturbed mode: chance per command of starting a burst
BURST_STEPS = (5, 15)
BURST_SCALE = 0.6
GRIPPER_NOISE = {"scripted": 0.0, "perturbed": 0.1, "random": 0.2}
SKILLS = ("pick_place", "pick_lift", "push", "poke", "wander")
SKILL_WEIGHTS = (0.40, 0.20, 0.12, 0.08, 0.20)
FAILURES = ("none", "miss", "offset", "early_close", "no_close", "drop", "abort")
FAILURE_WEIGHTS = (0.55, 0.08, 0.10, 0.07, 0.05, 0.08, 0.07)
DETOUR_P = {"scripted": 0.1, "perturbed": 0.4, "random": 0.4}
DETOUR_M = (0.03, 0.08)
REGRASP_P = 0.7
MAX_REGRASPS = 2
GAIN = (0.5, 1.0)
VMAX = (0.4, 1.0)
# S2's follower geometry: the fixed jaw clears the cube face by 5 mm (so101_followers.py v1).
FIXED_JAW_CLEARANCE_M = 0.005
PREGRASP_HEIGHT_M = (0.05, 0.08)
LIFT_HEIGHT_M = (0.06, 0.12)
RELEASE_CLEARANCE_M = (0.002, 0.02)
GRASP_CHECK_RISE_M = 0.02  # the regrasp check and the per-attempt grasp fact
# A rigid grasp-aim error beyond S2's 64/64 envelope (5 mm toward the fixed jaw, 10 mm along,
# 10 mm up): (closing, along, up) in the zero-offset grasp frame.
OFFSET_FAILURES = (
    ((0.008, 0.012), (0, 0)),
    ((0.0, 0.0), (0.012, 0.02)),
)

# Per-step phase labels (sidecar ``phase``).
PHASES = (
    "settle",
    "reach",
    "descend",
    "close",
    "lift",
    "hold",
    "transport",
    "lower",
    "release",
    "retreat",
    "regrasp_open",
    "push",
    "poke",
    "wander",
    "drop",
)

# ----- seeds, salts, sizes ------------------------------------------------------------------------
POLICY_SALT = 9901
SPLIT_SALT = 9902
DEV_SEEDS = range(990000, 990100)  # policy development (not stored)
PILOT_FIRST_SEED = 990100  # the timing/measurement pilot: 2 shards of 32
CORPUS_FIRST_SEED = 991000  # so101-play-v1: 30 shards of 100
CORPUS_SHARDS = 30
SHARD_SIZE = 100
EPISODE_COMMANDS = 400
SETTLE_COMMANDS = 5
MIN_STORED_COMMANDS = 50
VAL_PER_SHARD = 5
TEST_PER_SHARD = 5
DISK_FLOOR_GIB = 20.0


def corpus_seed_range(first: int = CORPUS_FIRST_SEED, shards: int = CORPUS_SHARDS) -> range:
    return range(first, first + SHARD_SIZE * shards)


def shard_seeds(shard: int, *, first: int, size: int) -> list[int]:
    return list(range(first + size * shard, first + size * (shard + 1)))


def split_rank(seed: int, *, salt: int = SPLIT_SALT) -> str:
    return hashlib.sha256(f"{salt}:{seed}".encode()).hexdigest()


def split_assignment(seeds, *, n_val: int, n_test: int) -> dict:
    """Rank by sha256("9902:<seed>"): lowest n_val val, next n_test test, the rest train."""
    ranked = sorted(seeds, key=split_rank)
    if len(ranked) < n_val + n_test + 1:
        raise ContractError("a shard needs more stored episodes than val + test")
    return {
        "val": sorted(ranked[:n_val]),
        "test": sorted(ranked[n_val : n_val + n_test]),
        "train": sorted(ranked[n_val + n_test :]),
    }


def episode_id(seed: int) -> str:
    return f"so101-play-{seed}"


# ----- the robot ---------------------------------------------------------------------------------
def make_play_robot(*, image_size: int = IMAGE_SIZE, render: bool = True):
    """The S1 scene and embodiment; with ``render`` the renderer is warmed up."""
    from embodied_jepa.so101 import SO101Embodiment
    from embodied_jepa.so101_simulation import SO101Simulation

    sim = SO101Simulation(width=image_size, height=image_size, render=render)
    robot = SO101Embodiment(sim)
    if render:
        sim.render("wrist_rgb")
        sim.render("overview_rgb")
    return robot


def sample_reset(rng, home) -> dict:
    """Cube xy and yaw, target xy (at least 8 cm from the cube) and a jittered home pose."""
    cube = np.array([rng.uniform(*CUBE_REGION[0]), rng.uniform(*CUBE_REGION[1])])
    yaw = float(rng.uniform(-np.pi / 4, np.pi / 4))
    for _ in range(1000):
        target = np.array([rng.uniform(*TARGET_REGION[0]), rng.uniform(*TARGET_REGION[1])])
        if np.linalg.norm(target - cube) >= MIN_CUBE_TARGET_M:
            break
    else:  # pragma: no cover - the regions make this practically impossible
        raise ContractError("could not draw a target away from the cube")
    joints = np.asarray(home, float).copy()
    joints[:5] += rng.uniform(-HOME_JITTER_RAD, HOME_JITTER_RAD, 5)
    joints[5] = rng.uniform(*HOME_GRIPPER_RAD)
    return {
        "cube_xy": cube.tolist(),
        "cube_yaw": yaw,
        "target_xy": target.tolist(),
        "joint_positions": joints.tolist(),
    }


# ----- kinematic helpers (scratch MjData; the live state is never touched) ----------------------
def _scratch_site(sim, q):
    data = sim.mj.MjData(sim.model)
    data.qpos[:] = sim.data.qpos
    data.qpos[sim.qadr[:5]] = q
    sim.mj.mj_kinematics(sim.model, data)
    site = data.site("gripperframe")
    return site.xpos.copy(), site.xmat.reshape(3, 3).copy()


def _grasp_frame(cube_position, cube_quaternion):
    """S2's rule: the jaw closes along the cube face normal most across the base-to-cube ray."""
    w, x, y, z = cube_quaternion
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    face = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    other = np.array([-face[1], face[0], 0.0])
    ray = np.array([cube_position[0], cube_position[1], 0.0])
    ray /= max(np.linalg.norm(ray), 1e-9)
    return face if abs(face @ ray) < abs(other @ ray) else other


def _reachable(cube_position) -> bool:
    """A cube resting on the band the pick and push skills aim at (else the policy wanders)."""
    x, y, z = cube_position
    return 0.12 <= x <= 0.30 and abs(y) <= 0.18 and abs(z - REST_Z) < 0.01


@dataclass
class Segment:
    """Track an arm joint goal for ``steps`` commands with an absolute gripper command."""

    phase: str
    q: np.ndarray
    gripper: float
    steps: int
    gain: float = 1.0
    vmax: float = 1.0


@dataclass
class Attempt:
    """One pick attempt, for the per-attempt grasp fact."""

    skill: str
    failure: str
    regrasp: int
    lift_steps: list = field(default_factory=list)


class PlayPolicy:
    """Per-episode mixture of noisy scripted skills with injected failures and perturbations.

    ``act(robot)`` returns a normalized ``joint_delta_so101_v0`` request and the phase label of
    the command. Queue items are ``Segment`` objects or callables that expand, when reached, into
    more items from the then-current simulator truth (the place after a lift, the regrasp check).
    """

    def __init__(self, rng, robot, *, mode=None):
        self.rng = rng
        self.robot = robot
        self.sim = robot.sim
        self.mode = mode or str(rng.choice(MODES, p=MODE_WEIGHTS))
        self.sigma = NOISE_SIGMA[self.mode]
        self.noise = np.zeros(5)
        self.burst = 0
        self.burst_dir = np.zeros(5)
        self.queue: list = []
        self.skill_log: list[list] = []
        self.attempts: list[Attempt] = []
        self.q_seed = np.asarray(robot.home[:5], float).copy()
        self.speed = (1.0, 1.0)

    # -- kinematics ------------------------------------------------------------------------
    def ik(self, point, closing):
        from embodied_jepa.so101_scripted import solve_ik

        q, residual = solve_ik(self.sim, self.q_seed, np.asarray(point, float), closing)
        self.q_seed = q
        return q, residual

    def line(self, start, end, n, gripper, steps, closing, phase):
        """Straight Cartesian line of the fingertip site, as ``n`` IK goals."""
        rng = self.rng
        start, end = np.asarray(start, float), np.asarray(end, float)
        points = [start + (end - start) * k / n for k in range(1, n + 1)]
        if phase in ("reach", "transport") and rng.random() < DETOUR_P[self.mode]:
            k = int(rng.integers(0, max(1, n - 1)))
            angle = rng.uniform(0, 2 * np.pi)
            size = rng.uniform(*DETOUR_M)
            detour = np.array([np.cos(angle), np.sin(angle), abs(rng.normal(0, 0.5))]) * size
            points[k] = points[k] + detour
        per = max(1, steps // n)
        gain, vmax = self.speed
        out = []
        for point in points:
            q, _ = self.ik(point, closing)
            out.append(Segment(phase, q, gripper, per, gain, vmax))
        return out

    def _site(self):
        position, rotation = _scratch_site(self.sim, self.sim.targets[:5])
        return position, rotation

    # -- skills ----------------------------------------------------------------------------
    def plan(self):
        rng = self.rng
        truth = self.sim.task_truth()
        cube = truth["cube_position"]
        if (self.mode == "random" and rng.random() < 0.6) or not _reachable(cube):
            skill = "wander"
        else:
            skill = str(rng.choice(SKILLS, p=SKILL_WEIGHTS))
        failure = "none"
        if skill in ("pick_place", "pick_lift"):
            failure = str(rng.choice(FAILURES, p=FAILURE_WEIGHTS))
            if skill == "pick_lift" and failure == "drop":
                failure = "none"  # pick_lift drops by its own end choice
        self.speed = (float(rng.uniform(*GAIN)), float(rng.uniform(*VMAX)))
        self.skill_log.append([skill, failure])
        getattr(self, f"_{skill}")(truth, failure)

    def _pick_segments(self, truth, failure, *, regrasp=0):
        """Pre-grasp, descend, close, lift (and the regrasp check); returns the lift point."""
        rng = self.rng
        cube = truth["cube_position"]
        closing = _grasp_frame(cube, truth["cube_quaternion"])
        if rng.random() < 0.15:  # a deliberately mis-turned wrist (a grasp across an edge)
            angle = rng.uniform(0.3, 0.8) * rng.choice([-1, 1])
            c, s = np.cos(angle), np.sin(angle)
            closing = np.array(
                [c * closing[0] - s * closing[1], s * closing[0] + c * closing[1], 0]
            )
        from embodied_jepa.so101_scripted import JAW_CENTRE_OFFSET_M, _closing_axis

        shift = JAW_CENTRE_OFFSET_M - FIXED_JAW_CLEARANCE_M

        def aim(height, offset):
            point = cube + np.array([0.0, 0.0, height])
            q, _ = self.ik(point, closing)
            return point + _closing_axis(self.sim, q) * shift + offset

        _, rotation = _scratch_site(self.sim, self.ik(aim(0.0, np.zeros(3)), closing)[0])
        g = np.zeros(3)
        if failure == "offset":
            (c0, c1), (a0, a1) = OFFSET_FAILURES[int(rng.integers(len(OFFSET_FAILURES)))]
            g[0] = rng.uniform(c0, c1) if c1 > 0 else 0.0
            g[1] = rng.uniform(a0, a1) * rng.choice([-1, 1]) if a1 > 0 else 0.0
        offset = g[0] * rotation[:, 2] + g[1] * rotation[:, 1]
        offset[:2] += rng.normal(0, 0.002, 2)  # aim noise (inside S2's envelope)
        if failure == "miss":
            angle = rng.uniform(0, 2 * np.pi)
            offset[:2] += rng.uniform(0.02, 0.04) * np.array([np.cos(angle), np.sin(angle)])
        pre_h = float(rng.uniform(*PREGRASP_HEIGHT_M))
        pre, grasp = aim(pre_h, offset), aim(0.0, offset)
        start, _ = self._site()
        items = []
        open_cmd = float(rng.uniform(0.6, 1.0))
        items += self.line(start, pre, 3, open_cmd, int(rng.integers(25, 45)), closing, "reach")
        if failure == "early_close":
            mid = pre + (grasp - pre) * rng.uniform(0.3, 0.8)
            items += self.line(pre, mid, 2, open_cmd, int(rng.integers(8, 15)), closing, "descend")
            items.append(Segment("close", self.q_seed, -1.0, int(rng.integers(10, 16))))
            items += self.line(mid, grasp, 3, -1.0, int(rng.integers(10, 20)), closing, "descend")
        else:
            items += self.line(
                pre, grasp, 4, open_cmd, int(rng.integers(15, 30)), closing, "descend"
            )
        if failure == "abort":
            items += self._wander_segments(1, "retreat")
            return items, None
        close = -1.0 if failure != "no_close" else open_cmd
        items.append(Segment("close", self.q_seed, close, int(rng.integers(10, 18))))
        lift = grasp + np.array([0.0, 0.0, float(rng.uniform(*LIFT_HEIGHT_M))])
        items += self.line(grasp, lift, 3, close, int(rng.integers(15, 30)), closing, "lift")
        attempt = Attempt(self.skill_log[-1][0], failure, regrasp)
        self.attempts.append(attempt)
        items.append(lambda: self._mark_attempt(attempt))
        return items, (lift, closing, close)

    def _mark_attempt(self, attempt):
        attempt.lift_steps.append(self.step_index)
        return []

    def _grasped(self):
        truth = self.sim.task_truth()
        return (
            truth["cube_position"][2] - REST_Z >= GRASP_CHECK_RISE_M
            and truth["fixed_jaw_contact"]
            and truth["moving_jaw_contact"]
        )

    def _regrasp_check(self, regrasp, after_lift, info):
        """After a lift: if the cube is not held, maybe open, rise and pick it again."""

        def expand():
            rng = self.rng
            if self._grasped() or regrasp >= MAX_REGRASPS or rng.random() > REGRASP_P:
                return after_lift(info)
            self.skill_log.append(["regrasp", "none"])
            site, _ = self._site()
            items = [Segment("regrasp_open", self.q_seed, 1.0, int(rng.integers(6, 10)))]
            closing = np.cross(UP, np.array([site[0], site[1], 0.0]) / np.hypot(*site[:2]))
            items += self.line(site, site + [0, 0, 0.05], 2, 1.0, 10, closing, "retreat")

            def again():
                truth = self.sim.task_truth()
                if not _reachable(truth["cube_position"]):
                    return []
                picked, new_info = self._pick_segments(truth, "none", regrasp=regrasp + 1)
                if new_info is None:
                    return picked
                return picked + [self._regrasp_check(regrasp + 1, after_lift, new_info)]

            return items + [again]

        return expand

    def _pick_place(self, truth, failure):
        rng = self.rng
        target = truth["target_position"][:2]
        if rng.random() < 0.6:
            dest = target + rng.normal(0, 0.006, 2)
        else:
            dest = np.array([rng.uniform(*PLACE_REGION[0]), rng.uniform(*PLACE_REGION[1])])
        hold = int(rng.integers(0, 20))
        clearance = float(rng.uniform(*RELEASE_CLEARANCE_M))

        def after_lift(info):
            lift, _closing, close = info
            return self._place_segments(lift, close, dest, hold, clearance, failure)

        items, info = self._pick_segments(truth, failure)
        self.queue = items if info is None else items + [self._regrasp_check(0, after_lift, info)]

    def _place_segments(self, lift, close, dest, hold, clearance, failure):
        rng = self.rng
        truth = self.sim.task_truth()
        position, rotation = _scratch_site(self.sim, self.sim.targets[:5])
        in_hand = rotation.T @ (truth["cube_position"] - truth["gripper_site_position"])
        if np.linalg.norm(in_hand) > 0.06:  # nothing in the hand: place an empty grasp
            in_hand = np.array([0.0, 0.0, 0.0])
        items = []
        if hold:
            items.append(Segment("hold", self.q_seed, close, hold))
        ray = np.array([dest[0], dest[1], 0.0]) / np.hypot(*dest)
        closing = np.cross(UP, ray)
        cube_goal = np.array([*dest, REST_Z + clearance])
        q, _ = self.ik(cube_goal, closing)
        _, goal_rotation = _scratch_site(self.sim, q)
        down = cube_goal - goal_rotation @ in_hand
        above = down + np.array([0.0, 0.0, max(lift[2] - down[2], 0.04)])
        start = position
        if failure == "drop":
            mid = start + (above - start) * rng.uniform(0.3, 0.8)
            items += self.line(
                start, mid, 3, close, int(rng.integers(12, 25)), closing, "transport"
            )
            items.append(Segment("drop", self.q_seed, 1.0, int(rng.integers(10, 20))))
            items += self._wander_segments(1, "retreat")
            return items
        items += self.line(start, above, 4, close, int(rng.integers(25, 45)), closing, "transport")
        items += self.line(above, down, 3, close, int(rng.integers(15, 25)), closing, "lower")
        release = float(rng.uniform(0.0, 0.6))
        items.append(Segment("release", self.q_seed, release, int(rng.integers(8, 14))))
        jaw = goal_rotation[:, 2]
        clear = down - 0.008 * jaw
        items += self.line(down, clear, 2, release, 6, closing, "retreat")
        up = clear + np.array([0.0, 0.0, float(rng.uniform(0.05, 0.10))])
        items += self.line(clear, up, 3, release, int(rng.integers(12, 20)), closing, "retreat")
        return items

    def _pick_lift(self, truth, failure):
        rng = self.rng

        def after_lift(info):
            lift, closing, close = info
            items = [Segment("hold", self.q_seed, close, int(rng.integers(10, 40)))]
            here, _ = self._site()
            if rng.random() < 0.5:  # drop it from the lift height
                items.append(Segment("drop", self.q_seed, 1.0, int(rng.integers(8, 14))))
                items += self.line(here, here + [0, 0, 0.04], 2, 1.0, 10, closing, "retreat")
                return items
            rise = self.sim.task_truth()["cube_position"][2] - REST_Z
            down = here - np.array([0.0, 0.0, float(np.clip(rise - 0.004, 0.0, 0.15))])
            items += self.line(here, down, 3, close, int(rng.integers(15, 25)), closing, "lower")
            items.append(Segment("release", self.q_seed, 0.6, 10))
            items += self.line(down, down + [0, 0, 0.06], 2, 0.6, 12, closing, "retreat")
            return items

        items, info = self._pick_segments(truth, failure)
        self.queue = items if info is None else items + [self._regrasp_check(0, after_lift, info)]

    def _push(self, truth, _failure):
        rng = self.rng
        cube = truth["cube_position"]
        angle = rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        height = REST_Z + rng.uniform(-0.005, 0.01)
        start = cube - direction * 0.06
        end = cube + direction * rng.uniform(0.03, 0.08)
        for point in (start, end):  # stay on the reachable table band
            point[0] = np.clip(point[0], 0.12, 0.30)
            point[1] = np.clip(point[1], -0.18, 0.18)
        start[2] = end[2] = height
        closing = np.cross(UP, direction)
        grip = float(rng.uniform(-1.0, 0.2))
        here, _ = self._site()
        items = self.line(
            here, start + [0, 0, 0.06], 3, grip, int(rng.integers(20, 35)), closing, "reach"
        )
        items += self.line(start + [0, 0, 0.06], start, 2, grip, 12, closing, "push")
        items += self.line(start, end, 4, grip, int(rng.integers(20, 35)), closing, "push")
        items += self.line(end, end + [0, 0, 0.06], 2, grip, 10, closing, "retreat")
        self.queue = items

    def _poke(self, truth, _failure):
        rng = self.rng
        cube = truth["cube_position"]
        closing = _grasp_frame(cube, truth["cube_quaternion"])
        point = cube + np.r_[rng.normal(0, 0.015, 2), rng.uniform(0.0, 0.03)]
        grip = float(rng.uniform(-1.0, 1.0))
        here, _ = self._site()
        items = self.line(
            here, point + [0, 0, 0.07], 3, grip, int(rng.integers(20, 35)), closing, "reach"
        )
        items += self.line(
            point + [0, 0, 0.07], point, 2, grip, int(rng.integers(10, 20)), closing, "poke"
        )
        items += self.line(point, point + [0, 0, 0.07], 2, grip, 10, closing, "retreat")
        self.queue = items

    def _wander(self, _truth, _failure):
        self.queue = self._wander_segments(int(self.rng.integers(2, 5)), "wander")

    def _wander_segments(self, count, phase):
        rng = self.rng
        items = []
        home = np.asarray(self.robot.home[:5], float)
        limits = self.sim.limits[:5]
        for _ in range(count):
            grip = float(rng.choice([-1.0, 1.0, rng.uniform(-1, 1)]))
            steps = int(rng.integers(15, 40))
            if rng.random() < 0.5:  # joint space
                for _ in range(50):
                    q = np.clip(
                        home + rng.uniform(-WANDER_JOINT_SPAN, WANDER_JOINT_SPAN),
                        limits[:, 0],
                        limits[:, 1],
                    )
                    site, _ = _scratch_site(self.sim, q)
                    if site[2] >= WANDER_MIN_SITE_Z and site[0] >= 0.05:
                        break
                else:
                    q = home.copy()
                self.q_seed = q
                items.append(Segment(phase, q, grip, steps, *self.speed))
            else:  # Cartesian, approach down, random jaw yaw
                point = np.array([rng.uniform(lo, hi) for lo, hi in WANDER_REGION])
                angle = rng.uniform(-np.pi, np.pi)
                closing = np.array([np.cos(angle), np.sin(angle), 0.0])
                q, _ = self.ik(point, closing)
                items.append(Segment(phase, q, grip, steps, *self.speed))
        return items

    # -- acting ----------------------------------------------------------------------------
    def act(self, step_index: int):
        rng = self.rng
        self.step_index = step_index
        guard = 0
        while not self.queue or callable(self.queue[0]):
            guard += 1
            if guard > 20:  # pragma: no cover - a planning loop that produces nothing
                self.queue = self._wander_segments(1, "wander")
                break
            if not self.queue:
                self.plan()
                continue
            expand = self.queue.pop(0)
            self.queue[:0] = expand()
        seg = self.queue[0]
        seg.steps -= 1
        if seg.steps <= 0:
            self.queue.pop(0)
        scales = self.robot.scales.astype(float)
        arm = np.clip(seg.gain * (seg.q - self.sim.targets[:5]) / scales, -seg.vmax, seg.vmax)
        self.noise = OU_THETA * self.noise + rng.normal(0, self.sigma * np.sqrt(1 - OU_THETA**2), 5)
        arm = arm + self.noise
        if self.mode == "perturbed":
            if self.burst == 0 and rng.random() < BURST_RATE:
                self.burst = int(rng.integers(*BURST_STEPS))
                self.burst_dir = rng.normal(0, BURST_SCALE, 5)
            if self.burst > 0:
                self.burst -= 1
                arm = arm + self.burst_dir
        gripper = (
            seg.gripper + rng.normal(0, GRIPPER_NOISE[self.mode])
            if GRIPPER_NOISE[self.mode]
            else seg.gripper
        )
        action = np.concatenate((np.clip(arm, -1, 1), [np.clip(gripper, -1, 1)]))
        return action.astype(np.float32), seg.phase


# ----- one episode ------------------------------------------------------------------------------
TRUTH_KEYS = (
    "cube_position",
    "cube_quaternion",
    "cube_velocity",
    "cube_angular_velocity",
    "gripper_site_position",
)
CONTACT_KEYS = ("fixed_jaw_contact", "moving_jaw_contact", "gripper_contact")


def run_episode(robot, seed: int, *, salt: int = POLICY_SALT, commands: int = EPISODE_COMMANDS):
    """Collect one play episode. Returns arrays and facts; stores nothing."""
    rng = np.random.default_rng([salt, seed])
    reset = sample_reset(rng, robot.home)
    robot.reset(seed, **reset)
    sim = robot.sim
    for _ in range(SETTLE_COMMANDS):
        robot.observe()
        result = robot.execute(robot.hold_action())
        if result.status not in ("applied", "clipped"):
            raise ContractError(f"settle command rejected: {result.reason}")
    policy = PlayPolicy(rng, robot)
    frames = {camera: [] for camera in CAMERAS}
    rows = {
        k: [] for k in ("state", "time", "targets", "phase", "skill", *TRUTH_KEYS, *CONTACT_KEYS)
    }
    requested, applied, physical, clipped = [], [], [], []
    stop_reason = ""

    def record(observation, phase):
        for camera in CAMERAS:
            frames[camera].append(observation.images[camera][0])
        rows["state"].append(observation.robot_state[0])
        rows["time"].append(float(observation.timestamps[0]))
        rows["targets"].append(sim.targets.copy())
        rows["phase"].append(PHASES.index(phase))
        rows["skill"].append(max(len(policy.skill_log) - 1, 0))
        truth = sim.task_truth()
        for key in TRUTH_KEYS:
            rows[key].append(np.asarray(truth[key], float))
        for key in CONTACT_KEYS:
            rows[key].append(bool(truth[key]))

    for step in range(commands):
        observation = robot.observe()
        request, phase = policy.act(step)
        record(observation, phase)
        result = robot.execute(request)
        if result.status not in ("applied", "clipped"):
            stop_reason = f"execute {result.status}: {result.reason}"
            break
        requested.append(request)
        applied.append(result.applied_action)
        physical.append(robot.denormalize_action(result.applied_action))
        clipped.append(result.status == "clipped")
    n = len(applied)
    for values in (*frames.values(), *rows.values()):
        del values[n:]
    record(robot.observe(), "settle" if not n else PHASES[rows["phase"][-1]])
    empty = np.zeros((0, 6), np.float32)
    out = {
        "seed": int(seed),
        "reset": reset,
        "mode": policy.mode,
        "skill_log": policy.skill_log,
        "attempts": [
            {
                "skill": a.skill,
                "failure": a.failure,
                "regrasp": a.regrasp,
                "lift_end_step": a.lift_steps[0] if a.lift_steps else None,
            }
            for a in policy.attempts
        ],
        "stop_reason": stop_reason,
        "frames": {camera: np.stack(frames[camera]) for camera in CAMERAS},
        "states": np.stack(rows["state"]).astype(np.float32),
        "timestamps": np.asarray(rows["time"], np.float64),
        "actions": np.stack(applied).astype(np.float32) if applied else empty,
        "requested": np.stack(requested).astype(np.float32) if requested else empty,
        "physical": np.stack(physical).astype(np.float32) if physical else empty,
        "clipped": np.asarray(clipped, bool),
        "targets": np.stack(rows["targets"]).astype(np.float32),
        "phase": np.asarray(rows["phase"], np.int8),
        "skill_index": np.asarray(rows["skill"], np.int16),
        "target_position": np.asarray(sim.task_truth()["target_position"], np.float32),
    }
    for key in TRUTH_KEYS:
        out[key] = np.stack(rows[key]).astype(np.float32)
    for key in CONTACT_KEYS:
        out[key] = np.asarray(rows[key], bool)
    return out


SIDECAR_KEYS = (
    "requested",
    "clipped",
    "targets",
    "phase",
    "skill_index",
    *TRUTH_KEYS,
    *CONTACT_KEYS,
    "target_position",
)


# ----- facts (offline, from sidecar truth; S2's scorers) ----------------------------------------
def _runs(mask):
    out, count = np.zeros(len(mask), int), 0
    for i, value in enumerate(mask):
        count = count + 1 if value else 0
        out[i] = count
    return out


def _windows(mask, length):
    """Disjoint runs of at least ``length`` consecutive True values."""
    runs = _runs(np.asarray(mask, bool))
    ends = [
        i
        for i in range(len(runs))
        if runs[i] >= length and (i + 1 == len(runs) or runs[i + 1] == 0)
    ]
    return len(ends)


def episode_facts(ep) -> dict:
    """Per-episode facts from sidecar arrays: S2's scorers and window counts."""
    from embodied_jepa.so101_tasks import LIFT_V0, PLACE_V0, lift_outcome, place_outcome

    p = np.asarray(ep["cube_position"], float)
    contact = np.asarray(ep["gripper_contact"], bool)
    both = np.asarray(ep["fixed_jaw_contact"], bool) & np.asarray(ep["moving_jaw_contact"], bool)
    rise = p[:, 2] - REST_Z
    lift = lift_outcome(p, contact, LIFT_V0)
    place = place_outcome(
        p,
        np.asarray(ep["cube_velocity"], float),
        np.asarray(ep["cube_angular_velocity"], float),
        contact,
        np.asarray(ep["target_position"], float)[:2],
        PLACE_V0,
    )
    held = (rise >= LIFT_V0.lift_height_m) & contact
    # A placement anywhere: after a pick, the cube resting free on the table for 20 steps.
    v = np.linalg.norm(np.asarray(ep["cube_velocity"], float), axis=1)
    w = np.linalg.norm(np.asarray(ep["cube_angular_velocity"], float), axis=1)
    picked = _runs((rise >= PLACE_V0.pick_height_m) & contact) >= PLACE_V0.pick_steps
    after_pick = np.cumsum(picked) > 0
    resting = (
        (np.abs(rise) <= PLACE_V0.rest_height_tolerance_m)
        & (v <= PLACE_V0.max_speed_m_s)
        & (w <= PLACE_V0.max_angular_speed_rad_s)
        & ~contact
    )
    attempts = []
    for a in ep["attempts"]:
        step = a["lift_end_step"]
        ok = None
        if step is not None and step < len(p):
            ok = bool(rise[step] >= GRASP_CHECK_RISE_M and both[step])
        attempts.append({**a, "grasped": ok})
    commands = int(len(p) - 1)
    return {
        "commands": commands,
        "lift_success": bool(lift["success"]),
        "lift_windows": _windows(held, LIFT_V0.hold_steps),
        "held_steps": int(held.sum()),
        "max_rise_m": float(rise.max()),
        "picked": bool(place["picked"]),
        "place_success": bool(place["success"]),
        "place_windows_anywhere": _windows(after_pick & resting, PLACE_V0.rest_steps),
        "contact_steps": int(contact.sum()),
        "both_jaw_steps": int(both.sum()),
        "cube_displacement_m": float(np.linalg.norm(p[:, :2] - p[0, :2], axis=1).max()),
        "cube_lost": lift["lost_step"] is not None,
        "attempts": attempts,
    }


# ----- storage ------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def create_store(root, robot, provenance):
    from embodied_jepa.schema_data import SchemaDatasetStore

    return SchemaDatasetStore.create(
        root,
        fps=FPS,
        state_schema=robot.state_schema,
        action_manifest=robot.manifest,
        provenance=provenance,
        robot_type="so101",
        action_schema=JOINT_DELTA_SO101_V0,
    )


def write_episode(store, robot, ep) -> dict:
    """Write the sidecar, then the episode; returns the episode's metadata."""
    from embodied_jepa.schema_data import SchemaEpisode

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
        "reset": ep["reset"],
        "mode": ep["mode"],
        "skill_log": ep["skill_log"],
        "stop_reason": ep["stop_reason"],
        "facts": facts,
        "sidecar": f"play/{eid}.npz",
        "sidecar_sha256": sha256_file(sidecar),
    }
    episode = SchemaEpisode(
        episode_id=eid,
        session_id=eid,
        task=TASK_NAME,
        object_id="cube",
        container_id="target",
        observations=ep["frames"],
        robot_states=ep["states"],
        state_mask=np.ones(ep["states"].shape, dtype=bool),
        actions=ep["actions"],
        timestamps=ep["timestamps"],
        state_schema=robot.state_schema,
        terminated=False,
        truncated=True,
        raw_actions=ep["physical"],
        metadata=metadata,
        action_schema=JOINT_DELTA_SO101_V0,
    )
    store.write_episode(episode)
    return metadata


def load_sidecar(store_root, row) -> dict:
    path = Path(store_root) / row["metadata"]["sidecar"]
    if sha256_file(path) != row["metadata"]["sidecar_sha256"]:
        raise ContractError(f"sidecar hash mismatch: {path}")
    with np.load(path) as data:
        out = {k: data[k] for k in data.files}
    out["reset"] = row["metadata"]["reset"]
    return out


def free_gib(path) -> float:
    return shutil.disk_usage(path).free / 2**30


def corpus_manifest_path(root) -> Path:
    return Path(root) / "corpus.json"


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


# Files whose bytes define the collection (recorded in the corpus manifest).
CODE_FILES = (
    "src/embodied_jepa/so101_play_corpus.py",
    "src/embodied_jepa/so101_scripted.py",
    "src/embodied_jepa/so101_tasks.py",
    "src/embodied_jepa/so101.py",
    "src/embodied_jepa/so101_simulation.py",
    "src/embodied_jepa/schema_data.py",
    "src/embodied_jepa/action_schemas.py",
    "src/embodied_jepa/data.py",
    "configs/so101_sim_action.json",
    "assets/so101_manifest.json",
    "scripts/collect_so101_play.py",
)
