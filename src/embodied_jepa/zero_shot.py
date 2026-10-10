"""TASK-088 (Phase 3 of ``docs/JEPA_ZERO_SHOT_PLAN.md``): zero-shot reach and grasp-and-lift by
short-horizon planning with the TASK-087 world models, on G1's right arm and right Dex3 in MuJoCo.

Protocol: ``docs/experiments/zero_shot_reach_grasp_v1.md``. This module holds the NumPy part:
seeds, salts and cohorts, the reset layouts, the task's goal demonstrators (privileged scripted
code that *produces* goals; never a controller arm that is counted as learned), the success
scorers, the subgoal switching rule, the non-learned baseline controllers (hold, random, the
scripted IK follower) and the statistics and rows. The torch part (the world-model adapter for
:class:`embodied_jepa.planning.CEMPlanner`, the frozen encoder and the episode loop) is
``zero_shot_runtime``.

Information flow (protocol §4): a controller arm sees only the onboard 112 px frame, the robot's
own measured joint state (proprioception) and the goal: goal image(s) and the goal robot pose (the
right arm and hand joint positions recorded when the task's demonstrator reached the goal, and the
palm pose the embodiment's forward kinematics gives for them). Object truth from the simulator is
read only by the task's demonstrator (to make goals) and by the scorer.

NumPy only at import; MuJoCo is imported lazily by :mod:`embodied_jepa.simulation`. Nothing here
is registered or imported by ``embodied_jepa``.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from embodied_jepa import play_corpus as pc
from embodied_jepa.contracts import ContractError
from embodied_jepa.embodiment import rotation_delta

TASK = "TASK-088"
PROTOCOL = "docs/experiments/zero_shot_reach_grasp_v1.md"
TASKS = ("reach", "grasp")

# ----- seeds and salts (protocol §9) -------------------------------------------------------------
SALTS = {"layout": 8801, "goal": 8802, "cem": 8803, "random": 8804, "bootstrap": 8805, "lam": 8806}
COHORTS = {
    "debug-reach": range(88900, 88950),
    "debug-grasp": range(88950, 89000),
    "k0-reach": range(88000, 88032),
    "k0-grasp": range(88100, 88132),
    "dev-reach": range(88200, 88216),
    "dev-grasp": range(88300, 88316),
    "gated-reach": range(880000, 880064),
    "gated-grasp": range(881000, 881064),
}


def cohort_task(cohort: str) -> str:
    if cohort not in COHORTS:
        raise ContractError(f"unknown cohort {cohort}")
    return cohort.rsplit("-", 1)[1]


def check_seed(cohort: str, seed: int) -> None:
    if int(seed) not in COHORTS[cohort]:
        raise ContractError(f"seed {seed} is not in cohort {cohort}")


# ----- scene, layouts (protocol §3) --------------------------------------------------------------
OBJECT_REGION = pc.TEST_OBJECT_REGION  # world xy, the Phase 3 test object region of TASK-085
PLATE_REGION = pc.PLATE_REGION
CLEARANCE = 0.01
GRASP_OBJECTS = pc.OBJECT_NAMES  # apple, cube, banana, can


def _place(rng, names, plate):
    placed = [(np.asarray(plate, float), pc.PLATE_RADIUS)]
    objects = {}
    for name in names:
        radius = pc.OBJECTS[name]["radius"]
        for _ in range(200):
            xy = np.array([rng.uniform(*OBJECT_REGION[0]), rng.uniform(*OBJECT_REGION[1])])
            if all(np.linalg.norm(xy - p) >= radius + r + CLEARANCE for p, r in placed):
                break
        else:
            return None
        placed.append((xy, radius))
        objects[name] = xy.tolist()
    return objects


def sample_layout(task: str, seed: int, attempt: int = 0) -> dict:
    """A fresh layout: the plate in TASK-085's plate region, every object in the Phase 3 test
    object region. Reach: 1-3 of the four objects. Grasp: a target (uniform over the four) and
    0-2 distractors. ``attempt`` > 0 draws a replacement after a demonstrator rejection."""
    rng = np.random.default_rng([SALTS["layout"], int(seed), int(attempt)])
    for _ in range(1000):
        plate = [rng.uniform(*PLATE_REGION[0]), rng.uniform(*PLATE_REGION[1])]
        if task == "reach":
            count = int(rng.integers(1, 4))
            names = [pc.OBJECT_NAMES[i] for i in sorted(rng.choice(4, count, replace=False))]
            target = None
        elif task == "grasp":
            target = pc.OBJECT_NAMES[int(rng.integers(4))]
            others = [n for n in pc.OBJECT_NAMES if n != target]
            k = int(rng.integers(0, 3))
            names = [target] + [others[i] for i in sorted(rng.choice(3, k, replace=False))]
        else:
            raise ContractError(f"unknown task {task}")
        objects = _place(rng, names, plate)
        if objects is None:
            continue
        yaw = {name: float(rng.uniform(-np.pi, np.pi)) for name in objects}
        return {"plate": plate, "objects": objects, "yaw": yaw, "target": target}
    raise ContractError("could not sample a layout")


# ----- robot, control (protocol §3, §5) ----------------------------------------------------------
SETTLE_COMMANDS = pc.SETTLE_COMMANDS  # 10 hold commands after the reset, as play-v1
TOP_DOWN = pc.TOP_DOWN
TRACK_CLIP = 0.4  # e9's tracking law (play-v1's scripted skills)
ROT_CLIP = pc.ROT_CLIP
GRASP_RAMP = pc.GRASP_RAMP
HOLD = np.zeros(14, np.float32)
HOLD[12:] = -1.0
PLAYED = (6, 7, 8, 9, 10, 11, 13)
FIXED_LOWER = np.array([0.0] * 6 + [-1.0] * 6 + [-1.0, -1.0], np.float32)
FIXED_UPPER = np.array([0.0] * 6 + [1.0] * 6 + [-1.0, 1.0], np.float32)


def tracking_action(robot, position, rotation, target, target_rotation, grasp, *, clip=TRACK_CLIP):
    """e9's tracking law (play-v1's scripted skills): arm command = (target - palm) / 1.5 cm
    clipped at ``clip``; orientation error / 0.06 rad clipped at 0.8; the right grasp as given."""
    action = HOLD.copy()
    scale = robot.manifest["translation_per_step_m"]
    action[6:9] = np.clip((np.asarray(target) - position) / scale, -clip, clip)
    error = 0.5 * sum(np.cross(rotation[:, i], target_rotation[:, i]) for i in range(3))
    action[9:12] = np.clip(error / robot.manifest["rotation_per_step_rad"], -ROT_CLIP, ROT_CLIP)
    action[13] = np.float32(grasp)
    return action


def synergy_grasp(robot, hand_q) -> float:
    """The right grasp scalar whose synergy is nearest the 7 hand joint angles (least squares on
    the open -> closed line), clipped to [-1, 1]."""
    opened = np.asarray(robot.manifest["right_open_rad"])
    closed = np.asarray(robot.manifest["right_closed_rad"])
    d = closed - opened
    t = float(np.dot(np.asarray(hand_q) - opened, d) / np.dot(d, d))
    return float(np.clip(2 * t - 1, -1, 1))


class Stopped(Exception):
    """The embodiment refused the command (infeasible projection, the measured-velocity stop or a
    rejected execution): the episode ends there, a failure for every arm alike."""


def execute(robot, request) -> np.ndarray:
    """play-v1's execution path: the request is projected by the embodiment's own backtracking
    (1, 1/2, ..., 1/64, 0) on the current observation and then executed. Returns the applied
    action; raises :class:`Stopped` when the robot refuses."""
    try:
        projection = robot.project_candidates(np.asarray(request, np.float32)[None, None, None])
    except ContractError as error:
        raise Stopped(f"projection: {error}") from error
    if not bool(projection.feasible[0, 0]):
        raise Stopped("projection infeasible")
    result = robot.execute(projection.actions[0, 0, 0].copy())
    if result.status not in ("applied", "clipped"):
        raise Stopped(f"execute {result.status}: {result.reason}")
    return result.applied_action


def reset(robot, layout) -> None:
    """Reset to the layout, open both hands and settle with 10 hold commands (play-v1 §5)."""
    robot.reset(0, layout=layout)
    for _ in range(SETTLE_COMMANDS):
        robot.observe()
        result = robot.execute(HOLD)
        if result.status not in ("applied", "clipped"):
            raise ContractError(f"settle command rejected: {result.reason}")


def proprio(robot, observation) -> dict:
    """What a controller may read of the robot besides the image: the measured joint state and
    the palm pose by the embodiment's forward kinematics of the *measured* joints."""
    names = list(robot.state_schema.names)
    from embodied_jepa import grounded_wm as gw

    cols = [names.index(n) for n in gw.STATE_NAMES]
    data = robot._snapshot_kinematics()
    position, rotation = robot.ee_pose("right", data=data)
    return {
        "state28": observation.robot_state[0, cols].astype(np.float32),
        "palm": position,
        "rotation": rotation,
    }


# ----- goals (protocol §4): the task's demonstrators ---------------------------------------------
REACH_REGION = ((0.24, 0.44), (-0.30, -0.02), (0.10, 0.26))  # pelvis xyz of the goal palm
REACH_MIN_DISTANCE = 0.10  # m from the start palm
REACH_YAW = 0.3  # rad, the goal palm yaw about the top-down orientation
DEMO_REACH_STEPS = 120
DEMO_SETTLE = 10
DEMO_TOLERANCE = 0.005
GOAL_ATTEMPTS = 30
OBJECT_STILL = 0.005  # m: a reach goal may not move an object
GRASP_HOVER = 0.08
GRASP_LIFT = 0.12
GRASP_PHASES = {"hover": 60, "descend": 50, "close": 35, "lift": 50, "hold": 20}
GRASP_CLOSURE = 1.0
SUBGOALS = ("hover", "pregrasp", "grasp", "lift")
PRESS_DEPTH = 0.05


def _track_until(robot, target, target_rotation, grasp, steps, *, stop_within=None, log=None):
    """Drive the tracking law for up to ``steps`` commands (stop early when the palm is within
    ``stop_within`` of the target); ``log`` receives per-step truth."""
    sim = robot.sim
    for _ in range(steps):
        obs = robot.observe()
        p = proprio(robot, obs)
        if stop_within is not None and np.linalg.norm(p["palm"] - target) < stop_within:
            return True
        execute(
            robot, tracking_action(robot, p["palm"], p["rotation"], target, target_rotation, grasp)
        )
        if log is not None:
            log.append(sim.play_truth())
    return stop_within is None


def _goal_record(robot):
    obs = robot.observe()
    p = proprio(robot, obs)
    return {
        "frame": obs.images[pc.CAMERA][0].copy(),
        "state28": p["state28"],
        "palm": p["palm"].astype(np.float32),
        "rotation": p["rotation"].astype(np.float32),
    }


def make_reach_goal(robot, seed: int) -> dict:
    """Protocol §4.1: draw a palm target and yaw; the demonstrator (e9's tracking law) drives the
    palm there from the episode's start; accepted when it converges without touching or moving an
    object; the goal is its final frame and pose."""
    rng = np.random.default_rng([SALTS["goal"], int(seed)])
    rejected = []
    for attempt in range(GOAL_ATTEMPTS):
        layout = sample_layout("reach", seed, attempt)
        reset(robot, layout)
        start = proprio(robot, robot.observe())
        target = np.array([rng.uniform(*r) for r in REACH_REGION])
        yaw = float(rng.uniform(-REACH_YAW, REACH_YAW))
        if np.linalg.norm(target - start["palm"]) < REACH_MIN_DISTANCE:
            rejected.append("too close")
            continue
        rotation = rotation_delta([0, 0, yaw]) @ TOP_DOWN
        objects0 = robot.sim.play_truth()["position"].copy()
        log = []
        try:
            ok = _track_until(
                robot, target, rotation, -1.0, DEMO_REACH_STEPS, stop_within=DEMO_TOLERANCE, log=log
            )
            for _ in range(DEMO_SETTLE):
                robot.observe()
                execute(robot, HOLD)
                log.append(robot.sim.play_truth())
        except Stopped as error:
            rejected.append(f"stopped: {error}")
            continue
        moved = max(np.linalg.norm(t["position"] - objects0, axis=1).max() for t in log)
        touched = any(t["robot_contact"].any() for t in log)
        if not ok:
            rejected.append("did not converge")
            continue
        if moved > OBJECT_STILL or touched:
            rejected.append("touched an object")
            continue
        goal = _goal_record(robot)
        if np.linalg.norm(goal["palm"] - target) > 0.02:
            rejected.append("settled off target")
            continue
        return {
            "task": "reach",
            "seed": int(seed),
            "attempt": attempt,
            "rejected": rejected,
            "layout": layout,
            "start_palm": start["palm"].tolist(),
            "target": target.tolist(),
            "yaw": yaw,
            "subgoals": [goal],
        }
    raise ContractError(f"no reach goal for seed {seed}: {rejected}")


def _target_index(layout) -> int:
    return pc.OBJECT_NAMES.index(layout["target"])


def grasp_truth_success(log, index, z0) -> bool:
    return lift_success(
        [t["position"][index][2] - z0 for t in log], [bool(t["grasp_contact"][index]) for t in log]
    )


def make_grasp_goal(robot, seed: int) -> dict:
    """Protocol §4.2: the demonstrator (e9's pick recipe, no noise) picks the target; accepted when
    it meets the lift criterion itself; the four subgoals are its frame and pose at the end of
    the hover, of the descent (open), of the close and of the hold after the lift."""
    rng = np.random.default_rng([SALTS["goal"], int(seed)])
    rejected = []
    for attempt in range(GOAL_ATTEMPTS):
        layout = sample_layout("grasp", seed, attempt)
        yaw = float(rng.uniform(-pc.YAW_RANGE, pc.YAW_RANGE))
        reset(robot, layout)
        start = proprio(robot, robot.observe())
        index = _target_index(layout)
        truth = robot.sim.play_truth()
        obj = pc.to_pelvis(truth["position"][index])
        z0 = float(truth["position"][index][2])
        aim = obj + pc.GRASP_OFFSET
        rotation = rotation_delta([0, 0, yaw]) @ TOP_DOWN
        log, subgoals = [], []
        grasp = -1.0

        def run(target, want, steps, rotation=rotation, log=log):
            nonlocal grasp
            for _ in range(steps):
                grasp = float(np.clip(want, grasp - GRASP_RAMP, grasp + GRASP_RAMP))
                obs = robot.observe()
                p = proprio(robot, obs)
                execute(
                    robot, tracking_action(robot, p["palm"], p["rotation"], target, rotation, grasp)
                )
                log.append(robot.sim.play_truth())

        try:
            run(aim + [0, 0, GRASP_HOVER], -1.0, GRASP_PHASES["hover"])
            subgoals.append(_goal_record(robot))
            run(aim, -1.0, GRASP_PHASES["descend"])
            subgoals.append(_goal_record(robot))
            run(aim, GRASP_CLOSURE, GRASP_PHASES["close"])
            subgoals.append(_goal_record(robot))
            run(aim + [0, 0, GRASP_LIFT], GRASP_CLOSURE, GRASP_PHASES["lift"])
            run(aim + [0, 0, GRASP_LIFT], GRASP_CLOSURE, GRASP_PHASES["hold"])
            subgoals.append(_goal_record(robot))
        except Stopped as error:
            rejected.append(f"stopped: {error}")
            continue
        if not grasp_truth_success(log, index, z0):
            rejected.append("demonstrator did not lift")
            continue
        return {
            "task": "grasp",
            "seed": int(seed),
            "attempt": attempt,
            "rejected": rejected,
            "layout": layout,
            "start_palm": start["palm"].tolist(),
            "target": layout["target"],
            "target_z0": z0,
            "yaw": yaw,
            "subgoals": subgoals,
        }
    raise ContractError(f"no grasp goal for seed {seed}: {rejected}")


def make_goal(robot, task: str, seed: int) -> dict:
    return make_reach_goal(robot, seed) if task == "reach" else make_grasp_goal(robot, seed)


# ----- scoring (protocol §6) ---------------------------------------------------------------------
REACH_TOLERANCE = 0.05  # m
REACH_DWELL = 10  # consecutive control steps (0.5 s)
REACH_REPORT_TOLERANCES = (0.03, 0.05, 0.08)
LIFT_HEIGHT = 0.05  # m above the object's start height
LIFT_DWELL = 20  # consecutive control steps (1 s) lifted and in grasp contact
BUDGET = {"reach": 150, "grasp": 400}  # grasp: 3 x the timeout + 100 on the last subgoal
SUBGOAL_TIMEOUT = 100
SWITCH_PALM = 0.01
SWITCH_HAND_RMS = 0.15
SWITCH_ROT = 0.17  # rad (about 10 degrees)
SWITCH_DWELL = 3


def dwell_reached(flags, dwell: int) -> int:
    """The step index at which ``dwell`` consecutive True flags are first complete, else -1.
    A transient crossing shorter than ``dwell`` never counts (the lesson of TASK-067)."""
    run = 0
    for i, flag in enumerate(flags):
        run = run + 1 if flag else 0
        if run >= dwell:
            return i
    return -1


def reach_success(distances, tolerance=REACH_TOLERANCE, dwell=REACH_DWELL) -> bool:
    return dwell_reached([d <= tolerance for d in distances], dwell) >= 0


def lift_success(rise, contact, height=LIFT_HEIGHT, dwell=LIFT_DWELL) -> bool:
    return (
        dwell_reached([r >= height and c for r, c in zip(rise, contact, strict=True)], dwell) >= 0
    )


def rotation_angle(a, b) -> float:
    """The angle (rad) between two rotation matrices."""
    c = (np.trace(np.asarray(a, float).T @ np.asarray(b, float)) - 1) / 2
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


class SubgoalSwitch:
    """Protocol §5.3: advance from subgoal k when the measured palm (forward kinematics of the
    measured joints) is within 1 cm and 0.17 rad of subgoal k's palm pose and the 7 measured
    hand joints are within 0.15 rad RMS of its hand joints, for 3 consecutive steps; or after
    100 steps on it. Reads proprioception and the goal pose only."""

    def __init__(self, subgoals):
        self.subgoals = subgoals
        self.index = 0
        self.steps = 0
        self.run = 0
        self.log = []

    def update(self, palm, hand_q, rotation) -> int:
        if self.index >= len(self.subgoals) - 1:
            return self.index
        g = self.subgoals[self.index]
        near = np.linalg.norm(np.asarray(palm) - g["palm"]) <= SWITCH_PALM
        near = near and rotation_angle(rotation, g["rotation"]) <= SWITCH_ROT
        hand = math.sqrt(float(np.mean((np.asarray(hand_q) - g["state28"][7:14]) ** 2)))
        self.run = self.run + 1 if (near and hand <= SWITCH_HAND_RMS) else 0
        self.steps += 1
        if self.run >= SWITCH_DWELL or self.steps >= SUBGOAL_TIMEOUT:
            self.log.append(
                {
                    "from": self.index,
                    "steps": self.steps,
                    "reason": "reached" if self.run >= SWITCH_DWELL else "timeout",
                }
            )
            self.index += 1
            self.steps = self.run = 0
        return self.index


# ----- baseline controllers (protocol §5.2) ------------------------------------------------------
class HoldController:
    name = "hold"

    def act(self, robot, observation, goal, index):
        return HOLD.copy()


class RandomController:
    """Uniform random right-arm and right-grasp commands in [-1, 1], fresh every step."""

    name = "random"

    def __init__(self, seed: int):
        self.rng = np.random.default_rng([SALTS["random"], int(seed)])

    def act(self, robot, observation, goal, index):
        action = HOLD.copy()
        action[list(PLAYED)] = self.rng.uniform(-1, 1, len(PLAYED)).astype(np.float32)
        return action


class IKController:
    """The scripted IK follower: e9's tracking law to the current subgoal's palm pose (the goal
    pose); grasp +1 when the synergy value nearest the subgoal's hand joints is above 0, else -1,
    ramped by 0.2 per step.
    On the grasp subgoal it commands the palm 5 cm below the goal palm, as e9's recipe commands
    its palm below where the table stops it (without the press the hand closes beside the object;
    disclosed development, protocol §12). It reads proprioception and the goal pose only (no
    image, no object truth)."""

    name = "ik"

    def __init__(self, press: bool = True):
        self.grasp = -1.0
        self.press = press
        if not press:
            self.name = "ik-nopress"

    def act(self, robot, observation, goal, index):
        p = proprio(robot, observation)
        g = goal["subgoals"][index]
        want = 1.0 if synergy_grasp(robot, g["state28"][7:14]) > 0 else -1.0
        self.grasp = float(np.clip(want, self.grasp - GRASP_RAMP, self.grasp + GRASP_RAMP))
        target = np.asarray(g["palm"], float).copy()
        if self.press and goal["task"] == "grasp" and SUBGOALS[index] == "grasp":
            target[2] -= PRESS_DEPTH  # e9's press: the palm is commanded below where it stops
        return tracking_action(robot, p["palm"], p["rotation"], target, g["rotation"], self.grasp)


# ----- arms (protocol §5) ------------------------------------------------------------------------
PRIMARY_SEED = 87100
MODEL_SEEDS = (87100, 87101, 87102)
# arm -> (model arm of TASK-087, latent weight, pose weight)
MODEL_ARMS = {
    "P": ("P", 1.0, 0.0),
    "G": ("G", 1.0, 1.0),
    "G-lat": ("G", 1.0, 0.0),
    "G-pose": ("G", 0.0, 1.0),
}
BASELINE_ARMS = ("hold", "random", "ik", "ik-nopress")
REPLICATION_SEEDS = (87101, 87102)  # P and G, on reach only (§5.4)


def check_run(cohort: str, arm: str, model_seed) -> None:
    """§5.4: which (arm, model seed) runs a cohort admits."""
    if arm in BASELINE_ARMS:
        if model_seed is not None:
            raise ContractError("baseline arms take no model seed")
        return
    if arm not in MODEL_ARMS:
        raise ContractError(f"unknown arm {arm}")
    if cohort.startswith("k0"):
        raise ContractError("K0 runs the baselines only (§7)")
    if model_seed == PRIMARY_SEED:
        return
    if model_seed in REPLICATION_SEEDS and arm in ("P", "G"):
        if cohort == "gated-reach" or cohort.startswith("debug"):
            return
    raise ContractError(f"{arm}-{model_seed} is not a run of {cohort}")


CEM = {"horizon": 8, "samples": 300, "iterations": 10, "elites": 30, "minimum_std": 0.05}


def cem_seed(seed: int) -> int:
    return SALTS["cem"] * 10**7 + int(seed)


# ----- statistics and rows (protocol §7, §8) -----------------------------------------------------
PLAN_BARS = {"reach": 0.80, "grasp": 0.40}
CEILING_FACTOR = 0.9
STOP_REACH = 0.50


def bar_from_ceiling(task: str, ceiling_successes: int, n: int) -> float:
    """§7.1: bar = min(the plan's example bar, 0.9 x the K0 ceiling rate)."""
    return min(PLAN_BARS[task], CEILING_FACTOR * ceiling_successes / n)


def bar_count(bar: float, n: int) -> int:
    return int(math.ceil(bar * n - 1e-9))


def exact_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson interval by bisection on the binomial tail (no SciPy)."""
    alpha = 1 - level

    def tail_ge(p):  # P(X >= k)
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def tail_le(p):  # P(X <= k)
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))

    def solve(f, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(tail_ge, alpha / 2, True)
    upper = 1.0 if k == n else solve(tail_le, alpha / 2, False)
    return lower, upper


def mcnemar_one_sided(b: int, c: int) -> float:
    """Exact one-sided McNemar p for b discordant pairs in favour of the first arm and c against."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(b, n + 1)) / 2**n


def row(results: dict, bars: dict, *, void: str = "") -> dict:
    """§8, first match, on the primary arm P (seed 87100); G is reported against the same bars."""
    n = {t: len(results[t]["P"]) for t in TASKS}
    k = {t: int(sum(results[t]["P"])) for t in TASKS}
    kg = {t: int(sum(results[t]["G"])) for t in TASKS}
    # the plan's stop level takes precedence: a reach pass needs at least half (§8)
    need = {t: bar_count(bars[t], n[t]) for t in TASKS}
    need["reach"] = max(need["reach"], bar_count(STOP_REACH, n["reach"]))
    if void:
        name = "Z3-VOID"
    elif k["reach"] >= need["reach"] and k["grasp"] >= need["grasp"]:
        name = "Z3-PASS"
    elif k["reach"] >= need["reach"]:
        name = "Z3-REACH"
    elif max(k["reach"], kg["reach"]) >= bar_count(STOP_REACH, n["reach"]):
        name = "Z3-LOW"
    else:
        name = "Z3-STOP-CANDIDATE"
    return {
        "row": name,
        "void_reason": void,
        "need": need,
        "P": k,
        "G": kg,
        "G_meets": {t: kg[t] >= need[t] for t in TASKS},
    }


# ----- record helpers ----------------------------------------------------------------------------
def goal_digest(goal: dict) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {k: v for k, v in goal.items() if k not in ("subgoals", "digest")}, sort_keys=True
        ).encode()
    )
    for g in goal["subgoals"]:
        for key in ("frame", "state28", "palm", "rotation"):
            digest.update(np.ascontiguousarray(g[key]).tobytes())
    return digest.hexdigest()


# ----- the pose weight lambda (protocol §5.1) ----------------------------------------------------
LAMBDA_PAIRS = 10_000


def pose_lambda(latents, states, episodes, state_mean, state_scale, *, pairs=LAMBDA_PAIRS) -> dict:
    """lambda = mean latent MSE / mean standardised joint-position MSE over ``pairs`` pairs of val
    frames from different episodes (salt 8806), so both cost terms have the same mean on
    unrelated frames. ``episodes`` is the val store's episode table (offset, frames)."""
    rng = np.random.default_rng([SALTS["lam"]])
    owner = np.concatenate([np.full(e["frames"], i) for i, e in enumerate(episodes)])
    total = len(owner)
    i = rng.integers(0, total, pairs)
    j = rng.integers(0, total, pairs)
    keep = owner[i] != owner[j]
    i, j = i[keep], j[keep]
    lat = np.asarray(latents[i], np.float32) - np.asarray(latents[j], np.float32)
    lat_mse = float((lat.astype(np.float64) ** 2).mean())
    scale = np.maximum(np.asarray(state_scale, np.float64), 1e-12)
    si = np.clip((np.asarray(states[i], np.float64) - state_mean) / scale, -5, 5)[:, :14]
    sj = np.clip((np.asarray(states[j], np.float64) - state_mean) / scale, -5, 5)[:, :14]
    pose_mse = float(((si - sj) ** 2).mean())
    return {
        "lambda": lat_mse / pose_mse,
        "latent_mse": lat_mse,
        "pose_mse": pose_mse,
        "pairs": int(keep.sum()),
    }


# ----- the bars (§7): set by the Stage 0 record from K0, frozen with the protocol -----------------
# K0 (Stage 0 record, protocol §15): ik reached 32/32 (reach) and 24/32 (grasp) on the calibration
# cohorts, so bar = min(0.80, 0.9 x 1.0) and min(0.40, 0.9 x 0.75): 52/64 and 26/64.
BARS: dict = {"reach": 0.80, "grasp": 0.40}
LAMBDA = 2.0552  # §5.1's rule on TASK-087's val store (Stage 0, R25.18); runs check it to 1e-4
GATED_EPISODES = 64
