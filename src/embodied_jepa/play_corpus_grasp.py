"""TASK-090: ``play_policy_grasp_v0``, a grasp-enriched variant of TASK-085's play policy, and its
collection (protocol ``docs/experiments/grasp_play_corpus_v1.md`` §2.1).

TASK-085's ``play_corpus.py`` and its constants are not edited: this module reuses its scene,
layouts, waypoint tracking, storage and sidecars. The changes against ``play_policy_v0`` are only
the ones §2.1 declares:

* modes: scripted 0.55, perturbed 0.45 (no random mode); TASK-085's noise, burst rate and lengths;
* skills: pick_place 0.85, push 0.10, wander 0.05 (no poke);
* the pick recipe of ``play_policy_v0`` (closure, aim noise, ramp, offset, waypoint step ranges,
  the 0.10-0.20 m lift) with two additions: a **lifted hold** of U(40, 80) commands at the lift
  height with the grasp held, after the lift waypoint; and **one retry** of the same object
  (hover, descend, close, lift; fresh aim noise, failure mode ``none``) when the collector's
  privileged check at the end of the lift waypoint finds no lift state of the target;
* ``FAILURE_WEIGHTS``, the uniform pick target, layouts, plate, camera, embodiment, velocity stop,
  episode length, settle commands, minimum length and stored fields unchanged;
* policy salt 9003; the new collections' split rank uses salt 9005 (5 val, 5 test per 100 seeds).

Like ``play_policy_v0`` this is a **privileged scripted** collector (it reads simulator object
truth to choose where to reach and whether to retry); it is never a learned result.

NumPy only at import; MuJoCo is imported lazily by :mod:`embodied_jepa.simulation`.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embodied_jepa import play_corpus as pc
from embodied_jepa.contracts import ContractError

TASK = "TASK-090"
PROTOCOL = "docs/experiments/grasp_play_corpus_v1.md"
POLICY_VERSION = "play_policy_grasp_v0"

# ----- the declared changes (§2.1) ---------------------------------------------------------------
MODES = ("scripted", "perturbed")
MODE_WEIGHTS = (0.55, 0.45)
SKILLS = ("pick_place", "push", "wander")
SKILL_WEIGHTS = (0.85, 0.10, 0.05)
HOLD_COMMANDS = (40, 80)  # inclusive: U(40, 80) commands at the lift height, grasp held
RETRIES = 1
LIFT_CHECK_RISE = 0.03  # m above the target's episode-start height, with grasp contact (TASK-085)
RETRY_SKILL = "pick_retry"

# ----- seeds and salts (§10) ---------------------------------------------------------------------
POLICY_SALT = 9003  # play_policy_grasp_v0
SPLIT_SALT = 9005  # the new collections' split rank
V0_POLICY_SALT = pc.POLICY_SALT  # 8701, play-v1p
CORPORA = {
    # name -> (policy, first seed, shards of 100)
    "play-v2g": ("grasp", 920000, 32),
    "play-v1p": ("v0", 910000, 32),
}
DEBUG_COLLECTION = {"grasp": range(930000, 930300), "v0": range(930300, 930400)}


def corpus_seeds(name: str) -> range:
    policy, first, shards = CORPORA[name]
    return range(first, first + pc.SHARD_SIZE * shards)


def check_collection_seed(policy: str, seed: int, *, debug: bool) -> None:
    allowed = (
        DEBUG_COLLECTION[policy]
        if debug
        else corpus_seeds("play-v2g" if policy == "grasp" else "play-v1p")
    )
    if int(seed) not in allowed:
        kind = "debug" if debug else "corpus"
        raise ContractError(f"seed {seed} is not a {kind} seed of policy {policy}")


def policy_salt(policy: str) -> int:
    return {"grasp": POLICY_SALT, "v0": V0_POLICY_SALT}[policy]


def split_rank(seed: int) -> str:
    return hashlib.sha256(f"{SPLIT_SALT}:{seed}".encode()).hexdigest()


def split_assignment(seeds, *, n_val=pc.VAL_PER_SHARD, n_test=pc.TEST_PER_SHARD) -> dict:
    """TASK-085's per-shard rank rule with salt 9005: the lowest n_val val, the next n_test test,
    the rest train."""
    ranked = sorted(seeds, key=split_rank)
    if len(ranked) < n_val + n_test + 1:
        raise ContractError("a shard needs more stored episodes than val + test")
    return {
        "val": sorted(ranked[:n_val]),
        "test": sorted(ranked[n_val : n_val + n_test]),
        "train": sorted(ranked[n_val + n_test :]),
    }


# ----- the policy (§2.1) -------------------------------------------------------------------------
@dataclass
class TaggedWaypoint(pc.Waypoint):
    tag: str = ""


class GraspPlayPolicy(pc.PlayPolicy):
    """``play_policy_v0`` with §2.1's mode and skill mix, a lifted hold and one retry."""

    def __init__(self, rng):
        mode = str(rng.choice(MODES, p=MODE_WEIGHTS))
        super().__init__(rng, mode=mode)
        self.z0 = None  # the objects' episode-start heights (the first truth the policy sees)
        self.pick = None  # the current pick's parameters (for the retry)
        self.pending_check = None
        self.checks: list[dict] = []

    # -- planning ------------------------------------------------------------------------------
    def _plan(self, truth, ee):
        rng = self.rng
        on_table = [
            i
            for i, _ in enumerate(pc.OBJECT_NAMES)
            if truth["position"][i][2] > pc.TABLE_TOP_Z - 0.01 and truth["position"][i][0] > 0.1
        ]
        skill = "wander" if not on_table else str(rng.choice(SKILLS, p=SKILL_WEIGHTS))
        failure = "none"
        if skill == "pick_place":
            failure = str(rng.choice(pc.FAILURES, p=pc.FAILURE_WEIGHTS))
        self.skill, self.failure = skill, failure
        self.skill_log.append((skill, failure))
        self.pick = None
        yaw = float(rng.uniform(-pc.YAW_RANGE, pc.YAW_RANGE))
        if skill == "wander":  # play_policy_v0's wander, unchanged
            self.queue = [
                pc.Waypoint(
                    self._random_hand_point(),
                    float(rng.choice([-1.0, rng.uniform(-1, 1)])),
                    int(rng.integers(15, 45)),
                    float(rng.uniform(-0.5, 0.5)),
                )
                for _ in range(int(rng.integers(2, 6)))
            ]
            return
        index = int(rng.choice(on_table))
        obj = pc.to_pelvis(truth["position"][index])
        aim = (
            obj
            + pc.GRASP_OFFSET
            + np.r_[rng.normal(0, pc.AIM_SIGMA, 2), rng.normal(0, pc.AIM_SIGMA / 2)]
        )
        if failure == "miss":
            angle = rng.uniform(0, 2 * np.pi)
            aim[:2] += rng.uniform(0.04, 0.08) * np.array([np.cos(angle), np.sin(angle)])
        if skill == "push":  # play_policy_v0's push, unchanged
            angle = rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            low = obj[2] + rng.uniform(0.02, 0.05)
            start = obj - direction * 0.09
            end = obj + direction * rng.uniform(0.05, 0.15)
            grasp = float(rng.uniform(-1, 0.3))
            self.queue = [
                pc.Waypoint(np.r_[start[:2], low + 0.10], grasp, int(rng.integers(20, 40)), yaw),
                pc.Waypoint(np.r_[start[:2], low], grasp, int(rng.integers(12, 25)), yaw),
                pc.Waypoint(np.r_[end[:2], low], grasp, int(rng.integers(20, 40)), yaw),
                pc.Waypoint(np.r_[end[:2], low + 0.10], -1.0, int(rng.integers(10, 20)), yaw),
            ]
            return
        # pick_place: play_policy_v0's recipe, plus the hold after the lift and the retry check
        closure = float(rng.uniform(*pc.CLOSURE))
        lift = float(rng.uniform(0.10, 0.20))
        if rng.random() < 0.4:
            dest = pc.to_pelvis(truth["plate_position"])[:2]
        else:
            dest = np.array([rng.uniform(*pc.OBJECT_REGION[0]), rng.uniform(*pc.OBJECT_REGION[1])])
        dest = dest + pc.GRASP_OFFSET[:2] + rng.normal(0, 0.01, 2)
        from embodied_jepa.resting_expert import RestingPlaceExpert

        reach = RestingPlaceExpert.release_pose(dest, reach_radius_m=0.485, floor=0.0, ceiling=0.26)
        dest = reach[:2]
        place_z = max(aim[2] + float(rng.uniform(0.0, 0.05)), reach[2])
        close = closure if failure != "no_close" else -1.0
        hover = aim + [0, 0, pc.HOVER_DZ]
        q = [pc.Waypoint(hover, -1.0, int(rng.integers(50, 80)), yaw)]
        if failure == "early_close":
            q.append(pc.Waypoint(hover, closure, int(rng.integers(15, 25)), yaw))
        q.append(
            pc.Waypoint(
                aim, close if failure == "early_close" else -1.0, int(rng.integers(40, 60)), yaw
            )
        )
        if failure == "abort":  # leave for a random point right after the descent (no lift)
            self.queue = q + [pc.Waypoint(self._random_hand_point(), -1.0, 30, yaw)]
            return
        q.append(pc.Waypoint(aim, close, int(rng.integers(25, 40)), yaw))
        q.append(TaggedWaypoint(aim + [0, 0, lift], close, int(rng.integers(40, 70)), yaw, "lift"))
        self.pick = {
            "index": index,
            "closure": closure,
            "lift": lift,
            "dest": dest,
            "place_z": place_z,
            "yaw": yaw,
            "attempt": 0,
        }
        self.queue = q + self._after_lift(aim, close, failure)

    def _after_lift(self, aim, close, failure) -> list:
        """The hold at the lift height, then play_policy_v0's drop or transfer and place."""
        rng, p = self.rng, self.pick
        lift, dest, place_z, yaw = p["lift"], p["dest"], p["place_z"], p["yaw"]
        lo, hi = HOLD_COMMANDS
        top = aim + [0, 0, lift]
        q = [pc.Waypoint(top, close, int(rng.integers(lo, hi + 1)), yaw)]
        transfer = np.r_[dest, max(aim[2] + lift, place_z + 0.03)]
        if failure == "drop":
            mid = (top + transfer) / 2
            return q + [
                pc.Waypoint(mid, close, int(rng.integers(15, 30)), yaw),
                pc.Waypoint(mid, -1.0, 20, yaw),
            ]
        return q + [
            pc.Waypoint(transfer, close, int(rng.integers(40, 70)), yaw),
            pc.Waypoint(np.r_[dest, place_z], close, int(rng.integers(25, 40)), yaw),
            pc.Waypoint(np.r_[dest, place_z], -1.0, int(rng.integers(20, 30)), yaw),
            pc.Waypoint(np.r_[dest, place_z + 0.10], -1.0, int(rng.integers(20, 30)), yaw),
        ]

    def _retry(self, truth):
        """Hover, descend, close and lift on the same object again (fresh aim noise, failure
        ``none``), then the hold and the transfer and place."""
        rng, p = self.rng, self.pick
        p["attempt"] = 1
        self.skill_log.append((RETRY_SKILL, "none"))
        self.failure = "none"
        obj = pc.to_pelvis(truth["position"][p["index"]])
        aim = (
            obj
            + pc.GRASP_OFFSET
            + np.r_[rng.normal(0, pc.AIM_SIGMA, 2), rng.normal(0, pc.AIM_SIGMA / 2)]
        )
        yaw, closure = p["yaw"], p["closure"]
        q = [
            pc.Waypoint(aim + [0, 0, pc.HOVER_DZ], -1.0, int(rng.integers(50, 80)), yaw),
            pc.Waypoint(aim, -1.0, int(rng.integers(40, 60)), yaw),
            pc.Waypoint(aim, closure, int(rng.integers(25, 40)), yaw),
            TaggedWaypoint(
                aim + [0, 0, p["lift"]], closure, int(rng.integers(40, 70)), yaw, "lift"
            ),
        ]
        self.queue = q + self._after_lift(aim, closure, "none")

    @staticmethod
    def lift_state(truth, z0, index) -> bool:
        rise = float(truth["position"][index][2] - z0[index])
        return bool(rise >= LIFT_CHECK_RISE and truth["grasp_contact"][index])

    # -- acting --------------------------------------------------------------------------------
    def act(self, robot, truth):
        if self.z0 is None:
            self.z0 = np.asarray(truth["position"], float)[:, 2].copy()
        if self.pending_check is not None:  # the end of a lift waypoint: the privileged check
            p = self.pending_check
            self.pending_check = None
            lifted = self.lift_state(truth, self.z0, p["index"])
            self.checks.append(
                {
                    "skill_entry": len(self.skill_log) - 1,
                    "object": pc.OBJECT_NAMES[p["index"]],
                    "attempt": int(p["attempt"]),
                    "lifted": lifted,
                }
            )
            if not lifted and p["attempt"] < RETRIES:
                self._retry(truth)
        head = self.queue[0] if self.queue else None
        action = super().act(robot, truth)
        if (
            head is not None
            and head.steps <= 0
            and getattr(head, "tag", "") == "lift"
            and self.pick is not None
        ):
            self.pending_check = dict(self.pick)
        return action


def make_policy(policy: str, rng):
    if policy == "grasp":
        return GraspPlayPolicy(rng)
    if policy == "v0":
        return pc.PlayPolicy(rng)
    raise ContractError(f"unknown policy {policy}")


# ----- one episode (TASK-085's run_episode with the policy as a parameter) -----------------------
def run_episode(robot, seed: int, *, policy: str, commands: int = pc.EPISODE_COMMANDS) -> dict:
    """``play_corpus.run_episode`` with the policy chosen by ``policy`` ("grasp" or "v0") and its
    salt; for "v0" it is TASK-085's episode exactly (checked by a test). Adds the collector's lift
    checks (``pick_checks``; empty for "v0")."""
    if policy == "v0":
        ep = pc.run_episode(robot, seed, salt=V0_POLICY_SALT, commands=commands)
        ep["pick_checks"] = []
        ep["policy"] = pc.POLICY_VERSION
        return ep
    rng = np.random.default_rng([POLICY_SALT, seed])
    layout = pc.sample_layout(rng)
    robot.reset(seed, layout=layout)
    sim = robot.sim
    hold = np.zeros(14, np.float32)
    hold[12:] = -1.0
    for _ in range(pc.SETTLE_COMMANDS):
        robot.observe()
        result = robot.execute(hold)
        if result.status not in ("applied", "clipped"):
            raise ContractError(f"settle command rejected: {result.reason}")
    agent = make_policy(policy, rng)
    rows = {k: [] for k in ("frame", "state", "time", "palms", "targets", "truth", "skill")}
    requested, applied, physical, reduced = [], [], [], []
    stop_reason = ""

    def record(observation, skill):
        rows["frame"].append(observation.images[pc.CAMERA][0])
        rows["state"].append(observation.robot_state[0])
        rows["time"].append(float(observation.timestamps[0]))
        rows["palms"].append(pc._palms(robot))
        rows["targets"].append(sim.targets.copy())
        rows["truth"].append(sim.play_truth())
        rows["skill"].append(skill)

    for _ in range(commands):
        observation = robot.observe()
        request = agent.act(robot, sim.play_truth())
        record(observation, len(agent.skill_log) - 1)
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
    record(robot.observe(), len(agent.skill_log) - 1)
    truths = rows.pop("truth")
    empty = np.zeros((0, 14), np.float32)
    return {
        "seed": int(seed),
        "policy": POLICY_VERSION,
        "layout": layout,
        "mode": agent.mode,
        "skill_log": [list(x) for x in agent.skill_log],
        "pick_checks": agent.checks,
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


def write_episode(store, robot, ep) -> dict:
    """``play_corpus.write_episode`` with the episode's own policy version and the collector's lift
    checks in the metadata; the sidecar keys are TASK-085's."""
    from embodied_jepa.data import Episode

    eid = pc.episode_id(ep["seed"])
    sidecar = Path(store.root) / "play" / f"{eid}.npz"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    if sidecar.exists():
        raise FileExistsError(f"sidecar exists: {sidecar}")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{k: np.asarray(ep[k]) for k in pc.SIDECAR_KEYS})
    sidecar.write_bytes(buffer.getvalue())
    metadata = {
        "seed": ep["seed"],
        "scene": pc.SCENE_VERSION,
        "policy": ep["policy"],
        "layout": ep["layout"],
        "mode": ep["mode"],
        "skill_log": ep["skill_log"],
        "pick_checks": ep["pick_checks"],
        "stop_reason": ep["stop_reason"],
        "facts": pc.episode_facts(ep),
        "sidecar": f"play/{eid}.npz",
        "sidecar_sha256": pc._sha256(sidecar),
    }
    episode = Episode(
        episode_id=eid,
        session_id=eid,
        task=pc.TASK_NAME,
        object_id="play",
        container_id="plate",
        observations={pc.CAMERA: ep["frames"]},
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
