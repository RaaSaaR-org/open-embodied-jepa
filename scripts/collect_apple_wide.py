"""TASK-048: wide-jitter Apple->Plate TRAIN corpus for world model v2.

A PRIVILEGED scripted collector (``scripted.apple_collector_policy``) with injected,
seeded action perturbations produces root episodes on the TASK-047 wide-jitter reset
rule, and from the pre-grasp state of each root it branches perturbed grasp-phase
continuations (including failed grasps). Nothing here is a learned result.

Model-input streams: ``onboard_rgb`` (112 px, the simulator's own render) and
``hand_crop_rgb`` (112 px window of a 320 px onboard render centred on the right palm
via robot kinematics only), plus proprioception and the executed 14-D actions.
Simulator truth goes ONLY into per-episode label sidecars (``training_labels``).

Subcommands: ``plan`` (write the frozen plan), ``run`` (workers -> assemble -> seal ->
audit -> acceptance), ``worker`` (internal).
"""

from __future__ import annotations

# ruff: noqa: E402
import time

ENTRY_WALL, ENTRY_MONO = time.time(), time.monotonic()

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import training_labels

COLLECTION = "apple_wide_v1"
FROZEN_SEEDS = tuple(range(48000, 48200))
PILOT_SEEDS = tuple(range(48900, 48932))
RESET_CENTERS = {"object_xy": (0.34, -0.18), "plate_xy": (0.49, -0.09)}
WIDE_JITTER_M = {"object_xy": 0.03, "plate_xy": 0.02}
IMAGE_SIZE = 112
CROP_RENDER_SIZE = 320
FPS = 20
# Per-root perturbation level cycles 0..3 over the seed order (level 0 = nominal script).
NOISE_LEVELS = (0, 1, 2, 3)
OU_THETA = 0.85
OU_SIGMA = {"translation": 0.04, "rotation": 0.025, "grasp": 0.1}  # per level
BURST_PROBABILITY = 0.01  # per level, per command
BURST_COMMANDS = 4
AIM_OFFSET_EVERY = 5  # every fifth root aims its grasp 1.5-3.0 cm off the apple
AIM_OFFSET_M = (0.015, 0.03)
BRANCH_KINDS = ("noise_only", "shift_close", "weak_close", "early_lift", "open_during_lift")
BRANCHES_PER_ROOT = 3
BRANCH_PHASE = 2  # the policy's "close" phase: branches start at the pre-grasp state
BRANCH_END_PHASE = 4  # stop after "lift"
ROOT_MAX_COMMANDS = 745  # the collector's full phase budget
SPLIT_SEED = 48
LOWER = np.array((0.0,) * 6 + (-0.5,) * 6 + (-1.0, -1.0), np.float32)
UPPER = np.array((0.0,) * 6 + (0.5,) * 6 + (-1.0, 1.0), np.float32)
GUARD_REFUSAL = "measured joint velocity limit exceeded"
STAGES = ("reach", "grasp", "transport", "place", "success")
PERTURBATION_CODES = {"none": 0, "ou": 1, "burst": 2}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def elapsed():
    return max(time.time() - ENTRY_WALL, time.monotonic() - ENTRY_MONO)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def wide_reset(seed):
    """Identical to ``evaluate_apple.wide_reset`` (TASK-047); a test pins the equality."""
    rng = np.random.default_rng(seed)
    return {
        "object_xy": (
            np.array(RESET_CENTERS["object_xy"])
            + rng.uniform(-WIDE_JITTER_M["object_xy"], WIDE_JITTER_M["object_xy"], 2)
        ).tolist(),
        "plate_xy": (
            np.array(RESET_CENTERS["plate_xy"])
            + rng.uniform(-WIDE_JITTER_M["plate_xy"], WIDE_JITTER_M["plate_xy"], 2)
        ).tolist(),
    }


def split_assignment(seeds):
    """Whole-reset (session) partition fixed before any outcome: 10% val, 5% test."""
    order = list(seeds)
    np.random.default_rng(SPLIT_SEED).shuffle(order)
    n_val, n_test = max(1, round(0.10 * len(order))), max(1, round(0.05 * len(order)))
    return {
        seed: ("val" if i < n_val else "test" if i < n_val + n_test else "train")
        for i, seed in enumerate(order)
    }


def branch_params(kind, rng):
    if kind == "noise_only":
        return {}
    if kind == "shift_close":
        angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(0.01, 0.025)
        return {"offset_xy_m": [radius * np.cos(angle), radius * np.sin(angle)]}
    if kind == "weak_close":
        return {"grasp_target": float(rng.uniform(-0.2, 0.8))}
    if kind == "early_lift":
        return {"close_commands": int(rng.integers(3, 31))}
    if kind == "open_during_lift":
        return {"open_after_lift_commands": int(rng.integers(2, 41))}
    raise ValueError(f"unknown branch kind {kind}")


def make_plan(seeds):
    splits = split_assignment(seeds)
    roots = []
    for index, seed in enumerate(seeds):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 48]))
        aim = None
        if index % AIM_OFFSET_EVERY == AIM_OFFSET_EVERY - 1:
            angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(*AIM_OFFSET_M)
            aim = [radius * np.cos(angle), radius * np.sin(angle)]
        branches = []
        for b in range(BRANCHES_PER_ROOT):
            # The index // AIM_OFFSET_EVERY term decouples kinds from aim-offset roots.
            slot = BRANCHES_PER_ROOT * index + b + index // AIM_OFFSET_EVERY
            kind = BRANCH_KINDS[slot % len(BRANCH_KINDS)]
            branches.append(
                {
                    "episode_id": f"wide-{seed}-b{b}-{kind}",
                    "kind": kind,
                    "params": branch_params(kind, rng),
                    "noise_seed": int(rng.integers(2**31)),
                }
            )
        roots.append(
            {
                "episode_id": f"wide-{seed}",
                "session_id": f"wide-reset-{seed}",
                "seed": seed,
                "split": splits[seed],
                **wide_reset(seed),
                "noise_level": NOISE_LEVELS[index % len(NOISE_LEVELS)],
                "noise_seed": int(rng.integers(2**31)),
                "aim_offset_xy_m": aim,
                "branches": branches,
            }
        )
    return {
        "collection": COLLECTION,
        "task": "TASK-048",
        "privileged_scripted_collector": True,
        "learned_control": False,
        "seeds": list(seeds),
        "reset_rule": "evaluate_apple.wide_reset (TASK-047): apple +-3 cm, plate +-2 cm",
        "reset_centers_xy": {k: list(v) for k, v in RESET_CENTERS.items()},
        "image_size": IMAGE_SIZE,
        "crop_render_size": CROP_RENDER_SIZE,
        "fps": FPS,
        "noise": {
            "levels": list(NOISE_LEVELS),
            "ou_theta": OU_THETA,
            "ou_sigma_per_level": OU_SIGMA,
            "burst_probability_per_level": BURST_PROBABILITY,
            "burst_commands": BURST_COMMANDS,
        },
        "aim_offset_every": AIM_OFFSET_EVERY,
        "aim_offset_m": list(AIM_OFFSET_M),
        "branch_kinds": list(BRANCH_KINDS),
        "branches_per_root": BRANCHES_PER_ROOT,
        "bounds": {"lower": LOWER.tolist(), "upper": UPPER.tolist()},
        "root_max_commands": ROOT_MAX_COMMANDS,
        "split_seed": SPLIT_SEED,
        "roots": roots,
    }


class Perturber:
    """Seeded Ornstein-Uhlenbeck noise on the right arm and grasp, plus random bursts."""

    def __init__(self, level, seed):
        self.level = level
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(7)
        self.burst, self.burst_left = np.zeros(6), 0
        self.sigma = level * np.array(
            [OU_SIGMA["translation"]] * 3 + [OU_SIGMA["rotation"]] * 3 + [OU_SIGMA["grasp"]]
        )

    def __call__(self, base):
        action = base.astype(np.float32).copy()
        if self.level == 0:
            return action, "none"
        noise = self.rng.normal(size=7)
        self.state = OU_THETA * self.state + np.sqrt(1 - OU_THETA**2) * self.sigma * noise
        start_burst = self.rng.random() < BURST_PROBABILITY * self.level
        burst_vector = self.rng.uniform(-0.5, 0.5, 6)
        if self.burst_left == 0 and start_burst:
            self.burst, self.burst_left = burst_vector, BURST_COMMANDS
        action[6:12] += self.state[:6].astype(np.float32)
        action[13] += np.float32(self.state[6])
        kind = "ou"
        if self.burst_left:
            action[6:12] = self.burst.astype(np.float32)
            self.burst_left -= 1
            kind = "burst"
        return action, kind


def shifted(policy, phases, offset_xy):
    items = list(policy.phases)
    for index in phases:
        target = items[index].target_base.copy()
        target[:2] += np.asarray(offset_xy)
        items[index] = replace(items[index], target_base=target)
    policy.phases = tuple(items)


class Controller:
    """Collector policy + optional branch modification + perturbation."""

    def __init__(self, policy, perturber):
        self.policy, self.perturber = policy, perturber
        self.grasp_override = None  # (phases, value)
        self.open_after_lift = None

    @property
    def done(self):
        return self.policy.done

    @property
    def phase_index(self):
        return self.policy.phase_index

    def apply_branch(self, kind, params, noise_seed):
        # Every branch draws its own perturbation stream (level >= 1), so sibling
        # branches never share an action prefix; the kind adds its modification.
        self.perturber = Perturber(max(1, self.perturber.level), noise_seed)
        if kind == "noise_only":
            pass
        elif kind == "shift_close":
            shifted(self.policy, (2, 3), params["offset_xy_m"])
        elif kind == "weak_close":
            self.grasp_override = ((2, 3), params["grasp_target"])
        elif kind == "early_lift":
            phases = list(self.policy.phases)
            phases[2] = replace(phases[2], commands=params["close_commands"])
            self.policy.phases = tuple(phases)
        elif kind == "open_during_lift":
            self.open_after_lift = params["open_after_lift_commands"]
        else:
            raise ValueError(kind)

    def command(self, robot):
        base = self.policy.action(robot)
        if self.grasp_override and self.phase_index in self.grasp_override[0]:
            base[13] = self.grasp_override[1]
        if (
            self.open_after_lift is not None
            and self.phase_index == 3
            and self.policy.phase_step >= self.open_after_lift
        ):
            base[13] = -1.0
        requested, kind = self.perturber(base)
        return base, requested, kind

    def advance(self, result):
        self.policy.advance(result)


class Recorder:
    def __init__(self):
        self.rows = {
            k: []
            for k in (
                "onboard",
                "crop",
                "window",
                "state",
                "mask",
                "time",
                "apple",
                "apple_velocity",
                "plate",
                "palm",
                "contact",
                "dropped",
                "stages",
            )
        }
        self.steps = {
            k: [] for k in ("applied", "raw", "base", "requested", "projected", "phase", "perturb")
        }

    def observe(self, robot, crop, stages):
        obs = robot.observe()
        image, window = crop.capture()
        truth = robot.sim.task_truth()
        row = self.rows
        row["onboard"].append(obs.images["onboard_rgb"][0].copy())
        row["crop"].append(image)
        row["window"].append(window)
        row["state"].append(obs.robot_state[0].copy())
        row["mask"].append(obs.state_mask[0].copy())
        row["time"].append(float(obs.timestamps[0]))
        row["apple"].append(np.asarray(truth["object_position"], float))
        row["apple_velocity"].append(np.asarray(truth["object_velocity"], float))
        row["plate"].append(np.asarray(truth["plate_position"], float))
        row["palm"].append(np.asarray(robot.sim.data.site("right_ee").xpos, float).copy())
        row["contact"].append(bool(truth["hand_contact"]))
        row["dropped"].append(bool(truth["dropped"]))
        row["stages"].append([bool(stages.get(k, False)) for k in STAGES])

    def sensors(self, index=-1):
        r = self.rows
        return (r["onboard"][index], r["crop"][index], r["state"][index], r["time"][index])

    def labels(self):
        r, s = self.rows, self.steps
        apple, palm = np.asarray(r["apple"]), np.asarray(r["palm"])
        return {
            "robot__hand_crop_window": np.asarray(r["window"], np.int16),
            "collector__phase_index": np.asarray(s["phase"], np.int8),
            "collector__base_action": np.asarray(s["base"], np.float32),
            "collector__requested_action": np.asarray(s["requested"], np.float32),
            "collector__projected_action": np.asarray(s["projected"], np.float32),
            "collector__perturbation": np.asarray(s["perturb"], np.int8),
            "privileged__apple_position_world": apple.astype(np.float32),
            "privileged__apple_velocity_world": np.asarray(r["apple_velocity"], np.float32),
            "privileged__plate_position_world": np.asarray(r["plate"], np.float32),
            "privileged__palm_position_world": palm.astype(np.float32),
            "privileged__palm_minus_apple_world": (palm - apple).astype(np.float32),
            "privileged__hand_contact": np.asarray(r["contact"], bool),
            "privileged__apple_dropped": np.asarray(r["dropped"], bool),
            "privileged__stages": np.asarray(r["stages"], bool),
        }


def step(robot, scorer, crop, controller, rec):
    """One projected, executed command. Returns (score, stop_reason or None)."""
    from embodied_jepa.contracts import ContractError

    phase = controller.phase_index
    base, requested, kind = controller.command(robot)
    requested = np.clip(requested, LOWER, UPPER).astype(np.float32)
    try:
        projected = robot.project_candidates(requested[None, None, None])
    except ContractError as error:
        if str(error) == GUARD_REFUSAL:
            return None, "guard_refused"
        raise
    if not projected.feasible[0, 0]:
        return None, "projection_infeasible"
    command = projected.actions[0, 0, 0].astype(np.float32)
    result = robot.execute(command)
    if result.applied_action is None:
        return None, f"command_rejected: {result.reason}"
    controller.advance(result)
    score = scorer.evaluate()
    s = rec.steps
    s["applied"].append(result.applied_action.copy())
    s["raw"].append(robot.denormalize_action(result.applied_action))
    s["base"].append(base.astype(np.float32))
    s["requested"].append(requested)
    s["projected"].append(command)
    s["phase"].append(phase)
    s["perturb"].append(PERTURBATION_CODES[kind])
    rec.observe(robot, crop, score)
    return score, None


def snapshot(robot, scorer, controller):
    data = robot.mj.MjData(robot.model)
    robot.mj.mj_copyData(data, robot.model, robot.sim.data)
    return {
        "data": data,
        "targets": robot.sim.targets.copy(),
        "grasp": robot._grasp.copy(),
        "stopped_reason": robot.sim.stopped_reason,
        "scorer": {k: copy.deepcopy(v) for k, v in vars(scorer).items() if k != "robot"},
        "controller": copy.deepcopy(controller),
    }


def restore(robot, scorer, saved):
    robot.mj.mj_copyData(robot.sim.data, robot.model, saved["data"])
    robot.sim.targets[:] = saved["targets"]
    robot._grasp[:] = saved["grasp"]
    robot.sim.stopped_reason = saved["stopped_reason"]
    robot._observation = None
    for key, value in saved["scorer"].items():
        setattr(scorer, key, copy.deepcopy(value))
    return copy.deepcopy(saved["controller"])


def same_sensors(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def outcome(rec, termination, *, grasp_phase_attempted):
    stages = dict(zip(STAGES, map(bool, rec.rows["stages"][-1]), strict=True))
    grasped = np.asarray(rec.rows["stages"])[:, 1]
    first = next((k for k in STAGES if not stages[k]), None)
    dropped_after = bool(grasped.any() and rec.rows["dropped"][-1])
    return {
        "stages": stages,
        "failure_stage": first,
        "grasp_phase_attempted": bool(grasp_phase_attempted),
        "grasp_phase_failure": bool(grasp_phase_attempted and not stages["grasp"]),
        "dropped_after_grasp": dropped_after,
        "final_hand_contact": bool(rec.rows["contact"][-1]),
        "termination": termination,
    }


def save_episode(work, episode_id, rec, meta):
    folder = work / "episodes" / episode_id
    folder.mkdir(parents=True, exist_ok=False)
    r, s = rec.rows, rec.steps
    np.savez_compressed(
        folder / "arrays.npz",
        onboard_rgb=np.asarray(r["onboard"], np.uint8),
        hand_crop_rgb=np.asarray(r["crop"], np.uint8),
        robot_states=np.asarray(r["state"], np.float32),
        state_mask=np.asarray(r["mask"], bool),
        timestamps=np.asarray(r["time"], np.float64),
        actions=np.asarray(s["applied"], np.float32),
        raw_actions=np.asarray(s["raw"], np.float32),
    )
    payload, label_sha = training_labels.encode(rec.labels())
    (folder / "labels.npz").write_bytes(payload)
    write(folder / "meta.json", meta | {"labels_sha256": label_sha})


def run_root(root, work, report):
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.hand_crop import HandCrop
    from embodied_jepa.scripted import apple_collector_policy
    from embodied_jepa.simulation import MuJoCoSimulation
    from embodied_jepa.task import AppleToPlateTask

    sim = MuJoCoSimulation(
        object_kind="apple", container_kind="plate", width=IMAGE_SIZE, height=IMAGE_SIZE
    )
    robot = G1Embodiment(sim)
    crop = None
    try:
        truth = robot.reset(root["seed"], object_xy=root["object_xy"], plate_xy=root["plate_xy"])
        if not np.allclose(truth["base_rotation_world"], np.eye(3), atol=1e-7):
            raise ValueError("collector assumes the fixed upright pelvis")
        crop = HandCrop(robot, render_size=CROP_RENDER_SIZE, crop_size=IMAGE_SIZE)
        policy = apple_collector_policy(sim.task_truth())
        if root["aim_offset_xy_m"] is not None:
            shifted(policy, (0, 1, 2, 3), root["aim_offset_xy_m"])
        controller = Controller(policy, Perturber(root["noise_level"], root["noise_seed"]))
        scorer = AppleToPlateTask(robot)
        rec = Recorder()
        rec.observe(robot, crop, scorer.evaluate())
        saved, branch_frame, termination = None, None, None
        while termination is None:
            if controller.done:
                termination = "policy_complete"
                break
            if len(rec.steps["applied"]) >= ROOT_MAX_COMMANDS:
                termination = "step_limit"
                break
            if (
                saved is None
                and controller.phase_index == BRANCH_PHASE
                and controller.policy.phase_step == 0
            ):
                saved = snapshot(robot, scorer, controller)
                branch_frame = len(rec.steps["applied"])
            score, stop = step(robot, scorer, crop, controller, rec)
            if stop:
                termination = stop
            elif score["success"]:
                termination = "success"
        entry = {
            "episode_id": root["episode_id"],
            "kind": "root",
            "termination": termination,
            "transitions": len(rec.steps["applied"]),
            "branch_frame": branch_frame,
        }
        root_outcome = outcome(rec, termination, grasp_phase_attempted=branch_frame is not None)
        entry["outcome"] = root_outcome
        report["episodes"].append(entry)
        if len(rec.steps["applied"]) >= 8:
            save_episode(
                work,
                root["episode_id"],
                rec,
                {
                    "kind": "root",
                    "root": root,
                    "branch": None,
                    "branch_frame": branch_frame,
                    "termination": termination,
                    "outcome": root_outcome,
                },
            )
            entry["stored"] = True
        root_rec = rec
        if saved is None:
            for branch in root["branches"]:
                report["episodes"].append(
                    {"episode_id": branch["episode_id"], "kind": "branch", "stored": False}
                    | {"termination": "no_pre_grasp_state"}
                )
            return
        # Restoration exactness: the restored state must reproduce the root's sensors at the
        # branch frame, and re-executing the root's own next command must reproduce frame+1.
        reference = root_rec.sensors(branch_frame)
        probe = restore(robot, scorer, saved)
        check = Recorder()
        check.observe(robot, crop, {})
        exact = same_sensors(check.sensors(0), reference)
        if exact and branch_frame < len(root_rec.steps["applied"]):
            command = root_rec.steps["projected"][branch_frame]
            result = robot.execute(command)
            probe.advance(result)
            check.observe(robot, crop, {})
            exact = bool(
                result.applied_action is not None
                and np.array_equal(result.applied_action, root_rec.steps["applied"][branch_frame])
                and same_sensors(check.sensors(1), root_rec.sensors(branch_frame + 1))
            )
        entry["restore_exact"] = exact
        if not exact:
            raise ValueError(f"inexact restoration at {root['episode_id']}")
        for branch in root["branches"]:
            controller = restore(robot, scorer, saved)
            controller.apply_branch(branch["kind"], branch["params"], branch["noise_seed"])
            rec = Recorder()
            rec.observe(robot, crop, {})
            if not same_sensors(rec.sensors(0), reference):
                raise ValueError("branch start sensors differ from the root")
            rec.rows["stages"][0] = list(root_rec.rows["stages"][branch_frame])
            termination = None
            while termination is None:
                if controller.done or controller.phase_index >= BRANCH_END_PHASE:
                    termination = "branch_complete"
                    break
                score, stop = step(robot, scorer, crop, controller, rec)
                if stop:
                    termination = stop
                elif score["success"]:
                    termination = "success"
            branch_outcome = outcome(rec, termination, grasp_phase_attempted=True)
            row = {
                "episode_id": branch["episode_id"],
                "kind": "branch",
                "termination": termination,
                "transitions": len(rec.steps["applied"]),
                "outcome": branch_outcome,
                "stored": False,
            }
            report["episodes"].append(row)
            if len(rec.steps["applied"]) >= 8:
                save_episode(
                    work,
                    branch["episode_id"],
                    rec,
                    {
                        "kind": "branch",
                        "root": root,
                        "branch": branch,
                        "branch_frame": branch_frame,
                        "termination": termination,
                        "outcome": branch_outcome,
                    },
                )
                row["stored"] = True
    finally:
        if crop is not None:
            crop.close()
        robot.close()


def worker(plan_path, work, index, count, max_seconds):
    plan = json.loads(Path(plan_path).read_text())
    report_path = work / f"worker-{index:02d}.json"
    report = {"worker": index, "roots": [], "episodes": [], "status": "running"}
    cpu = time.process_time()
    for position, root in enumerate(plan["roots"]):
        if position % count != index:
            continue
        if elapsed() >= max_seconds:
            report["roots"].append({"seed": root["seed"], "status": "not_started_budget"})
            continue
        started = time.monotonic()
        try:
            run_root(root, work, report)
            status = "completed"
        except Exception as error:  # recorded, never silently dropped
            status = f"runtime_error: {type(error).__name__}: {error}"
        report["roots"].append(
            {"seed": root["seed"], "status": status, "seconds": time.monotonic() - started}
        )
        report["process_cpu_seconds"] = time.process_time() - cpu
        write(report_path, report)
        print(json.dumps(report["roots"][-1]), flush=True)
    report["status"] = "finished"
    write(report_path, report)


def episode_metadata(meta, labels_reference):
    root, branch = meta["root"], meta["branch"]
    return {
        "collection": COLLECTION,
        "task_protocol": "TASK-048",
        "privileged_scripted_collector": True,
        "kind": meta["kind"],
        "root_episode_id": root["episode_id"],
        "reset_seed": root["seed"],
        "noise_level": root["noise_level"] if branch is None else max(1, root["noise_level"]),
        "root_noise_level": root["noise_level"],
        "aim_offset_applied": root["aim_offset_xy_m"] is not None,
        "branch_kind": None if branch is None else branch["kind"],
        "branch_params": None if branch is None else branch["params"],
        "branch_frame_in_root": meta["branch_frame"],
        "termination": meta["termination"],
        "model_input_cameras": ["onboard_rgb", "hand_crop_rgb"],
        "privileged_outcome_labels": meta["outcome"],
        "training_labels": labels_reference,
        "label_policy": training_labels.USE_POLICY,
    }


def assemble(plan, work, dataset, provenance):
    from embodied_jepa.data import DatasetStore, Episode
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import MuJoCoSimulation

    bootstrap = G1Embodiment(
        MuJoCoSimulation(
            object_kind="apple", container_kind="plate", width=IMAGE_SIZE, height=IMAGE_SIZE
        )
    )
    try:
        store = DatasetStore.create(
            dataset,
            fps=FPS,
            state_schema=bootstrap.state_schema,
            action_manifest=bootstrap.manifest,
            provenance=provenance,
        )
    finally:
        bootstrap.close()
    assignments = {"train": [], "val": [], "test": [], "holdout": []}
    for root in plan["roots"]:
        for episode_id in [root["episode_id"]] + [b["episode_id"] for b in root["branches"]]:
            folder = work / "episodes" / episode_id
            if not (folder / "meta.json").is_file():
                continue
            meta = json.loads((folder / "meta.json").read_text())
            payload = (folder / "labels.npz").read_bytes()
            if hashlib.sha256(payload).hexdigest() != meta["labels_sha256"]:
                raise ValueError(f"label shard hash changed: {episode_id}")
            reference = training_labels.write_encoded(store.root, episode_id, payload)
            with np.load(folder / "arrays.npz") as arrays:
                episode = Episode(
                    episode_id=episode_id,
                    session_id=root["session_id"],
                    task="apple_to_plate",
                    object_id="apple",
                    container_id="plate",
                    observations={
                        "onboard_rgb": arrays["onboard_rgb"],
                        "hand_crop_rgb": arrays["hand_crop_rgb"],
                    },
                    robot_states=arrays["robot_states"],
                    state_mask=arrays["state_mask"],
                    actions=arrays["actions"],
                    timestamps=arrays["timestamps"],
                    state_schema=store.state_schema,
                    terminated=meta["termination"] == "success",
                    truncated=meta["termination"] != "success",
                    raw_actions=arrays["raw_actions"],
                    metadata=episode_metadata(meta, reference),
                )
            store.write_episode(episode)
            assignments[root["split"]].append(episode_id)
    store.freeze_split_assignments(
        assignments,
        provenance={
            "method": "whole-reset sessions assigned by the frozen plan before collection",
            "split_seed": SPLIT_SEED,
            "plan_sha256": provenance["plan_sha256"],
        },
        heldout_combinations=(),
    )
    store.fit_normalization()
    store.verify()
    return store


def acceptance(dataset, plan, *, decode=True):
    """Compute every preregistered acceptance quantity from the sealed corpus."""
    from embodied_jepa.audit import audit_dataset
    from embodied_jepa.data import DatasetStore

    store = DatasetStore(dataset)
    rows = store.manifest["episodes"]
    roots = [r for r in rows if r["metadata"]["kind"] == "root"]
    branches = [r for r in rows if r["metadata"]["kind"] == "branch"]

    def o(row):
        return row["metadata"]["privileged_outcome_labels"]

    attempted = [r for r in rows if o(r)["grasp_phase_attempted"]]
    grasp_fail = [r for r in attempted if o(r)["grasp_phase_failure"]]
    info = json.loads((store.root / "meta/info.json").read_text())["features"]
    result = {
        "dataset_manifest_sha256": store.manifest_hash,
        "episodes": len(rows),
        "root_episodes": len(roots),
        "branch_episodes": len(branches),
        "transitions": sum(r["length"] - 1 for r in rows),
        "root_full_success": sum(o(r)["stages"]["success"] for r in roots),
        "root_grasp": sum(o(r)["stages"]["grasp"] for r in roots),
        "grasp_phase_attempts": len(attempted),
        "grasp_phase_failures": len(grasp_fail),
        "grasp_phase_failure_fraction": len(grasp_fail) / max(1, len(attempted)),
        "grasp_phase_successes": len(attempted) - len(grasp_fail),
        "branch_grasp": sum(o(r)["stages"]["grasp"] for r in branches),
        "branch_dropped_after_grasp": sum(o(r)["dropped_after_grasp"] for r in branches),
        "terminations": {},
        "failure_stage_counts": {},
        "branch_kind_outcomes": {},
        "noise_level_outcomes": {},
        "camera_shapes": {
            k.removeprefix("observation.images."): v["shape"]
            for k, v in info.items()
            if v["dtype"] == "image"
        },
    }
    for r in rows:
        t = r["metadata"]["termination"]
        result["terminations"][t] = result["terminations"].get(t, 0) + 1
        f = str(o(r)["failure_stage"])
        result["failure_stage_counts"][f] = result["failure_stage_counts"].get(f, 0) + 1
    for r in branches:
        k = r["metadata"]["branch_kind"]
        item = result["branch_kind_outcomes"].setdefault(k, {"episodes": 0, "grasp": 0})
        item["episodes"] += 1
        item["grasp"] += o(r)["stages"]["grasp"]
    for r in roots:
        k = str(r["metadata"]["noise_level"])
        item = result["noise_level_outcomes"].setdefault(
            k, {"episodes": 0, "grasp": 0, "success": 0}
        )
        item["episodes"] += 1
        item["grasp"] += o(r)["stages"]["grasp"]
        item["success"] += o(r)["stages"]["success"]
    # Split integrity and normalization scope.
    splits = store.manifest["splits"]
    by_session = {}
    for r in rows:
        name = next(n for n, ids in splits.items() if r["episode_id"] in ids)
        by_session.setdefault(r["session_id"], set()).add(name)
    planned = {root["session_id"]: root["split"] for root in plan["roots"]}
    result["split_counts"] = {k: len(v) for k, v in splits.items()}
    result["split_sessions"] = {
        n: sum(1 for s in by_session.values() if s == {n}) for n in ("train", "val", "test")
    }
    result["sessions_spanning_splits"] = sum(len(s) > 1 for s in by_session.values())
    result["sessions_off_plan"] = sum(
        planned.get(session) not in names for session, names in by_session.items()
    )
    result["normalization_is_train_only"] = sorted(
        store.manifest["normalization"]["episode_ids"]
    ) == sorted(splits["train"])
    # Label sidecars: presence, hash, privileged gating and row counts.
    label_errors = []
    actions, bases, sensitivity = [], [], []
    endpoints = {}
    for r in rows:
        reference = r["metadata"]["training_labels"]
        try:
            training_labels.load(store.root, reference, groups=("robot",))
            labels = training_labels.load(
                store.root,
                reference,
                groups=("robot", "collector", "privileged"),
                acknowledge_privileged_training_labels=True,
            )
            n = r["length"]
            if labels["privileged__apple_position_world"].shape != (n, 3) or labels[
                "collector__base_action"
            ].shape != (n - 1, 14):
                label_errors.append(f"{r['episode_id']}: label rows")
        except Exception as error:
            label_errors.append(f"{r['episode_id']}: {type(error).__name__}: {error}")
    gated = False
    try:
        training_labels.load(
            store.root, rows[0]["metadata"]["training_labels"], groups=("privileged",)
        )
    except Exception:
        gated = True
    result["label_errors"] = label_errors
    result["privileged_labels_gated"] = gated
    if decode:
        for r in rows:
            episode = store.read_episode(r["episode_id"])
            actions.append(episode.actions)
            bases.append(
                training_labels.load(
                    store.root,
                    r["metadata"]["training_labels"],
                    groups=("collector",),
                    acknowledge_privileged_training_labels=True,
                )["collector__base_action"]
            )
            if r["metadata"]["kind"] == "branch" and episode.transitions >= 16:
                endpoints.setdefault(r["metadata"]["root_episode_id"], []).append(
                    episode.observations["onboard_rgb"][16].astype(float)
                )
        for frames in endpoints.values():
            for i in range(len(frames)):
                for j in range(i + 1, len(frames)):
                    rms = np.sqrt(np.mean(((frames[i] - frames[j]) / 255) ** 2))
                    sensitivity.append(rms >= 1 / 255)
        action = np.concatenate(actions)
        base = np.concatenate(bases)
        right = action[:, 6:12]
        result["action_coverage"] = {
            "std": action.std(axis=0).tolist(),
            "right_arm_fraction_abs_above_0_1": (np.abs(right) > 0.1).mean(axis=0).tolist(),
            "right_arm_positive_fraction_above_0_1": (right > 0.1).mean(axis=0).tolist(),
            "right_arm_negative_fraction_below_minus_0_1": (right < -0.1).mean(axis=0).tolist(),
            "grasp_fraction_interior": float((np.abs(action[:, 13]) < 0.9).mean()),
            "off_script_fraction": float((np.abs(action - base).max(axis=1) > 0.02).mean()),
        }
        result["branch_pair_rgb_distinct_at_16"] = {
            "pairs": len(sensitivity),
            "distinct": int(np.sum(sensitivity)),
        }
        audit = audit_dataset(dataset)
        result["audit"] = {
            "episodes": audit["episodes"],
            "transitions": audit["transitions"],
            "time_delta_seconds": audit["time_delta_seconds"],
            "storage_bytes": audit["storage_bytes"],
            "split_counts": audit["split_counts"],
        }
    return result


def checks(result):
    """Preregistered acceptance checks (docs/experiments/apple_wide_collection_v1.md)."""
    cov = result.get("action_coverage", {})
    audit = result.get("audit", {})
    items = {
        "A1_root_episodes_ge_190": result["root_episodes"] >= 190,
        "A2_branch_episodes_ge_450": result["branch_episodes"] >= 450,
        "A3_root_full_success_ge_80": result["root_full_success"] >= 80,
        "A4_grasp_phase_failure_fraction_0_20_to_0_80": 0.20
        <= result["grasp_phase_failure_fraction"]
        <= 0.80,
        "A5_grasp_phase_successes_ge_250": result["grasp_phase_successes"] >= 250,
        "A6_right_arm_both_signs_ge_0_03": bool(cov)
        and min(cov["right_arm_positive_fraction_above_0_1"]) >= 0.03
        and min(cov["right_arm_negative_fraction_below_minus_0_1"]) >= 0.03,
        "A7_off_script_fraction_ge_0_50": bool(cov) and cov["off_script_fraction"] >= 0.50,
        "A8_branch_pairs_distinct_ge_0_90": bool(result.get("branch_pair_rgb_distinct_at_16"))
        and result["branch_pair_rgb_distinct_at_16"]["distinct"]
        >= 0.9 * result["branch_pair_rgb_distinct_at_16"]["pairs"]
        > 0,
        "A9_no_split_leakage": result["sessions_spanning_splits"] == 0
        and result["sessions_off_plan"] == 0
        and result["normalization_is_train_only"]
        and all(result["split_sessions"][k] > 0 for k in ("train", "val", "test")),
        "A10_audit_passes": bool(audit)
        and audit["episodes"] == result["episodes"]
        and abs(audit["time_delta_seconds"]["min"] - 1 / FPS) < 1e-6
        and abs(audit["time_delta_seconds"]["max"] - 1 / FPS) < 1e-6,
        "A11_labels_separated": not result["label_errors"] and result["privileged_labels_gated"],
        "A12_resolution": result["camera_shapes"]
        == {
            "onboard_rgb": [IMAGE_SIZE, IMAGE_SIZE, 3],
            "hand_crop_rgb": [IMAGE_SIZE, IMAGE_SIZE, 3],
        },
        "A13_run_integrity": result.get("integrity", {}).get("ok", False),
    }
    return {"checks": items, "all_passed": all(items.values())}


def runtime_versions():
    import platform

    import mujoco

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "uv_lock_sha256": digest(ROOT / "uv.lock"),
    }


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def run(args):
    dataset, work = args.output.resolve(), args.work.resolve()
    for path in (dataset, work):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        raise RuntimeError("the frozen run requires a clean tracked checkout")
    seeds = PILOT_SEEDS if args.seeds == "pilot" else FROZEN_SEEDS
    if args.limit:
        seeds = seeds[: args.limit]
    plan = make_plan(seeds)
    work.mkdir(parents=True)
    write(work / "plan.json", plan)
    source_hashes = {
        str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "src").rglob("*.py"))
    }
    provenance = {
        "producer": "scripts/collect_apple_wide.py (privileged scripted collector, TASK-048)",
        "license": "project-generated data; upstream robot assets retain BSD-3-Clause",
        "task_specific_development": True,
        "learned_control": False,
        "seed_set": args.seeds,
        "plan_sha256": digest(work / "plan.json"),
        "source_revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(dirty),
        **tracked_inputs(),
        "runtime_source_hashes": source_hashes,
        "runtime": runtime_versions(),
        "workers": args.workers,
        "max_seconds": args.max_seconds,
        "training_label_schema": training_labels.SCHEMA_VERSION,
        "training_label_policy": training_labels.USE_POLICY,
    }
    write(work / "provenance.json", provenance)
    env = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--plan",
                str(work / "plan.json"),
                "--work",
                str(work),
                "--index",
                str(i),
                "--count",
                str(args.workers),
                "--max-seconds",
                str(args.worker_seconds),
            ],
            cwd=ROOT,
            env=env,
            stdout=(work / f"worker-{i:02d}.log").open("w"),
            stderr=subprocess.STDOUT,
        )
        for i in range(args.workers)
    ]
    timed_out = False
    while any(p.poll() is None for p in processes):
        if elapsed() >= args.max_seconds:
            timed_out = True
            for p in processes:
                p.kill()
            break
        time.sleep(1)
    for p in processes:
        p.wait()
    collection_seconds = elapsed()
    write(
        work / "supervisor.json",
        {
            "timed_out": timed_out,
            "worker_returncodes": [p.returncode for p in processes],
            "collection_seconds": collection_seconds,
        },
    )
    return finalize(work, dataset)


def tracked_inputs():
    return {
        "collector_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(ROOT / "docs/experiments/apple_wide_collection_v1.md"),
        "action_manifest_sha256": digest(ROOT / "configs/g1_sim_action.json"),
        "asset_manifest_sha256": digest(ROOT / "assets/manifest.json"),
    }


def finalize(work, dataset):
    """Assemble shards, seal, audit and score. Re-runnable from an existing work directory
    (``finalize`` subcommand) into a new dataset directory; never re-simulates."""
    work, dataset = Path(work).resolve(), Path(dataset).resolve()
    if dataset.exists():
        raise FileExistsError(f"refusing to overwrite {dataset}")
    plan = json.loads((work / "plan.json").read_text())
    provenance = json.loads((work / "provenance.json").read_text())
    supervisor = json.loads((work / "supervisor.json").read_text())
    timed_out, returncodes = supervisor["timed_out"], supervisor["worker_returncodes"]
    report = {"status": "finalizing", "dataset": str(dataset), "supervisor": supervisor}
    report_path = work / "collection_report.json"
    verdict = None
    try:
        source_changed = [
            k for k, v in provenance["runtime_source_hashes"].items() if digest(ROOT / k) != v
        ]
        inputs_changed = [k for k, v in tracked_inputs().items() if provenance[k] != v]
        report["source_changed"] = source_changed
        report["inputs_changed"] = inputs_changed
        if digest(work / "plan.json") != provenance["plan_sha256"]:
            raise RuntimeError("plan changed after collection started")
        workers = [json.loads(p.read_text()) for p in sorted(work.glob("worker-*.json"))]
        started = elapsed()
        store = assemble(plan, work, dataset, provenance)
        report["assembly_seconds"] = elapsed() - started
        report["dataset_manifest_sha256"] = store.manifest_hash
        result = acceptance(dataset, plan)
        statuses = [r for w in workers for r in w["roots"]]
        branched = [
            e
            for w in workers
            for e in w["episodes"]
            if e["kind"] == "root" and e.get("branch_frame")
        ]
        integrity = {
            "roots_planned": len(plan["roots"]),
            "roots_reported": len(statuses),
            "roots_completed": sum(r["status"] == "completed" for r in statuses),
            "runtime_errors": [r for r in statuses if r["status"].startswith("runtime_error")],
            "not_started": sum(r["status"].startswith("not_started") for r in statuses),
            "branched_roots": len(branched),
            "restore_exact": sum(e.get("restore_exact") is True for e in branched),
            "supervisor_timeout": timed_out,
            "worker_returncodes": returncodes,
            "source_or_inputs_changed": bool(source_changed or inputs_changed),
        }
        integrity["ok"] = bool(
            integrity["roots_completed"] == integrity["roots_planned"]
            and integrity["restore_exact"] == integrity["branched_roots"]
            and not timed_out
            and not any(returncodes)
            and not integrity["source_or_inputs_changed"]
        )
        result["integrity"] = integrity
        verdict = checks(result)
        report.update(
            status="completed" if not timed_out else "supervisor_timeout",
            root_statuses=sorted(statuses, key=lambda r: r["seed"]),
            plan_sha256=provenance["plan_sha256"],
            acceptance=result,
            verdict=verdict,
        )
    except BaseException as error:
        report.update(status="finalize_error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["total_seconds"] = elapsed()
        write(report_path, report)
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["all_passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--seeds", choices=("frozen", "pilot"), default="frozen")
    p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--output", type=Path, required=True, help="new dataset directory")
    r.add_argument("--work", type=Path, required=True, help="new shard/report directory")
    r.add_argument("--seeds", choices=("frozen", "pilot"), default="frozen")
    r.add_argument("--limit", type=int, default=0, help="smoke only: first N seeds")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--max-seconds", type=float, default=5400)
    r.add_argument("--worker-seconds", type=float, default=4500)
    r.add_argument("--allow-dirty", action="store_true", help="pilot/smoke only")
    f = sub.add_parser("finalize", help="assemble/score existing shards; never simulates")
    f.add_argument("--work", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True, help="new dataset directory")
    w = sub.add_parser("worker")
    w.add_argument("--plan", type=Path, required=True)
    w.add_argument("--work", type=Path, required=True)
    w.add_argument("--index", type=int, required=True)
    w.add_argument("--count", type=int, required=True)
    w.add_argument("--max-seconds", type=float, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        write(args.out, make_plan(PILOT_SEEDS if args.seeds == "pilot" else FROZEN_SEEDS))
        print(digest(args.out))
        return 0
    if args.command == "worker":
        worker(args.plan, args.work, args.index, args.count, args.max_seconds)
        return 0
    if args.command == "finalize":
        return finalize(args.work, args.output)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
