"""TASK-064: the look-prefix 112 px Apple->Plate corpus (``apple_look_corpus_v1``).

A PRIVILEGED scripted collector (``scripted.apple_collector_policy``, which reads simulator truth
at reset) produces root episodes on the TASK-047 wide-jitter reset rule. Every episode starts with
TASK-061's constant, unperturbed 8-command look; the collector policy then takes over from the
post-look pose, with TASK-048's seeded perturbations, aim offsets and grasp-phase branches.
Nothing here is a learned result: learned Apple->Plate is still 0 successes.

The plan, the look, the checks and the rows are fixed in ``embodied_jepa.look_corpus``. The
episode mechanics (perturbation, controller, projected step, snapshot/restore, outcome labels)
are TASK-048's ``scripts/collect_apple_wide.py``, loaded unchanged (its hash is pinned).

Model-input stream: ``onboard_rgb`` at 112 px only (the simulator's own render; every renderer
renders once and discards the result before any frame is kept), plus proprioception and the
executed 14-D actions. Simulator truth goes ONLY into per-episode label sidecars.

**The test split is never decoded.** Every per-episode fact the checks need on all splits is
recorded by the worker at collection time; decoding for QA is restricted to train + val.

Subcommands: ``plan``, ``run`` (workers -> assemble -> seal -> acceptance -> readability gate ->
decision), ``worker`` (internal), ``finalize`` (assemble and score existing shards; never
simulates).
"""

from __future__ import annotations

# ruff: noqa: E402
import time

ENTRY_WALL, ENTRY_MONO = time.time(), time.monotonic()

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import look_corpus as lc
from embodied_jepa import observation_reprobe as orp
from embodied_jepa import training_labels

PROTOCOL_DOC = ROOT / "docs" / "experiments" / "apple_look_corpus_v1.md"
MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-look-corpus-v1.json"


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# TASK-048's collector, loaded unchanged (G-hash pins its bytes).
WIDE = _load("_collect_apple_wide", "scripts/collect_apple_wide.py")


def digest(path) -> str:
    return WIDE.digest(path)


def elapsed() -> float:
    return max(time.time() - ENTRY_WALL, time.monotonic() - ENTRY_MONO)


def write(path, value) -> None:
    WIDE.write(path, value)


# ----- episode mechanics ----------------------------------------------------------------------
class LookController:
    """Issues the constant look through the collector's own step, unperturbed."""

    phase_index = lc.LOOK_PHASE_INDEX

    def __init__(self):
        self.sequence = orp.look_sequence()
        self.executed = 0

    @property
    def done(self) -> bool:
        return self.executed >= lc.LOOK_STEPS

    def command(self, robot):
        base = np.asarray(self.sequence[self.executed], np.float32).copy()
        return base, base.copy(), "none"

    def advance(self, result) -> None:
        self.executed += 1


class Recorder(WIDE.Recorder):
    """TASK-048's recorder without the hand crop: onboard 112 px, state and truth labels."""

    def observe(self, robot, crop, stages):
        del crop  # no hand crop in this corpus
        obs = robot.observe()
        truth = robot.sim.task_truth()
        row = self.rows
        row["onboard"].append(obs.images["onboard_rgb"][0].copy())
        row["state"].append(obs.robot_state[0].copy())
        row["mask"].append(obs.state_mask[0].copy())
        row["time"].append(float(obs.timestamps[0]))
        row["apple"].append(np.asarray(truth["object_position"], float))
        row["apple_velocity"].append(np.asarray(truth["object_velocity"], float))
        row["plate"].append(np.asarray(truth["plate_position"], float))
        row["palm"].append(np.asarray(robot.sim.data.site("right_ee").xpos, float).copy())
        row["contact"].append(bool(truth["hand_contact"]))
        row["dropped"].append(bool(truth["dropped"]))
        row["stages"].append([bool(stages.get(k, False)) for k in WIDE.STAGES])

    def sensors(self, index=-1):
        r = self.rows
        return (r["onboard"][index], r["state"][index], r["time"][index])

    def labels(self):
        out = super().labels()
        del out["robot__hand_crop_window"]
        return out


def state_vector(robot) -> np.ndarray:
    sim = robot.sim
    return np.concatenate((sim.data.qpos[sim.qadr], sim.data.qvel[sim.vadr])).astype(np.float64)


def make_robot():
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import MuJoCoSimulation

    robot = G1Embodiment(
        MuJoCoSimulation(
            object_kind="apple", container_kind="plate", width=lc.IMAGE_SIZE, height=lc.IMAGE_SIZE
        )
    )
    robot.sim.render()  # warm-up: creates the renderer and discards its first render
    return robot


def run_look(robot, scorer, rec) -> tuple[str | None, dict]:
    """The 8 look commands through TASK-048's step. Returns (stop or None, look facts)."""
    controller = LookController()
    start_apple = np.asarray(rec.rows["apple"][0], float)
    start_plate = np.asarray(rec.rows["plate"][0], float)
    stop = None
    while not controller.done:
        _score, stop = WIDE.step(robot, scorer, None, controller, rec)
        if stop:
            break
    rows = rec.rows
    frames = range(1, len(rows["apple"]))
    facts = lc.look_facts(
        applied=np.asarray(rec.steps["applied"][: controller.executed], np.float64),
        post_state=state_vector(robot),
        apple_xy_moves=[
            float(np.abs(np.asarray(rows["apple"][i])[:2] - start_apple[:2]).max()) for i in frames
        ],
        plate_moves=[
            float(np.abs(np.asarray(rows["plate"][i]) - start_plate).max()) for i in frames
        ],
        contacts=[bool(rows["contact"][i]) for i in frames],
    )
    facts["stop"] = stop
    facts["post_look_frame_sha256"] = (
        hashlib.sha256(np.ascontiguousarray(rows["onboard"][-1]).tobytes()).hexdigest()
        if stop is None
        else None
    )
    return stop, facts


def intervals(rec) -> list[float]:
    return np.diff(np.asarray(rec.rows["time"], np.float64)).tolist()


def save_episode(work, episode_id, rec, meta) -> None:
    folder = work / "episodes" / episode_id
    folder.mkdir(parents=True, exist_ok=False)
    r, s = rec.rows, rec.steps
    np.savez_compressed(
        folder / "arrays.npz",
        onboard_rgb=np.asarray(r["onboard"], np.uint8),
        robot_states=np.asarray(r["state"], np.float32),
        state_mask=np.asarray(r["mask"], bool),
        timestamps=np.asarray(r["time"], np.float64),
        actions=np.asarray(s["applied"], np.float32),
        raw_actions=np.asarray(s["raw"], np.float32),
    )
    payload, label_sha = training_labels.encode(rec.labels())
    (folder / "labels.npz").write_bytes(payload)
    write(folder / "meta.json", meta | {"labels_sha256": label_sha})


def episode_record(episode_id, kind, termination, rec, outcome) -> dict:
    gaps = intervals(rec)
    return {
        "episode_id": episode_id,
        "kind": kind,
        "termination": termination,
        "transitions": len(rec.steps["applied"]),
        "outcome": outcome,
        "stored": False,
        "interval_min": min(gaps) if gaps else None,
        "interval_max": max(gaps) if gaps else None,
        "intervals_ok": lc.frame_interval_ok(gaps),
        "frames": len(rec.rows["onboard"]),
        "frame_shape": list(np.asarray(rec.rows["onboard"][0]).shape),
    }


def run_root(root, work, report):
    from embodied_jepa.scripted import apple_collector_policy
    from embodied_jepa.task import AppleToPlateTask

    robot = make_robot()
    try:
        truth = robot.reset(root["seed"], object_xy=root["object_xy"], plate_xy=root["plate_xy"])
        if not np.allclose(truth["base_rotation_world"], np.eye(3), atol=1e-7):
            raise ValueError("collector assumes the fixed upright pelvis")
        reset_truth = robot.sim.task_truth()
        scorer = AppleToPlateTask(robot)
        rec = Recorder()
        rec.observe(robot, None, scorer.evaluate())
        # --- the look prefix: constant, unperturbed, reads nothing about the reset ---
        stop, look = run_look(robot, scorer, rec)
        look["seed"] = root["seed"]
        report["looks"].append(look)
        termination = None if stop is None else f"look_{stop}"
        saved, branch_frame = None, None
        controller = None
        if termination is None:
            policy = apple_collector_policy(reset_truth)
            if root["aim_offset_xy_m"] is not None:
                WIDE.shifted(policy, (0, 1, 2, 3), root["aim_offset_xy_m"])
            controller = WIDE.Controller(
                policy, WIDE.Perturber(root["noise_level"], root["noise_seed"])
            )
        while termination is None:
            if controller.done:
                termination = "policy_complete"
                break
            if len(rec.steps["applied"]) - lc.LOOK_STEPS >= lc.POLICY_MAX_COMMANDS:
                termination = "step_limit"
                break
            if (
                saved is None
                and controller.phase_index == lc.BRANCH_PHASE
                and controller.policy.phase_step == 0
            ):
                saved = WIDE.snapshot(robot, scorer, controller)
                branch_frame = len(rec.steps["applied"])
            score, stop = WIDE.step(robot, scorer, None, controller, rec)
            if stop:
                termination = stop
            elif score["success"]:
                termination = "success"
        root_outcome = WIDE.outcome(
            rec, termination, grasp_phase_attempted=branch_frame is not None
        )
        entry = episode_record(root["episode_id"], "root", termination, rec, root_outcome)
        entry["branch_frame"] = branch_frame
        report["episodes"].append(entry)
        if len(rec.steps["applied"]) >= lc.MIN_STORED_TRANSITIONS:
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
                    "look": look,
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
        # Restoration exactness (TASK-048): the restored state reproduces the root's sensors at
        # the branch frame, and re-executing the root's next command reproduces frame + 1.
        reference = root_rec.sensors(branch_frame)
        probe = WIDE.restore(robot, scorer, saved)
        check = Recorder()
        check.observe(robot, None, {})
        exact = WIDE.same_sensors(check.sensors(0), reference)
        if exact and branch_frame < len(root_rec.steps["applied"]):
            command = root_rec.steps["projected"][branch_frame]
            result = robot.execute(command)
            probe.advance(result)
            check.observe(robot, None, {})
            exact = bool(
                result.applied_action is not None
                and np.array_equal(result.applied_action, root_rec.steps["applied"][branch_frame])
                and WIDE.same_sensors(check.sensors(1), root_rec.sensors(branch_frame + 1))
            )
        entry["restore_exact"] = exact
        if not exact:
            raise ValueError(f"inexact restoration at {root['episode_id']}")
        for branch in root["branches"]:
            controller = WIDE.restore(robot, scorer, saved)
            controller.apply_branch(branch["kind"], branch["params"], branch["noise_seed"])
            rec = Recorder()
            rec.observe(robot, None, {})
            if not WIDE.same_sensors(rec.sensors(0), reference):
                raise ValueError("branch start sensors differ from the root")
            rec.rows["stages"][0] = list(root_rec.rows["stages"][branch_frame])
            # A branch continues the root's clock; its first frame's time is the root's.
            termination = None
            while termination is None:
                if controller.done or controller.phase_index >= lc.BRANCH_END_PHASE:
                    termination = "branch_complete"
                    break
                score, stop = WIDE.step(robot, scorer, None, controller, rec)
                if stop:
                    termination = stop
                elif score["success"]:
                    termination = "success"
            branch_outcome = WIDE.outcome(rec, termination, grasp_phase_attempted=True)
            row = episode_record(branch["episode_id"], "branch", termination, rec, branch_outcome)
            report["episodes"].append(row)
            if len(rec.steps["applied"]) >= lc.MIN_STORED_TRANSITIONS:
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
        robot.close()


def worker(plan_path, work, index, count, max_seconds):
    plan = json.loads(Path(plan_path).read_text())
    report_path = work / f"worker-{index:02d}.json"
    report = {"worker": index, "roots": [], "episodes": [], "looks": [], "status": "running"}
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


# ----- assembly -------------------------------------------------------------------------------
def episode_metadata(meta, labels_reference):
    root, branch = meta["root"], meta["branch"]
    return {
        "collection": lc.COLLECTION,
        "task_protocol": lc.TASK,
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
        "look_prefix_steps": lc.LOOK_STEPS if branch is None else 0,
        "decision_frame": lc.DECISION_FRAME if branch is None else None,
        "termination": meta["termination"],
        "model_input_cameras": list(lc.CAMERAS),
        "privileged_outcome_labels": meta["outcome"],
        "training_labels": labels_reference,
        "label_policy": training_labels.USE_POLICY,
    }


def assemble(plan, work, dataset, provenance):
    from embodied_jepa.data import DatasetStore, Episode

    bootstrap = make_robot()
    try:
        store = DatasetStore.create(
            dataset,
            fps=lc.FPS,
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
                    observations={"onboard_rgb": arrays["onboard_rgb"]},
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
            "split_seed": lc.SPLIT_SEED,
            "plan_sha256": provenance["plan_sha256"],
        },
        heldout_combinations=(),
    )
    store.fit_normalization()
    store.verify()
    return store


# ----- acceptance (protocol §6) ---------------------------------------------------------------
class TrainValReader:
    """Reads ONLY train + val episodes; any other id is refused (the test split is never
    decoded during QA)."""

    def __init__(self, store, allowed_ids):
        self.store = store
        self.allowed = frozenset(allowed_ids)

    def episode(self, episode_id):
        if episode_id not in self.allowed:
            raise lc.GuardError(f"refusing to decode {episode_id}: not a train or val episode")
        return self.store.read_episode(episode_id)

    def labels(self, row, groups, acknowledge=False):
        if row["episode_id"] not in self.allowed:
            raise lc.GuardError(f"refusing to read labels of {row['episode_id']}: not train/val")
        return training_labels.load(
            self.store.root,
            row["metadata"]["training_labels"],
            groups=groups,
            acknowledge_privileged_training_labels=acknowledge,
        )


def acceptance(dataset, plan, workers) -> dict:
    """Every preregistered acceptance quantity. Counts come from manifest metadata and the
    workers' collection-time records (all splits); decoding is train + val only."""
    from embodied_jepa.data import DatasetStore

    store = DatasetStore(dataset)
    rows = store.manifest["episodes"]
    roots = [r for r in rows if r["metadata"]["kind"] == "root"]
    branches = [r for r in rows if r["metadata"]["kind"] == "branch"]
    splits = store.manifest["splits"]
    readable = set(splits["train"]) | set(splits["val"])
    reader = TrainValReader(store, readable)

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
    by_split = {}
    for name, ids in splits.items():
        chosen = [r for r in rows if r["episode_id"] in set(ids)]
        by_split[name] = {
            "episodes": len(chosen),
            "root_success": sum(
                o(r)["stages"]["success"] for r in chosen if r["metadata"]["kind"] == "root"
            ),
            "transitions": sum(r["length"] - 1 for r in chosen),
        }
    result["by_split"] = by_split
    # A9: split integrity and normalisation scope (manifest only).
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
    # A11: sidecars. Train + val are loaded (hash, rows, gating); test sidecars are checked by
    # their sha256 against the episode metadata only, never loaded.
    label_errors = []
    for r in rows:
        reference = r["metadata"]["training_labels"]
        try:
            if r["episode_id"] in readable:
                reader.labels(r, ("robot",))
                labels = reader.labels(r, ("robot", "collector", "privileged"), acknowledge=True)
                n = r["length"]
                if labels["privileged__apple_position_world"].shape != (n, 3) or labels[
                    "collector__base_action"
                ].shape != (n - 1, 14):
                    label_errors.append(f"{r['episode_id']}: label rows")
                if r["metadata"]["kind"] == "root":
                    phases = labels["collector__phase_index"][: lc.LOOK_STEPS]
                    if not np.all(phases == lc.LOOK_PHASE_INDEX):
                        label_errors.append(f"{r['episode_id']}: look phase labels")
            else:
                payload = (store.root / reference["path"]).read_bytes()
                if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
                    label_errors.append(f"{r['episode_id']}: sidecar hash")
        except Exception as error:
            label_errors.append(f"{r['episode_id']}: {type(error).__name__}: {error}")
    gated = False
    first_readable = next(r for r in rows if r["episode_id"] in readable)
    try:
        training_labels.load(
            store.root, first_readable["metadata"]["training_labels"], groups=("privileged",)
        )
    except Exception:
        gated = True
    result["label_errors"] = label_errors
    result["privileged_labels_gated"] = gated
    # A6-A8 and the A10 audit: decoded train + val episodes only.
    actions, bases, sensitivity, gaps = [], [], [], []
    endpoints = {}
    decoded = 0
    for r in rows:
        if r["episode_id"] not in readable:
            continue
        episode = reader.episode(r["episode_id"])
        decoded += 1
        gaps.extend(np.diff(episode.timestamps).tolist())
        actions.append(episode.actions)
        bases.append(reader.labels(r, ("collector",), acknowledge=True)["collector__base_action"])
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
        "scope": "train + val episodes",
        "std": action.std(axis=0).tolist(),
        "right_arm_positive_fraction_above_0_1": (right > 0.1).mean(axis=0).tolist(),
        "right_arm_negative_fraction_below_minus_0_1": (right < -0.1).mean(axis=0).tolist(),
        "grasp_fraction_interior": float((np.abs(action[:, 13]) < 0.9).mean()),
        "off_script_fraction": float((np.abs(action - base).max(axis=1) > 0.02).mean()),
    }
    result["branch_pair_rgb_distinct_at_16"] = {
        "scope": "train + val sibling branches",
        "pairs": len(sensitivity),
        "distinct": int(np.sum(sensitivity)),
    }
    recorded = {e["episode_id"]: e for w in workers for e in w["episodes"] if e.get("stored")}
    stored_ids = {r["episode_id"] for r in rows}
    result["audit"] = {
        "scope": "train + val decoded; every episode's frame intervals recorded at collection",
        "episodes_expected": len(readable),
        "episodes_decoded": decoded,
        "decoded_interval_min": min(gaps),
        "decoded_interval_max": max(gaps),
        "decoded_intervals_ok": lc.frame_interval_ok(gaps),
        "collection_records_cover_every_stored_episode": stored_ids <= set(recorded),
        "collection_intervals_ok": all(
            recorded.get(e, {}).get("intervals_ok", False) for e in stored_ids
        ),
        "storage_bytes": sum(p.stat().st_size for p in Path(dataset).rglob("*") if p.is_file()),
    }
    result["audit"]["all_intervals_ok"] = bool(
        result["audit"]["decoded_intervals_ok"]
        and result["audit"]["collection_records_cover_every_stored_episode"]
        and result["audit"]["collection_intervals_ok"]
    )
    return result


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


def tracked_inputs():
    return {
        "collector_sha256": digest(Path(__file__)),
        "wide_collector_sha256": digest(ROOT / "scripts/collect_apple_wide.py"),
        "readability_gate_sha256": digest(ROOT / "scripts/read_apple_look.py"),
        "probe_info_ceiling_sha256": digest(ROOT / "scripts/probe_info_ceiling.py"),
        "look_corpus_module_sha256": digest(ROOT / "src/embodied_jepa/look_corpus.py"),
        "protocol_sha256": digest(PROTOCOL_DOC) if PROTOCOL_DOC.is_file() else None,
        "manifest_sha256": digest(MANIFEST) if MANIFEST.is_file() else None,
        "action_manifest_sha256": digest(ROOT / "configs/g1_sim_action.json"),
        "asset_manifest_sha256": digest(ROOT / "assets/manifest.json"),
    }


def check_pins(manifest: dict, root=ROOT) -> dict:
    """G-hash: every pinned file matches. Returns the per-file sha256 that was found."""
    found, mismatched = {}, []
    for name, want in manifest["hashes"].items():
        path = Path(root) / name
        found[name] = digest(path) if path.is_file() else None
        if found[name] != want:
            mismatched.append(name)
    if mismatched:
        raise lc.GuardError(f"G-hash: {len(mismatched)} pinned files differ: {mismatched[:5]}")
    return found


def preflight(plan, dirty, manifest_path=MANIFEST, root=ROOT) -> dict:
    """G-hash and G-plan before any frozen seed is simulated (protocol §9)."""
    if dirty:
        raise lc.GuardError("G-hash: the tracked tree is not clean")
    if not Path(manifest_path).is_file():
        raise lc.GuardError(f"G-hash: manifest {manifest_path} is missing")
    manifest = json.loads(Path(manifest_path).read_text())
    found = check_pins(manifest, root)
    for script in ("scripts/collect_apple_look.py", "scripts/read_apple_look.py"):
        if script not in manifest["hashes"]:
            raise lc.GuardError(f"G-hash: {script} is not pinned in the manifest")
    pins = manifest["plan"]
    rounded = lc.plan_sha256_rounded(plan)
    if rounded != pins["sha256_rounded_1e-9"]:
        raise lc.GuardError(f"G-plan: rounded plan hash {rounded[:12]} != pinned")
    exact = lc.plan_sha256(plan)
    import platform

    exact_checked = platform.system() == "Darwin" and platform.machine() == "arm64"
    if exact_checked and exact != pins["sha256_macos_arm64"]:
        raise lc.GuardError(f"G-plan: plan hash {exact[:12]} != pinned")
    return {
        "hashes_checked": len(manifest["hashes"]),
        "hashes": found,
        "plan_sha256": exact,
        "plan_sha256_rounded": rounded,
        "exact_plan_hash_checked": exact_checked,
    }


def run(args):
    dataset, work = args.output.resolve(), args.work.resolve()
    for path in (dataset, work):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        raise RuntimeError("the frozen run requires a clean tracked checkout")
    seeds = lc.PILOT_SEEDS if args.seeds == "pilot" else lc.FROZEN_SEEDS
    if args.limit:
        seeds = seeds[: args.limit]
    plan = lc.make_plan(seeds)
    work.mkdir(parents=True)
    (work / "plan.json").write_bytes(lc.plan_bytes(plan))
    if args.seeds == "frozen":
        try:
            report_preflight = preflight(plan, dirty)
        except lc.GuardError as error:
            report = {
                "protocol": lc.PROTOCOL,
                "task": lc.TASK,
                "status": "void",
                "outcome": "V",
                "void_reason": f"guard: {error}",
                "decision": {"outcome": "V", "reason": str(error)},
                "learned_apple_to_plate_successes": 0,
                "exemption_spent": False,
                "total_seconds": elapsed(),
            }
            write_report(work / "collection_report.json", report)
            print(json.dumps({k: report[k] for k in ("status", "outcome", "void_reason")}))
            return 1
        write(work / "preflight.json", report_preflight)
    try:
        collect(args, work, plan, dirty)
    except BaseException as error:
        report = {
            "protocol": lc.PROTOCOL,
            "task": lc.TASK,
            "status": "void: crashed during collection",
            "outcome": "V",
            "void_reason": f"exception: {type(error).__name__}: {error}",
            "traceback": "".join(traceback.format_exception(error)),
            "decision": {"outcome": "V", "reason": f"{type(error).__name__}: {error}"},
            "learned_apple_to_plate_successes": 0,
            "exemption_spent": False,
            "total_seconds": elapsed(),
        }
        write_report(work / "collection_report.json", report)
        raise
    return finalize(work, dataset, readability=not args.skip_readability, invoked_via="run")


def collect(args, work, plan, dirty) -> None:
    """Provenance, workers and the supervisor loop; writes ``supervisor.json``. Workers are
    killed if anything here raises."""
    source_hashes = {
        str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "src").rglob("*.py"))
    }
    provenance = {
        "producer": "scripts/collect_apple_look.py (privileged scripted collector, TASK-064)",
        "license": "project-generated data; upstream robot assets retain BSD-3-Clause",
        "task_specific_development": True,
        "learned_control": False,
        "seed_set": args.seeds,
        "plan_sha256": digest(work / "plan.json"),
        "plan_sha256_rounded": lc.plan_sha256_rounded(plan),
        "source_revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(dirty),
        **tracked_inputs(),
        "runtime_source_hashes": source_hashes,
        "runtime": runtime_versions(),
        "workers": args.workers,
        "max_seconds": args.max_seconds,
        "worker_seconds": args.worker_seconds,
        "training_label_schema": training_labels.SCHEMA_VERSION,
        "training_label_policy": training_labels.USE_POLICY,
        "look_sequence_sha256": lc.LOOK_SEQUENCE_SHA256,
        "cameras": list(lc.CAMERAS),
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
    try:
        while any(p.poll() is None for p in processes):
            if elapsed() >= args.max_seconds:
                timed_out = True
                break
            time.sleep(1)
    finally:
        for p in processes:
            if p.poll() is None:
                p.kill()
        for p in processes:
            p.wait()
    write(
        work / "supervisor.json",
        {
            "timed_out": timed_out,
            "worker_returncodes": [p.returncode for p in processes],
            "collection_seconds": elapsed(),
        },
    )


def integrity_of(plan, workers, supervisor, source_changed, inputs_changed) -> dict:
    statuses = [r for w in workers for r in w["roots"]]
    branched = [
        e for w in workers for e in w["episodes"] if e["kind"] == "root" and e.get("branch_frame")
    ]
    integrity = {
        "roots_planned": len(plan["roots"]),
        "roots_reported": len(statuses),
        "roots_completed": sum(r["status"] == "completed" for r in statuses),
        "runtime_errors": [r for r in statuses if r["status"].startswith("runtime_error")],
        "not_started": sum(r["status"].startswith("not_started") for r in statuses),
        "branched_roots": len(branched),
        "restore_exact": sum(e.get("restore_exact") is True for e in branched),
        "supervisor_timeout": supervisor["timed_out"],
        "worker_returncodes": supervisor["worker_returncodes"],
        "source_or_inputs_changed": bool(source_changed or inputs_changed),
    }
    integrity["ok"] = bool(
        integrity["roots_completed"] == integrity["roots_planned"]
        and integrity["restore_exact"] == integrity["branched_roots"]
        and not integrity["supervisor_timeout"]
        and not any(integrity["worker_returncodes"])
        and not integrity["source_or_inputs_changed"]
    )
    return integrity


def void_reason_of(supervisor, source_changed, inputs_changed) -> str | None:
    """Protocol §11: the run stopped early (supervisor cap, a worker crash) or its code or
    inputs changed while it ran. A root-level runtime error is not a stop: it fails A13."""
    if supervisor["timed_out"]:
        return "supervisor wall cap reached; collection stopped early"
    if any(supervisor["worker_returncodes"]):
        return f"a worker exited non-zero: {supervisor['worker_returncodes']}"
    if source_changed or inputs_changed:
        return f"source or tracked inputs changed during the run: {source_changed + inputs_changed}"
    return None


def _guards():
    from embodied_jepa import info_ceiling, pretrained_encoder

    return (lc.GuardError, info_ceiling.GuardError, pretrained_encoder.WeightsError)


GUARDS = _guards()


def load_reader():
    return _load("_read_apple_look", "scripts/read_apple_look.py")


def finalize(work, dataset, *, readability=True, invoked_via="finalize"):
    """Assemble shards, seal, score, run the readability gate and decide. Re-runnable from an
    existing work directory into a NEW dataset directory; never re-simulates. Always writes
    ``<work>/collection_report.json``, with outcome V and a ``void_reason`` on any stop."""
    work, dataset = Path(work).resolve(), Path(dataset).resolve()
    if dataset.exists():
        raise FileExistsError(f"refusing to overwrite {dataset}")
    report_path = work / "collection_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}; copy it aside first")
    plan = json.loads((work / "plan.json").read_text())
    provenance = json.loads((work / "provenance.json").read_text())
    supervisor = json.loads((work / "supervisor.json").read_text())
    report = {
        "protocol": lc.PROTOCOL,
        "task": lc.TASK,
        "status": "finalizing",
        "dataset": str(dataset),
        "supervisor": supervisor,
        "learned_apple_to_plate_successes": 0,
        "exemption_spent": False,
        "seed_set": provenance["seed_set"],
        "invoked_via": invoked_via,
    }
    try:
        source_changed = [
            k for k, v in provenance["runtime_source_hashes"].items() if digest(ROOT / k) != v
        ]
        inputs_changed = [k for k, v in tracked_inputs().items() if provenance.get(k) != v]
        report["source_changed"] = source_changed
        report["inputs_changed"] = inputs_changed
        if provenance["seed_set"] == "frozen":
            # G-hash again, now covering the gate code loaded below (protocol §9).
            report["pinned_hashes_at_finalize"] = check_pins(json.loads(MANIFEST.read_text()))
        if digest(work / "plan.json") != provenance["plan_sha256"]:
            raise lc.GuardError("plan changed after collection started")
        reason = void_reason_of(supervisor, source_changed, inputs_changed)
        if reason:
            raise lc.GuardError(reason)
        workers = [json.loads(p.read_text()) for p in sorted(work.glob("worker-*.json"))]
        not_started = [
            r["seed"] for w in workers for r in w["roots"] if r["status"].startswith("not_started")
        ]
        if not_started:
            # The worker cap stopped the collection before every root ran: an early stop.
            raise lc.GuardError(f"G-stop: {len(not_started)} roots never started (worker cap)")
        started = elapsed()
        store = assemble(plan, work, dataset, provenance)
        report["assembly_seconds"] = elapsed() - started
        report["dataset_manifest_sha256"] = store.manifest_hash
        result = acceptance(dataset, plan, workers)
        result["integrity"] = integrity_of(plan, workers, supervisor, [], [])
        looks = sorted((x for w in workers for x in w["looks"]), key=lambda x: x["seed"])
        result["look"] = lc.check_look_records(looks, [r["seed"] for r in plan["roots"]])
        verdict = lc.checks(result)
        report["acceptance"] = result
        report["verdict"] = verdict
        report["root_statuses"] = sorted(
            (r for w in workers for r in w["roots"]), key=lambda r: r["seed"]
        )
        report["plan_sha256"] = provenance["plan_sha256"]
        report["plan_sha256_rounded"] = provenance["plan_sha256_rounded"]
        if readability:
            reader = load_reader()
            gate = reader.readability(
                dataset, plan, clock_start=elapsed(), smoke=provenance["seed_set"] == "pilot"
            )
            report["readability"] = gate
            read_ok = gate["passes"]
        else:
            report["readability"] = {"skipped": True}
            read_ok = False
        decision = lc.decide(
            void=False,
            data_all_passed=verdict["all_passed"],
            readability_passed=read_ok,
            look_ok=verdict["checks"]["L1_look_prefix"],
        )
        if provenance["seed_set"] != "frozen":
            # A pilot/smoke run is never a row (its readability targets are noise).
            decision = {"outcome": "smoke (not a row)", "would_be_row_on_noise": decision}
        report["decision"] = decision
        report["outcome"] = decision["outcome"]
        report["status"] = "complete"
    except GUARDS as error:
        report.update(status="void", outcome="V", void_reason=f"guard: {error}")
        report["decision"] = {"outcome": "V", "reason": str(error)}
    except BaseException as error:
        report.update(
            status="void: crashed before the report was complete",
            outcome="V",
            void_reason=f"exception: {type(error).__name__}: {error}",
            traceback="".join(traceback.format_exception(error)),
        )
        report["decision"] = {"outcome": "V", "reason": f"{type(error).__name__}: {error}"}
        report["total_seconds"] = elapsed()
        write_report(report_path, report)
        raise
    report["total_seconds"] = elapsed()
    write_report(report_path, report)
    print(json.dumps({k: report.get(k) for k in ("status", "outcome", "void_reason")}, indent=2))
    return 0 if report["status"] == "complete" else 1


def write_report(path, report):
    """Non-finite floats become null and are listed (TASK-059's rule)."""
    base = _load("_probe_info_ceiling", "scripts/probe_info_ceiling.py")
    clean, found = base.finite_json(report)
    clean["non_finite_fields"] = found
    write(path, clean)


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
    r.add_argument("--workers", type=int, default=lc.WORKERS)
    r.add_argument("--max-seconds", type=float, default=lc.SUPERVISOR_SECONDS)
    r.add_argument("--worker-seconds", type=float, default=lc.WORKER_SECONDS)
    r.add_argument("--allow-dirty", action="store_true", help="pilot/smoke only")
    r.add_argument("--skip-readability", action="store_true", help="pilot/smoke only")
    f = sub.add_parser("finalize", help="assemble/score existing shards; never simulates")
    f.add_argument("--work", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True, help="new dataset directory")
    f.add_argument("--skip-readability", action="store_true", help="pilot/smoke only")
    w = sub.add_parser("worker")
    w.add_argument("--plan", type=Path, required=True)
    w.add_argument("--work", type=Path, required=True)
    w.add_argument("--index", type=int, required=True)
    w.add_argument("--count", type=int, required=True)
    w.add_argument("--max-seconds", type=float, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        plan = lc.make_plan(lc.PILOT_SEEDS if args.seeds == "pilot" else lc.FROZEN_SEEDS)
        args.out.write_bytes(lc.plan_bytes(plan))
        print(digest(args.out))
        return 0
    if args.command == "worker":
        worker(args.plan, args.work, args.index, args.count, args.max_seconds)
        return 0
    if args.command == "finalize":
        provenance = json.loads((args.work / "provenance.json").read_text())
        if provenance["seed_set"] == "frozen":
            raise SystemExit("finalize is not a recovery path for the frozen run (protocol §12)")
        return finalize(args.work, args.output, readability=not args.skip_readability)
    if args.seeds == "frozen" and (args.limit or args.allow_dirty or args.skip_readability):
        raise SystemExit("--limit, --allow-dirty and --skip-readability are pilot/smoke only")
    preregistered = (lc.WORKERS, lc.SUPERVISOR_SECONDS, lc.WORKER_SECONDS)
    if (
        args.seeds == "frozen"
        and (
            args.workers,
            args.max_seconds,
            args.worker_seconds,
        )
        != preregistered
    ):
        raise SystemExit("the frozen run uses the preregistered workers and caps (protocol §11)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
