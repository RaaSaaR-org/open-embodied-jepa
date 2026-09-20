"""Supervised dense-image-waypoint development evaluation; final cohort is opt-in."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ENTRY_CLOCK = (time.time(), time.monotonic())

from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa.contracts import ContractError  # noqa: E402
from embodied_jepa.training import json_hash  # noqa: E402
from embodied_jepa.waypoint_planning import (  # noqa: E402
    ImageWaypoint,
    WaypointConfig,
    WaypointController,
)

MODES = ("learned", "persistence", "dynamics_shuffle", "hold", "random")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def make_plan(
    *,
    stage="development",
    seeds=None,
    modes=MODES,
    max_seconds=600,
    max_steps=1000,
    stride=10,
    dwell=3,
    horizon=4,
    candidates=16,
    iterations=2,
    proposals=True,
    control_timeout=5.0,
):
    if stage not in ("development", "final"):
        raise ValueError("stage must be development or final")
    cohort = tuple(range(43000, 43005)) if stage == "development" else tuple(range(44000, 44020))
    seeds = list(cohort if seeds is None else seeds)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s not in cohort for s in seeds)
    ):
        raise ValueError("seeds must be unique members of the declared stage cohort")
    if stage == "final" and seeds != list(cohort):
        raise ValueError("final evaluation requires the complete frozen20-reset cohort")
    modes = list(modes)
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError("unknown or repeated control mode")
    if not np.isfinite(max_seconds) or not 0 < max_seconds <= 1800:
        raise ValueError("hard wall budget must lie in (0,1800]seconds")
    if not np.isfinite(control_timeout) or control_timeout <= 0:
        raise ValueError("control timeout must be positive and finite")
    if type(stride) is not int or stride < 1 or type(dwell) is not int or dwell < 1:
        raise ValueError("waypoint stride and dwell must be positive integers")
    lower = (0.0,) * 6 + (-0.5,) * 6 + (-1.0, -1.0)
    upper = (0.0,) * 6 + (0.5,) * 6 + (-1.0, 1.0)
    config = WaypointConfig(
        horizon=horizon,
        candidates=candidates,
        iterations=iterations,
        max_steps=max_steps,
        lower_bounds=lower,
        upper_bounds=upper,
    )
    attempts = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        reset = {
            "seed": seed,
            "object_xy": (np.array([0.34, -0.18]) + rng.uniform(-0.006, 0.006, 2)).tolist(),
            "plate_xy": (np.array([0.49, -0.09]) + rng.uniform(-0.006, 0.006, 2)).tolist(),
        }
        for mode in modes:
            attempts.append(
                {"attempt_id": f"{seed}-{mode}", "seed": seed, "mode": mode, "reset": reset}
            )
    return {
        "format_version": 1,
        "stage": stage,
        "seeds": seeds,
        "modes": modes,
        "max_seconds": max_seconds,
        "finalization_reserve_seconds": min(5.0, max_seconds / 10),
        "control_timeout_seconds": float(control_timeout),
        "controller": asdict(config),
        "waypoint_stride": stride,
        "waypoint_dwell": dwell,
        "demonstration_proposals": bool(proposals),
        "demo_selection": "lexicographically first successful NOMINAL apple/plate TRAIN episode",
        "threshold_rule": (
            "max(1.1*max(distance(frame[t+-2],frame[t])),"
            "median positive adjacent TRAIN-demo frame distance,1e-12,"
            "1.1*q90(successful NOMINAL TRAIN same-frame distances))"
        ),
        "proposal_rule": (
            "horizon-length applied-action windows entirely within prior stride frames; "
            "evenly spaced starts if budget limited"
        ),
        "reset_jitter_m": 0.006,
        "attempts": attempts,
        "success_rule": (
            "unchanged ordered AppleToPlateTask; scorer never drives waypoint advancement"
        ),
    }


def checkpoint_model(dataset, checkpoint, *, device="cpu"):
    import torch

    from embodied_jepa.config import MODELS
    from embodied_jepa.data import DatasetStore

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    store = DatasetStore(dataset)
    envelope = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {
        "dataset_hash": store.manifest_hash,
        "split_hash": json_hash(
            {"splits": store.manifest.get("splits"), "policy": store.manifest.get("split_policy")}
        ),
        "action_hash": json_hash(store.manifest["action_manifest"]),
    }
    for key, value in expected.items():
        if envelope.get("metadata", {}).get(key) != value:
            raise ContractError(f"checkpoint {key} does not match sealed dataset")
    if json_hash(envelope.get("state_schema")) != json_hash(asdict(store.state_schema)):
        raise ContractError("checkpoint robot-state schema mismatch")
    training = set(store.manifest.get("splits", {}).get("train", []))
    normalized = set(envelope.get("metadata", {}).get("normalization", {}).get("episode_ids", []))
    if not training or normalized != training:
        raise ContractError("checkpoint normalization must cover exactly the frozen training split")
    model = MODELS.create(
        envelope["backend"],
        state_schema=store.state_schema,
        device=device,
        seed=envelope["seed"],
        config=envelope["config"],
        metadata=expected,
    )
    model.load(checkpoint)
    if not callable(getattr(model, "observed_distance", None)):
        raise ContractError("waypoint model requires an observed-image distance metric")
    return store, model


def select_demonstration(store):
    for episode_id in sorted(store.manifest.get("splits", {}).get("train", [])):
        episode = store.read_episode(episode_id)
        if (episode.object_id, episode.container_id) != ("apple", "plate"):
            continue
        if (
            episode.metadata.get("collection_success") is True
            and episode.terminated
            and episode.metadata.get("variant") == "nominal"
        ):
            return episode
    raise ContractError("no successful apple/plate training demonstration")


def nominal_calibration_episodes(store):
    result = []
    for name in sorted(store.manifest.get("splits", {}).get("train", [])):
        episode = store.read_episode(name)
        if (
            (episode.object_id, episode.container_id) == ("apple", "plate")
            and episode.metadata.get("variant") == "nominal"
            and episode.metadata.get("collection_success") is True
            and episode.terminated
        ):
            result.append(episode)
    return result


def calibrate_waypoints(
    model,
    episode,
    *,
    stride=10,
    dwell=3,
    horizon=4,
    candidates=16,
    proposals=True,
    calibration_episodes=None,
    training_episode_ids=None,
):
    references = list(calibration_episodes) if calibration_episodes is not None else [episode]
    allowed = (
        set(training_episode_ids) if training_episode_ids is not None else {episode.episode_id}
    )
    if episode.episode_id not in allowed or any(e.episode_id not in allowed for e in references):
        raise ContractError("waypoint calibration received non-training episode")
    frames = episode.observations["onboard_rgb"]
    indices = list(range(0, len(frames), stride))
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)

    def distance(i, j):
        cost = np.asarray(
            model.observed_distance(
                {"onboard_rgb": frames[i : i + 1]}, {"onboard_rgb": frames[j : j + 1]}
            )
        )
        if cost.shape != (1,) or not np.isfinite(cost).all() or cost[0] < 0:
            raise ContractError("invalid training image calibration distance")
        return float(cost[0])

    adjacent = [distance(i - 1, i) for i in range(1, len(frames))]
    positive = [v for v in adjacent if v > 0]
    floor = max(float(np.median(positive)) if positive else 0.0, 1e-12)
    waypoints, records = [], []
    for waypoint_number, index in enumerate(indices):
        neighborhood = list(range(max(0, index - 2), min(len(frames), index + 3)))
        distances = [distance(i, index) for i in neighborhood]
        cross = []
        for reference in references:
            source = reference.observations["onboard_rgb"]
            if index >= len(source):
                continue  # No end clamping: only observed corresponding frames.
            value = np.asarray(
                model.observed_distance(
                    {"onboard_rgb": source[index : index + 1]},
                    {"onboard_rgb": frames[index : index + 1]},
                )
            )
            if value.shape != (1,) or not np.isfinite(value).all() or value[0] < 0:
                raise ContractError("invalid cross-training image distance")
            cross.append(
                {"episode_id": reference.episode_id, "frame": index, "distance": float(value[0])}
            )
        if not cross:
            raise ContractError("waypoint has no training frame coverage")
        cross_quantile = float(np.quantile([v["distance"] for v in cross], 0.9))
        threshold = max(1.1 * max(distances), floor, 1.1 * cross_quantile)
        starts = list(range(max(0, index - stride), index - horizon + 1)) if proposals else []
        if len(starts) > candidates - 2:
            starts = [starts[i] for i in np.linspace(0, len(starts) - 1, candidates - 2, dtype=int)]
        action_proposals = (
            np.stack([episode.actions[i : i + horizon] for i in starts]) if starts else None
        )
        waypoint = ImageWaypoint(
            {"onboard_rgb": frames[index : index + 1]},
            threshold,
            dwell,
            f"{episode.episode_id}/frame-{index}",
            action_proposals=action_proposals,
        )
        waypoints.append(waypoint)
        neighbors = [i for i in (waypoint_number - 1, waypoint_number + 1) if 0 <= i < len(indices)]
        separability = [
            {"frame": indices[i], "distance": distance(indices[i], index)} for i in neighbors
        ]
        for entry in separability:
            entry["inside_threshold"] = entry["distance"] <= threshold
        records.append(
            {
                "frame": index,
                "adjacent_waypoint_separability": separability,
                "overlapping_adjacent_waypoint_count": sum(
                    e["inside_threshold"] for e in separability
                ),
                "threshold": threshold,
                "neighborhood_frames": neighborhood,
                "neighborhood_distances": distances,
                "cross_training_distances": cross,
                "cross_training_q90": cross_quantile,
                "cross_training_reference_count": len(cross),
                "proposal_start_frames": starts,
                "image_sha256": waypoint.image_hash,
                "source_id": waypoint.source_id,
            }
        )
    return waypoints, {
        "episode_id": episode.episode_id,
        "adjacent_distance_floor": floor,
        "adjacent_distances": adjacent,
        "calibration_training_episode_ids": [e.episode_id for e in references],
        "waypoints_with_adjacent_overlap": sum(
            r["overlapping_adjacent_waypoint_count"] > 0 for r in records
        ),
        "waypoints": records,
    }


def verify_plan_inputs(plan, *, source_root=None):
    source_root = ROOT if source_root is None else Path(source_root)
    if digest(plan["checkpoint"]) != plan["checkpoint_sha256"]:
        raise ContractError("frozen checkpoint changed")
    if digest(Path(plan["dataset"]) / "meta/jepa_manifest.json") != plan["dataset_manifest_sha256"]:
        raise ContractError("frozen dataset manifest changed")
    manifest = json.loads((Path(plan["dataset"]) / "meta/jepa_manifest.json").read_text())
    for relative, expected in manifest.get("sha256", {}).items():
        if digest(Path(plan["dataset"]) / relative) != expected:
            raise ContractError("frozen dataset contents changed")
    for relative, expected in plan["source_hashes"].items():
        if digest(source_root / relative) != expected:
            raise ContractError("frozen evaluation source changed")
    for relative, key in (
        ("configs/g1_sim_action.json", "action_manifest_sha256"),
        ("assets/manifest.json", "asset_manifest_sha256"),
    ):
        if digest(source_root / relative) != plan[key]:
            raise ContractError("frozen action or asset manifest changed")


def prepare_worker(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    verify_plan_inputs(plan)
    store, model = checkpoint_model(plan["dataset"], plan["checkpoint"])
    demo = select_demonstration(store)
    waypoints, calibration = calibrate_waypoints(
        model,
        demo,
        stride=plan["waypoint_stride"],
        dwell=plan["waypoint_dwell"],
        horizon=plan["controller"]["horizon"],
        candidates=plan["controller"]["candidates"],
        proposals=plan["demonstration_proposals"],
        calibration_episodes=nominal_calibration_episodes(store),
        training_episode_ids=store.manifest["splits"]["train"],
    )
    payload = {"images": np.concatenate([w.images["onboard_rgb"] for w in waypoints])}
    for index, waypoint in enumerate(waypoints):
        if waypoint.action_proposals is not None:
            payload[f"proposals_{index}"] = waypoint.action_proposals
    np.savez_compressed(output / "waypoints.npz", **payload)
    write(output / "calibration.json", calibration)
    resolved = plan | {
        "waypoints_sha256": digest(output / "waypoints.npz"),
        "calibration_sha256": digest(output / "calibration.json"),
        "demonstration_episode_id": demo.episode_id,
    }
    write(output / "resolved_plan.json", resolved)


def load_waypoints(output, plan):
    for name, field in (
        ("waypoints.npz", "waypoints_sha256"),
        ("calibration.json", "calibration_sha256"),
    ):
        if digest(output / name) != plan[field]:
            raise ContractError("frozen waypoint artifact changed")
    calibration = json.loads((output / "calibration.json").read_text())
    with np.load(output / "waypoints.npz", allow_pickle=False) as arrays:
        return [
            ImageWaypoint(
                {"onboard_rgb": arrays["images"][i : i + 1]},
                row["threshold"],
                plan["waypoint_dwell"],
                row["source_id"],
                action_proposals=arrays[f"proposals_{i}"] if f"proposals_{i}" in arrays else None,
            )
            for i, row in enumerate(calibration["waypoints"])
        ]


def attempt_worker(output, attempt_id):
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import MuJoCoSimulation
    from embodied_jepa.task import AppleToPlateTask

    output = Path(output)
    plan = json.loads((output / "resolved_plan.json").read_text())
    trial = next(t for t in plan["attempts"] if t["attempt_id"] == attempt_id)
    folder = output / "attempts" / attempt_id
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "started.json", trial | {"start_unix_time": time.time()})
    robot = None
    score, failure, reason, steps = {"success": False}, "", "step_limit", 0
    current = {}
    stage = "load"
    start = time.time()

    def record(row):
        with (folder / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()

    try:
        verify_plan_inputs(plan)
        store, model = checkpoint_model(plan["dataset"], plan["checkpoint"])
        waypoints = load_waypoints(output, plan)
        config = WaypointConfig(
            **(
                plan["controller"]
                | {
                    "seed": trial["seed"],
                    "ablation": trial["mode"] if trial["mode"] in MODES[:3] else "persistence",
                }
            )
        )
        controller = WaypointController(
            model, waypoints, progress_distance=model.observed_distance, config=config
        )
        stage = "construct"
        robot = G1Embodiment(
            MuJoCoSimulation(object_kind="apple", container_kind="plate", width=96, height=96)
        )
        if robot.state_schema != store.state_schema or json_hash(robot.manifest) != json_hash(
            store.manifest["action_manifest"]
        ):
            raise ContractError(
                "evaluation embodiment disagrees with dataset state/action contracts"
            )
        stage = "reset"
        robot.reset(**trial["reset"])
        scorer = AppleToPlateTask(robot)
        rng = np.random.default_rng(trial["seed"])
        last_grasps = np.array(config.initial_grasps, np.float32)
        for step in range(config.max_steps):
            control_start_wall, control_start_mono = time.time(), time.monotonic()
            stage = "observe"
            observation = robot.observe()
            stage = "plan"
            if trial["mode"] in MODES[:3]:
                decision = controller.step(observation, robot.project_candidates)
                if decision.action is None:
                    reason = decision.termination_reason
                    break
                command, current = decision.action, decision.trace.copy()
            else:
                requested = np.zeros(14, np.float32)
                requested[12:] = last_grasps
                if trial["mode"] == "random":
                    requested = rng.uniform(config.lower_bounds, config.upper_bounds).astype(
                        np.float32
                    )
                projected = robot.project_candidates(requested[None, None, None])
                if not projected.feasible[0, 0]:
                    raise ContractError("baseline command has no feasible projection")
                command = projected.actions[0, 0, 0]
                current = {
                    "step": step,
                    "sampled_action": requested.tolist(),
                    "projected_action": command.tolist(),
                    "observation_timestamp": float(observation.timestamps[0]),
                    "goal_index": None,
                }
            control_elapsed = max(
                time.time() - control_start_wall, time.monotonic() - control_start_mono
            )
            current["control_seconds"] = control_elapsed
            if control_elapsed > plan["control_timeout_seconds"]:
                reason = "deadline_miss"
                record(current | {"event": "deadline_miss", "executed": False, "reason": reason})
                break
            record(current | {"event": "command_pending", "executed": None})
            stage = "execute"
            result = robot.execute(command)
            if trial["mode"] in MODES[:3]:
                current = controller.acknowledge(result)
            else:
                current.update(
                    status=result.status,
                    reason=result.reason,
                    applied_action=None
                    if result.applied_action is None
                    else result.applied_action.tolist(),
                    execution_timestamp=result.timestamp,
                )
            current["control_seconds"] = control_elapsed
            accepted = result.applied_action is not None
            steps += int(accepted)
            if accepted:
                last_grasps = result.applied_action[12:].copy()
            stage = "score"
            score = scorer.evaluate()
            record(current | {"event": "result", "executed": accepted, "score": score})
            if not accepted:
                reason = "execution_rejected"
                break
            if score["success"]:
                reason = "success"
                break
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        reason = "runtime_error"
        record(
            {
                "event": "error",
                "stage": stage,
                "error": failure,
                "execution_uncertain": stage == "execute",
                "partial_trace": current,
            }
        )
    finally:
        if robot is not None:
            robot.stop(reason)
            robot.close()
    report = trial | {
        "status": "completed",
        "termination_reason": reason,
        "error": failure,
        "executed_steps": steps,
        "score": score,
        "success": bool(score["success"]),
        "wall_seconds": time.time() - start,
    }
    write(folder / "report.json", report)


def supervise(command, *, cwd, log, remaining, popen=subprocess.Popen, sleep=time.sleep):
    """Hard wall supervision includes suspension; kill the isolated child process group."""
    start_wall, start_mono = time.time(), time.monotonic()
    with Path(log).open("w") as stream:
        process = popen(
            command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        while process.poll() is None:
            elapsed = max(time.time() - start_wall, time.monotonic() - start_mono)
            if elapsed >= remaining:
                try:
                    os.killpg(process.pid, 9)
                except ProcessLookupError:
                    pass
                process.wait()
                return {"status": "hard_wall_timeout", "returncode": process.returncode}
            sleep(min(0.1, max(0.001, remaining - elapsed)))
        return {"status": "exited", "returncode": process.returncode}


def summarize_attempt(output, trial, fallback):
    folder = Path(output) / "attempts" / trial["attempt_id"]
    if (folder / "report.json").exists():
        return json.loads((folder / "report.json").read_text())
    results, pending = [], False
    if (folder / "trace.jsonl").exists():
        for line in (folder / "trace.jsonl").read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A killed writer can leave an incomplete final JSON line.
            if row.get("event") == "command_pending":
                pending = True
            elif row.get("event") == "result":
                results.append(row)
                pending = False
    return trial | {
        "status": fallback,
        "success": False,
        "executed_steps": sum(r.get("executed") is True for r in results),
        "execution_uncertain": pending,
        "score": results[-1].get("score", {}) if results else {},
        "started": (folder / "started.json").exists(),
    }


def clean_completion(records):
    return bool(records) and all(
        r.get("status") == "completed"
        and r.get("termination_reason") not in ("runtime_error", "deadline_miss")
        and r.get("provenance_valid", True)
        for r in records
    )


def verify_generated(output, resolved, expected_digest):
    if digest(output / "resolved_plan.json") != expected_digest:
        raise ContractError("resolved evaluation plan changed")
    for name, key in (
        ("waypoints.npz", "waypoints_sha256"),
        ("calibration.json", "calibration_sha256"),
    ):
        if digest(output / name) != resolved[key]:
            raise ContractError("frozen waypoint calibration changed")


def run(args, *, start_clock=None):
    start_wall, start_mono = start_clock or (time.time(), time.monotonic())
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("evaluation output must be new")
    plan = make_plan(
        stage=args.stage,
        seeds=args.seeds,
        modes=args.modes,
        max_seconds=args.max_seconds,
        max_steps=args.max_steps,
        stride=args.stride,
        dwell=args.dwell,
        horizon=args.horizon,
        candidates=args.candidates,
        iterations=args.iterations,
        proposals=not args.no_proposals,
        control_timeout=args.control_timeout,
    )
    if args.stage == "final" and args.selection is None:
        raise ValueError("final cohort requires an explicit frozen development-selection JSON")
    dataset, checkpoint = Path(args.dataset).resolve(), Path(args.checkpoint).resolve()
    plan.update(
        dataset=str(dataset),
        checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint),
        dataset_manifest_sha256=digest(dataset / "meta/jepa_manifest.json"),
    )
    if args.selection is not None:
        selection = json.loads(Path(args.selection).read_text())
        for key in ("checkpoint_sha256", "dataset_manifest_sha256"):
            if selection.get(key) != plan[key]:
                raise ValueError("development selection does not match frozen input hashes")
        for key in (
            "waypoint_stride",
            "waypoint_dwell",
            "demonstration_proposals",
            "control_timeout_seconds",
        ):
            if selection.get(key) != plan[key]:
                raise ValueError("development selection waypoint protocol mismatch")
        if selection.get("controller") != json.loads(json.dumps(plan["controller"])):
            raise ValueError("development selection does not match controller configuration")
        plan["selection_sha256"] = digest(args.selection)
    output.mkdir(parents=True)
    snapshot = output / "source"
    snapshot.mkdir()
    shutil.copytree(
        ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    (snapshot / "scripts").mkdir()
    shutil.copyfile(Path(__file__), snapshot / "scripts/evaluate_apple.py")
    for directory, filename in (("configs", "g1_sim_action.json"), ("assets", "manifest.json")):
        (snapshot / directory).mkdir()
        shutil.copyfile(ROOT / directory / filename, snapshot / directory / filename)
    (snapshot / "third_party").symlink_to(ROOT / "third_party", target_is_directory=True)
    plan["source_hashes"] = {
        str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.py"))
    }
    plan["action_manifest_sha256"] = digest(snapshot / "configs/g1_sim_action.json")
    plan["asset_manifest_sha256"] = digest(snapshot / "assets/manifest.json")
    plan["source_revision"] = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    write(output / "plan.json", plan)

    def remaining():
        return (
            plan["max_seconds"]
            - plan["finalization_reserve_seconds"]
            - max(time.time() - start_wall, time.monotonic() - start_mono)
        )

    worker = [
        sys.executable,
        str(snapshot / "scripts/evaluate_apple.py"),
        "--worker-output",
        str(output),
    ]
    preparation = (
        {"status": "not_started_budget", "returncode": None}
        if remaining() <= 0
        else supervise(
            worker + ["--worker", "prepare"],
            cwd=snapshot,
            log=output / "prepare.log",
            remaining=remaining(),
        )
    )
    records = []
    ready = preparation["returncode"] == 0 and (output / "resolved_plan.json").exists()
    resolved = json.loads((output / "resolved_plan.json").read_text()) if ready else None
    resolved_digest = digest(output / "resolved_plan.json") if ready else None
    invalid = ""
    for trial in plan["attempts"]:
        outcome = None
        fallback = "preparation_failed" if not ready else "not_started_budget"
        if invalid:
            fallback = "not_started_invalid_inputs"
        elif ready and remaining() > 0:
            try:
                verify_plan_inputs(plan, source_root=snapshot)
                verify_generated(output, resolved, resolved_digest)
                if remaining() <= 0:
                    fallback = "not_started_budget"
                else:
                    outcome = supervise(
                        worker + ["--worker", "attempt", "--attempt-id", trial["attempt_id"]],
                        cwd=snapshot,
                        log=output / f"{trial['attempt_id']}.log",
                        remaining=remaining(),
                    )
                    fallback = outcome["status"] if outcome["returncode"] else "missing_report"
                    verify_plan_inputs(plan, source_root=snapshot)
                    verify_generated(output, resolved, resolved_digest)
            except (ContractError, OSError, ValueError) as error:
                invalid = f"{type(error).__name__}: {error}"
                fallback = "invalid_inputs"
        record = summarize_attempt(output, trial, fallback)
        record["supervisor"] = outcome
        if outcome is not None and (outcome["returncode"] != 0 or outcome["status"] != "exited"):
            record.update(
                worker_report_status=record["status"],
                reported_success_before_failure=record["success"],
                success=False,
                status="hard_wall_timeout"
                if outcome["status"] == "hard_wall_timeout"
                else "child_failed",
            )
        if invalid:
            record.update(status=fallback, provenance_valid=False, provenance_error=invalid)
            for prior in records:
                prior["provenance_valid"] = False
        records.append(record)
        write(
            output / "report.json",
            {
                "plan_sha256": digest(output / "plan.json"),
                "resolved_plan_sha256": resolved_digest,
                "preparation": preparation,
                "planned": len(plan["attempts"]),
                "records": records,
                "status": "completed"
                if len(records) == len(plan["attempts"]) and clean_completion(records)
                else "incomplete",
                "provenance_valid": not bool(invalid),
                "provenance_error": invalid,
                "successes": 0 if invalid else sum(r["success"] for r in records),
                "denominator": len(plan["attempts"]),
                "wall_seconds": max(time.time() - start_wall, time.monotonic() - start_mono),
            },
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--stage", choices=("development", "final"), default="development")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--control-timeout", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--dwell", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--no-proposals", action="store_true")
    parser.add_argument("--worker", choices=("prepare", "attempt"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--attempt-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        prepare_worker(args.worker_output) if args.worker == "prepare" else attempt_worker(
            args.worker_output, args.attempt_id
        )
        return
    if not all((args.dataset, args.checkpoint, args.output)):
        parser.error("dataset, checkpoint and output are required")
    records = run(args, start_clock=ENTRY_CLOCK)
    return 0 if clean_completion(records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
