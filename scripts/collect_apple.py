"""Task-specific Apple→Plate development collection, explicitly privileged oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa.data import DatasetStore, Episode  # noqa: E402
from embodied_jepa.embodiment import G1Embodiment  # noqa: E402
from embodied_jepa.manipulation_collection import _json  # noqa: E402
from embodied_jepa.scripted import EarlyReleaseOracleManipulationPolicy  # noqa: E402
from embodied_jepa.simulation import MuJoCoSimulation  # noqa: E402
from embodied_jepa.task import AppleToPlateTask  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_json(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_plan(transfer_x_shift):
    if transfer_x_shift not in (0, -0.02, -0.035):
        raise ValueError("transfer shift must be selected by the declared mechanics probe")
    attempts = []
    for index, seed in enumerate(range(42000, 42032)):
        rng = np.random.default_rng(seed)
        attempts.append(
            {
                "seed": seed,
                "variant": ("nominal", "nominal", "position_burst", "closure_burst")[index % 4],
                "object_xy": (np.array([0.34, -0.18]) + rng.uniform(-0.006, 0.006, 2)).tolist(),
                "plate_xy": (np.array([0.49, -0.09]) + rng.uniform(-0.006, 0.006, 2)).tolist(),
                "status": "not_started",
            }
        )
    return {
        "attempts": attempts,
        "oracle_only": True,
        "learned_control": False,
        "task_specific_development": True,
        "heldout_combinations": [],
        "split_seed": 42,
        "max_true_wall_seconds": 600,
        "execution_cutoff_seconds": 540,
        "minimum_transitions": 8,
        "max_commands_per_attempt": 745,
        "palm_x_offset": -0.015,
        "transfer_x_shift": transfer_x_shift,
        "opening_ramp": 0.08,
        "perturbation_block_commands": 4,
        "position_uniform_xyz": 0.04,
        "position_uniform_rotation": 0.02,
        "closure_uniform_grasp": 0.08,
        "successful_train_waypoint_stride": 10,
    }


def perturb_action(base, step, seed, variant):
    """Four nominal commands then four constant-perturbation commands, repeat."""
    result = base.copy()
    if variant not in ("nominal", "position_burst", "closure_burst"):
        raise ValueError("unknown perturbation variant")
    if variant == "nominal" or (step // 4) % 2 == 0:
        return result
    rng = np.random.default_rng(np.random.SeedSequence([seed, step // 8, 731]))
    if variant == "position_burst":
        result[6:9] += rng.uniform(-0.04, 0.04, 3).astype(np.float32)
        result[9:12] += rng.uniform(-0.02, 0.02, 3).astype(np.float32)
    else:
        result[13] += np.float32(rng.uniform(-0.08, 0.08))
    return np.clip(result, -1, 1).astype(np.float32)


def make_policy(truth, transfer_x_shift):
    policy = EarlyReleaseOracleManipulationPolicy(truth, opening_ramp=0.08)
    phases = []
    for index, phase in enumerate(policy.phases):
        target = phase.target_base.copy()
        target[0] += 0.015 + (transfer_x_shift if index >= 4 else 0)
        phases.append(replace(phase, target_base=target))
    policy.phases = tuple(phases)
    return policy


def verify_snapshot(root, expected_hash):
    path = root / "snapshot_manifest.json"
    if digest(path) != expected_hash:
        raise ValueError("snapshot manifest changed")
    for relative, expected in json.loads(path.read_text())["files"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"frozen snapshot changed: {relative}")


def worker(output, mechanics_report, expected_hash, snapshot_hash=None):
    from PIL import Image

    if snapshot_hash is not None:
        verify_snapshot(ROOT, snapshot_hash)
    if digest(mechanics_report) != expected_hash:
        raise ValueError("mechanics selection report hash changed")
    selection = json.loads(mechanics_report.read_text())["selection"]["selected"]
    if selection is None or selection["successes"] < 1:
        raise ValueError("no mechanically successful collection policy selected")
    declared = make_plan(selection["transfer_x_shift"])
    output.mkdir(parents=True, exist_ok=False)
    wall, monotonic, cpu = time.time(), time.monotonic(), time.process_time()

    def timing():
        return {
            "true_wall_seconds": max(time.time() - wall, time.monotonic() - monotonic),
            "process_cpu_seconds": time.process_time() - cpu,
        }

    hashes = {str(path.relative_to(ROOT)): digest(path) for path in (ROOT / "src").rglob("*.py")}
    provenance = {
        "producer": "privileged task-specific Apple→Plate collector",
        "license": "project-generated data; upstream robot assets retain BSD-3-Clause",
        "task_specific_development": True,
        "historical_mvp_corpus_unchanged": True,
        "mechanics_report_sha256": expected_hash,
        "runtime_source_hashes": hashes,
        "collector_sha256": digest(Path(__file__)),
        "plan": declared,
        "asset_manifest_sha256": digest(ROOT / "assets/manifest.json"),
        "action_manifest_sha256": digest(ROOT / "configs/g1_sim_action.json"),
        "protocol_sha256": digest(ROOT / "docs/experiments/apple_collection_v1.md"),
        "snapshot_manifest_sha256": snapshot_hash,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    bootstrap = G1Embodiment(
        MuJoCoSimulation(object_kind="apple", container_kind="plate", width=96, height=96)
    )
    try:
        store = DatasetStore.create(
            output,
            fps=20,
            state_schema=bootstrap.state_schema,
            action_manifest=bootstrap.manifest,
            provenance=provenance,
        )
    finally:
        bootstrap.close()
    write(output / "collection_plan.json", declared | {"provenance": provenance})
    shutil.copyfile(mechanics_report, output / "mechanics_selection.json")
    records = [dict(trial) for trial in declared["attempts"]]
    write(output / "collection_report.json", {"attempts": records, "timing": timing()})
    for record in records:
        trial = dict(record)
        if timing()["true_wall_seconds"] >= 540:
            record["status"] = "not_started_budget"
            continue
        robot = None
        frames, states, masks, times, actions, requests, proposals, raw, phases, scores = (
            [] for _ in range(10)
        )
        try:
            record["status"] = "attempted"
            write(output / "collection_report.json", {"attempts": records, "timing": timing()})
            sim = MuJoCoSimulation(object_kind="apple", container_kind="plate", width=96, height=96)
            robot = G1Embodiment(sim)
            truth = _json(
                robot.reset(trial["seed"], object_xy=trial["object_xy"], plate_xy=trial["plate_xy"])
            )
            policy, scorer = (
                make_policy(truth, declared["transfer_x_shift"]),
                AppleToPlateTask(robot),
            )
            record["status"] = "running"

            def capture(robot=robot, frames=frames, states=states, masks=masks, times=times):
                obs = robot.observe()
                frames.append(obs.images["onboard_rgb"][0].copy())
                states.append(obs.robot_state[0].copy())
                masks.append(obs.state_mask[0].copy())
                times.append(float(obs.timestamps[0]))

            capture()
            while not policy.done:
                if timing()["true_wall_seconds"] >= 540:
                    record["status"] = "budget_truncated"
                    break
                phase = policy.phase
                base = policy.action(robot)
                requested = perturb_action(base, policy.step_count, trial["seed"], trial["variant"])
                result = robot.execute(requested)
                policy.advance(result)
                score = scorer.evaluate()
                record["final_score"] = score
                if result.applied_action is None:
                    record.update(
                        status="command_rejected",
                        failure=result.reason,
                        rejected_request=requested.tolist(),
                        rejected_phase=phase,
                    )
                    break
                proposals.append(base.copy())
                requests.append(requested.copy())
                actions.append(result.applied_action.copy())
                raw.append(robot.denormalize_action(result.applied_action))
                phases.append(phase)
                scores.append(score)
                capture()
                if score["success"]:
                    record["status"] = "success"
                    break
            if record["status"] == "running":
                record["status"] = "policy_complete_without_success"
            if len(actions) >= 8:
                episode_id = f"apple-{trial['seed']}"
                episode = Episode(
                    episode_id=episode_id,
                    session_id=f"apple-reset-{trial['seed']}",
                    task="apple_to_plate",
                    object_id="apple",
                    container_id="plate",
                    observations={"onboard_rgb": np.array(frames)},
                    robot_states=np.array(states),
                    state_mask=np.array(masks),
                    actions=np.array(actions),
                    timestamps=np.array(times),
                    state_schema=robot.state_schema,
                    terminated=record["status"] == "success",
                    truncated=record["status"] != "success",
                    raw_actions=np.array(raw),
                    metadata={
                        "oracle_collection_only": True,
                        "variant": trial["variant"],
                        "reset_truth": truth,
                        "phase_labels": phases,
                        "policy_labels": [
                            (
                                "nominal"
                                if trial["variant"] == "nominal"
                                else "perturbation"
                                if (step // 4) % 2
                                else "nominal_recovery"
                            )
                            for step in range(len(phases))
                        ],
                        "perturbation_seed": trial["seed"],
                        "perturbation_bounds": {
                            "xyz": 0.04,
                            "rotation": 0.02,
                            "grasp": 0.08,
                            "block_commands": 4,
                            "variant": trial["variant"],
                        },
                        "stage_scores": scores,
                        "base_policy_actions": np.array(proposals).tolist(),
                        "requested_actions": np.array(requests).tolist(),
                        "model_scored_actions": None,
                        "model_scored_missing_reason": "oracle does not score a world model",
                        "collection_success": record["status"] == "success",
                        "final_score": record.get("final_score"),
                        "termination": record["status"],
                        "failure": record.get("failure"),
                        "trial": trial,
                    },
                )
                # Oracle truth/score metadata is never part of canonical SequenceBatch inputs.
                store.write_episode(episode)
                record["episode_id"] = episode_id
        except Exception as error:
            record.update(status="runtime_error", failure=f"{type(error).__name__}: {error}")
        finally:
            if robot is not None:
                robot.stop(record["status"])
                robot.close()
        record["executed_steps"] = len(actions)
        write(output / "collection_report.json", {"attempts": records, "timing": timing()})
        print(
            json.dumps(
                {
                    k: record.get(k)
                    for k in ("seed", "variant", "status", "executed_steps", "failure")
                }
            ),
            flush=True,
        )
    if store is not None and len(store.episode_ids) >= 3:
        store.freeze_splits(seed=42, heldout_combinations=())
        store.fit_normalization()
        goals = []
        for episode_id in store.manifest["splits"]["train"]:
            episode = store.read_episode(episode_id)
            if not episode.metadata["collection_success"]:
                continue
            directory = output / "train_waypoints" / episode_id
            directory.mkdir(parents=True)
            entries = []
            for index in sorted(set(range(0, episode.transitions + 1, 10)) | {episode.transitions}):
                path = directory / f"frame-{index:04d}.png"
                Image.fromarray(episode.observations["onboard_rgb"][index]).save(path)
                entries.append(
                    {
                        "frame_index": index,
                        "timestamp": float(episode.timestamps[index]),
                        "image": str(path.relative_to(output)),
                        "sha256": digest(path),
                        "phase": episode.metadata["phase_labels"][max(0, index - 1)],
                    }
                )
            goals.append({"episode_id": episode_id, "split": "train", "images": entries})
        write(
            output / "train_waypoints.json",
            {
                "source_manifest_sha256": store.manifest_hash,
                "image_only_goals": True,
                "successful_train_episodes": goals,
            },
        )
        store.verify()
    if snapshot_hash is not None:
        verify_snapshot(ROOT, snapshot_hash)
    complete = all(
        row["status"] in ("success", "command_rejected", "policy_complete_without_success")
        for row in records
    )
    successful = {
        row["episode_id"]
        for row in store.manifest["episodes"]
        if row["metadata"]["collection_success"]
    }
    success_by_split = {
        name: len(set(ids) & successful) for name, ids in (store.manifest["splits"] or {}).items()
    }
    ready = bool(
        store.manifest["splits"]
        and (output / "train_waypoints.json").is_file()
        and success_by_split.get("train", 0)
        and success_by_split.get("val", 0)
    )
    report = {
        "status": "completed" if complete and ready else "incomplete",
        "training_ready": complete and ready,
        "successful_episodes_by_split": success_by_split,
        "stored_episodes": len(store.episode_ids),
        "stored_transitions": sum(row["length"] - 1 for row in store.manifest["episodes"]),
        "phase_counts": dict(
            Counter(
                phase
                for row in store.manifest["episodes"]
                for phase in row["metadata"]["phase_labels"]
            )
        ),
        "attempts": records,
        "timing": timing(),
        "dataset_sha256": store.manifest_hash if store is not None else None,
        "split_counts": {k: len(v) for k, v in store.manifest["splits"].items()}
        if store is not None and store.manifest["splits"]
        else None,
    }
    write(
        output / "collection_report.json",
        report,
    )
    return report


def finalize_supervision(output, declared, *, returncode, timed_out, integrity_error=None):
    path = output / "collection_report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    existing = {row["seed"]: row for row in report.get("attempts", [])}
    records = [existing.get(row["seed"], dict(row)) for row in declared["attempts"]]
    for row in records:
        if row["status"] in ("running", "attempted"):
            row.update(status="interrupted_unknown_execution", executed_steps=None)
        elif row["status"] == "not_started":
            row["status"] = (
                "not_started_supervisor_budget" if timed_out else "not_started_worker_failure"
            )
    ready = bool(
        not returncode
        and not timed_out
        and not integrity_error
        and report.get("status") == "completed"
        and report.get("training_ready")
    )
    report.update(
        attempts=records,
        status="completed" if ready else "incomplete",
        training_ready=ready,
        supervisor_timeout=timed_out,
        returncode=returncode,
        integrity_error=integrity_error,
    )
    write(path, report)
    return report


def main():
    wall, mono = time.time(), time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mechanics-report", type=Path, required=True)
    parser.add_argument("--mechanics-sha256", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.snapshot_sha256:
            raise ValueError("worker requires a frozen snapshot manifest")
        report = worker(
            args.output.resolve(),
            args.mechanics_report.resolve(),
            args.mechanics_sha256,
            args.snapshot_sha256,
        )
        return 0 if report["status"] == "completed" else 2
    if args.output.exists():
        raise FileExistsError("refusing to overwrite collection")
    if digest(args.mechanics_report) != args.mechanics_sha256:
        raise ValueError("mechanics report hash changed")
    selection = json.loads(args.mechanics_report.read_text())["selection"]["selected"]
    if selection is None or selection["successes"] < 1:
        raise ValueError("no successful mechanics selection")
    declared = make_plan(selection["transfer_x_shift"])
    snapshot = args.output.resolve().with_name(args.output.name + "-runtime")
    if snapshot.exists():
        raise FileExistsError("refusing to overwrite source snapshot")
    snapshot.mkdir(parents=True)
    shutil.copytree(ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for relative in (
        "scripts/collect_apple.py",
        "configs/g1_sim_action.json",
        "assets/manifest.json",
        "docs/experiments/apple_collection_v1.md",
    ):
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    shutil.copyfile(args.mechanics_report, snapshot / "mechanics_selection.json")
    write(snapshot / "frozen_plan.json", declared)
    files = {
        str(path.relative_to(snapshot)): digest(path)
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    write(snapshot / "snapshot_manifest.json", {"files": files})
    snapshot_hash = digest(snapshot / "snapshot_manifest.json")
    verify_snapshot(snapshot, snapshot_hash)
    (snapshot / "third_party").symlink_to(
        (ROOT / "third_party").resolve(), target_is_directory=True
    )
    command = [
        sys.executable,
        str(snapshot / "scripts/collect_apple.py"),
        "--worker",
        "--output",
        str(args.output.resolve()),
        "--mechanics-report",
        str(snapshot / "mechanics_selection.json"),
        "--mechanics-sha256",
        args.mechanics_sha256,
        "--snapshot-sha256",
        snapshot_hash,
    ]
    env = os.environ | {
        "PYTHONPATH": str(snapshot / "src"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    if max(time.time() - wall, time.monotonic() - mono) >= 600:
        finalize_supervision(args.output, declared, returncode=2, timed_out=True)
        return 2
    process = subprocess.Popen(command, cwd=ROOT, env=env)
    timed_out = False
    try:
        while process.poll() is None:
            if max(time.time() - wall, time.monotonic() - mono) >= 600:
                process.kill()
                timed_out = True
                break
            time.sleep(0.1)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        integrity_error = None
        try:
            verify_snapshot(snapshot, snapshot_hash)
        except Exception as error:
            integrity_error = f"{type(error).__name__}: {error}"
        report = finalize_supervision(
            args.output,
            declared,
            returncode=process.returncode,
            timed_out=timed_out,
            integrity_error=integrity_error,
        )
        write(
            args.output / "supervisor.json",
            {
                "returncode": process.returncode,
                "true_wall_seconds": max(time.time() - wall, time.monotonic() - mono),
                "max_true_wall_seconds": 600,
                "runtime_snapshot": str(snapshot),
                "snapshot_manifest_sha256": snapshot_hash,
                "status": report["status"],
                "training_ready": report["training_ready"],
            },
        )
    return 0 if report["training_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
