"""Privileged, bounded Apple→Plate mechanics probes; never learned control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa.embodiment import G1Embodiment  # noqa: E402
from embodied_jepa.manipulation_collection import _json  # noqa: E402
from embodied_jepa.scripted import (  # noqa: E402
    EarlyReleaseOracleManipulationPolicy,
    OracleManipulationPolicy,
)
from embodied_jepa.simulation import MuJoCoSimulation  # noqa: E402
from embodied_jepa.task import AppleToPlateTask, TaskThresholds  # noqa: E402

CANDIDATES = (
    ("original", -0.03, 0.052),
    ("early_release", -0.03, 0.052),
    ("lower_grasp", -0.03, 0.037),
    ("higher_grasp", -0.03, 0.067),
    ("rear_grasp", -0.045, 0.052),
    ("forward_grasp", -0.015, 0.052),
)


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_json(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(refine_landing=False):
    attempts = []
    candidates = (
        tuple((f"landing_shift_{shift}", -0.015, 0.052) for shift in (0, -0.02, -0.035))
        if refine_landing
        else CANDIDATES
    )
    for seed in (41100, 41101) if refine_landing else (41000, 41001):
        rng = np.random.default_rng(seed)
        obj = (np.array([0.34, -0.18]) + rng.uniform(-0.004, 0.004, 2)).tolist()
        plate = (np.array([0.49, -0.09]) + rng.uniform(-0.004, 0.004, 2)).tolist()
        for index, (name, x, z) in enumerate(candidates):
            attempts.append(
                dict(
                    seed=seed,
                    candidate=name,
                    x_offset=x,
                    grasp_height=z,
                    transfer_x_shift=(0, -0.02, -0.035)[index] if refine_landing else 0,
                    object_xy=obj,
                    plate_xy=plate,
                    status="not_started",
                )
            )
    return {
        "oracle_only": True,
        "learned_control": False,
        "max_cpu_seconds": 60 if refine_landing else 180,
        "max_true_wall_seconds": 60 if refine_landing else 180,
        "execution_cutoff_seconds": 55 if refine_landing else 170,
        "thresholds": asdict(TaskThresholds()),
        "attempts": attempts,
    }


def policy_for(trial, truth):
    policy = (
        OracleManipulationPolicy(truth)
        if trial["candidate"] == "original"
        else EarlyReleaseOracleManipulationPolicy(truth, opening_ramp=0.08)
    )
    phases = []
    for index, phase in enumerate(policy.phases):
        target = phase.target_base.copy()
        target[0] += trial["x_offset"] + 0.03
        if index >= 4:
            target[0] += trial.get("transfer_x_shift", 0)
        if phase.name in ("descend", "close"):
            target[2] += trial["grasp_height"] - 0.052
        phases.append(replace(phase, target_base=target))
    policy.phases = tuple(phases)
    return policy


def worker(output, refine_landing=False):
    from PIL import Image

    declared = plan(refine_landing)
    limit, cutoff = declared["max_cpu_seconds"], declared["execution_cutoff_seconds"]
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
    output.mkdir(parents=True, exist_ok=False)
    start_wall, start_mono, start_cpu = time.time(), time.monotonic(), time.process_time()

    def timing():
        return {
            "true_wall_seconds": max(time.time() - start_wall, time.monotonic() - start_mono),
            "cpu_seconds": time.process_time() - start_cpu,
        }

    source_paths = [
        Path(__file__),
        ROOT / "configs/g1_sim_action.json",
        ROOT / "assets/manifest.json",
    ]
    source_paths += [
        ROOT / "src/embodied_jepa" / name
        for name in ("simulation.py", "embodiment.py", "scripted.py", "task.py", "contracts.py")
    ]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    frozen = declared | {
        "source_hashes": hashes,
        "source_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    write(output / "plan.json", frozen)
    for path in source_paths:
        target = output / "sources" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    report = json.loads(json.dumps(frozen))
    write(output / "report.json", report)
    for index, trial in enumerate(report["attempts"]):
        if max(timing().values()) >= cutoff:
            trial["status"] = "not_started_budget"
            continue
        if any(digest(path) != hashes[str(path.relative_to(ROOT))] for path in source_paths):
            trial["status"] = "not_started_source_changed"
            continue
        folder = output / f"{index:02d}-{trial['seed']}-{trial['candidate']}"
        folder.mkdir()
        frames, states, masks, times, actions, requests, raw, phases, rows = ([] for _ in range(9))
        robot, policy = None, None
        try:
            sim = MuJoCoSimulation(object_kind="apple", container_kind="plate", width=96, height=96)
            robot = G1Embodiment(sim)
            truth = robot.reset(
                trial["seed"], object_xy=trial["object_xy"], plate_xy=trial["plate_xy"]
            )
            trial["initial_truth"] = _json(truth)
            policy = policy_for(trial, truth)
            scorer = AppleToPlateTask(robot)
            trial.update(status="running", state_schema=asdict(robot.state_schema))

            def capture(robot=robot, frames=frames, states=states, masks=masks, times=times):
                observation = robot.observe()
                frames.append(observation.images["onboard_rgb"][0].copy())
                states.append(observation.robot_state[0].copy())
                masks.append(observation.state_mask[0].copy())
                times.append(float(observation.timestamps[0]))

            capture()
            while not policy.done:
                if max(timing().values()) >= cutoff:
                    trial["status"] = "budget_truncated"
                    break
                phase = policy.phase
                requested = policy.action(robot)
                result = robot.execute(requested)
                policy.advance(result)
                truth, score = sim.task_truth(), scorer.evaluate()
                velocity = sim.data.qvel[sim.vadr].copy()
                joint = int(np.argmax(np.abs(velocity)))
                rows.append(
                    _json(
                        {
                            "phase": phase,
                            "requested": requested,
                            "applied": result.applied_action,
                            "status": result.status,
                            "reason": result.reason,
                            "truth": truth,
                            "score": score,
                            "ee_pose_base": robot.ee_pose("right")[0],
                            "max_joint_velocity": velocity[joint],
                            "max_velocity_joint": sim.joint_names[joint],
                        }
                    )
                )
                trial["final_score"] = score
                if result.applied_action is None:
                    trial.update(status="command_rejected", failure=result.reason)
                    break
                actions.append(result.applied_action.copy())
                requests.append(requested.copy())
                raw.append(robot.denormalize_action(result.applied_action))
                phases.append(phase)
                capture()
                if policy.phase != phase or score["success"]:
                    Image.fromarray(frames[-1]).save(folder / f"phase-{phase}.png")
                if score["success"]:
                    trial["status"] = "success"
                    break
            if trial["status"] == "running":
                trial["status"] = "policy_complete_without_success"
        except Exception as error:
            trial.update(status="runtime_error", failure=f"{type(error).__name__}: {error}")
        finally:
            if robot is not None:
                robot.stop(trial["status"])
                robot.close()
        if actions:
            np.savez_compressed(
                folder / "transitions.npz",
                rgb=np.array(frames),
                state=np.array(states),
                state_mask=np.array(masks),
                timestamps=np.array(times),
                actions=np.array(actions),
                requested_actions=np.array(requests),
                raw_actions=np.array(raw),
                phase=np.array(phases),
            )
        write(folder / "trace.json", rows)
        trial.update(
            executed_steps=len(actions),
            last_phase=rows[-1]["phase"] if rows else None,
            source_unchanged=all(
                digest(p) == hashes[str(p.relative_to(ROOT))] for p in source_paths
            ),
        )
        report["timing"] = timing()
        write(output / "report.json", report)
        print(
            json.dumps(
                {
                    key: trial.get(key)
                    for key in (
                        "seed",
                        "candidate",
                        "status",
                        "executed_steps",
                        "last_phase",
                        "failure",
                    )
                }
            ),
            flush=True,
        )
    report["timing"] = timing()
    if refine_landing:
        candidates = []
        for shift in (0, -0.02, -0.035):
            successful = [
                row
                for row in report["attempts"]
                if row["transfer_x_shift"] == shift and row["status"] == "success"
            ]
            margins = [0.04 - row["final_score"]["object_plate_distance_m"] for row in successful]
            candidates.append(
                {
                    "transfer_x_shift": shift,
                    "successes": len(successful),
                    "mean_success_margin_m": float(np.mean(margins)) if margins else None,
                }
            )
        eligible = [candidate for candidate in candidates if candidate["successes"]]
        report["selection"] = {
            "candidates": candidates,
            "selected": max(eligible, key=lambda x: (x["successes"], x["mean_success_margin_m"]))
            if eligible
            else None,
        }
    write(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--refine-landing", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.output.resolve(), args.refine_landing)
        return
    if args.output.exists():
        raise FileExistsError("refusing to overwrite probe")
    snapshot = args.output.resolve().with_name(args.output.name + "-runtime")
    if snapshot.exists():
        raise FileExistsError("refusing to overwrite runtime snapshot")
    snapshot.mkdir(parents=True)
    shutil.copytree(ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (snapshot / "scripts").mkdir()
    copied_script = snapshot / "scripts" / Path(__file__).name
    shutil.copyfile(Path(__file__), copied_script)
    (snapshot / "configs").mkdir()
    shutil.copyfile(ROOT / "configs/g1_sim_action.json", snapshot / "configs/g1_sim_action.json")
    (snapshot / "assets").mkdir()
    shutil.copyfile(ROOT / "assets/manifest.json", snapshot / "assets/manifest.json")
    (snapshot / "third_party").symlink_to(
        (ROOT / "third_party").resolve(), target_is_directory=True
    )
    command = [
        sys.executable,
        str(copied_script),
        "--worker",
        "--output",
        str(args.output.resolve()),
    ]
    if args.refine_landing:
        command.append("--refine-landing")
    max_seconds = 60 if args.refine_landing else 180
    env = os.environ | {
        "PYTHONPATH": str(snapshot / "src"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    start_wall, start_mono = time.time(), time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, env=env)
    try:
        while process.poll() is None:
            if max(time.time() - start_wall, time.monotonic() - start_mono) >= max_seconds:
                process.kill()
                break
            time.sleep(0.1)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        if args.output.is_dir():
            write(
                args.output / "supervisor.json",
                {
                    "returncode": process.returncode,
                    "runtime_snapshot": str(snapshot),
                    "source_hashes": {
                        str(p.relative_to(snapshot)): digest(p)
                        for p in (snapshot / "src").rglob("*.py")
                    },
                    "true_wall_seconds": max(
                        time.time() - start_wall, time.monotonic() - start_mono
                    ),
                    "max_true_wall_seconds": max_seconds,
                },
            )


if __name__ == "__main__":
    main()
