"""Bounded privileged manipulation collection; no held-out Apple→Plate data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import signal
import subprocess
import threading
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from embodied_jepa.collection import PAIRS
from embodied_jepa.data import DatasetStore, Episode
from embodied_jepa.embodiment import G1Embodiment
from embodied_jepa.scripted import EarlyReleaseOracleManipulationPolicy, OracleManipulationPolicy
from embodied_jepa.simulation import MuJoCoSimulation
from embodied_jepa.task import AppleToPlateTask


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _write(path, value):
    path.write_text(json.dumps(_json(value), indent=2, allow_nan=False) + "\n")


def elapsed_seconds(start_wall, start_active):
    """Count host suspension without letting a backward wall-clock change extend a run."""
    return max(time.time() - start_wall, time.monotonic() - start_active)


def make_plan(
    *,
    episodes=100,
    seed=1000,
    max_commands=805,
    max_seconds=600,
    width=96,
    policy_name="original",
    object_center=(0.34, -0.18),
    container_center=(0.49, -0.09),
):
    if policy_name not in ("original", "early_release_v1"):
        raise ValueError("unknown manipulation policy")
    for center in (object_center, container_center):
        if np.asarray(center).shape != (2,) or not np.isfinite(center).all():
            raise ValueError("reset centers must be finite xy pairs")
    policy_budget = 745 if policy_name == "early_release_v1" else 805
    if any(type(v) is not int or v < 1 for v in (episodes, seed, width)):
        raise ValueError("episodes, seed and width must be positive integers")
    if type(max_commands) is not int or not 8 <= max_commands <= policy_budget:
        raise ValueError(
            f"max_commands must lie between 8 and the fixed policy budget{policy_budget}"
        )
    if not np.isfinite(max_seconds) or max_seconds <= 0 or max_seconds > 600:
        raise ValueError("max_seconds must be positive and at most600")
    trials = []
    for index in range(episodes):
        trial_seed = seed + index
        rng = np.random.default_rng(trial_seed)
        obj, container = PAIRS[index % len(PAIRS)]
        assert (obj, container) != ("apple", "plate")
        trials.append(
            {
                "index": index,
                "seed": trial_seed,
                "object": obj,
                "container": container,
                "object_xy": (np.array(object_center) + rng.uniform(-0.006, 0.006, 2)).tolist(),
                "container_xy": (
                    np.array(container_center) + rng.uniform(-0.006, 0.006, 2)
                ).tolist(),
            }
        )
    return {
        "kind": "manipulation_release_v1"
        if policy_name == "early_release_v1"
        else "manipulation_pilot_v0",
        "policy_name": policy_name,
        "policy_options": {"opening_ramp": 0.08} if policy_name == "early_release_v1" else {},
        "object_center": list(object_center),
        "container_center": list(container_center),
        "reset_jitter_m": 0.006,
        "requested_episodes": episodes,
        "trials": trials,
        "max_commands": max_commands,
        "max_wall_seconds": max_seconds,
        "finalization_reserve_seconds": min(30.0, max_seconds / 10),
        "seed": seed,
        "width": width,
        "fps": 20,
        "minimum_stored_transitions": 8,
        "split_ratios": [0.8, 0.1, 0.1],
        "held_out_pair": ["apple", "plate"],
        "policy": (
            "fixed EarlyReleaseOracleManipulationPolicy opening_ramp=0.08; "
            "no tuning during collection"
            if policy_name == "early_release_v1"
            else "fixed OracleManipulationPolicy; no tuning during collection"
        ),
        "success": "ordered AppleToPlateTask scorer, applied equally to all training pairs",
    }


def collect(
    root,
    *,
    episodes=100,
    seed=1000,
    max_commands=805,
    max_seconds=600,
    width=96,
    policy_name="original",
    object_center=(0.34, -0.18),
    container_center=(0.49, -0.09),
):
    """Collect complete20Hz transitions; retain failures, budget stops and observed prefixes."""
    plan = make_plan(
        episodes=episodes,
        seed=seed,
        max_commands=max_commands,
        max_seconds=max_seconds,
        width=width,
        policy_name=policy_name,
        object_center=object_center,
        container_center=container_center,
    )
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite {root}")
    start, start_wall = time.monotonic(), time.time()
    execution_budget = max_seconds - plan["finalization_reserve_seconds"]
    directory = Path(__file__).parent
    source_paths = [
        directory / name
        for name in (
            "manipulation_collection.py",
            "collection.py",
            "simulation.py",
            "embodiment.py",
            "scripted.py",
            "task.py",
            "data.py",
        )
    ]
    source_bytes = {p.name: p.read_bytes() for p in source_paths}
    asset_path = directory.parents[1] / "assets/manifest.json"
    provenance = {
        "producer": "privileged scripted MuJoCo manipulation collector",
        "policy": plan["policy"],
        "simulated": True,
        "physical_hardware": False,
        "license": "project-generated data; upstream robot assets retain BSD-3-Clause",
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "implementation_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in source_bytes.items()
        },
        "asset_manifest_sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
        "asset_revision": json.loads(asset_path.read_text())["revision"],
        "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("mujoco", "numpy", "pillow", "pyarrow")
        },
        "held_out_pair": plan["held_out_pair"],
        "collection_seed": seed,
        "clock_policy": "max(wall-clock elapsed, monotonic elapsed); includes host suspension",
        "start_unix_time": start_wall,
    }
    simulations = {}
    attempts = []
    phase_counts = Counter()
    store = None
    interrupted = [False]
    previous_sigint = None
    if threading.current_thread() is threading.main_thread():
        previous_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))
    try:
        # Create the store and save the frozen plan before the first recorded trial.
        first = PAIRS[0]
        simulations[first] = G1Embodiment(
            MuJoCoSimulation(
                width=width, height=width, object_kind=first[0], container_kind=first[1]
            )
        )
        store = DatasetStore.create(
            root,
            fps=20,
            state_schema=simulations[first].state_schema,
            action_manifest=simulations[first].manifest,
            provenance=provenance,
        )
        _write(root / "collection_plan.json", plan)
        source_directory = root / "meta/collection_source"
        source_directory.mkdir()
        for name, content in source_bytes.items():
            path = source_directory / name
            path.write_bytes(content)
            store._record_hash(str(path.relative_to(root)))
        print(f"Frozen plan written before execution: {root / 'collection_plan.json'}", flush=True)
        for trial in plan["trials"]:
            if interrupted[0] or elapsed_seconds(start_wall, start) >= execution_budget:
                break
            pair = trial["object"], trial["container"]
            snapshots, actions, raw_actions, requested, phases, scores = [], [], [], [], [], []
            reasons = []
            last_score = {"success": False}
            reset_truth = None
            policy = None
            robot = None
            stage = "construct"
            trial_start = time.monotonic()
            successful_execution_count = 0
            try:
                if pair not in simulations:
                    simulations[pair] = G1Embodiment(
                        MuJoCoSimulation(
                            width=width, height=width, object_kind=pair[0], container_kind=pair[1]
                        )
                    )
                robot = simulations[pair]
                stage = "reset"
                reset_truth = robot.reset(
                    trial["seed"], object_xy=trial["object_xy"], plate_xy=trial["container_xy"]
                )
                scorer = AppleToPlateTask(robot)
                last_score = scorer.evaluate()
                policy = (
                    EarlyReleaseOracleManipulationPolicy(reset_truth, opening_ramp=0.08)
                    if policy_name == "early_release_v1"
                    else OracleManipulationPolicy(reset_truth)
                )
                stage = "observe_initial"
                snapshots.append(robot.observe())
                while not policy.done and len(actions) < max_commands:
                    if interrupted[0]:
                        reasons.append("interrupt_requested")
                        break
                    if elapsed_seconds(start_wall, start) >= execution_budget:
                        reasons.append("wall_budget")
                        break
                    phase = policy.phase
                    command = policy.action(robot)
                    stage = "execute"
                    result = robot.execute(command)
                    policy.advance(result)
                    if result.applied_action is None:
                        reasons.append(result.reason or "command_rejected")
                        break
                    successful_execution_count += 1
                    stage = "observe_following"
                    following = robot.observe()
                    # Only complete before/action/after transitions enter the dataset.
                    actions.append(result.applied_action.copy())
                    raw_actions.append(robot.denormalize_action(result.applied_action))
                    requested.append(command.copy())
                    snapshots.append(following)
                    phases.append(phase)
                    stage = "evaluate"
                    last_score = scorer.evaluate()
                    scores.append(last_score)
                if not reasons and (policy is None or not policy.done):
                    reasons.append("command_budget")
            except Exception as error:
                reasons.append(f"{stage}: {type(error).__name__}: {error}")
            finally:
                if robot is not None:
                    robot.stop(reasons[-1] if reasons else "oracle_complete")
            complete = policy is not None and policy.done and not policy.failure and not reasons
            success = bool(last_score.get("success", False))
            attempt = trial | {
                "episode_id": f"manip-{trial['seed']:06d}",
                "complete_transitions": len(actions),
                "successful_execution_count": successful_execution_count,
                "incomplete_execution_count": successful_execution_count - len(actions),
                "execution_uncertain": bool(
                    reasons and stage == "execute" and "Error" in reasons[-1]
                ),
                "stored": False,
                "policy_complete": complete,
                "success": success,
                "final_score": last_score,
                "reasons": reasons,
                "seconds": time.monotonic() - trial_start,
            }
            if len(scores) < len(actions):
                scores.append({"success": False, "scoring_error": reasons[-1]})
            if len(actions) >= 8:
                metadata = {
                    "trial": trial,
                    "reset_truth": _json(reset_truth),
                    "phase_labels": phases,
                    "stage_scores": scores,
                    "requested_actions": np.asarray(requested).tolist(),
                    "final_score": last_score,
                    "collection_success": success,
                    "policy_complete": complete,
                    "stop_reasons": reasons,
                    "camera": {
                        "name": "onboard_rgb",
                        "width": width,
                        "height": width,
                        "fovy_deg": float(robot.sim.model.camera("onboard_rgb").fovy[0]),
                    },
                    "state_schema": asdict(robot.state_schema),
                    "requested_max_commands": max_commands,
                    "incomplete_execution_count": attempt["incomplete_execution_count"],
                }
                episode = Episode(
                    episode_id=attempt["episode_id"],
                    session_id=f"manip-reset-{trial['seed']:06d}",
                    task="scripted_tabletop_manipulation",
                    object_id=pair[0],
                    container_id=pair[1],
                    observations={
                        "onboard_rgb": np.concatenate([s.images["onboard_rgb"] for s in snapshots])
                    },
                    robot_states=np.concatenate([s.robot_state for s in snapshots]),
                    state_mask=np.concatenate([s.state_mask for s in snapshots]),
                    actions=np.stack(actions),
                    timestamps=np.concatenate([s.timestamps for s in snapshots]),
                    state_schema=robot.state_schema,
                    raw_actions=np.stack(raw_actions),
                    terminated=bool(success),
                    truncated=not success,
                    metadata=metadata,
                )
                store.write_episode(episode)
                attempt["stored"] = True
                phase_counts.update(phases)
            attempts.append(attempt)
            with (root / "attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(_json(attempt), allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        key: attempt[key]
                        for key in (
                            "index",
                            "seed",
                            "object",
                            "container",
                            "complete_transitions",
                            "success",
                            "reasons",
                        )
                    }
                ),
                flush=True,
            )
        splits = None
        if len(store.episode_ids) >= 3:
            splits = store.freeze_splits(seed=seed, ratios=(0.8, 0.1, 0.1))
            store.fit_normalization()
        stage_counts = {
            key: sum(bool(a["final_score"].get(key)) for a in attempts)
            for key in ("reach", "grasp", "transport", "place", "release")
        }
        report = {
            "plan": plan,
            "provenance": provenance,
            "attempted_episodes": len(attempts),
            "unattempted_episodes": len(plan["trials"]) - len(attempts),
            "unattempted_budget_episodes": 0
            if interrupted[0]
            else len(plan["trials"]) - len(attempts),
            "unattempted_interrupted_episodes": len(plan["trials"]) - len(attempts)
            if interrupted[0]
            else 0,
            "interrupt_requested": interrupted[0],
            "stored_episodes": len(store.episode_ids),
            "transitions": sum(a["complete_transitions"] for a in attempts if a["stored"]),
            "successes": sum(a["success"] for a in attempts),
            "success_denominator": len(attempts),
            "stage_counts": stage_counts,
            "pair_attempts": dict(Counter(f"{a['object']}->{a['container']}" for a in attempts)),
            "phase_transition_counts": dict(phase_counts),
            "attempts": attempts,
            "splits": splits,
            "dataset_hash": store.manifest_hash,
            "wall_seconds": elapsed_seconds(start_wall, start),
            "active_seconds": time.monotonic() - start,
            "storage_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        }
        _write(root / "collection_report.json", report)
        return report
    finally:
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        for robot in simulations.values():
            robot.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/manipulation-pilot-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--policy", choices=("original", "early_release_v1"), default="original")
    parser.add_argument("--max-commands", type=int, default=805)
    parser.add_argument("--container-center", type=float, nargs=2, default=(0.49, -0.09))
    args = parser.parse_args()
    report = collect(
        args.output,
        episodes=args.episodes,
        seed=args.seed,
        max_seconds=args.max_seconds,
        policy_name=args.policy,
        max_commands=args.max_commands,
        container_center=args.container_center,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "attempted_episodes",
                    "stored_episodes",
                    "transitions",
                    "successes",
                    "dataset_hash",
                    "wall_seconds",
                    "storage_bytes",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
