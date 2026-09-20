"""Bounded simulation collection. Privileged kinematics are used only by the data policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from embodied_jepa.data import DatasetStore, Episode
from embodied_jepa.embodiment import G1Embodiment
from embodied_jepa.simulation import MuJoCoSimulation

PAIRS = (("cube", "target"), ("banana", "bowl"), ("apple", "bowl"), ("cube", "plate"))


def scripted_action(embodiment, target, *, grasp=-1.0, amplitude=0.35):
    action = np.zeros(14, dtype=np.float32)
    action[12:] = [-1, grasp]
    position, _ = embodiment.ee_pose("right")
    scale = embodiment.manifest["translation_per_step_m"]
    action[6:9] = np.clip((np.asarray(target) - position) / scale, -amplitude, amplitude)
    return action


def reach_goal(embodiment, target, *, max_steps=100, tolerance=0.012):
    """Oracle control used to render a reachable goal and establish task feasibility."""
    trace = []
    for _ in range(max_steps):
        embodiment.observe()
        command = scripted_action(embodiment, target)
        result = embodiment.execute(command)
        trace.append({"status": result.status, "reason": result.reason})
        if result.applied_action is None:
            break
        error = float(np.linalg.norm(embodiment.ee_pose("right")[0] - target))
        if error < tolerance:
            return {"success": True, "error_m": error, "steps": len(trace), "trace": trace}
    return {
        "success": False,
        "error_m": float(np.linalg.norm(embodiment.ee_pose("right")[0] - target)),
        "steps": len(trace),
        "trace": trace,
    }


def collect(root, *, episodes=100, steps=50, seed=123, width=96):
    if type(episodes) is not int or episodes < 12 or type(steps) is not int or steps < 8:
        raise ValueError("collection requires >=12 episodes and >=8 steps")
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite {root}")
    start = time.perf_counter()
    store = None
    failures, summaries = [], []
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            Path(__file__),
            Path(__file__).with_name("embodiment.py"),
            Path(__file__).with_name("simulation.py"),
        )
    }
    provenance = {
        "producer": "open-embodied-jepa synthetic MuJoCo collection",
        "license": "project-generated simulation observations; robot assets BSD-3-Clause",
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "implementation_hashes": source_hashes,
        "collection_seed": seed,
        "policy": "bounded right-arm target tracking with diverse offsets and noise",
        "held_out_pair": ["apple", "plate"],
        "render_width": width,
        "simulated": True,
        "physical_hardware": False,
    }
    simulations = {}
    try:
        for index in range(episodes):
            pair = PAIRS[index % len(PAIRS)]
            if pair not in simulations:
                sim = MuJoCoSimulation(
                    width=width, height=width, object_kind=pair[0], container_kind=pair[1]
                )
                simulations[pair] = G1Embodiment(sim)
            robot = simulations[pair]
            if store is None:
                store = DatasetStore.create(
                    root,
                    fps=20,
                    state_schema=robot.state_schema,
                    action_manifest=robot.manifest,
                    provenance=provenance,
                )
            episode_seed = seed + index
            rng = np.random.default_rng(episode_seed)
            try:
                reset_truth = robot.reset(episode_seed)
            except Exception as error:
                failures.append(
                    {
                        "episode": index,
                        "transitions": 0,
                        "reasons": [f"reset_error: {type(error).__name__}: {error}"],
                    }
                )
                continue
            initial = robot.ee_pose("right")[0].copy()
            try:
                snapshots, actions, raw_actions = [robot.observe()], [], []
            except Exception as error:
                failures.append(
                    {
                        "episode": index,
                        "transitions": 0,
                        "reasons": [f"initial_observation_error: {type(error).__name__}: {error}"],
                    }
                )
                robot.stop("initial observation failed")
                continue
            target = initial.copy()
            reasons = []
            for step in range(steps):
                if step % 12 == 0:
                    target = initial + rng.uniform([-0.035, -0.07, -0.035], [0.045, 0.035, 0.055])
                command = scripted_action(robot, target, amplitude=0.4)
                command[6:9] = np.clip(command[6:9] + rng.normal(0, 0.12, 3), -0.5, 0.5)
                try:
                    result = robot.execute(command)
                    if result.applied_action is None:
                        reasons.append(result.reason)
                        break
                    # Append only a complete observed transition, preserving the valid prefix.
                    following = robot.observe()
                    actions.append(result.applied_action.copy())
                    raw_actions.append(robot.denormalize_action(result.applied_action))
                    snapshots.append(following)
                except Exception as error:
                    reasons.append(f"incomplete_transition: {type(error).__name__}: {error}")
                    robot.stop(reasons[-1])
                    break
            if len(actions) < 8:
                failures.append({"episode": index, "transitions": len(actions), "reasons": reasons})
                continue
            episode = Episode(
                episode_id=f"sim-{episode_seed:06d}",
                session_id=f"reset-{episode_seed:06d}",
                task="right_arm_reach",
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
                metadata={
                    "seed": episode_seed,
                    "reset_truth": {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in reset_truth.items()
                    },
                    "camera": {
                        "name": "onboard_rgb",
                        "width": width,
                        "height": width,
                        "fovy_deg": float(robot.sim.model.camera("onboard_rgb").fovy[0]),
                    },
                    "policy": provenance["policy"],
                    "stop_reasons": reasons,
                    "requested_steps": steps,
                    "final_ee_displacement_m": (robot.ee_pose("right")[0] - initial).tolist(),
                },
            )
            store.write_episode(episode)
            summaries.append(
                {
                    "episode_id": episode.episode_id,
                    "transitions": len(actions),
                    "object": pair[0],
                    "container": pair[1],
                    "reasons": reasons,
                }
            )
            if index % 10 == 0:
                print(
                    json.dumps(
                        {
                            "collected": index + 1,
                            "episodes": episodes,
                            "elapsed_s": time.perf_counter() - start,
                        }
                    ),
                    flush=True,
                )
        if store is None or len(store.episode_ids) < 3:
            raise RuntimeError("collection produced too few complete episodes")
        splits = store.freeze_splits(seed=seed)
        store.fit_normalization()
        report = {
            "kind": "reach_pilot",
            "requested_episodes": episodes,
            "stored_episodes": len(summaries),
            "failed_short_episodes": failures,
            "episodes": summaries,
            "transitions": sum(s["transitions"] for s in summaries),
            "splits": splits,
            "dataset_hash": store.manifest_hash,
            "wall_seconds": time.perf_counter() - start,
            "state_schema": asdict(store.state_schema),
            "provenance": provenance,
        }
        (root / "collection_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        for robot in simulations.values():
            robot.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    report = collect(args.output, episodes=args.episodes, steps=args.steps, seed=args.seed)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("stored_episodes", "transitions", "dataset_hash", "wall_seconds")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
