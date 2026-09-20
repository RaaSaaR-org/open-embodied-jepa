"""Bounded scripted contact-grasp feasibility on a training pair, never learned control."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from embodied_jepa.embodiment import G1Embodiment, rotation_delta
from embodied_jepa.simulation import MuJoCoSimulation


def run_suite(args):
    """Five predeclared training-pair trials; every outcome is retained."""
    from embodied_jepa.scripted import OracleManipulationPolicy

    trials = [
        {
            "object_kind": "cube",
            "container_kind": "target",
            "object_xy": [0.34, -0.18],
            "plate_xy": [0.43, -0.18],
        },
        {
            "object_kind": "cube",
            "container_kind": "target",
            "object_xy": [0.35, -0.19],
            "plate_xy": [0.43, -0.12],
        },
        {
            "object_kind": "apple",
            "container_kind": "bowl",
            "object_xy": [0.34, -0.18],
            "plate_xy": [0.43, -0.16],
        },
        {
            "object_kind": "banana",
            "container_kind": "bowl",
            "object_xy": [0.34, -0.18],
            "plate_xy": [0.43, -0.16],
        },
        {
            "object_kind": "cube",
            "container_kind": "plate",
            "object_xy": [0.34, -0.18],
            "plate_xy": [0.43, -0.16],
        },
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "trials": trials,
        "max_seconds_per_trial": args.max_seconds,
        "success_rule": "task_truth placed true throughout final 10 accepted steps (0.5s)",
        "held_out_pair": "apple_plate excluded",
        "policy": "privileged OracleManipulationPolicy",
    }
    plan_text = json.dumps(plan, indent=2) + "\n"
    (args.output / "suite_plan.json").write_text(plan_text)
    summaries = []
    for seed, trial in enumerate(trials):
        folder = args.output / f"trial_{seed}"
        folder.mkdir(exist_ok=True)
        sim = MuJoCoSimulation(
            object_kind=trial["object_kind"],
            container_kind=trial["container_kind"],
            width=96,
            height=96,
        )
        robot = G1Embodiment(sim)
        rows = []
        start = time.monotonic()
        failure = ""
        policy = None
        truth = sim.task_truth()
        try:
            truth = robot.reset(seed, object_xy=trial["object_xy"], plate_xy=trial["plate_xy"])
            policy = OracleManipulationPolicy(truth)
            while not policy.done:
                if time.monotonic() - start > args.max_seconds:
                    failure = "wall budget"
                    break
                phase = policy.phase
                robot.observe()
                requested = policy.action(robot)
                result = robot.execute(requested)
                policy.advance(result)
                truth = sim.task_truth()
                rows.append(
                    {
                        "phase": phase,
                        "status": result.status,
                        "reason": result.reason,
                        "requested_action": requested.tolist(),
                        "applied_action": None
                        if result.applied_action is None
                        else result.applied_action.tolist(),
                        "object_world": truth["object_position"].tolist(),
                        "object_velocity": truth["object_velocity"].tolist(),
                        "ee_base": robot.ee_pose("right")[0].tolist(),
                        "time": truth["timestamp"],
                        "contact": truth["hand_contact"],
                        "lifted": truth["lifted"],
                        "placed": truth["placed"],
                    }
                )
                if policy.phase != phase:
                    Image.fromarray(sim.render()).save(folder / f"{phase}.png")
            failure = failure or policy.failure
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        finally:
            robot.close()
        summary = trial | {
            "seed": seed,
            "accepted_commands": 0 if policy is None else policy.step_count,
            "failure": failure,
            "elapsed_seconds": time.monotonic() - start,
            "any_lift": any(r["lifted"] for r in rows),
            "final_placed": bool(rows and rows[-1]["placed"]),
            "success": len(rows) >= 10 and not failure and all(r["placed"] for r in rows[-10:]),
            "final_object_world": truth["object_position"].tolist(),
        }
        (folder / "report.json").write_text(
            json.dumps(summary | {"records": rows}, indent=2) + "\n"
        )
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    paths = [
        Path("src/embodied_jepa") / name
        for name in ("simulation.py", "embodiment.py", "scripted.py")
    ]
    report = {
        "plan_sha256": hashlib.sha256(plan_text.encode()).hexdigest(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "action_manifest_sha256": hashlib.sha256(
            Path("configs/g1_sim_action.json").read_bytes()
        ).hexdigest(),
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "trials": summaries,
        "successes": sum(t["success"] for t in summaries),
        "count": len(summaries),
    }
    (args.output / "suite_report.json").write_text(json.dumps(report, indent=2) + "\n")


def run_controller_v1(args):
    """Six frozen current-reset trials, preserving every observed transition."""
    from embodied_jepa.manipulation_collection import _json, elapsed_seconds
    from embodied_jepa.scripted import (
        EarlyReleaseOracleManipulationPolicy,
        OracleManipulationPolicy,
        SlowOracleManipulationPolicy,
    )
    from embodied_jepa.task import AppleToPlateTask

    start_wall, start_active = time.time(), time.monotonic()

    def elapsed():
        return elapsed_seconds(start_wall, start_active)

    configurations = [
        ("original", None),
        ("slow_full", {"closure": 1.0, "grasp_ramp": 0.04}),
        ("slow_gentle", {"closure": 0.5, "grasp_ramp": 0.03}),
    ]
    if args.controller_v2:
        configurations = [
            ("early_open_08", {"opening_ramp": 0.08}),
            ("early_open_04", {"opening_ramp": 0.04}),
        ]
    max_seconds = 300 if args.controller_v2 else 600
    policy_class = (
        EarlyReleaseOracleManipulationPolicy if args.controller_v2 else SlowOracleManipulationPolicy
    )
    trials = [
        {
            "seed": (3000 if args.controller_v2 else 2000) + i * len(configurations) + j,
            "object_kind": "cube",
            "container_kind": container,
            "object_xy": [0.34, -0.18],
            "plate_xy": [0.45, -0.10],
            "candidate": name,
            "policy_options": options,
        }
        for i, container in enumerate(("plate", "target"))
        for j, (name, options) in enumerate(configurations)
    ]
    args.output.mkdir(parents=True)
    source_paths = [
        Path(__file__),
        Path("configs/g1_sim_action.json"),
        Path("assets/manifest.json"),
    ]
    source_paths += [
        Path("src/embodied_jepa") / name
        for name in ("simulation.py", "embodiment.py", "scripted.py", "task.py", "contracts.py")
    ]
    sources = {}
    for path in source_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sources[str(path)] = digest
        destination = args.output / "sources" / path.name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(path.read_bytes())
    plan = {
        "trials": trials,
        "max_true_wall_seconds": max_seconds,
        "execution_reserve_seconds": 30,
        "policy": "privileged oracle; not learned control",
        "heldout_apple_plate": "excluded",
        "scorer": "ordered AppleToPlateTask with default frozen thresholds",
        "source_sha256": sources,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    reports = []
    for trial in trials:
        if elapsed() >= max_seconds - 30:
            break
        folder = args.output / f"trial_{trial['seed']}"
        folder.mkdir()
        robot, sim, policy = None, None, None
        rows, frames, states, timestamps, actions, requested, raw_actions = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        failure = ""
        score = {"success": False}
        initial_truth = None
        try:
            sim = MuJoCoSimulation(
                object_kind=trial["object_kind"],
                container_kind=trial["container_kind"],
                width=96,
                height=96,
            )
            robot = G1Embodiment(sim)
            initial_truth = robot.reset(
                trial["seed"], object_xy=trial["object_xy"], plate_xy=trial["plate_xy"]
            )
            policy = (
                OracleManipulationPolicy(initial_truth)
                if trial["policy_options"] is None
                else policy_class(initial_truth, **trial["policy_options"])
            )
            scorer = AppleToPlateTask(robot)
            observation = robot.observe()
            frames.append(observation.images["onboard_rgb"][0].copy())
            states.append(observation.robot_state[0].copy())
            timestamps.append(float(observation.timestamps[0]))
            while not policy.done:
                if elapsed() >= max_seconds - 30:
                    failure = "true_wall_budget"
                    break
                phase = policy.phase
                action = policy.action(robot)
                result = robot.execute(action)
                policy.advance(result)
                truth = sim.task_truth()
                score = scorer.evaluate()
                velocity = sim.data.qvel[sim.vadr]
                max_joint = int(np.argmax(np.abs(velocity)))
                rows.append(
                    _json(
                        {
                            "phase": phase,
                            "status": result.status,
                            "reason": result.reason,
                            "requested_action": action,
                            "applied_action": result.applied_action,
                            "truth": truth,
                            "score": score,
                            "max_velocity_joint": sim.joint_names[max_joint],
                            "max_joint_velocity": velocity[max_joint],
                            "joint_velocity": velocity.copy(),
                        }
                    )
                )
                if result.applied_action is None:
                    failure = result.reason
                    break
                observation = robot.observe()
                frames.append(observation.images["onboard_rgb"][0].copy())
                states.append(observation.robot_state[0].copy())
                timestamps.append(float(observation.timestamps[0]))
                actions.append(result.applied_action.copy())
                requested.append(action.copy())
                raw_actions.append(robot.denormalize_action(result.applied_action))
                if policy.phase != phase:
                    Image.fromarray(frames[-1]).save(folder / f"{phase}.png")
        except (Exception, KeyboardInterrupt) as error:
            failure = f"{type(error).__name__}: {error}"
        finally:
            if robot is not None:
                robot.stop(failure or "probe complete")
                robot.close()
        if actions:
            np.savez_compressed(
                folder / "transitions.npz",
                rgb=np.array(frames),
                state=np.array(states),
                timestamps=np.array(timestamps),
                actions=np.array(actions),
                requested_actions=np.array(requested),
                raw_actions=np.array(raw_actions),
            )
        report = trial | {
            "failure": failure,
            "complete_transitions": len(actions),
            "policy_complete": bool(policy and policy.done and not policy.failure),
            "success": bool(score["success"]),
            "final_score": score,
            "initial_truth": _json(initial_truth),
            "final_record": rows[-1] if rows else None,
        }
        (folder / "trace.json").write_text(json.dumps(rows, indent=2) + "\n")
        (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
        summary = {
            "trials": reports,
            "planned": len(trials),
            "attempted": len(reports),
            "successes": sum(r["success"] for r in reports),
            "true_wall_seconds": elapsed(),
            "plan_sha256": hashlib.sha256((args.output / "plan.json").read_bytes()).hexdigest(),
        }
        (args.output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("initial_truth", "final_record")}
            ),
            flush=True,
        )
        if failure.startswith("KeyboardInterrupt"):
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/grasp_probe"))
    parser.add_argument("--max-seconds", type=float, default=45)
    parser.add_argument(
        "--suite", action="store_true", help="run the frozen five-trial transfer/release suite"
    )
    parser.add_argument(
        "--controller-v1",
        action="store_true",
        help="run frozen six-trial slow-controller comparison",
    )
    parser.add_argument(
        "--controller-v2",
        action="store_true",
        help="run frozen four-trial early-release comparison",
    )
    args = parser.parse_args()
    if not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("max-seconds must be finite and positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite experiment artifacts in {args.output}")
    if args.controller_v1 or args.controller_v2:
        run_controller_v1(args)
        return
    if args.suite:
        run_suite(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    sim = MuJoCoSimulation(object_kind="cube", container_kind="target", width=96, height=96)
    robot = G1Embodiment(sim)
    start = time.monotonic()
    records = []
    stopped = ""
    try:
        truth = robot.reset(0, object_xy=[0.34, -0.18], plate_xy=[0.48, -0.10])
        object_base = truth["base_rotation_world"].T @ (
            truth["object_position"] - truth["base_position_world"]
        )
        rotation = rotation_delta([-np.pi / 2, 0, 0])
        phases = [
            ("orient", object_base + [-0.03, 0, 0.13], -1.0, 130),
            ("descend", object_base + [-0.03, 0, 0.052], -1.0, 80),
            ("close", object_base + [-0.03, 0, 0.052], 1.0, 45),
            ("lift", object_base + [-0.03, 0, 0.21], 1.0, 150),
        ]
        for phase, target, grasp, budget in phases:
            for _ in range(budget):
                if time.monotonic() - start > args.max_seconds:
                    stopped = "wall budget"
                    break
                robot.observe()
                position, current_rotation = robot.ee_pose("right")
                dr = 0.5 * sum(np.cross(current_rotation[:, i], rotation[:, i]) for i in range(3))
                action = np.zeros(14, dtype=np.float32)
                action[6:9] = np.clip((target - position) / 0.015, -0.4, 0.4)
                action[9:12] = np.clip(dr / 0.06, -0.5, 0.5)
                action[12:] = [-1, grasp]
                result = robot.execute(action)
                truth = sim.task_truth()
                records.append(
                    {
                        "phase": phase,
                        "status": result.status,
                        "reason": result.reason,
                        "time": truth["timestamp"],
                        "ee_base": robot.ee_pose("right")[0].tolist(),
                        "object_world": truth["object_position"].tolist(),
                        "object_velocity": truth["object_velocity"].tolist(),
                        "contact": truth["hand_contact"],
                        "lifted": truth["lifted"],
                    }
                )
                if result.applied_action is None:
                    stopped = result.reason
                    break
            Image.fromarray(sim.render()).save(args.output / f"{phase}.png")
            print(phase, records[-1] if records else {}, flush=True)
            if stopped:
                break
        report = {
            "question": (
                "Can one bounded scripted top-down Dex3 synergy contact-grasp "
                "and lift a cube proxy?"
            ),
            "controller": "privileged scripted collector feasibility, not a learned policy",
            "pair": "cube_target",
            "seed": 0,
            "max_seconds": args.max_seconds,
            "elapsed_seconds": time.monotonic() - start,
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "action_manifest_sha256": hashlib.sha256(
                Path("configs/g1_sim_action.json").read_bytes()
            ).hexdigest(),
            "stopped": stopped,
            "any_contact": any(r["contact"] for r in records),
            "any_lift": any(r["lifted"] for r in records),
            "final_object_z": float(sim.task_truth()["object_position"][2]),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "phase_budgets": [
                {"name": p, "target_base": target.tolist(), "grasp": grasp, "max_commands": n}
                for p, target, grasp, n in phases
            ],
            "final_lifted": bool(sim.task_truth()["lifted"]),
            "records": records,
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    finally:
        robot.close()


if __name__ == "__main__":
    main()
