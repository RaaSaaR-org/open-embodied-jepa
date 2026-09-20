"""Common image-goal benchmark; model selection is exclusively registry/config based."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import subprocess
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from embodied_jepa.collection import reach_goal, scripted_action
from embodied_jepa.config import ExperimentConfig
from embodied_jepa.data import DatasetStore
from embodied_jepa.planning import MPC, CEMPlanner
from embodied_jepa.registry import EMBODIMENTS, MODELS, TASKS
from embodied_jepa.simulation import MuJoCoSimulation


def json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(successes, total):
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - radius), min(1, center + radius)]


def summarize(episodes):
    if not episodes:
        raise ValueError("no evaluation episodes")
    successes = sum(bool(e["score"].get("success", False)) for e in episodes)
    latencies = [
        t["planning_seconds"] for e in episodes for t in e["trace"] if "planning_seconds" in t
    ]
    candidates = sum(t.get("candidate_evaluations", 0) for e in episodes for t in e["trace"])
    return {
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes),
        "wilson95": wilson(successes, len(episodes)),
        "termination_reasons": dict(Counter(e["termination_reason"] for e in episodes)),
        "stages": {
            stage: sum(bool(e["score"].get(stage, False)) for e in episodes)
            for stage in ("reach", "grasp", "transport", "place", "release")
        },
        "plan_latency_seconds": {
            "p50": float(np.median(latencies)) if latencies else None,
            "p95": float(np.quantile(latencies, 0.95)) if latencies else None,
            "missing_reason": None if latencies else "control policy does not plan",
        },
        "candidates_per_second": candidates / sum(latencies)
        if latencies and sum(latencies) > 0
        else None,
        "candidates_missing_reason": None if latencies else "control policy does not plan",
        "deadline_misses": sum(e["termination_reason"] == "deadline_miss" for e in episodes),
    }


def validate_results(run_manifest, episodes):
    required = {
        "schema_version",
        "run_id",
        "timestamp",
        "source_revision",
        "backend",
        "checkpoint_hash",
        "mode",
        "dataset_hash",
        "split_hash",
        "action_hash",
        "environment",
        "planner",
        "train_seed",
        "evaluation_seeds",
        "task_version",
    }
    missing = required - run_manifest.keys()
    if missing:
        raise ValueError(f"missing run fields: {sorted(missing)}")
    if run_manifest["schema_version"] != 1 or run_manifest["mode"] != "common":
        raise ValueError("unsupported result schema/mode")
    if [e["seed"] for e in episodes] != run_manifest["evaluation_seeds"]:
        raise ValueError("all frozen evaluation seeds must be recorded exactly once in order")
    for episode in episodes:
        if (
            not isinstance(episode.get("score"), dict)
            or type(episode["score"].get("success")) is not bool
        ):
            raise ValueError("episode requires an explicit success outcome")
        if episode["termination_reason"] == "success" and not episode["score"]["success"]:
            raise ValueError("success termination conflicts with scoring evidence")
        if episode["executed_steps"] != sum(t.get("executed") is True for t in episode["trace"]):
            raise ValueError("executed step count conflicts with trace")
    json.dumps({"run": run_manifest, "episodes": episodes}, allow_nan=False)


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _png(path, array):
    from PIL import Image

    Image.fromarray(array).save(path)


def _control(robot, task, cfg, policy, rng, max_steps):
    trace = []
    reason, score = "step_limit", {}
    stage, step = "evaluate", 0
    try:
        score = task.evaluate()
        for step in range(max_steps):
            stage = "observe"
            robot.observe()
            if policy == "oracle":
                command = scripted_action(robot, task.target)
            elif policy == "random":
                command = rng.uniform(cfg.lower_bounds, cfg.upper_bounds).astype(np.float32)
            else:
                command = np.zeros(14, np.float32)
                command[12:] = -1
            stage = "execute"
            result = robot.execute(command)
            trace.append(
                {
                    "step": step,
                    "status": result.status,
                    "reason": result.reason,
                    "executed": result.applied_action is not None,
                    "action": None
                    if result.applied_action is None
                    else result.applied_action.tolist(),
                }
            )
            if result.applied_action is None:
                reason = result.status
                break
            stage = "evaluate"
            score = task.evaluate()
            if score["success"]:
                reason = "success"
                break
    except Exception as error:
        reason = "runtime_error"
        trace.append(
            {
                "step": step,
                "stage": stage,
                "error": f"{type(error).__name__}: {error}",
                "executed": None if stage == "execute" else False,
                "execution_uncertain": stage == "execute",
            }
        )
    finally:
        robot.stop(reason)
    return {
        "trace": trace,
        "score": score,
        "termination_reason": reason,
        "replans": 0,
        "executed_steps": sum(t["executed"] is True for t in trace),
    }


def run(config_path, *, policy="model", run_id=None):
    if policy not in ("model", "hold", "random", "oracle"):
        raise ValueError("unknown evaluation policy")
    cfg = ExperimentConfig.load(config_path, require_checkpoint=policy == "model")
    if cfg.task != "reach" and policy == "oracle":
        raise ValueError("full-task oracle is a separate labeled scripted probe")
    store = DatasetStore(cfg.dataset_root)
    action_manifest = json.loads(cfg.action_manifest.read_text())
    if action_manifest != store.manifest["action_manifest"]:
        raise ValueError("dataset/runtime action manifests differ")
    run_id = (
        run_id
        or f"{cfg.task}-{cfg.backend}-{policy}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}"
    )
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a single directory name")
    output = cfg.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    cfg.save(output / "resolved_config.json")
    import torch

    torch.set_num_threads(4)
    model = None
    if policy == "model":
        model = MODELS.create(
            cfg.backend,
            state_schema=store.state_schema,
            device=cfg.device,
            seed=cfg.seed,
            config=cfg.model_settings,
            metadata={
                "dataset_hash": store.manifest_hash,
                "action_hash": json_hash(store.manifest["action_manifest"]),
            },
        )
        model.load(cfg.checkpoint)
    run_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "implementation_hashes": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
        "mode": "common",
        "backend": cfg.backend if policy == "model" else f"control:{policy}",
        "checkpoint_hash": file_hash(cfg.checkpoint) if model else None,
        "checkpoint_missing_reason": None if model else "control policy",
        "dataset_hash": store.manifest_hash,
        "split_hash": json_hash(
            {"splits": store.manifest.get("splits"), "policy": store.manifest.get("split_policy")}
        ),
        "action_hash": json_hash(action_manifest),
        "train_seed": model.seed if model else None,
        "train_seed_missing_reason": None if model else "control policy",
        "evaluation_seeds": list(cfg.evaluation_seeds),
        "planner": asdict(cfg.planner),
        "timeout_seconds": cfg.timeout_seconds,
        "task": cfg.task,
        "task_version": "tabletop_proxy_v0",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "device": cfg.device,
            "engine": "mujoco",
            "mujoco": importlib.metadata.version("mujoco"),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "model_diagnostics": None,
        "model_diagnostics_reason": "see training report for within-backend metrics",
    }
    asset_path = Path(__file__).resolve().parents[2] / "assets/manifest.json"
    run_manifest["asset_manifest_hash"] = file_hash(asset_path)
    run_manifest["asset_revision"] = json.loads(asset_path.read_text())["revision"]
    _write(output / "run.json", run_manifest)
    _write(
        output / "dependencies.json",
        sorted(
            (
                {"name": d.metadata["Name"], "version": d.version}
                for d in importlib.metadata.distributions()
            ),
            key=lambda d: d["name"].lower(),
        ),
    )
    episodes = []
    for seed in cfg.evaluation_seeds:
        episode_dir = output / f"episode-{seed}"
        episode_dir.mkdir()
        start = time.perf_counter()
        record = {
            "seed": seed,
            "score": {},
            "trace": [],
            "termination_reason": "runtime_error",
            "executed_steps": 0,
            "replans": 0,
        }
        robot = None
        try:
            # Development reach cohort uses a seen pairing; final full task is the held-out pair.
            kinds = ("cube", "target") if cfg.task == "reach" else ("apple", "plate")
            sim = MuJoCoSimulation(object_kind=kinds[0], container_kind=kinds[1])
            robot = EMBODIMENTS.create(
                "unitree_g1_dex3", simulation=sim, action_manifest=cfg.action_manifest
            )
            robot.reset(seed)
            target = None
            if cfg.task == "reach":
                rng = np.random.default_rng(seed)
                target = robot.ee_pose("right")[0] + rng.uniform(
                    [-0.015, -0.045, -0.03], [0.025, -0.025, 0.025]
                )
                oracle = reach_goal(robot, target)
                record["goal_oracle"] = oracle
                if not oracle["success"]:
                    record["termination_reason"] = "unreachable_goal"
                    raise ValueError("scripted goal reachability failed")
                goal = robot.observe().images
                _write(
                    episode_dir / "goal_manifest.json",
                    {"seed": seed, "target_base": target.tolist(), "oracle": oracle},
                )
            else:
                # A separate goal reset depicts a supported object and retracted open hand.
                plate = sim.task_truth()["plate_position"][:2]
                robot.reset(seed, object_xy=plate, object_on_container=True)
                for _ in range(30):
                    robot.observe()
                    hold = np.zeros(14, np.float32)
                    hold[12:] = -1
                    if robot.execute(hold).applied_action is None:
                        raise ValueError("goal settling failed")
                if not sim.task_truth()["placed"]:
                    raise ValueError("goal object is not stably supported on plate")
                goal = robot.observe().images
            _png(episode_dir / "goal.png", goal["onboard_rgb"][0])
            record["goal_image_sha256"] = file_hash(episode_dir / "goal.png")
            robot.reset(seed)
            initial = robot.observe()
            _png(episode_dir / "initial.png", initial.images["onboard_rgb"][0])
            record["initial_image_sha256"] = file_hash(episode_dir / "initial.png")
            task = TASKS.create(
                cfg.task,
                embodiment=robot,
                **({"target_base": target} if target is not None else {}),
            )
            record["initial_score"] = task.evaluate()
            record["thresholds"] = asdict(task.thresholds)
            budget = replace(cfg.planner, seed=seed)
            if model:

                def save_observation(step, observation, destination=episode_dir):
                    if step % 5 == 0:
                        _png(
                            destination / f"frame-{step:04d}.png",
                            observation.images["onboard_rgb"][0],
                        )

                result = MPC(
                    model,
                    CEMPlanner(budget),
                    timeout_seconds=cfg.timeout_seconds,
                    device=cfg.device,
                ).run(
                    robot,
                    goal,
                    max_steps=cfg.max_steps,
                    evaluate=task.evaluate,
                    record_observation=save_observation,
                )
            else:
                result = _control(
                    robot, task, budget, policy, np.random.default_rng(seed), cfg.max_steps
                )
            record.update(result)
            record["final_score"] = task.evaluate()
            _png(episode_dir / "final.png", sim.render())
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        finally:
            if robot is not None:
                robot.stop(record["termination_reason"])
                robot.close()
        original_score = record["score"]
        stages = {
            stage: original_score.get(stage)
            for stage in ("reach", "grasp", "transport", "place", "release")
        }
        record["score"] = {
            **stages,
            **original_score,
            "success": bool(original_score.get("success", False)),
            "missing_stage_reason": "not applicable or outcome not observed"
            if any(v is None for v in stages.values())
            else None,
        }
        record["limit_rejections"] = sum(
            "limit" in t.get("reason", "") and not t.get("executed", False) for t in record["trace"]
        )
        record["collision_violations"] = None
        record["collision_missing_reason"] = (
            "contact occurs in manipulation; forbidden-contact classification is not implemented"
        )
        record["wall_seconds"] = time.perf_counter() - start
        record["rollout_directory"] = episode_dir.name
        episodes.append(record)
        _write(episode_dir / "episode.json", record)
        with (output / "episodes.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "seed": seed,
                    "termination": record["termination_reason"],
                    "score": record["score"],
                }
            ),
            flush=True,
        )
    validate_results(run_manifest, episodes)
    summary = summarize(episodes)
    summary["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if platform.system() == "Darwin" else 1024
    )
    _write(output / "summary.json", summary)
    return output, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", choices=("model", "hold", "random", "oracle"), default="model")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    output, summary = run(args.config, policy=args.policy, run_id=args.run_id)
    print(json.dumps({"output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
