"""Bounded, paired TASK-029 development diagnostic; never retrains checkpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The shared editable virtualenv can point to another checkout.
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa.config import ExperimentConfig  # noqa: E402
from embodied_jepa.goals import RUNTIME_SOURCES, load_goals  # noqa: E402
from embodied_jepa.training import RunClock, source_identity  # noqa: E402

MAX_SECONDS = 600.0
SEEDS = tuple(range(20000, 20005))
BACKENDS = ("native_jepa", "leworldmodel")
PINNED_INPUTS = {
    "configs/mvp_common.yaml": ("292c7484e522f7cc076a8e122950a5dfe9f097e7bb49dff947a004a563ac1bd2"),
    "configs/mvp_seed1.yaml": ("19e52c1cf96c04bde0ec53fc4f1dd6625365dc347de80d2da520e9b74e34bbb6"),
    "data/mvp-v0/meta/jepa_manifest.json": (
        "200246b34bd18cb30c8c711879c42a5ed2f31a62c99b96c1b347e37aa57512b7"
    ),
    "checkpoints/mvp-v0/seed-1/native_jepa.pt": (
        "62125697464d56bcb79ece37008d6afd81b62505f0fab369b5aaa13510b1c33a"
    ),
    "checkpoints/mvp-v0/seed-1/leworldmodel.pt": (
        "ab00894e48c1fac0c862fa5b37cb648f0d673704d6afb839980c2f45611b2dd5"
    ),
    "configs/g1_sim_action.json": (
        "f247effe9a9cd5e9ae3a9ad8e13ed036fa652f5a0d39f96b14c98979642c38b1"
    ),
    "data/goals/mvp-v0/manifest.json": (
        "eef30af440d40064efecc98a84a68d4388f9f60d1151104b4859a8f9934ee38e"
    ),
    "assets/manifest.json": ("2421e194d4102a1ff3b59c9767ca5314d335d7e00dba010816177f06ed17e993"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def planned_runs():
    return [
        {
            "run_id": f"{seed}-{backend}-projection-{'on' if projected else 'off'}",
            "seed": seed,
            "backend": backend,
            "project_candidates": projected,
            "status": "planned",
            "metrics": None,
        }
        for seed in SEEDS
        for backend in BACKENDS
        for projected in (False, True)
    ]


def verify_inputs(root):
    for relative, expected in PINNED_INPUTS.items():
        if digest(root / relative) != expected:
            raise ValueError(f"frozen input changed: {relative}")


def source_hash():
    return source_identity()["python_source_sha256"]


def verify_generated(record):
    if digest(record["config"]) != record["config_sha256"]:
        raise ValueError("generated configuration changed")
    config = json.loads(Path(record["config"]).read_text())
    goal_path = Path(config["task"]["goal_manifest"])
    if digest(goal_path) != record["goal_manifest_sha256"]:
        raise ValueError("derived goal manifest changed")
    goal = json.loads(goal_path.read_text())
    for episode in goal.get("episodes", []):
        image = (goal_path.parent / episode["image"]).resolve()
        if (
            not image.is_relative_to(goal_path.parent.resolve())
            or digest(image) != episode["sha256"]
        ):
            raise ValueError("derived goal image changed")


def derive_goal(root, destination, seed):
    """Rebind reused pixels to revised runtime explicitly; never alter original goals."""
    parent_path = root / "data/goals/mvp-v0/manifest.json"
    original = json.loads(parent_path.read_text())
    entry = next(item for item in original["episodes"] if item["seed"] == seed)
    source_image = (parent_path.parent / entry["image"]).resolve()
    if not source_image.is_relative_to(parent_path.parent.resolve()):
        raise ValueError("goal image escapes frozen source directory")
    if digest(source_image) != entry["sha256"]:
        raise ValueError("original goal image hash mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    filename = source_image.name
    shutil.copyfile(source_image, destination / filename)
    manifest = copy.deepcopy(original)
    manifest.update(
        seeds=[seed],
        episodes=[dict(entry, image=filename)],
        runtime_source_hashes={
            name: digest(root / "src/embodied_jepa" / name) for name in RUNTIME_SOURCES
        },
        development_reuse={
            "parent_manifest_sha256": digest(parent_path),
            "original_runtime_source_hashes": original["runtime_source_hashes"],
            "reason": "TASK-029 revised feasibility runtime; original pixels/reset unchanged",
            "independent_test": False,
        },
    )
    path = destination / "manifest.json"
    write(path, manifest)
    load_goals(
        path,
        task="apple_to_plate",
        seeds=[seed],
        action_manifest=root / "configs/g1_sim_action.json",
    )
    return path


def prepare(root, output, records):
    # Resolve inherited paths with the production configuration API.
    base = ExperimentConfig.load(root / "configs/mvp_seed1.yaml").resolved
    goals = {seed: derive_goal(root, output / "goals" / str(seed), seed) for seed in SEEDS}
    configs = output / "configs"
    configs.mkdir()
    for record in records:
        config = copy.deepcopy(base)
        config["seed"] = 1
        config["runtime"] = {"device": "cpu"}
        config["world_model"]["backend"] = record["backend"]
        config["planner"].update(
            horizon=4,
            samples=64,
            iterations=3,
            elites=8,
            timeout_seconds=5.0,
            project_candidates=record["project_candidates"],
        )
        config["task"].update(max_steps=100, goal_manifest=str(goals[record["seed"]]))
        config["evaluation"] = {"seeds": [record["seed"]], "output_root": str(output / "runs")}
        path = configs / f"{record['run_id']}.json"
        write(path, config)
        ExperimentConfig.load(path)
        record.update(
            config=str(path),
            config_sha256=digest(path),
            goal_manifest_sha256=digest(goals[record["seed"]]),
        )


def statistics(values):
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "p50": float(np.median(values)) if values else None,
        "p95": float(np.quantile(values, 0.95)) if values else None,
        "max": float(np.max(values)) if values else None,
    }


def episode_metrics(episode):
    trace = episode.get("trace", [])
    termination = episode["termination_reason"]
    score = episode.get("final_score", episode.get("score", {}))
    result = {
        "joint_rate_stop": any("joint rate" in row.get("reason", "").lower() for row in trace),
        "any_guard_or_runtime_stop": termination not in ("success", "step_limit"),
        "termination_reason": termination,
        "executed_steps": episode["executed_steps"],
        "stages": {
            stage: bool(score.get(stage, False))
            for stage in ("reach", "grasp", "transport", "place", "release", "success")
        },
        "latency_seconds": {
            name: statistics([row[name] for row in trace if name in row])
            for name in ("planning_seconds", "projection_seconds", "control_seconds")
        },
        "action_disagreement": {},
    }
    for label, left, right in (
        ("sampled_vs_projected", "sampled_action", "projected_action"),
        ("execute_requested_vs_applied", "requested_action", "action"),
        ("sampled_vs_applied", "sampled_action", "action"),
    ):
        deltas = [
            np.abs(np.asarray(row[left]) - np.asarray(row[right]))
            for row in trace
            if row.get(left) is not None and row.get(right) is not None
        ]
        if any(value.shape != (14,) or not np.isfinite(value).all() for value in deltas):
            raise ValueError("invalid action trace")
        result["action_disagreement"][label] = {
            "paired_steps": len(deltas),
            "mean_absolute_error": float(np.mean(deltas)) if deltas else None,
            "max_absolute_error": float(np.max(deltas)) if deltas else None,
            "missing_reason": None if deltas else "no paired trace actions; not assumed zero",
        }
    return result


def supervise(command, log_path, seconds, root):
    clock = RunClock()
    environment = os.environ | {
        "PYTHONPATH": str(root / "src"),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    with log_path.open("x") as log:
        process = subprocess.Popen(
            command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        timeout = False
        try:
            while process.poll() is None:
                if clock.elapsed() >= seconds:
                    process.kill()
                    timeout = True
                    break
                time.sleep(min(0.1, max(0.001, seconds - clock.elapsed())))
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    return {"returncode": process.returncode, "timeout": timeout, "timing": clock.snapshot()}


def paired_summary(records):
    """Use only completed matched pairs; always disclose omitted/incomplete pairs."""
    result = {}
    for backend in BACKENDS:
        pairs = []
        for seed in SEEDS:
            selected = [r for r in records if r["backend"] == backend and r["seed"] == seed]
            if len(selected) != 2 or any(row["status"] != "completed" for row in selected):
                continue
            off = next(r["metrics"] for r in selected if not r["project_candidates"])
            on = next(r["metrics"] for r in selected if r["project_candidates"])
            pairs.append(
                {
                    "seed": seed,
                    "on_minus_off": {
                        "joint_rate_stops": int(on["joint_rate_stop"])
                        - int(off["joint_rate_stop"]),
                        "any_stops": int(on["any_guard_or_runtime_stop"])
                        - int(off["any_guard_or_runtime_stop"]),
                        "executed_steps": on["executed_steps"] - off["executed_steps"],
                        "stages": {
                            name: int(on["stages"][name]) - int(off["stages"][name])
                            for name in on["stages"]
                        },
                    },
                }
            )
        result[backend] = {
            "planned_pairs": len(SEEDS),
            "completed_pairs": len(pairs),
            "incomplete_pairs": len(SEEDS) - len(pairs),
            "paired_differences": pairs,
        }
    return result


def run(output, expected_source_sha256, *, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    clock = RunClock()
    records = planned_runs()
    report = {
        "schema_version": 1,
        "status": "running",
        "development_only": True,
        "max_seconds": MAX_SECONDS,
        "max_steps": 100,
        "training_seed": 1,
        "source": source_identity(),
        "expected_source_sha256": expected_source_sha256,
        "pinned_inputs": PINNED_INPUTS,
        "runner_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(root / "docs/experiments/feasibility_v1.md"),
        "runs": records,
    }
    report_path = output / "comparison.json"
    write(report_path, report)
    try:
        verify_inputs(root)
        if source_hash() != expected_source_sha256:
            raise ValueError("source hash differs from authorized revision")
        prepare(root, output, records)
    except Exception as error:
        report.update(status="setup_failed", error=f"{type(error).__name__}: {error}")
        for row in records:
            row["status"] = "not_started_setup_failure"
        report["timing"] = clock.snapshot()
        write(report_path, report)
        return report
    invalidated = None
    for row in records:
        if clock.elapsed() >= MAX_SECONDS:
            row["status"] = "not_started_total_budget"
        elif invalidated:
            row.update(status="not_started_integrity_changed", error=invalidated)
        else:
            try:
                verify_inputs(root)
                if source_hash() != expected_source_sha256:
                    raise ValueError("source changed during comparison")
                verify_generated(row)
                remaining = MAX_SECONDS - clock.elapsed()
                if remaining <= 0:
                    row["status"] = "not_started_total_budget"
                    continue
                command = [
                    sys.executable,
                    "-m",
                    "embodied_jepa.benchmark",
                    "--config",
                    row["config"],
                    "--policy",
                    "model",
                    "--run-id",
                    row["run_id"],
                ]
                row.update(status="attempted", command=command, remaining_budget_seconds=remaining)
                write(report_path, report)
                row.update(supervise(command, output / f"{row['run_id']}.log", remaining, root))
                episode_path = (
                    output / "runs" / row["run_id"] / f"episode-{row['seed']}" / "episode.json"
                )
                if episode_path.exists():
                    episode = json.loads(episode_path.read_text())
                    row.update(metrics=episode_metrics(episode), episode=str(episode_path))
                row["status"] = (
                    "incomplete_total_budget"
                    if row["timeout"]
                    else "process_failed"
                    if row["returncode"]
                    else "completed"
                    if row["metrics"] is not None
                    else "missing_episode_result"
                )
                verify_inputs(root)
                verify_generated(row)
                if source_hash() != expected_source_sha256:
                    raise ValueError("source changed during episode")
            except Exception as error:
                invalidated = f"{type(error).__name__}: {error}"
                row.update(status="failed_integrity_or_orchestration", error=invalidated)
        report["timing"] = clock.snapshot()
        write(report_path, report)
    report["status"] = (
        "completed" if all(r["status"] == "completed" for r in records) else "incomplete"
    )
    report["paired_summary"] = paired_summary(records)
    report["timing"] = clock.snapshot()
    write(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    report = run(args.output, args.source_sha256)
    print(json.dumps({"status": report["status"], "timing": report["timing"]}, indent=2))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
