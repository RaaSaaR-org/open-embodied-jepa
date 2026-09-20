"""Read-only phase-balanced sensor-model diagnostics on TRAIN/VAL, never TEST."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ENTRY_CLOCKS = time.time(), time.monotonic()

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa.data import DatasetStore, concatenate_batches  # noqa: E402
from embodied_jepa.registry import MODELS  # noqa: E402
from embodied_jepa.training import json_hash, source_identity  # noqa: E402

DATASET_SHA = "d70edd9daaeec58e80a4e6a743980c049335a03f2b32b7d42bbac5ec5bc579df"
CHECKPOINT_SHA = "3a5c5e7ba77556b6ce4251f05af72dc86df0121807d913ca8b444db60026e3ee"
PHASES = ("orient", "descend", "close", "lift", "transfer", "release_high")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def cohorts(manifest):
    rows = {row["episode_id"]: row for row in manifest["episodes"]}
    groups = []
    for split_index, split in enumerate(("train", "val")):
        for phase_index, phase in enumerate(PHASES):
            for horizon in (4, 8):
                candidates = []
                for episode_id in sorted(manifest["splits"][split]):
                    labels = rows[episode_id]["metadata"]["phase_labels"]
                    for start in range(len(labels) - horizon + 1):
                        if all(label == phase for label in labels[start : start + horizon]):
                            candidates.append((episode_id, start))
                rng = np.random.default_rng(
                    np.random.SeedSequence([123, split_index, phase_index, horizon])
                )
                indices = rng.choice(len(candidates), min(32, len(candidates)), replace=False)
                chosen = [candidates[int(index)] for index in indices]
                groups.append(
                    {
                        "split": split,
                        "phase": phase,
                        "horizon": horizon,
                        "available_windows": len(candidates),
                        "selected_windows": chosen,
                        "cohort_sha256": json_hash(chosen),
                        "status": "planned",
                    }
                )
    return groups


def worker(args):
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    import torch

    torch.set_num_threads(4)
    wall, mono, cpu = time.time(), time.monotonic(), time.process_time()

    def clocks():
        return {
            "true_wall_seconds": max(time.time() - wall, time.monotonic() - mono),
            "process_cpu_seconds": time.process_time() - cpu,
        }

    def check_budget():
        if clocks()["true_wall_seconds"] >= 115 or clocks()["process_cpu_seconds"] >= 55:
            raise TimeoutError("diagnostic execution budget exhausted; reserve finalization")

    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "test_images_decoded": False,
        "read_only": True,
        "source": source_identity(),
        "groups": [],
        "decoded_episode_ids": [],
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(ROOT / "docs/experiments/apple_sensor_diagnostics_v1.md"),
    }
    write(args.output / "report.json", report)
    try:
        if digest(args.checkpoint) != CHECKPOINT_SHA:
            raise ValueError("checkpoint differs from frozen selected checkpoint")
        store = DatasetStore(args.dataset)
        if store.manifest_hash != DATASET_SHA:
            raise ValueError("dataset differs from frozen corpus")
        report.update(dataset_sha256=store.manifest_hash, checkpoint_sha256=CHECKPOINT_SHA)
        report["groups"] = cohorts(store.manifest)
        write(args.output / "plan.json", report)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["backend"] != "sensor_wm":
            raise ValueError("wrong backend")
        normalization_ids = checkpoint["metadata"]["normalization"]["episode_ids"]
        if set(normalization_ids) != set(store.manifest["splits"]["train"]):
            raise ValueError("normalization IDs do not match frozen training partition")
        model = MODELS.create(
            "sensor_wm",
            state_schema=store.state_schema,
            device="cpu",
            seed=checkpoint["seed"],
            config=checkpoint["config"],
            metadata={
                "dataset_hash": DATASET_SHA,
                "split_hash": json_hash(
                    {"splits": store.manifest["splits"], "policy": store.manifest["split_policy"]}
                ),
                "action_hash": json_hash(store.manifest["action_manifest"]),
            },
        )
        model.load(args.checkpoint)
        report.update(
            checkpoint_step=model.updates,
            model_config=model.config,
            implementation_sha256=model.implementation_sha256,
        )
        allowed = set(store.manifest["splits"]["train"] + store.manifest["splits"]["val"])
        cache = {}
        for group in report["groups"]:
            check_budget()
            if not group["selected_windows"]:
                group.update(status="unavailable", reason="no complete single-phase windows")
                continue
            batches = []
            for episode_id, start in group["selected_windows"]:
                if episode_id not in allowed:
                    raise ValueError("cohort attempts to read an excluded episode")
                if episode_id not in cache:
                    check_budget()
                    cache[episode_id] = store.read_episode(episode_id)
                    report["decoded_episode_ids"].append(episode_id)
                batches.append(cache[episode_id].sequence(start, group["horizon"]))
            batch = concatenate_batches(batches)
            metrics = model.diagnostics(batch)
            actions = batch.actions
            shuffled = np.roll(actions.reshape(-1, 14), 1, axis=0).reshape(actions.shape)
            pixels = batch.observations[model.config["camera"]].astype(np.float32) / 255
            motion = np.diff(pixels, axis=1)
            group.update(
                status="measured",
                metrics=dict(metrics),
                prediction_persistence_ratio=metrics["prediction_mse"]
                / max(metrics["persistence_mse"], 1e-12),
                shuffled_relative_improvement=(
                    metrics["shuffled_action_mse"] - metrics["prediction_mse"]
                )
                / max(metrics["shuffled_action_mse"], 1e-12),
                consecutive_rgb_rms=float(np.sqrt(np.mean(motion**2))),
                action_channel_std=actions.reshape(-1, 14).std(axis=0).tolist(),
                shuffled_action_rms=float(np.sqrt(np.mean((actions - shuffled) ** 2))),
                shuffled_command_fraction=float(
                    np.any(np.abs(actions - shuffled) > 1e-6, axis=-1).mean()
                ),
            )
            write(args.output / "report.json", report | {"timing": clocks()})
        if digest(args.checkpoint) != CHECKPOINT_SHA or store.manifest_hash != DATASET_SHA:
            raise ValueError("frozen artifacts changed during diagnosis")
        store.verify()
        after = source_identity()
        if any(after[key] != report["source"][key] for key in ("revision", "python_source_sha256")):
            raise ValueError("source revision or Python implementation changed during diagnosis")
        if model.implementation_sha256 != report["implementation_sha256"]:
            raise ValueError("model implementation changed during diagnosis")
        if (
            digest(Path(__file__)) != report["script_sha256"]
            or digest(ROOT / "docs/experiments/apple_sensor_diagnostics_v1.md")
            != report["protocol_sha256"]
        ):
            raise ValueError("diagnostic script or protocol changed during diagnosis")
        report["integrity_verified_after"] = True
        report["status"] = "completed"
    except Exception as error:
        report.update(status="incomplete", error=f"{type(error).__name__}: {error}")
    finally:
        report["timing"] = clocks()
        write(args.output / "report.json", report)
    return 0 if report["status"] == "completed" else 2


def main():
    wall, mono = ENTRY_CLOCKS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/apple-task-v1")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "checkpoints/apple-sensor-v1/sensor.pt"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite diagnostic")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset",
        str(args.dataset.resolve()),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--output",
        str(args.output.resolve()),
    ]
    env = os.environ | {
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    process = subprocess.Popen(command, cwd=ROOT, env=env)
    timeout = False
    try:
        while process.poll() is None:
            if max(time.time() - wall, time.monotonic() - mono) >= 120:
                process.kill()
                timeout = True
                break
            time.sleep(0.1)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        args.output.mkdir(parents=True, exist_ok=True)
        report_path = args.output / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {"groups": []}
        if timeout or process.returncode:
            report.update(
                status="incomplete", supervisor_timeout=timeout, returncode=process.returncode
            )
            for group in report["groups"]:
                if group["status"] == "planned":
                    group["status"] = "not_completed_budget_or_failure"
            write(report_path, report)
        write(
            args.output / "supervisor.json",
            {
                "returncode": process.returncode,
                "timeout": timeout,
                "true_wall_seconds": max(time.time() - wall, time.monotonic() - mono),
                "max_wall_seconds": 120,
                "max_child_cpu_seconds": 60,
            },
        )
    return (
        0 if not process.returncode and not timeout and report.get("status") == "completed" else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
