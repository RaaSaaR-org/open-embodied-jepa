"""Bounded, read-only matched-action diagnostics; no TEST decoding or model fitting."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ENTRY = time.time(), time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/experiments/apple_branch_diagnostics_v1.md"
BASELINE_DATA_SHA = "d70edd9daaeec58e80a4e6a743980c049335a03f2b32b7d42bbac5ec5bc579df"
BASELINE_CHECKPOINT_SHA = "3a5c5e7ba77556b6ce4251f05af72dc86df0121807d913ca8b444db60026e3ee"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def elapsed():
    return max(time.time() - ENTRY[0], time.monotonic() - ENTRY[1])


def identities(args):
    files = {
        "branches": args.dataset / "meta/jepa_manifest.json",
        "branch_report": args.dataset / "branch_report.json",
        "baseline_dataset": args.baseline_dataset / "meta/jepa_manifest.json",
        "baseline_checkpoint": args.baseline_checkpoint,
        "new_checkpoint": args.new_checkpoint,
        "protocol": PROTOCOL,
        "script": Path(__file__),
        "checkpoint_helper": ROOT / "scripts/evaluate_apple.py",
    }
    result = {key: digest(path) for key, path in files.items()}
    hasher = hashlib.sha256()
    for path in sorted((ROOT / "src/embodied_jepa").rglob("*.py")):
        hasher.update(str(path.relative_to(ROOT / "src/embodied_jepa")).encode())
        hasher.update(path.read_bytes())
    result["python_source_sha256"] = hasher.hexdigest()
    result["source_revision"] = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    return result


def roots_from_report(report, manifest):
    """Metadata-only cohort construction; never inspect a TEST payload."""
    rows = {r["episode_id"]: r for r in manifest["episodes"]}
    roots = {}
    for attempt in report["attempts"]:
        split, parent = attempt["split"], attempt["parent_episode_id"]
        if split not in ("train", "val") or parent not in manifest["splits"][split]:
            raise ValueError("branch parent is not in its declared TRAIN/VAL partition")
        if rows[parent]["session_id"] != attempt["session_id"]:
            raise ValueError("branch session differs from parent")
        root = roots.setdefault(
            attempt["root_id"],
            {
                key: attempt[key]
                for key in (
                    "root_id",
                    "split",
                    "phase",
                    "parent_episode_id",
                    "session_id",
                )
            }
            | {"attempts": [], "status": "planned"},
        )
        if any(
            root[k] != attempt[k] for k in ("split", "phase", "parent_episode_id", "session_id")
        ):
            raise ValueError("inconsistent sibling root identity")
        if any(r["branch"] == attempt["branch"] for r in root["attempts"]):
            raise ValueError("duplicate branch identity")
        episode_id = attempt["episode_id"]
        if episode_id in rows:
            row = rows[episode_id]
            meta = row["metadata"]
            if (
                episode_id not in manifest["splits"][split]
                or row["session_id"] != root["session_id"]
                or meta["root_id"] != root["root_id"]
                or meta["parent_episode_id"] != parent
                or meta["branch"] != attempt["branch"]
            ):
                raise ValueError("stored branch lineage/split mismatch")
        root["attempts"].append(dict(attempt, stored=episode_id in rows))
    return list(roots.values())


def same_start(episodes, horizon):
    import numpy as np

    first = episodes[0]
    for episode in episodes[1:]:
        if set(first.observations) != set(episode.observations):
            raise ValueError("siblings have different cameras")
        arrays = [
            (first.observations[k][0], episode.observations[k][0]) for k in first.observations
        ]
        arrays += [
            (getattr(first, k)[0], getattr(episode, k)[0])
            for k in ("robot_states", "state_mask", "timestamps")
        ]
        arrays += [
            (
                first.timestamps[: horizon + 1] - first.timestamps[0],
                episode.timestamps[: horizon + 1] - episode.timestamps[0],
            )
        ]
        if any(not np.array_equal(a, b) for a, b in arrays):
            raise ValueError("siblings must have identical starting sensors and aligned timestamps")


def decision(own, wrong):
    tie = abs(own - wrong) <= 1e-10 + 1e-6 * max(abs(own), abs(wrong))
    return (0.5 if tie else float(own < wrong)), int(tie)


def pair_metrics(costs, raw_errors, proprio_errors, images, states, actions, persistence):
    import numpy as np

    rows = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            rgb = float(np.sqrt(np.mean(((images[i].astype(float) - images[j]) / 255) ** 2)))
            action = float(np.sqrt(np.mean((actions[i] - actions[j]) ** 2)))
            left, tie_left = decision(float(costs[i, i]), float(costs[j, i]))
            right, tie_right = decision(float(costs[j, j]), float(costs[i, j]))
            rows.append(
                dict(
                    left=i,
                    right=j,
                    rgb_rms=rgb,
                    action_rms=action,
                    state_rms=float(np.sqrt(np.mean((states[i] - states[j]) ** 2))),
                    informative=rgb >= 1 / 255 and action > 1e-8,
                    ranking=(left + right) / 2,
                    ties=tie_left + tie_right,
                    own=float((raw_errors[i, i] + raw_errors[j, j]) / 2),
                    wrong=float((raw_errors[j, i] + raw_errors[i, j]) / 2),
                    proprio_own=float((proprio_errors[i, i] + proprio_errors[j, j]) / 2),
                    proprio_wrong=float((proprio_errors[j, i] + proprio_errors[i, j]) / 2),
                    persistence=float((persistence[i] + persistence[j]) / 2),
                )
            )
    return rows


def aggregate(rows, *, bootstrap=2000):
    import numpy as np

    informative = []
    for row in rows:
        pairs = [p for p in row["pairs"] if p["informative"]]
        if pairs:
            informative.append(
                (
                    row["session_id"],
                    [np.mean([p[k] for p in pairs]) for k in ("ranking", "own", "wrong")],
                )
            )
    result = dict(
        roots=len(rows),
        informative_roots=len(informative),
        pairs=sum(len(r["pairs"]) for r in rows),
        informative_pairs=sum(p["informative"] for r in rows for p in r["pairs"]),
        ties=sum(p["ties"] for r in rows for p in r["pairs"] if p["informative"]),
        gate_passed=False,
    )
    if not informative:
        return result | {
            "accuracy": None,
            "error_reduction": None,
            "reason": "no informative roots",
        }
    values = np.array([x[1] for x in informative])
    mean = values.mean(0)

    def ratio(x):
        return float(1 - x[1] / x[2]) if x[2] > 0 else None

    rng = np.random.default_rng(20260921)
    sampled = values[rng.integers(len(values), size=(bootstrap, len(values)))].mean(1)
    reduction = np.divide(
        sampled[:, 1],
        sampled[:, 2],
        out=np.full(bootstrap, np.nan),
        where=sampled[:, 2] > 0,
    )

    def ci(x):
        return np.quantile(x, [0.025, 0.975]).tolist() if np.isfinite(x).all() else None

    sessions = sorted({x[0] for x in informative})
    clustered = []
    for selection in rng.integers(len(sessions), size=(bootstrap, len(sessions))):
        sample = np.concatenate(
            [values[[x[0] == sessions[i] for x in informative]] for i in selection]
        )
        avg = sample.mean(0)
        clustered.append([avg[0], ratio(avg)])
    cluster = np.array(clustered, dtype=float)
    result.update(
        accuracy=float(mean[0]),
        error_reduction=ratio(mean),
        own_visual_mse=float(mean[1]),
        wrong_visual_mse=float(mean[2]),
        accuracy_ci95=ci(sampled[:, 0]),
        error_reduction_ci95=ci(1 - reduction),
        parent_sessions=len(sessions),
        parent_accuracy_ci95=ci(cluster[:, 0]),
        parent_error_reduction_ci95=ci(cluster[:, 1]),
    )
    result["gate_passed"] = bool(
        len(values) >= 6
        and len(sessions) >= 2
        and mean[0] >= 0.70
        and result["error_reduction"] is not None
        and result["error_reduction"] >= 0.10
        and result["error_reduction_ci95"] is not None
        and result["error_reduction_ci95"][0] > 0
    )
    return result


def evaluate_root(model, episodes, horizon):
    import numpy as np
    import torch

    from embodied_jepa.contracts import RobotState

    same_start(episodes, horizon)
    first = episodes[0]
    initial = {k: v[0:1] for k, v in first.observations.items()}
    state = RobotState(
        first.robot_states[0:1],
        first.state_mask[0:1],
        first.timestamps[0:1],
        first.state_schema,
    )
    actions = np.stack([e.actions[:horizon] for e in episodes])
    with torch.no_grad():
        latent = model.encode(initial, state)
        predicted = model.predict(latent, actions[None])
        terminal = predicted.values[0, :, -1]
        n = model.visual_dimension
        observed_images = np.stack(
            [e.observations[model.config["camera"]][horizon] for e in episodes]
        )
        images = {model.config["camera"]: observed_images}
        pixels, _ = model.pixels(images)
        observed = pixels.flatten(1)
        raw_predicted = terminal[:, :n] * model.sensor_scale[:n] + model.sensor_mean[:n]
        raw_errors = (raw_predicted[:, None] - observed[None]).square().mean(-1).cpu().numpy()
        measured = np.stack([e.robot_states[horizon] for e in episodes])
        norm_state = (
            torch.from_numpy(measured.copy()) - model.sensor_mean[n:]
        ) / model.sensor_scale[n:]
        proprio_errors = (terminal[:, None, n:] - norm_state[None]).square().mean(-1).cpu().numpy()
        start_pixels, _ = model.pixels(initial)
        persistence = (start_pixels.flatten(1) - observed).square().mean(-1).cpu().numpy()
        costs = np.stack(
            [
                model.distance(
                    predicted,
                    model.encode_goal({model.config["camera"]: observed_images[j : j + 1]}),
                )[0, :, -1]
                for j in range(len(episodes))
            ],
            axis=1,
        )
    return pair_metrics(
        costs,
        raw_errors,
        proprio_errors,
        observed_images,
        measured,
        actions,
        persistence,
    )


def invalidate(report, reason):
    report.update(status="incomplete", gate_passed=False, error=reason)
    for summary in report.get("summaries", []):
        summary["gate_passed"] = False
        for phase in summary.get("phases", {}).values():
            phase["gate_passed"] = False
    return report


def worker(args):
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    registration = json.loads((args.output / "registration.json").read_text())
    report = {
        "status": "running",
        "environment": registration["environment"],
        "identity": registration["identity"],
        "test_images_decoded": False,
        "roots": registration["roots"],
        "results": [],
        "decoded_episode_ids": [],
        "models": {},
        "gate_passed": False,
    }
    write(args.output / "report.json", report)
    try:
        import torch
        from evaluate_apple import checkpoint_model

        from embodied_jepa.data import DatasetStore

        torch.set_num_threads(1)

        def budget():
            if elapsed() >= 110 or time.process_time() >= 55:
                raise TimeoutError("matched diagnostic budget exhausted")

        if identities(args) != registration["identity"]:
            raise ValueError("registered inputs changed before worker")
        if (
            registration["identity"]["baseline_dataset"] != BASELINE_DATA_SHA
            or registration["identity"]["baseline_checkpoint"] != BASELINE_CHECKPOINT_SHA
        ):
            raise ValueError("original frozen baseline identity mismatch")
        store = DatasetStore(args.dataset)
        baseline_store, baseline = checkpoint_model(args.baseline_dataset, args.baseline_checkpoint)
        new_store, new = checkpoint_model(args.dataset, args.new_checkpoint)
        models = {"original": baseline, "branches": new}
        if baseline.backend != "sensor_wm" or new.backend != "sensor_wm":
            raise ValueError("diagnostic supports the explicit sensor backend only")
        if (
            baseline.state_schema != new.state_schema
            or baseline_store.manifest["action_manifest"] != store.manifest["action_manifest"]
        ):
            raise ValueError("source corpora do not share state/action contracts")
        for name, model in models.items():
            report["models"][name] = {
                "normalization": model.metadata["normalization"],
                "dataset_hash": model.metadata["dataset_hash"],
                "implementation_sha256": model.implementation_sha256,
            }
        for root in report["roots"]:
            budget()
            episodes = []
            for attempt in root["attempts"]:
                if attempt["stored"]:
                    episode = store.read_episode(attempt["episode_id"])
                    episodes.append(episode)
                    report["decoded_episode_ids"].append(episode.episode_id)
            for horizon in (8, 16):
                eligible = [e for e in episodes if len(e.actions) >= horizon]
                for name, model in models.items():
                    budget()
                    row = {k: root[k] for k in ("root_id", "split", "phase", "session_id")}
                    row.update(
                        model=name,
                        horizon=horizon,
                        eligible_branches=len(eligible),
                        planned_branches=len(root["attempts"]),
                        episode_ids=[e.episode_id for e in eligible],
                        pairs=[],
                    )
                    if len(eligible) >= 2:
                        row["pairs"] = evaluate_root(model, eligible, horizon)
                    report["results"].append(row)
            root["status"] = "completed"
            write(args.output / "report.json", report)
        report["summaries"] = []
        for name in models:
            for split in ("train", "val"):
                for horizon in (8, 16):
                    selected = [
                        r
                        for r in report["results"]
                        if r["model"] == name and r["split"] == split and r["horizon"] == horizon
                    ]
                    summary = aggregate(selected)
                    summary.update(
                        model=name,
                        split=split,
                        horizon=horizon,
                        primary=(split == "val" and horizon == 16),
                    )
                    summary["phases"] = {
                        phase: aggregate([r for r in selected if r["phase"] == phase])
                        for phase in sorted({r["phase"] for r in selected})
                    }
                    report["summaries"].append(summary)
                    budget()
        for dataset in (store, baseline_store, new_store):
            dataset.verify()
        if identities(args) != registration["identity"]:
            raise ValueError("registered source/checkpoint/corpus/protocol drift")
        primary = next(s for s in report["summaries"] if s["model"] == "branches" and s["primary"])
        report.update(status="completed", gate_passed=primary["gate_passed"])
    except Exception as exc:
        invalidate(report, f"{type(exc).__name__}: {exc}")
    finally:
        report["true_wall_seconds"] = elapsed()
        report["process_cpu_seconds"] = time.process_time()
        report["completed_roots"] = sum(r["status"] == "completed" for r in report["roots"])
        report["planned_roots"] = len(report["roots"])
        write(args.output / "report.json", report)
    return 0 if report["status"] == "completed" else 2


def wait_for_worker(process):
    """Recheck civil and monotonic clocks after wakeups, including host suspension."""
    while process.poll() is None:
        if elapsed() >= 115:
            process.kill()
            process.wait()
            return True
        time.sleep(0.1)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-dataset", type=Path, default=ROOT / "data/apple-task-v1")
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/apple-sensor-v1/sensor.pt",
    )
    parser.add_argument("--new-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key in (
        "dataset",
        "baseline_dataset",
        "baseline_checkpoint",
        "new_checkpoint",
        "output",
    ):
        setattr(args, key, getattr(args, key).resolve())
    if args.worker:
        return worker(args)
    if args.output.exists():
        raise FileExistsError("refusing existing diagnostic output")
    identity = identities(args)
    manifest = json.loads((args.dataset / "meta/jepa_manifest.json").read_text())
    collection = json.loads((args.dataset / "branch_report.json").read_text())
    if collection.get("dataset_sha256") != identity["branches"]:
        raise ValueError("branch report does not identify the frozen dataset")
    roots = roots_from_report(collection, manifest)
    args.output.mkdir(parents=True)
    registration = {
        "identity": identity,
        "inputs": {
            key: str(getattr(args, key))
            for key in ("dataset", "baseline_dataset", "baseline_checkpoint", "new_checkpoint")
        },
        "roots": roots,
        "planned_roots": len(roots),
        "max_wall_seconds": 120,
        "worker_kill_wall_seconds": 115,
        "finalization_reserve_seconds": 5,
        "max_child_cpu_seconds": 60,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": 1,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "pyarrow", "pandas", "Pillow")
            },
        },
        "test_images_decoded": False,
        "source_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    write(args.output / "registration.json", registration)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    for key in (
        "dataset",
        "baseline_dataset",
        "baseline_checkpoint",
        "new_checkpoint",
        "output",
    ):
        command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
    timeout = False
    with (args.output / "child.log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ
            | {
                "PYTHONPATH": str(ROOT / "src"),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        timeout = wait_for_worker(process)
    path = args.output / "report.json"
    report = json.loads(path.read_text()) if path.exists() else {"roots": roots, "results": []}
    try:
        drift = identities(args) != identity
    except Exception:
        drift = True
    if timeout or process.returncode or drift or report.get("status") != "completed":
        invalidate(report, "supervisor timeout, child failure, incomplete report or input drift")
        report.update(supervisor_timeout=timeout, returncode=process.returncode, input_drift=drift)
        write(path, report)
    write(
        args.output / "supervisor.json",
        {
            "true_wall_seconds": elapsed(),
            "timeout": timeout,
            "returncode": process.returncode,
            "planned_roots": len(roots),
            "completed_roots": sum(r["status"] == "completed" for r in report["roots"]),
        },
    )
    return 0 if report.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
