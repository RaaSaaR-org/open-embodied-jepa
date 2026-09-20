"""Supervise the fixed apple-sensor-v1 training attempt; no retry or budget extension."""

from __future__ import annotations

import time

ENTRY_WALL, ENTRY_MONOTONIC = time.time(), time.perf_counter()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/experiments/apple_sensor_training_v1.md"
MAX_SECONDS = 1800.0
FIXED = {
    "backend": "sensor_wm",
    "seed": 0,
    "device": "cpu",
    "cpu_threads": 4,
    "steps": 3000,
    "batch_size": 16,
    "horizon": 8,
    "validation_every": 300,
    "validation_batches": 4,
    "selection": "raw_mse",
    "memory_limit_gib": 16,
}

# This child imports the actual model under the parent's existing time budget.
# It resolves configuration only; it cannot decode episodes or optimize weights.
RESOLVE_CONFIG = """
import json, sys
from dataclasses import asdict
from pathlib import Path
import torch
from embodied_jepa.contracts import StateSchema
from embodied_jepa.models.sensor import SensorWorldModel
torch.set_num_threads(4)
manifest = json.loads(Path(sys.argv[1]).read_text())
schema = StateSchema(**manifest['state_schema'])
model = SensorWorldModel(schema, device='cpu', seed=0)
value = {'backend': model.backend, 'config': model.config,
         'state_schema': asdict(schema), 'implementation_sha256': model.implementation_sha256,
         'source_revision': model.source_revision}
path = Path(sys.argv[2])
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\\n')
temporary.replace(path)
"""


class Clock:
    def __init__(self, *, include_entry=False):
        self.wall = ENTRY_WALL if include_entry else time.time()
        self.monotonic = ENTRY_MONOTONIC if include_entry else time.perf_counter()

    def snapshot(self):
        wall, monotonic = time.time() - self.wall, time.perf_counter() - self.monotonic
        return {
            "elapsed_seconds": max(0.0, wall, monotonic),
            "wall_clock_elapsed_seconds": wall,
            "monotonic_elapsed_seconds": monotonic,
        }

    def remaining(self):
        return max(0.0, MAX_SECONDS - self.snapshot()["elapsed_seconds"])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity(root):
    source = root / "src/embodied_jepa"
    paths = sorted(source.rglob("*.py"))
    if not paths:
        raise ValueError("Python source tree is missing")
    value = hashlib.sha256()
    for path in paths:
        value.update(str(path.relative_to(source)).encode())
        value.update(path.read_bytes())
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True, timeout=10
        ).strip()
    )
    return {"revision": revision, "dirty": dirty, "python_source_sha256": value.hexdigest()}


def verify(root, dataset, expected):
    source = source_identity(root)
    actual = {
        "source_sha256": source["python_source_sha256"],
        "dataset_sha256": digest(dataset / "meta/jepa_manifest.json"),
        "protocol_sha256": digest(root / PROTOCOL),
        "runner_sha256": digest(Path(__file__)),
    }
    for name, value in expected.items():
        if actual[name] != value:
            raise ValueError(f"frozen {name} changed")
    return source


def supervise(command, log_path, clock, root, *, on_start=None):
    """Use one outer deadline across setup/imports/training/finalization phases."""
    if clock.remaining() <= 0:
        return {"started": False, "timeout": True, "returncode": None}
    environment = os.environ | {
        "PYTHONPATH": str(root / "src"),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    with log_path.open("x") as stream:
        process = subprocess.Popen(
            command, cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT
        )
        timeout = False
        try:
            if on_start is not None:
                on_start()
            while process.poll() is None:
                remaining = clock.remaining()
                if remaining <= 0:
                    timeout = True
                    process.kill()
                    break
                time.sleep(min(0.1, remaining))
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    return {"started": True, "timeout": timeout, "returncode": process.returncode}


def run(
    dataset,
    output,
    expected_source_sha256,
    expected_dataset_sha256,
    expected_protocol_sha256,
    *,
    root=ROOT,
    clock=None,
):
    clock = clock or Clock()
    root, dataset, output = Path(root).resolve(), Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    expected = {
        "source_sha256": expected_source_sha256,
        "dataset_sha256": expected_dataset_sha256,
        "protocol_sha256": expected_protocol_sha256,
        "runner_sha256": digest(Path(__file__)),
    }
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in expected.values()):
        raise ValueError("expected identities must be lowercase SHA-256 digests")
    output.mkdir(parents=True)
    report_path = output / "supervisor.json"
    report = {
        "schema_version": 1,
        "status": "setup",
        "fixed": FIXED,
        "expected": expected,
        "planned_runs": 1,
        "started_runs": 0,
        "completed_runs": 0,
        "max_seconds": MAX_SECONDS,
        "dataset": str(dataset),
        "output": str(output),
        "test_samples_loaded_by_supervisor": False,
    }
    write(report_path, report)
    registered = None
    try:
        before = verify(root, dataset, expected)
        report["source_before"] = before
        manifest = json.loads((dataset / "meta/jepa_manifest.json").read_text())
        if not manifest.get("splits", {}).get("train") or not manifest["splits"].get("val"):
            raise ValueError("sealed nonempty training/validation splits are required")
        resolved_path = output / "resolved_model.json"
        config_command = [
            sys.executable,
            "-c",
            RESOLVE_CONFIG,
            str(dataset / "meta/jepa_manifest.json"),
            str(resolved_path),
        ]
        report["configuration_process"] = supervise(
            config_command, output / "configuration.log", clock, root
        )
        if report["configuration_process"]["timeout"]:
            report["status"] = "not_started_total_budget"
        elif report["configuration_process"]["returncode"] != 0:
            report["status"] = "configuration_failed"
        else:
            resolved = json.loads(resolved_path.read_text())
            if resolved["backend"] != "sensor_wm":
                raise ValueError("resolved the wrong model backend")
            config_path = output / "model_config.json"
            write(config_path, resolved["config"])
            registered = {resolved_path: digest(resolved_path), config_path: digest(config_path)}
            checkpoint = output / "sensor.pt"
            command = [
                sys.executable,
                "-m",
                "embodied_jepa.training",
                "--dataset",
                str(dataset),
                "--output",
                str(checkpoint),
                "--model-config",
                str(config_path),
            ]
            for key, value in FIXED.items():
                if key != "cpu_threads":
                    command.extend(("--" + key.replace("_", "-"), str(value)))
            command.extend(("--max-seconds", str(clock.remaining())))
            registration = {
                "schema_version": 1,
                "experiment": "apple_sensor_training_v1",
                "source": before,
                "expected": expected,
                "model": resolved,
                "model_config_sha256": digest(config_path),
                "fixed": FIXED,
                "dataset": str(dataset),
                "output": str(checkpoint),
                "split_hash": json_hash(
                    {"splits": manifest["splits"], "policy": manifest.get("split_policy")}
                ),
                "action_hash": json_hash(manifest["action_manifest"]),
                "command": command,
                "timing_before_training": clock.snapshot(),
            }
            registration_path = output / "registration.json"
            write(registration_path, registration)
            registered[registration_path] = digest(registration_path)
            report["registration_sha256"] = registered[registration_path]
            report["command"] = command
            verify(root, dataset, expected)
            report["status"] = "attempted"
            write(report_path, report)

            def started():
                report["started_runs"] = 1
                write(report_path, report)

            child = supervise(command, output / "training.log", clock, root, on_start=started)
            report["training_process"] = child
            report["started_runs"] = int(child["started"])
            child_path = checkpoint.with_suffix(".run.json")
            if child_path.exists():
                child_report = json.loads(child_path.read_text())
                report["child_report"] = {
                    "path": str(child_path),
                    "sha256": digest(child_path),
                    "status": child_report["status"],
                    "completed_steps": child_report.get("completed_steps"),
                    "best_step": child_report.get("best_step"),
                    "test_samples_loaded": child_report.get("test_samples_loaded"),
                }
                if child_report.get("test_samples_loaded") is not False:
                    raise ValueError("child does not certify test-split exclusion")
                provenance = child_report.get("provenance", {})
                for name, expected_value in (
                    ("dataset_hash", expected_dataset_sha256),
                    ("split_hash", registration["split_hash"]),
                    ("action_hash", registration["action_hash"]),
                    ("source_tree_hash", expected_source_sha256),
                ):
                    if name in provenance and provenance[name] != expected_value:
                        raise ValueError(f"child {name} provenance changed")
                if child_report["status"] == "completed":
                    required_provenance = {
                        "dataset_hash",
                        "split_hash",
                        "action_hash",
                        "source_tree_hash",
                    }
                    if (
                        not required_provenance <= provenance.keys()
                        or child_report.get("model_config") != resolved["config"]
                    ):
                        raise ValueError("completed child lacks registered model/provenance")
                    for name in (
                        "steps",
                        "batch_size",
                        "horizon",
                        "validation_every",
                        "validation_batches",
                    ):
                        if child_report.get("budget", {}).get(name) != FIXED[name]:
                            raise ValueError(f"child {name} budget changed")
                    if (
                        not checkpoint.exists()
                        or not checkpoint.with_name("sensor.latest.pt").exists()
                    ):
                        raise ValueError("completed child is missing best/latest checkpoint")
            report["status"] = (
                "supervisor_timeout"
                if child["timeout"] and child["started"]
                else "not_started_total_budget"
                if not child["started"]
                else "process_failed"
                if child["returncode"] != 0
                else "completed"
                if report.get("child_report", {}).get("status") == "completed"
                and report["child_report"]["completed_steps"] == FIXED["steps"]
                else "incomplete_child_report"
            )
        after = verify(root, dataset, expected)
        report["source_after"] = after
        if before["revision"] != after["revision"]:
            raise ValueError("Git revision changed during experiment")
        if registered:
            for path, expected_hash in registered.items():
                if digest(path) != expected_hash:
                    raise ValueError(f"registered configuration changed: {path.name}")
    except Exception as error:
        report.update(
            status="failed_integrity_or_orchestration", error=f"{type(error).__name__}: {error}"
        )
    finally:
        report["completed_runs"] = int(report["status"] == "completed")
        report["artifacts"] = {
            path.name: digest(path)
            for path in output.iterdir()
            if path.is_file() and path.name != "supervisor.json" and not path.name.endswith(".tmp")
        }
        report["timing"] = clock.snapshot()
        write(report_path, report)
    return report


def main():
    clock = Clock(include_entry=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    report = run(
        args.dataset,
        args.output,
        args.source_sha256,
        args.dataset_sha256,
        args.protocol_sha256,
        clock=clock,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "timing": report["timing"],
                "counts": {
                    k: report[k] for k in ("planned_runs", "started_runs", "completed_runs")
                },
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
