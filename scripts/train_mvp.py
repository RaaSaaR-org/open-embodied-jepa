"""Execute the preregistered six-run CPU experiment with a shared wall-time cap."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from embodied_jepa.contracts import ContractError
from embodied_jepa.data import DatasetStore
from embodied_jepa.training import RunClock, source_identity

RUNS = [(backend, seed) for seed in (0, 1, 2) for backend in ("native_jepa", "leworldmodel")]
TOTAL_SECONDS = 18 * 60
RUN_SECONDS = 600


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def identity():
    root = Path(__file__).resolve().parents[1]
    return {
        "python_source_sha256": source_identity()["python_source_sha256"],
        "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(
            (root / "docs/experiments/mvp_final.md").read_bytes()
        ).hexdigest(),
    }


def stability_error(dataset, expected_dataset_hash, frozen):
    try:
        if identity() != frozen:
            return "source_changed"
        if DatasetStore(dataset).manifest_hash != expected_dataset_hash:
            return "dataset_changed"
    except Exception as error:
        return f"integrity_check_failed: {type(error).__name__}: {error}"
    return None


def supervise(command, log_path, max_seconds):
    """Watch wall AND monotonic time, including imports and child finalization."""
    clock = RunClock()
    environment = os.environ | {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    with log_path.open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        killed = False
        try:
            while process.poll() is None:
                if clock.elapsed() >= max_seconds:
                    process.kill()
                    killed = True
                    break
                time.sleep(min(0.1, max(0.001, max_seconds - clock.elapsed())))
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    return {
        "returncode": process.returncode,
        "supervisor_timeout": killed,
        "supervisor_timing": clock.snapshot(),
    }


def run(dataset, output, expected_dataset_hash):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment {output}")
    store = DatasetStore(dataset)
    if store.manifest_hash != expected_dataset_hash:
        raise ContractError("dataset manifest does not match the preregistered hash")
    frozen = identity()
    output.mkdir(parents=True)
    report_path = output / "experiment.json"
    report = {
        "format_version": 1,
        "status": "running",
        "dataset": str(dataset),
        "dataset_sha256": expected_dataset_hash,
        "source": source_identity(),
        "frozen_identity": frozen,
        "total_max_seconds": TOTAL_SECONDS,
        "per_run_max_seconds": RUN_SECONDS,
        "run_order": [{"backend": backend, "seed": seed} for backend, seed in RUNS],
        "runs": [],
    }
    clock = RunClock()
    _write(report_path, report)
    invalidated = None
    for backend, seed in RUNS:
        record = {"backend": backend, "seed": seed}
        remaining = TOTAL_SECONDS - clock.elapsed()
        if remaining <= 0:
            record["status"] = "not_started_total_budget"
        elif invalidated or (
            invalidated := stability_error(dataset, expected_dataset_hash, frozen)
        ):
            record.update(status="not_started_integrity_changed", error=invalidated)
        else:
            budget = min(RUN_SECONDS, TOTAL_SECONDS - clock.elapsed())
            if budget <= 0:
                record["status"] = "not_started_total_budget"
                report["runs"].append(record)
                continue
            checkpoint = output / f"seed-{seed}" / f"{backend}.pt"
            checkpoint.parent.mkdir(exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "embodied_jepa.training",
                "--dataset",
                str(dataset),
                "--backend",
                backend,
                "--output",
                str(checkpoint),
                "--seed",
                str(seed),
                "--steps",
                "3000",
                "--batch-size",
                "16",
                "--horizon",
                "4",
                "--device",
                "cpu",
                "--max-seconds",
                str(budget),
                "--validation-every",
                "100",
                "--validation-batches",
                "4",
                "--selection",
                "noncollapsed_relative",
                "--memory-limit-gib",
                "16",
            ]
            record.update(command=command, max_seconds=budget)
            try:
                record.update(supervise(command, checkpoint.with_suffix(".log"), budget))
                run_report = checkpoint.with_suffix(".run.json")
                if run_report.exists():
                    child = json.loads(run_report.read_text())
                    record.update(
                        child_status=child["status"],
                        completed_steps=child.get("completed_steps"),
                        best_step=child.get("best_step"),
                        best_selection_score=child.get("best_selection_score"),
                        run_report=str(run_report),
                    )
                record["status"] = (
                    "supervisor_timeout"
                    if record["supervisor_timeout"]
                    else record.get("child_status", "failed_without_report")
                )
                if record["returncode"] != 0 and record["status"] == "completed":
                    record["status"] = "failed_exit_after_report"
                invalidated = stability_error(dataset, expected_dataset_hash, frozen)
                if invalidated:
                    record.update(status="invalid_integrity_changed", error=invalidated)
                record["checkpoint_sha256"] = {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (
                        checkpoint,
                        checkpoint.with_name(checkpoint.stem + ".latest.pt"),
                    )
                    if path.exists()
                }
            except Exception as error:
                record.update(
                    status="orchestration_failed", error=f"{type(error).__name__}: {error}"
                )
        report["runs"].append(record)
        report["timing"] = clock.snapshot()
        _write(report_path, report)
    report["status"] = (
        "completed"
        if all(item["status"] == "completed" for item in report["runs"])
        else "incomplete"
    )
    report["timing"] = clock.snapshot()
    _write(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.dataset, args.output, args.dataset_sha256)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
