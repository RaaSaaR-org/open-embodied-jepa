"""Run the declared final cohort serially; retain failures and incomplete attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def observed_episodes(directory):
    path = Path(directory) / "episodes.jsonl"
    seeds, errors = [], []
    if path.exists():
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
                seed = row["seed"]
                if type(seed) is not int or seed in seeds:
                    raise ValueError("invalid or duplicate seed")
                seeds.append(seed)
            except (ValueError, KeyError, TypeError) as error:
                errors.append({"line": line_number, "error": str(error)})
    return {
        "expected_episode_seeds": list(range(20000, 20050)),
        "recorded_episode_seeds": seeds,
        "unrecorded_episode_seeds": sorted(set(range(20000, 20050)) - set(seeds)),
        "record_errors": errors,
        "summary_present": (Path(directory) / "summary.json").is_file(),
    }


def stop_process(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/mvp-v0-evaluation")
    parser.add_argument("--max-seconds", type=float, default=1800)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 1800:
        parser.error("budget must be positive and at most 1800 seconds")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    start_wall, start_active = time.time(), time.monotonic()
    planned = []
    for seed in range(3):
        for backend in ("native_jepa", "leworldmodel"):
            suffix = "" if seed == 0 else f"_seed{seed}"
            stem = "mvp_common" if seed == 0 else f"mvp{suffix}"
            if backend == "leworldmodel":
                stem = "mvp_lewm" if seed == 0 else f"{stem}_lewm"
            planned.append(
                {
                    "run_id": f"mvp-v0-{backend}-seed{seed}",
                    "config": f"configs/{stem}.yaml",
                    "policy": "model",
                }
            )
    planned += [
        {"run_id": f"mvp-v0-{policy}", "config": "configs/mvp_common.yaml", "policy": policy}
        for policy in ("hold", "random")
    ]
    protocol = ROOT / "docs/experiments/mvp_evaluation.md"
    report = {
        "schema_version": 1,
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "max_seconds": args.max_seconds,
        "planned": planned,
        "attempts": [],
        "complete": False,
    }

    def save():
        report["elapsed_seconds"] = max(time.time() - start_wall, time.monotonic() - start_active)
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "report.json")

    save()  # Record every intended run before launching any model evaluation.
    for trial in planned:
        remaining = args.max_seconds - max(
            time.time() - start_wall, time.monotonic() - start_active
        )
        if remaining <= 0:
            break
        command = [
            sys.executable,
            "-m",
            "embodied_jepa.benchmark",
            "--config",
            trial["config"],
            "--policy",
            trial["policy"],
            "--run-id",
            trial["run_id"],
        ]
        attempt = trial | {"command": command, "status": "running"}
        report["attempts"].append(attempt)
        save()
        process = None
        try:
            with (output / f"{trial['run_id']}.log").open("x") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    elapsed = max(time.time() - start_wall, time.monotonic() - start_active)
                    if elapsed >= args.max_seconds:
                        attempt["status"] = "wall_budget"
                        stop_process(process)
                        break
                    time.sleep(0.25)
                else:
                    attempt["status"] = "completed" if process.returncode == 0 else "failed"
                attempt["returncode"] = process.returncode
        except Exception as error:
            attempt.update(status="orchestration_error", error=f"{type(error).__name__}: {error}")
        except BaseException as error:
            attempt.update(status="interrupted", error=type(error).__name__)
            raise
        finally:
            stop_process(process)
            attempt["episode_accounting"] = observed_episodes(ROOT / "outputs" / trial["run_id"])
            accounting = attempt["episode_accounting"]
            if attempt["status"] == "completed" and (
                accounting["unrecorded_episode_seeds"]
                or accounting["record_errors"]
                or not accounting["summary_present"]
            ):
                attempt["status"] = "incomplete_artifacts"
            report["unattempted"] = planned[len(report["attempts"]) :]
            save()
        print(json.dumps(attempt), flush=True)
    report["unattempted"] = planned[len(report["attempts"]) :]
    report["complete"] = not report["unattempted"] and all(
        a["status"] == "completed" for a in report["attempts"]
    )
    save()
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
