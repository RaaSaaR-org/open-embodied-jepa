"""CUDA software smoke: train both backends twice on cuda, compare, reload, run MPC on cuda.

Each backend is trained twice with the same seed, in two separate processes, on an existing
dataset (for example the one ``reproduce_smoke.py`` collected). The check passes only if the two
runs' per-step training metrics, validation metrics and final weights are bit-identical, which
is what the deterministic CUDA setup in ``embodied_jepa.devices`` is for. The first run's best
checkpoint of each backend is then reloaded by a short closed-loop MuJoCo MPC evaluation with
``runtime.device: cuda``, through the same backend-only config swap as ``reproduce_smoke.py``.

This is a software check, not research acceptance: 100 updates, raw-loss selection, two
five-step episodes. It refuses to overwrite ``outputs/<name>`` or ``checkpoints/<name>``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("native_jepa", "leworldmodel")
RUNS = ("a", "b")


def compare_runs(first: Path, second: Path) -> dict:
    """Bit-for-bit comparison of two training runs' curves and final weights."""
    import torch

    def events(path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        train = [row["metrics"] for row in rows if row["kind"] == "train"]
        validation = [row["horizons"] for row in rows if row["kind"] == "validation"]
        return train, validation

    train_a, validation_a = events(first.with_name(first.stem + ".metrics.jsonl"))
    train_b, validation_b = events(second.with_name(second.stem + ".metrics.jsonl"))
    weights = {}
    for label, suffix in (("best", ".pt"), ("latest", ".latest.pt")):
        state_a = torch.load(first.with_name(first.stem + suffix), weights_only=True)["weights"]
        state_b = torch.load(second.with_name(second.stem + suffix), weights_only=True)["weights"]
        weights[label] = state_a.keys() == state_b.keys() and all(
            torch.equal(state_a[key], state_b[key]) for key in state_a
        )
    return {
        "train_steps": [len(train_a), len(train_b)],
        "train_metrics_identical": bool(train_a) and train_a == train_b,
        "validation_metrics_identical": bool(validation_a) and validation_a == validation_b,
        "best_weights_identical": weights["best"],
        "latest_weights_identical": weights["latest"],
        "final_train_loss": [train_a[-1]["loss"], train_b[-1]["loss"]] if train_a else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--name", default="cuda-smoke")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be a directory name")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    dataset = args.dataset.resolve()
    output = ROOT / "outputs" / args.name
    checkpoints = ROOT / "checkpoints" / args.name
    if checkpoints.exists():
        raise SystemExit(f"refusing to overwrite {checkpoints}")
    output.mkdir(parents=True, exist_ok=False)

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    report = {
        "purpose": __doc__,
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "dataset": str(dataset),
        "commands": [],
        "complete": False,
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def execute(label, command):
        with (output / f"{label}.log").open("x") as log:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=False
            )
        report["commands"].append(
            {"label": label, "command": command, "returncode": result.returncode}
        )
        save()
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {output}")

    python = sys.executable
    report["comparisons"] = {}
    for backend in BACKENDS:
        for run in RUNS:
            execute(
                f"train-{backend}-{run}",
                [
                    python,
                    "-m",
                    "embodied_jepa.training",
                    "--dataset",
                    str(dataset),
                    "--backend",
                    backend,
                    "--output",
                    str(checkpoints / run / f"{backend}.pt"),
                    "--steps",
                    str(args.steps),
                    "--batch-size",
                    "8",
                    "--horizon",
                    "4",
                    "--seed",
                    "7",
                    "--device",
                    "cuda",
                    "--max-seconds",
                    "300",
                    "--validation-every",
                    "50",
                    "--validation-batches",
                    "1",
                    "--selection",
                    "raw_mse",
                ],
            )
        first, second = (checkpoints / run / f"{backend}.pt" for run in RUNS)
        comparison = compare_runs(first, second)
        run_report = json.loads(first.with_name(first.stem + ".run.json").read_text())
        comparison["environment"] = run_report["environment"]
        comparison["cuda_memory"] = {
            key: value for key, value in run_report.items() if "cuda" in key and "bytes" in key
        }
        comparison["update_seconds_note"] = "see the run's metrics.jsonl; not compared"
        report["comparisons"][backend] = comparison
        save()
    config = yaml.safe_load((ROOT / "configs/reach_pilot.yaml").read_text())
    config["seed"] = 7
    config["runtime"] = {"device": "cuda"}
    config["dataset"]["root"] = str(dataset)
    config["world_model"]["checkpoints"] = {
        b: str(checkpoints / RUNS[0] / f"{b}.pt") for b in BACKENDS
    }
    config["embodiment"]["action_manifest"] = str(ROOT / "configs/g1_sim_action.json")
    config["evaluation"] = {"seeds": [19000, 19001], "output_root": str(output)}
    config["task"]["max_steps"] = 5
    config["planner"].update(samples=16, iterations=2, elites=4)
    base = output / "native.yaml"
    base.write_text(yaml.safe_dump(config, sort_keys=False))
    external = output / "lewm.yaml"
    external.write_text("extends: native.yaml\nworld_model:\n  backend: leworldmodel\n")
    for backend, path in (("native_jepa", base), ("leworldmodel", external)):
        execute(
            f"evaluate-{backend}",
            [python, "-m", "embodied_jepa.benchmark", "--config", str(path), "--run-id", backend],
        )
        episodes = [
            json.loads(line)
            for line in (output / backend / "episodes.jsonl").read_text().splitlines()
        ]
        if len(episodes) != 2 or any(e["replans"] == 0 for e in episodes):
            raise RuntimeError(f"{backend} did not execute both model-planner integration episodes")
    identical = all(
        all(value for key, value in comparison.items() if key.endswith("_identical"))
        for comparison in report["comparisons"].values()
    )
    report["same_seed_runs_bit_identical"] = identical
    report["complete"] = True
    report["limitation"] = (
        f"{args.steps} updates, raw-loss selection, two five-step MPC episodes on cuda; "
        "software integration and determinism only, not a learned-control result"
    )
    save()
    print(json.dumps({"complete": True, "same_seed_runs_bit_identical": identical}))
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
