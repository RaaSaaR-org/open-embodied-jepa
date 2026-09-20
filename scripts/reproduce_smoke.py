"""Small real-simulation data→train→reload→backend-swap check, not research acceptance."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="clean-smoke")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be a directory name")
    output = ROOT / "outputs" / args.name
    output.mkdir(parents=True, exist_ok=False)
    dataset = ROOT / "data" / args.name
    checkpoints = ROOT / "checkpoints" / args.name
    report = {"purpose": __doc__, "commands": [], "complete": False}

    def execute(label, command):
        with (output / f"{label}.log").open("x") as log:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False
            )
        report["commands"].append(
            {"label": label, "command": command, "returncode": result.returncode}
        )
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {output}")

    python = sys.executable
    execute(
        "collect",
        [
            python,
            "-m",
            "embodied_jepa.collection",
            "--output",
            str(dataset),
            "--episodes",
            "12",
            "--steps",
            "30",
            "--seed",
            "5000",
        ],
    )
    for backend in ("native_jepa", "leworldmodel"):
        execute(
            f"train-{backend}",
            [
                python,
                "-m",
                "embodied_jepa.training",
                "--dataset",
                str(dataset),
                "--backend",
                backend,
                "--output",
                str(checkpoints / f"{backend}.pt"),
                "--steps",
                "100",
                "--batch-size",
                "8",
                "--horizon",
                "4",
                "--seed",
                "7",
                "--device",
                "cpu",
                "--max-seconds",
                "120",
                "--validation-every",
                "50",
                "--validation-batches",
                "1",
                "--selection",
                "raw_mse",
            ],
        )
    # This tiny software smoke deliberately does not assert learned-policy quality.
    config = yaml.safe_load((ROOT / "configs/reach_pilot.yaml").read_text())
    config["seed"] = 7
    config["dataset"]["root"] = str(dataset)
    config["world_model"]["checkpoints"] = {
        b: str(checkpoints / f"{b}.pt") for b in ("native_jepa", "leworldmodel")
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
    report["complete"] = True
    report["limitation"] = (
        "100 updates, raw-loss selection, two five-step episodes; software integration only"
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(output / "report.json")


if __name__ == "__main__":
    main()
