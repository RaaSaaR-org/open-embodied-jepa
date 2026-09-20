"""Validate and summarize completed frozen MVP runs without selecting or rerunning them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from embodied_jepa.benchmark import summarize as summarize_episodes
from embodied_jepa.result_schema import validate_results


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(root):
    root = Path(root)
    experiment = read(root / "checkpoints/mvp-v0/experiment.json")
    evaluation = read(root / "outputs/mvp-v0-evaluation/report.json")
    if experiment["status"] != "completed" or not evaluation["complete"]:
        raise ValueError("Final comparison is incomplete; retain the supervisor failure report")
    result = {
        "schema_version": 1,
        "purpose": (
            "All preregistered runs, no best-seed selection; repeated seeds share physical resets"
        ),
        "training_supervisor": experiment,
        "evaluation_supervisor": evaluation,
        "training": [],
        "evaluation": [],
    }
    for trial in experiment["runs"]:
        folder = root / "checkpoints/mvp-v0" / f"seed-{trial['seed']}"
        path = folder / f"{trial['backend']}.run.json"
        report = read(path)
        selected = next(v for v in report["validation"] if v["step"] == report["best_step"])
        if not selected["selection"]["eligible"]:
            raise ValueError("Ineligible selected checkpoint")
        result["training"].append(
            {
                "backend": report["backend"],
                "seed": report["seed"],
                "status": report["status"],
                "updates": report["completed_steps"],
                "selected_step": report["best_step"],
                "report_sha256": digest(path),
                "checkpoint_sha256": digest(folder / f"{trial['backend']}.pt"),
                "elapsed_seconds": report["elapsed_seconds"],
                "peak_host_rss_bytes": report["peak_host_rss_bytes"],
                "selected_validation": selected,
                "test_samples_loaded": report["test_samples_loaded"],
            }
        )
    selected_hashes = {
        (row["backend"], row["seed"]): row["checkpoint_sha256"] for row in result["training"]
    }
    if len(selected_hashes) != 6 or len(evaluation["attempts"]) != 8:
        raise ValueError("Expected six training runs and eight evaluation/control runs")
    reference = None
    for trial in evaluation["attempts"]:
        folder = root / "outputs" / trial["run_id"]
        run = read(folder / "run.json")
        episodes = [
            json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()
        ]
        validate_results(run, episodes)
        saved_summary = read(folder / "summary.json")
        if any(
            saved_summary.get(key) != value for key, value in summarize_episodes(episodes).items()
        ):
            raise ValueError("Saved summary disagrees with episode records")
        if run["evaluation_seeds"] != list(range(20000, 20050)):
            raise ValueError("Evaluation differs from the declared 50-reset cohort")
        if trial["policy"] == "model" and run["checkpoint_hash"] != selected_hashes.get(
            (run["backend"], run["train_seed"])
        ):
            raise ValueError("Evaluation did not use its selected training checkpoint")
        if run["source_tree_hash"] != experiment["frozen_identity"]["python_source_sha256"]:
            raise ValueError("Runtime source changed between training and evaluation")
        planner = dict(run["planner"])
        planner.pop("seed")  # Actual per-episode candidate RNG uses each fixed evaluation seed.
        shared = {
            k: run[k]
            for k in (
                "dataset_hash",
                "split_hash",
                "action_hash",
                "goal_manifest_hash",
                "evaluation_seeds",
                "task",
                "task_version",
                "max_steps",
                "timeout_seconds",
            )
        }
        shared["planner"] = planner
        shared["goal_images"] = [episode["goal_image_sha256"] for episode in episodes]
        if reference is None:
            reference = shared
        if shared != reference:
            raise ValueError(f"Common cohort/config mismatch: {trial['run_id']}")
        result["evaluation"].append(
            {
                "run_id": trial["run_id"],
                "backend": run["backend"],
                "train_seed": run["train_seed"],
                "checkpoint_sha256": run["checkpoint_hash"],
                "summary": saved_summary,
                "run_sha256": digest(folder / "run.json"),
                "episodes_sha256": digest(folder / "episodes.jsonl"),
                "total_executed_steps": sum(e["executed_steps"] for e in episodes),
                "total_replans": sum(e["replans"] for e in episodes),
                "limit_rejections": sum(e["limit_rejections"] for e in episodes),
                "rollouts": str(folder.relative_to(root)),
            }
        )
    result["common"] = reference
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="benchmarks/manifests/mvp-results-v0.json")
    args = parser.parse_args()
    result = summarize(args.root)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(path)


if __name__ == "__main__":
    main()
