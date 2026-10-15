"""Dataset coverage reports; aggregate observation only, never model selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from embodied_jepa.contracts import ACTION_NAMES
from embodied_jepa.data import DatasetStore


def audit_dataset(root):
    store = DatasetStore(root)
    actions, states, intervals = [], [], []
    outcomes = Counter()
    for episode_id in store.episode_ids:
        episode = store.read_episode(episode_id)
        actions.append(episode.actions)
        states.append(episode.robot_states)
        intervals.extend(np.diff(episode.timestamps).tolist())
        outcomes.update(episode.metadata.get("stop_reasons", []) or ["requested_step_budget"])
    action, state = np.concatenate(actions), np.concatenate(states)
    report = {
        "dataset_hash": store.manifest_hash,
        "episodes": len(store.episode_ids),
        "transitions": int(len(action)),
        "observations": int(len(state)),
        "action_coverage": {
            name: {
                "min": float(action[:, i].min()),
                "max": float(action[:, i].max()),
                "std": float(action[:, i].std()),
            }
            for i, name in enumerate(ACTION_NAMES)
        },
        "state_coverage": {
            name: {
                "min": float(state[:, i].min()),
                "max": float(state[:, i].max()),
                "std": float(state[:, i].std()),
            }
            for i, name in enumerate(store.state_schema.names)
        },
        "time_delta_seconds": {
            "min": min(intervals),
            "max": max(intervals),
            "mean": float(np.mean(intervals)),
        },
        "stop_reasons": dict(outcomes),
        "storage_bytes": sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file()),
        "split_counts": {k: len(v) for k, v in store.manifest["splits"].items()},
        "heldout_pair_present_in_train": any(
            r["object_id"] == "apple"
            and r["container_id"] == "plate"
            and r["episode_id"] in store.manifest["splits"]["train"]
            for r in store.manifest["episodes"]
        ),
        "scope": "aggregate corpus coverage only",
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit_dataset(args.dataset), indent=2) + "\n")
