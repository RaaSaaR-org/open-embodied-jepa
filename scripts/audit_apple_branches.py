"""Audit branch-corpus artifacts without physics, inference, or TEST image decoding."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embodied_jepa.data import MANIFEST, DatasetStore  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame(episode, index):
    return (
        episode.observations["onboard_rgb"][index],
        episode.robot_states[index],
        episode.state_mask[index],
        episode.timestamps[index],
    )


def identical(left, right):
    return all(np.array_equal(a, b) for a, b in zip(left, right, strict=True))


def expected_request(recorded, branch, root_grasp):
    result = recorded.copy()
    if branch == "hold":
        result[:12], result[12:] = 0, root_grasp
    elif branch == "half_motion":
        result[6:12] *= 0.5
    elif branch == "reverse_translation":
        result[6:9] *= -1
    elif branch == "reverse_rotation":
        result[9:12] *= -1
    elif branch == "up":
        result[6:12], result[8] = 0, 0.25
    elif branch in ("open", "closed"):
        result[13] = -1 if branch == "open" else 1
    elif branch != "recorded":
        raise ValueError(f"unknown branch {branch}")
    return np.clip(result, -1, 1).astype(np.float32)


def contrast(roots, endpoints):
    metrics, gates = [], {}
    for root in roots:
        siblings = endpoints.get(root["root_id"], {})
        pairs = []
        for horizon in (8, 16):
            for left, right in itertools.combinations(sorted(siblings), 2):
                if horizon not in siblings[left] or horizon not in siblings[right]:
                    continue
                (a, aa), (b, ba) = siblings[left][horizon], siblings[right][horizon]
                rgb_rms = float(np.sqrt(np.mean(((a[0].astype(float) - b[0]) / 255) ** 2)))
                pairs.append(
                    {
                        "horizon": horizon,
                        "left": left,
                        "right": right,
                        "rgb_rms": rgb_rms,
                        "state_rms": float(np.sqrt(np.mean((a[1] - b[1]) ** 2))),
                        "applied_action_rms": float(np.sqrt(np.mean((aa - ba) ** 2))),
                        "visually_distinct": rgb_rms >= 1 / 255,
                    }
                )
        complete = sum(16 in value for value in siblings.values())
        distinct = sum(x["horizon"] == 16 and x["visually_distinct"] for x in pairs)
        metrics.append(
            root
            | {
                "complete_h16_branches": complete,
                "distinct_h16_pairs": distinct,
                "passes": complete >= 4 and distinct >= 2,
                "pairs": pairs,
            }
        )
    for split in ("train", "val"):
        available = [r for r in metrics if r["split"] == split and r["frame_index"] is not None]
        passed = sum(r["passes"] for r in available)
        gates[split] = {
            "available_roots": len(available),
            "passing_roots": passed,
            "passed": bool(available) and passed / len(available) >= 0.75,
        }
    return metrics, gates


def audit(source_path, dataset, plan_path):
    started = time.monotonic()
    source = DatasetStore(source_path)
    plan = json.loads(plan_path.read_text())
    report_path = dataset / "branch_report.json"
    report = json.loads(report_path.read_text())
    findings = []

    def require(condition, message):
        if not condition:
            findings.append(message)

    require(source.manifest_hash == plan["parent_dataset_sha256"], "parent manifest drift")
    require(report.get("identity") == plan.get("identity"), "reported frozen identity differs")
    require(report.get("supervisor_wall_seconds", 601) <= 600, "declared wall budget exceeded")
    planned = {row["episode_id"]: row for row in plan["attempts"]}
    records = {row["episode_id"]: row for row in report["attempts"]}
    require(len(records) == len(report["attempts"]), "duplicate attempt report IDs")
    require(set(records) == set(planned), "attempt report does not cover the frozen cohort")
    stored = None
    try:
        stored = DatasetStore(dataset)
    except Exception as error:
        findings.append(f"corpus integrity/unsealed availability: {type(error).__name__}: {error}")
    originals = {row["episode_id"]: row for row in source.manifest["episodes"]}
    original_split = {
        episode: split for split, ids in source.manifest["splits"].items() for episode in ids
    }
    checks = {"test_payloads_byte_equal": 0, "original_payloads_byte_equal": 0}
    split_counts, canonical = {}, {}
    if stored is not None:
        for key in ("action_manifest", "state_schema", "fps", "robot_type"):
            require(stored.manifest[key] == source.manifest[key], f"inherited {key} changed")
        rows = {row["episode_id"]: row for row in stored.manifest["episodes"]}
        partitions = stored.manifest.get("splits") or {}
        assignment = {episode: split for split, ids in partitions.items() for episode in ids}
        require(bool(partitions), "corpus partitions are not sealed")
        require(len(assignment) == sum(map(len, partitions.values())), "duplicate split membership")
        require(set(assignment) == set(rows), "partitions do not cover all stored episodes")
        split_counts = {name: len(ids) for name, ids in partitions.items()}
        sessions = {}
        for name, row in rows.items():
            assigned = assignment.get(name)
            previous = sessions.setdefault(row["session_id"], assigned)
            require(previous == assigned, f"session crosses partitions: {row['session_id']}")
            if name in originals:
                original = originals[name]
                same_bytes = sha(source.root / original["path"]) == sha(dataset / row["path"])
                require(same_bytes, f"original payload changed: {name}")
                checks["original_payloads_byte_equal"] += int(same_bytes)
                if original_split[name] == "test":
                    checks["test_payloads_byte_equal"] += int(same_bytes)
                require(assigned == original_split[name], f"original split changed: {name}")
                require(row == original, f"original manifest episode metadata changed: {name}")
            else:
                require(name in planned, f"unplanned stored branch: {name}")
                if name not in planned:
                    continue
                parent = planned[name]["parent_episode_id"]
                require(assigned == original_split[parent], f"branch split differs: {name}")
                require(
                    row["session_id"] == originals[parent]["session_id"], f"session differs: {name}"
                )
                require(assigned in ("train", "val"), f"branch entered TEST: {name}")
                if assigned in ("train", "val"):
                    canonical[name] = stored.read_episode(name)
        require(set(originals) <= set(rows), "source demonstrations missing")
        if stored.manifest.get("normalization"):
            ids = set(stored.manifest["normalization"]["episode_ids"])
            require(
                ids == set(partitions.get("train", [])), "normalization IDs are not exact TRAIN"
            )
        else:
            findings.append("training normalization missing")
    parent_cache, root_frames, endpoints, prefix_rows = {}, {}, {}, []
    for name, specification in planned.items():
        row = records.get(name, {"status": "missing"})
        parent_id = specification["parent_episode_id"]
        require(original_split[parent_id] in ("train", "val"), f"excluded parent: {parent_id}")
        journal = dataset / "branch_journal" / name
        files = sorted(journal.glob("frame-*.npz"))
        count = max(0, len(files) - 1)
        prefix_rows.append(
            {
                "episode_id": name,
                "status": row["status"],
                "durable_transitions": count,
                "canonical_transitions": canonical[name].transitions if name in canonical else 0,
            }
        )
        require(
            [p.name for p in files] == [f"frame-{i:03d}.npz" for i in range(len(files))],
            f"journal gap: {name}",
        )
        if row["status"] == "completed":
            require(
                count == 16 and row.get("executed_steps") == 16, f"completed length differs: {name}"
            )
        if name in canonical:
            require(canonical[name].transitions == count, f"stored/journal prefix differs: {name}")
        if not files:
            continue
        if original_split[parent_id] not in ("train", "val"):
            # A corrupt plan is an audit finding, never permission to decode TEST.
            continue
        if parent_id not in parent_cache:
            parent_cache[parent_id] = source.read_episode(parent_id)
        parent = parent_cache[parent_id]
        index = specification["frame_index"]
        original_frame = frame(parent, index)
        grasp = parent.actions[index - 1, 12:] if index else np.array([-1, -1], np.float32)
        actions, captured = [], []
        for i, path in enumerate(files):
            with np.load(path, allow_pickle=False) as data:
                current = tuple(data[key].copy() for key in ("rgb", "state", "mask", "timestamp"))
                captured.append(current)
                if i == 0:
                    root_id = specification["root_id"]
                    previous = root_frames.setdefault(root_id, current)
                    require(identical(previous, current), f"sibling roots differ: {name}")
                    require(
                        np.array_equal(current[0], original_frame[0]),
                        f"source root RGB differs: {name}",
                    )
                    require(
                        np.allclose(current[1], original_frame[1], atol=1e-5, rtol=0),
                        f"source root state differs: {name}",
                    )
                    require(
                        np.array_equal(current[2], original_frame[2]),
                        f"source root mask differs: {name}",
                    )
                    require(
                        np.isclose(current[3], original_frame[3], atol=1e-9, rtol=0),
                        f"source root time differs: {name}",
                    )
                else:
                    require(
                        np.isclose(
                            current[3] - captured[i - 1][3],
                            1 / source.manifest["fps"],
                            atol=1e-9,
                            rtol=0,
                        ),
                        f"noncanonical action interval: {name}/{i}",
                    )
                    action, request, raw = (
                        data[key].copy() for key in ("action", "request", "raw_action")
                    )
                    actions.append(action)
                    require(
                        action.dtype == np.float32
                        and action.shape == (14,)
                        and np.isfinite(action).all()
                        and np.max(np.abs(action)) <= 1,
                        f"invalid action: {name}/{i}",
                    )
                    expected = action * np.array(
                        [0.015] * 3 + [0.06] * 3 + [0.015] * 3 + [0.06] * 3 + [1, 1], np.float32
                    )
                    expected[12:] = (action[12:] + 1) / 2
                    require(
                        np.array_equal(raw, expected), f"physical action scale differs: {name}/{i}"
                    )
                    recorded = np.array(
                        parent.metadata["requested_actions"][index + i - 1], np.float32
                    )
                    desired = expected_request(recorded, specification["branch"], grasp)
                    require(
                        np.array_equal(request, desired),
                        f"branch request semantics differ: {name}/{i}",
                    )
                    if specification["branch"] == "recorded":
                        require(
                            np.allclose(action, parent.actions[index + i - 1], atol=2e-6, rtol=0),
                            f"recorded applied action differs: {name}/{i}",
                        )
                    if name in canonical:
                        episode = canonical[name]
                        require(
                            np.array_equal(action, episode.actions[i - 1])
                            and np.array_equal(raw, episode.raw_actions[i - 1]),
                            f"canonical labels differ: {name}/{i}",
                        )
                if name in canonical:
                    require(
                        identical(current, frame(canonical[name], i)),
                        f"canonical sensors differ: {name}/{i}",
                    )
        endpoints.setdefault(specification["root_id"], {})[specification["branch"]] = {
            h: (captured[h], np.asarray(actions[:h])) for h in (8, 16) if count >= h
        }
    metrics, gates = contrast(plan["roots"], endpoints)
    calculated = {row["root_id"]: row for row in metrics}
    for reported in report.get("matched_root_metrics", []):
        actual = calculated[reported["root_id"]]["pairs"]
        require(len(actual) == len(reported["pairs"]), f"pair count differs: {reported['root_id']}")
        for left, right in zip(actual, reported["pairs"], strict=False):
            for key in ("left", "right", "horizon", "visually_distinct"):
                require(left[key] == right[key], f"pair identity differs: {reported['root_id']}")
            for ours, theirs in (
                ("rgb_rms", "rgb_rms"),
                ("state_rms", "measured_state_rms"),
                ("applied_action_rms", "applied_action_rms"),
            ):
                require(
                    np.isclose(left[ours], right[theirs], atol=1e-12, rtol=1e-10),
                    f"pair metric differs: {reported['root_id']}/{ours}",
                )
    if report.get("action_contrast_gate") is not None:
        require(gates == report["action_contrast_gate"], "recomputed contrast gates differ")
    ready = (
        not findings
        and report.get("status") == "completed"
        and all(g["passed"] for g in gates.values())
    )
    require(
        not report.get("training_ready") or ready, "reported training_ready unsupported by audit"
    )
    return {
        "source_dataset_sha256": source.manifest_hash,
        "dataset_sha256": sha(dataset / MANIFEST) if (dataset / MANIFEST).exists() else None,
        "plan_sha256": sha(plan_path),
        "report_sha256": sha(report_path),
        "audit_script_sha256": sha(__file__),
        "status": "passed" if not findings else "findings",
        "findings": findings,
        "training_ready": ready,
        "collector_status": report["status"],
        "planned_attempts": len(planned),
        "status_counts": dict(Counter(r["status"] for r in records.values())),
        "split_counts": split_counts,
        "byte_checks": checks,
        "decoded_parent_ids": sorted(parent_cache),
        "test_images_decoded": False,
        "root_initial_tensors_checked": len(root_frames),
        "canonical_branch_count": len(canonical),
        "canonical_transitions": sum(e.transitions for e in canonical.values()),
        "prefixes": prefix_rows,
        "contrast_gates": gates,
        "matched_roots": metrics,
        "wall_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/apple-task-v1")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/apple-branches-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite audit")
    plan = args.dataset.with_name(args.dataset.name + "-plan.json")
    try:
        result = audit(args.source, args.dataset, plan)
    except Exception as error:
        result = {
            "status": "audit_error",
            "training_ready": False,
            "error": f"{type(error).__name__}: {error}",
            "audit_script_sha256": sha(__file__),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
