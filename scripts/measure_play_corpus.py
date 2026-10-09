"""TASK-085: measure the play corpus against the protocol's gate (play_corpus_v1.md §10).

Reads every shard (``DatasetStore`` opens with its full hash check), every sidecar (hash checked)
and the action column (no image decoding), and writes one JSON report with the row.

    .venv/bin/python scripts/measure_play_corpus.py --corpus data/play-v1 --out report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import play_corpus as pc  # noqa: E402


def _list(counts) -> list:
    return np.asarray(counts).tolist()


def measure(corpus: Path, *, debug: bool = False) -> dict:
    import pyarrow.parquet as pq

    from embodied_jepa.data import DatasetStore

    summary = json.loads(pc.corpus_manifest_path(corpus).read_text())
    void = []
    if summary.get("dirty") and not debug:
        void.append("collected from a dirty tree")
    if summary.get("debug") and not debug:
        void.append("a debug collection")
    allowed = pc.DEBUG_SEEDS if debug else range(850000, 853200)
    per_split = {s: [] for s in ("train", "val", "test")}
    episodes_info = []
    actions = {s: [] for s in ("train", "val", "test")}
    bytes_total = 0
    shards = []
    for entry in summary["shards"]:
        if entry["status"] not in ("complete", "exists-complete"):
            shards.append({"shard": entry["shard"], "status": entry["status"]})
            continue
        root = corpus / f"shard-{entry['shard']:02d}"
        try:
            store = DatasetStore(root)
        except Exception as error:
            void.append(f"shard {entry['shard']}: {error!r}")
            continue
        if entry.get("manifest_hash") and store.manifest_hash != entry["manifest_hash"]:
            void.append(f"shard {entry['shard']}: manifest hash differs from corpus.json")
        if store.manifest["provenance"].get("revision") != summary["revision"]:
            void.append(f"shard {entry['shard']}: revision differs")
        split_of = {eid: name for name, ids in store.manifest["splits"].items() for eid in ids}
        bytes_total += sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        shards.append(
            {
                "shard": entry["shard"],
                "status": "measured",
                "episodes": len(store.manifest["episodes"]),
            }
        )
        for row in store.manifest["episodes"]:
            seed = row["metadata"]["seed"]
            if seed not in allowed:
                void.append(f"seed {seed} outside the declared range")
            split = split_of[row["episode_id"]]
            if split not in per_split:
                void.append(f"{row['episode_id']} in {split}")
                continue
            side = pc.load_sidecar(root, row)
            per_split[split].append(side)
            table = pq.read_table(root / row["path"], columns=["action", "action_valid"])
            valid = np.asarray(table.column("action_valid").to_pylist(), bool)
            acts = np.asarray(table.column("action").to_pylist(), np.float32)[valid]
            actions[split].append(acts)
            facts = row["metadata"]["facts"]
            episodes_info.append(
                {
                    "split": split,
                    "mode": row["metadata"]["mode"],
                    "skills": row["metadata"]["skill_log"],
                    "stop": row["metadata"]["stop_reason"],
                    "commands": facts["commands"],
                    "moved": facts["moved_any"],
                    "lifted": facts["lifted_any"],
                    "grasped": facts["grasp_contact_any"],
                    "objects": facts["objects_on_table"],
                    "moved_objects": [
                        n for n, m in zip(pc.OBJECT_NAMES, facts["moved"], strict=True) if m
                    ],
                    "lifted_objects": [
                        n for n, m in zip(pc.OBJECT_NAMES, facts["lifted"], strict=True) if m
                    ],
                }
            )
    counts = {s: pc.gate_counts(v) for s, v in per_split.items()}
    total_commands = sum(c["commands"] for c in counts.values())
    decision = pc.gate_row(
        counts["train"], total_commands=total_commands, void="; ".join(void[:20])
    )

    def rates(rows):
        n = len(rows)
        if not n:
            return {"episodes": 0}
        return {
            "episodes": n,
            "moved": sum(r["moved"] for r in rows) / n,
            "lifted": sum(r["lifted"] for r in rows) / n,
            "grasp_contact": sum(r["grasped"] for r in rows) / n,
            "truncated": sum(bool(r["stop"]) for r in rows) / n,
            "mean_commands": float(np.mean([r["commands"] for r in rows])),
        }

    train_rows = [r for r in episodes_info if r["split"] == "train"]
    stops = Counter(
        (r["stop"].split("[")[-1].split()[0] if "[" in r["stop"] else r["stop"]) or "none"
        for r in episodes_info
    )
    skills = Counter(tuple(s) for r in train_rows for s in r["skills"])
    action_stats = {}
    for split, chunks in actions.items():
        if not chunks:
            continue
        a = np.concatenate(chunks)
        action_stats[split] = {
            "commands": int(len(a)),
            "mean": a.mean(0).round(4).tolist(),
            "std": a.std(0).round(4).tolist(),
            "saturated_fraction": (np.abs(a) >= 0.99).mean(0).round(4).tolist(),
        }
    split_reports = {}
    for split, c in counts.items():
        n = c["episodes"]
        split_reports[split] = {
            "episodes": n,
            "commands": c["commands"],
            "hours": c["commands"] / pc.FPS / 3600,
            "hand_cells_ge_200": float((c["hand"] >= 200).mean()),
            "hand_min": int(c["hand"].min()),
            "object_cells_min_starts": int(c["starts"].min()),
            "object_cells_ge_10_contacts": float((c["contacts"] >= 10).mean()),
            "moved": c["moved"] / n if n else None,
            "lifted": c["lifted"] / n if n else None,
            "grasp_contact": c["grasped"] / n if n else None,
        }
    return {
        "task": "TASK-085",
        "protocol": "docs/experiments/play_corpus_v1.md",
        "corpus": str(corpus),
        "corpus_json_sha256": hashlib.sha256(
            pc.corpus_manifest_path(corpus).read_bytes()
        ).hexdigest(),
        "revision": summary["revision"],
        "renderers": summary.get("renderers"),
        "decision": decision,
        "void_reasons": void,
        "splits": split_reports,
        "bytes": bytes_total,
        "shards": shards,
        "reported": {
            "train_by_mode": {
                m: rates([r for r in train_rows if r["mode"] == m]) for m in pc.MODES
            },
            "train_all": rates(train_rows),
            "train_skills": {f"{k[0]}/{k[1]}": v for k, v in sorted(skills.items())},
            "stop_reasons_all_splits": dict(stops),
            "train_moved_by_object": dict(
                Counter(o for r in train_rows for o in r["moved_objects"])
            ),
            "train_lifted_by_object": dict(
                Counter(o for r in train_rows for o in r["lifted_objects"])
            ),
            "actions": action_stats,
        },
        "maps": {
            "train_hand_counts_xyz": _list(counts["train"]["hand"]),
            "train_object_starts_xy": _list(counts["train"]["starts"]),
            "train_object_contacts_xy": _list(counts["train"]["contacts"]),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    report = measure(Path(args.corpus), debug=args.debug)
    pc.write_json(out, report)
    d = report["decision"]
    print(
        json.dumps(
            {
                "row": d["row"],
                "failing": d["failing"],
                "gates": {k: v["pass"] for k, v in d["gates"].items()},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
