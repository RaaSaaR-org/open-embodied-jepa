"""SO-101 S3 (TASK-099): measure an SO-101 play corpus before any training (development, no gate).

Opens every shard with ``SchemaDatasetStore`` (full hash check) and every sidecar (hash
checked), and writes one JSON report and a contact sheet into a new ``--out`` directory:

- integrity: manifest hashes against ``corpus.json``, one revision, seeds inside their shard,
  no duplicates, the frozen splits equal the declared rule;
- split leakage: episode, session and seed sets disjoint across splits, no reset (cube xy) and
  no first frame (PNG bytes of either camera) shared across splits;
- action-range coverage per channel (histograms, quantiles, saturation) and joint-position
  coverage per joint against its limits;
- gripper open/close balance (commands, measured jaw angle, jaw contact);
- lift and place window counts with S2's scorers (``so101_tasks.py``), per split;
- the per-attempt successful-grasp fraction (cube 2 cm up with both jaws at the end of the lift);
- camera frame checks on decoded frames (blank, frozen and cube-visible fractions);
- storage per episode and per frame, and the collection's timing.

    .venv/bin/python scripts/measure_so101_play.py --corpus data/so101-play-v1 --out outputs/<new>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import so101_play_corpus as pc  # noqa: E402

SPLITS = ("train", "val", "test")
BINS = np.linspace(-1.0, 1.0, 21)
OCCUPIED = 1e-3  # a bin counts as covered with at least 0.1 % of the commands


def _red_fraction(frame):
    f = frame.astype(np.int16)
    red = (f[..., 0] > 120) & (f[..., 1] < 70) & (f[..., 2] < 70)
    return float(red.mean())


def frame_stats(task) -> dict:
    """Decode an episode's frames (every ``stride``-th) and summarise them per camera."""
    import pyarrow.parquet as pq
    from PIL import Image

    path, stride = task
    columns = [f"observation.images.{c}" for c in pc.CAMERAS]
    table = pq.read_table(path, columns=columns)
    out = {}
    for camera, column in zip(pc.CAMERAS, columns, strict=True):
        items = table.column(column).to_pylist()
        first = hashlib.sha256(items[0]["bytes"]).hexdigest()
        stds, means, reds, frozen, sizes = [], [], [], 0, []
        previous = None
        for i, item in enumerate(items):
            sizes.append(len(item["bytes"]))
            if i % stride:
                continue
            frame = np.asarray(Image.open(io.BytesIO(item["bytes"])).convert("RGB"))
            stds.append(float(frame.std()))
            means.append(float(frame.mean()))
            reds.append(_red_fraction(frame))
            if previous is not None and np.array_equal(previous, frame):
                frozen += 1
            previous = frame
        out[camera] = {
            "first_sha256": first,
            "decoded": len(stds),
            "shape": list(frame.shape),
            "min_std": min(stds),
            "mean_brightness": float(np.mean(means)),
            "blank": int(sum(s < 2.0 for s in stds)),
            "frozen_pairs": frozen,
            "cube_visible": int(sum(r > 2e-4 for r in reds)),
            "median_red_fraction": float(np.median(reds)),
            "png_bytes_mean": float(np.mean(sizes)),
        }
    return out


def _hist(values):
    counts, _ = np.histogram(np.clip(values, -1, 1), bins=BINS)
    return counts


def measure(corpus: Path, *, stride: int, workers: int) -> dict:
    import pyarrow.parquet as pq

    from embodied_jepa.schema_data import SchemaDatasetStore

    summary = json.loads(pc.corpus_manifest_path(corpus).read_text())
    problems = []
    if summary.get("dirty") and summary.get("kind") != "debug":
        problems.append("collected from a dirty tree")
    per_split = {s: [] for s in SPLITS}
    shards, frame_tasks, storage = [], [], {"bytes": 0, "episodes": 0, "frames": 0}
    seen = set()
    actions = {s: [] for s in SPLITS}
    requested_all, states_all = [], []
    limits = None
    for entry in summary["shards"]:
        if entry["status"] not in ("complete", "exists-complete"):
            shards.append({"shard": entry["shard"], "status": entry["status"]})
            continue
        root = corpus / f"shard-{entry['shard']:02d}"
        store = SchemaDatasetStore(root)
        if entry.get("manifest_hash") and store.manifest_hash != entry["manifest_hash"]:
            problems.append(f"shard {entry['shard']}: manifest hash differs from corpus.json")
        prov = store.manifest["provenance"]
        if prov.get("revision") != summary["revision"]:
            problems.append(f"shard {entry['shard']}: revision differs")
        if (prov.get("policy_salt"), prov.get("split_salt")) != (pc.POLICY_SALT, pc.SPLIT_SALT):
            problems.append(f"shard {entry['shard']}: salts differ")
        manifest = store.manifest["action_manifest"]
        limits = np.asarray(
            [manifest["joint_limits_rad"][n] for n in pc.JOINT_DELTA_SO101_V0.names]
        )
        first, last = entry["seeds"]
        seeds_here = [row["metadata"]["seed"] for row in store.manifest["episodes"]]
        if not all(first <= s <= last for s in seeds_here) or len(set(seeds_here)) != len(
            seeds_here
        ):
            problems.append(f"shard {entry['shard']}: seeds outside the shard or duplicated")
        if seen & set(seeds_here):
            problems.append(f"shard {entry['shard']}: seeds duplicated across shards")
        seen.update(seeds_here)
        n_val, n_test = (len(store.manifest["splits"][k]) for k in ("val", "test"))
        rule = pc.split_assignment(seeds_here, n_val=n_val, n_test=n_test)
        frozen = {
            k: sorted(int(e.rsplit("-", 1)[1]) for e in store.manifest["splits"][k]) for k in SPLITS
        }
        if frozen != rule:
            problems.append(f"shard {entry['shard']}: frozen splits differ from the rule")
        if store.manifest["splits"].get("holdout"):
            problems.append(f"shard {entry['shard']}: nonempty holdout")
        split_of = {eid: name for name, ids in store.manifest["splits"].items() for eid in ids}
        shard_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        storage["bytes"] += shard_bytes
        shards.append(
            {
                "shard": entry["shard"],
                "episodes": len(seeds_here),
                "bytes": shard_bytes,
                "manifest_hash": store.manifest_hash,
            }
        )
        for row in store.manifest["episodes"]:
            split = split_of[row["episode_id"]]
            side = pc.load_sidecar(root, row)
            table = pq.read_table(
                root / row["path"], columns=["action", "action_valid", "observation.state"]
            )
            valid = np.asarray(table.column("action_valid").to_pylist(), bool)
            acts = np.asarray(table.column("action").to_pylist(), np.float32)[valid]
            state = np.asarray(table.column("observation.state").to_pylist(), np.float32)
            if not np.array_equal(acts.shape, side["requested"].shape):
                problems.append(f"{row['episode_id']}: action and sidecar lengths differ")
            actions[split].append(acts)
            requested_all.append((side["requested"], acts))
            states_all.append(state[:, :6])
            storage["episodes"] += 1
            storage["frames"] += len(state)
            frame_tasks.append((str(root / row["path"]), stride))
            per_split[split].append(
                {
                    "episode_id": row["episode_id"],
                    "session_id": row["session_id"],
                    "seed": row["metadata"]["seed"],
                    "mode": row["metadata"]["mode"],
                    "skills": row["metadata"]["skill_log"],
                    "stop": row["metadata"]["stop_reason"],
                    "facts": row["metadata"]["facts"],
                    "cube_xy": tuple(np.round(row["metadata"]["reset"]["cube_xy"], 6)),
                    "phase": np.bincount(side["phase"], minlength=len(pc.PHASES)),
                    "gripper_q": state[:, 5],
                    "both_jaw": side["fixed_jaw_contact"] & side["moving_jaw_contact"],
                    "frames_path": str(root / row["path"]),
                }
            )
    # ----- frames (parallel decode) -----
    started = time.time()
    if workers <= 1:
        stats = [frame_stats(task) for task in frame_tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            stats = list(pool.map(frame_stats, frame_tasks, chunksize=4))
    frame_seconds = time.time() - started
    stats_by_path = {t[0]: s for t, s in zip(frame_tasks, stats, strict=True)}
    # ----- leakage -----
    leakage = {}
    for key in ("episode_id", "session_id", "seed", "cube_xy"):
        sets = {s: {e[key] for e in per_split[s]} for s in SPLITS}
        leakage[key] = {
            f"{a}&{b}": len(sets[a] & sets[b])
            for i, a in enumerate(SPLITS)
            for b in SPLITS[i + 1 :]
        }
    for camera in pc.CAMERAS:
        sets = {
            s: {stats_by_path[e["frames_path"]][camera]["first_sha256"] for e in per_split[s]}
            for s in SPLITS
        }
        leakage[f"first_frame_{camera}"] = {
            f"{a}&{b}": len(sets[a] & sets[b])
            for i, a in enumerate(SPLITS)
            for b in SPLITS[i + 1 :]
        }
    leak_total = sum(v for d in leakage.values() for v in d.values())
    # ----- actions -----
    names = list(pc.JOINT_DELTA_SO101_V0.names)
    every = np.concatenate([a for s in SPLITS for a in actions[s]])
    train = np.concatenate(actions["train"]) if actions["train"] else every[:0]
    coverage = {}
    for j, name in enumerate(names):
        hist = _hist(every[:, j])
        hist_train = _hist(train[:, j]) if len(train) else hist
        coverage[name] = {
            "quantiles_1_5_25_50_75_95_99": np.quantile(
                every[:, j], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            )
            .round(4)
            .tolist(),
            "std": float(every[:, j].std()),
            "at_minus_1": float((every[:, j] <= -0.999).mean()),
            "at_plus_1": float((every[:, j] >= 0.999).mean()),
            "near_zero_abs_lt_0.05": float((np.abs(every[:, j]) < 0.05).mean()),
            "histogram_20_bins": hist.tolist(),
            "bins_covered_all": int((hist / hist.sum() >= OCCUPIED).sum()),
            "bins_covered_train": int((hist_train / max(hist_train.sum(), 1) >= OCCUPIED).sum()),
        }
    req = np.concatenate([r for r, _ in requested_all])
    app = np.concatenate([a for _, a in requested_all])
    differs = req != app
    clip = {
        "commands_clipped": float(differs.any(axis=1).mean()),
        "per_channel": dict(zip(names, differs.mean(axis=0).round(4).tolist(), strict=True)),
    }
    q = np.concatenate(states_all)
    joint_cov = {}
    for j, name in enumerate(names):
        lo, hi = limits[j]
        hist, _ = np.histogram(q[:, j], bins=np.linspace(lo, hi, 21))
        joint_cov[name] = {
            "limits_rad": [float(lo), float(hi)],
            "min": float(q[:, j].min()),
            "max": float(q[:, j].max()),
            "quantiles_1_50_99": np.quantile(q[:, j], [0.01, 0.5, 0.99]).round(4).tolist(),
            "fraction_of_range_spanned_1_99": float(
                (np.quantile(q[:, j], 0.99) - np.quantile(q[:, j], 0.01)) / (hi - lo)
            ),
            "bins_covered_of_20": int((hist / hist.sum() >= OCCUPIED).sum()),
        }
    # ----- gripper -----
    eps = [e for s in SPLITS for e in per_split[s]]
    gq = np.concatenate([e["gripper_q"] for e in eps])
    both = np.concatenate([e["both_jaw"] for e in eps])
    g = every[:, 5]
    gripper = {
        "command_closing_half_lt_0": float((g < 0).mean()),
        "command_opening_half_ge_0": float((g >= 0).mean()),
        "command_at_minus_1": float((g <= -0.999).mean()),
        "command_at_plus_1": float((g >= 0.999).mean()),
        "measured_rad_quantiles_5_25_50_75_95": np.quantile(gq, [0.05, 0.25, 0.5, 0.75, 0.95])
        .round(3)
        .tolist(),
        "measured_closed_below_0.3_rad": float((gq < 0.3).mean()),
        "measured_open_above_1.0_rad": float((gq > 1.0).mean()),
        "both_jaws_on_cube_fraction_of_frames": float(both.mean()),
    }
    # ----- windows and grasps -----
    windows = {}
    for s in SPLITS:
        f = [e["facts"] for e in per_split[s]]
        windows[s] = {
            "episodes": len(f),
            "commands": int(sum(x["commands"] for x in f)),
            "lift_windows": int(sum(x["lift_windows"] for x in f)),
            "episodes_with_lift_success": int(sum(x["lift_success"] for x in f)),
            "held_steps_5cm_contact": int(sum(x["held_steps"] for x in f)),
            "episodes_picked": int(sum(x["picked"] for x in f)),
            "episodes_with_place_success_marker": int(sum(x["place_success"] for x in f)),
            "place_windows_anywhere": int(sum(x["place_windows_anywhere"] for x in f)),
            "episodes_cube_lost": int(sum(x["cube_lost"] for x in f)),
            "episodes_cube_moved_ge_2cm": int(sum(x["cube_displacement_m"] >= 0.02 for x in f)),
        }
    attempts = [a for e in eps for a in e["facts"]["attempts"]]
    finished = [a for a in attempts if a["grasped"] is not None]
    by_failure = defaultdict(lambda: [0, 0])
    for a in finished:
        key = a["failure"] if not a["regrasp"] else f"regrasp{a['regrasp']}"
        by_failure[key][0] += int(a["grasped"])
        by_failure[key][1] += 1
    grasps = {
        "attempts": len(attempts),
        "attempts_reaching_lift_end": len(finished),
        "grasped": int(sum(a["grasped"] for a in finished)),
        "grasped_fraction": float(np.mean([a["grasped"] for a in finished])) if finished else None,
        "by_failure_or_regrasp": {k: {"grasped": v[0], "of": v[1]} for k, v in by_failure.items()},
    }
    # ----- frames -----
    frames = {}
    for camera in pc.CAMERAS:
        cs = [s[camera] for s in stats]
        decoded = sum(c["decoded"] for c in cs)
        frames[camera] = {
            "shape": cs[0]["shape"],
            "decoded_frames": decoded,
            "stride": stride,
            "blank_frames_std_lt_2": int(sum(c["blank"] for c in cs)),
            "frozen_consecutive_pairs": int(sum(c["frozen_pairs"] for c in cs)),
            "cube_visible_fraction": float(sum(c["cube_visible"] for c in cs) / decoded),
            "min_frame_std": float(min(c["min_std"] for c in cs)),
            "mean_brightness": float(np.mean([c["mean_brightness"] for c in cs])),
            "png_bytes_mean": float(np.mean([c["png_bytes_mean"] for c in cs])),
        }
    # ----- timing -----
    reports = sorted((corpus / "reports").glob("shard-*.json"))
    per_ep = [
        e["collect_seconds"]
        for r in reports
        for e in json.loads(r.read_text())["episodes"]
        if "collect_seconds" in e
    ]
    episodes_out = [
        {k: e[k] for k in ("episode_id", "seed", "mode", "stop")}
        | {"split": s, "commands": e["facts"]["commands"], "skills": e["skills"]}
        for s in SPLITS
        for e in per_split[s]
    ]
    return {
        "corpus": str(corpus),
        "revision": summary["revision"],
        "kind": summary["kind"],
        "problems": problems,
        "shards": shards,
        "storage": storage
        | {
            "gib": storage["bytes"] / 2**30,
            "mib_per_episode": storage["bytes"] / max(storage["episodes"], 1) / 2**20,
            "kib_per_frame_both_cameras": storage["bytes"] / max(storage["frames"], 1) / 2**10,
        },
        "timing": {
            "collection_wall_seconds": summary["seconds"],
            "workers": summary.get("workers"),
            "episodes": len(per_ep),
            "episode_seconds_mean": float(np.mean(per_ep)) if per_ep else None,
            "episode_seconds_p95": float(np.quantile(per_ep, 0.95)) if per_ep else None,
            "frame_check_seconds": frame_seconds,
        },
        "modes": dict(Counter(e["mode"] for e in eps)),
        "skills": dict(Counter(s[0] for e in eps for s in e["skills"])),
        "failures": dict(Counter(s[1] for e in eps for s in e["skills"] if s[1] != "none")),
        "stop_reasons": dict(Counter(e["stop"] or "(full length)" for e in eps)),
        "phase_steps": dict(
            zip(pc.PHASES, np.sum([e["phase"] for e in eps], axis=0).tolist(), strict=True)
        ),
        "leakage": leakage,
        "leakage_total": int(leak_total),
        "action_coverage": coverage,
        "clipping": clip,
        "joint_coverage": joint_cov,
        "gripper": gripper,
        "windows": windows,
        "grasps": grasps,
        "frames": frames,
        "episodes": episodes_out,
    }


def contact_sheet(corpus: Path, report: dict, out: Path, *, count: int = 4, times: int = 8):
    """Both cameras at ``times`` evenly spaced frames of ``count`` episodes (one PNG)."""
    import pyarrow.parquet as pq
    from PIL import Image

    from embodied_jepa.schema_data import SchemaDatasetStore

    episodes = report["episodes"]
    picks = [episodes[i] for i in np.linspace(0, len(episodes) - 1, count).astype(int)]
    rows = []
    for e in picks:
        shard = next(
            s["shard"]
            for s in report["shards"]
            if "episodes" in s
            and any(
                r["episode_id"] == e["episode_id"]
                for r in SchemaDatasetStore(corpus / f"shard-{s['shard']:02d}").manifest["episodes"]
            )
        )
        root = corpus / f"shard-{shard:02d}"
        row = next(
            r
            for r in SchemaDatasetStore(root).manifest["episodes"]
            if r["episode_id"] == e["episode_id"]
        )
        table = pq.read_table(
            root / row["path"], columns=[f"observation.images.{c}" for c in pc.CAMERAS]
        )
        n = table.num_rows
        for camera in pc.CAMERAS:
            items = table.column(f"observation.images.{camera}").to_pylist()
            tiles = [
                np.asarray(Image.open(io.BytesIO(items[i]["bytes"])).convert("RGB"))
                for i in np.linspace(0, n - 1, times).astype(int)
            ]
            rows.append(np.concatenate(tiles, axis=1))
    sheet = np.concatenate(rows, axis=0)
    Image.fromarray(sheet).save(out / "contact_sheet.png")
    return [e["episode_id"] for e in picks]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=1, help="decode every k-th frame")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--no-sheet", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    corpus = Path(args.corpus)
    started = time.time()
    report = measure(corpus, stride=args.stride, workers=args.workers)
    report["measure_seconds"] = time.time() - started
    if not args.no_sheet:
        report["contact_sheet_episodes"] = contact_sheet(corpus, report, out)
    pc.write_json(out / "report.json", report)
    brief = {
        k: report[k]
        for k in ("problems", "leakage_total", "storage", "timing", "grasps", "gripper")
    }
    brief["windows"] = report["windows"]
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
