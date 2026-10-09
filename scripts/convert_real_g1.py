"""TASK-086: download, convert and measure open real G1 + Dex3 data (real_g1_dex3_prestep.md).

    .venv/bin/python scripts/convert_real_g1.py download --raw data/real-g1-raw
    .venv/bin/python scripts/convert_real_g1.py convert --raw data/real-g1-raw \
        --out data/real-g1-v1 --report <evidence>/conversion_report.json

``download`` pins each repository to the revision it resolves at that moment and records the
license tag, the files, their sizes and sha256. ``convert`` measures every set (states and
actions) and writes a ``DatasetStore`` for each set whose video was downloaded. Never overwrites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import real_g1_convert as rc  # noqa: E402

UNITREE = (
    "BlockStacking",
    "CameraPackaging",
    "ObjectPlacement",
    "Pouring",
    "ToastedBread",
    "PickApple",
    "PickBottle",
    "PickCharger",
    "PickGum",
    "PickSnack",
    "PickTissue",
    "PickDoll",
)
BASE = ["meta/**", "README*", "LICEN*", "NOTICE*"]
SOURCES = {
    **{
        f"unitree-{name.lower()}": {
            "repo": f"unitreerobotics/G1_Dex3_{name}_Dataset",
            "provider": "unitree",
            "license": "apache-2.0",
            "role": "train",
            "patterns": BASE
            + ["data/**"]
            + (["videos/observation.images.cam_left_high/**"] if name == "PickApple" else []),
            "video": name == "PickApple",
        }
        for name in UNITREE
    },
    "unitree-graspsquare": {
        "repo": "unitreerobotics/G1_Dex3_GraspSquare_Dataset",
        "provider": "unitree",
        "license": "apache-2.0",
        "role": "excluded-duplicate-check",
        "patterns": BASE + ["data/**"],
        "video": False,
    },
    "nvidia-teleop-g1": {
        "repo": "nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1",
        "provider": "nvidia",
        "license": "cc-by-4.0",
        "role": "train",
        "protocol_revision": "0d7bdd06e6",
        "patterns": ["README*", "LICEN*", "NOTICE*", "*/meta/**", "*/data/**", "*/videos/**"],
        "video": True,
    },
    "nvidia-appletoplate": {
        "repo": "nvidia/GR00T-N1.7-AppleToPlate",
        "provider": "nvidia",
        "license": "cc-by-4.0",
        "role": "holdout",
        "protocol_revision": "d89c126a71",
        "patterns": BASE + ["data/**", "videos/**"],
        "video": True,
    },
}
# AppleToPlate's state names are not joint names: its modality.json groups and, inside a group,
# the order of NVIDIA's Teleop set (arm in MJCF order; hand index, middle, thumb). An assumption
# for the held-out set only; it is outside the decision.
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
NVIDIA_GROUP_ORDER = {
    "arm": [f"{p}_joint" for _, p in rc.ARM_PARTS],
    "hand": [
        f"hand_{f}_joint"
        for f in ("index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2")
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


# ----- download ---------------------------------------------------------------------------------
def download(raw: Path, names, floor_gib: float) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    record = {}
    for name in names:
        spec = SOURCES[name]
        info = api.dataset_info(spec["repo"], files_metadata=True)
        tags = [t for t in (info.tags or []) if t.startswith("license:")]
        entry = {
            "repo": spec["repo"],
            "revision": info.sha,
            "license_tags": tags,
            "gated": info.gated,
            "role": spec["role"],
        }
        pinned = spec.get("protocol_revision")
        if pinned and not info.sha.startswith(pinned):
            entry["status"] = f"not-downloaded: head {info.sha[:10]} is not the protocol's {pinned}"
            record[name] = entry
            continue
        if info.gated or f"license:{spec['license']}" not in tags:
            entry["status"] = "not-downloaded: gated or license tag differs"
            record[name] = entry
            continue
        free = shutil.disk_usage(raw.parent if raw.parent.exists() else ".").free / 2**30
        if free < floor_gib:
            entry["status"] = f"not-downloaded: {free:.1f} GiB free < {floor_gib}"
            record[name] = entry
            continue
        target = raw / name
        snapshot_download(
            spec["repo"],
            repo_type="dataset",
            revision=info.sha,
            local_dir=target,
            allow_patterns=spec["patterns"],
        )
        lfs = {s.rfilename: (s.lfs or {}).get("sha256") for s in info.siblings}
        files = {}
        for p in sorted(target.rglob("*")):
            if p.is_file() and ".cache" not in p.parts:
                rel = str(p.relative_to(target))
                digest = sha256(p)
                if lfs.get(rel) and lfs[rel] != digest:
                    raise SystemExit(f"{name}/{rel}: sha256 differs from the Hub's LFS oid")
                files[rel] = {"bytes": p.stat().st_size, "sha256": digest}
        entry.update(
            status="downloaded",
            files=files,
            notice=[f for f in files if "NOTICE" in f.upper() or "LICEN" in f.upper()],
        )
        record[name] = entry
    return record


# ----- loading ------------------------------------------------------------------------------------
def load_unitree(root: Path):
    """Yield (episode_key, task, names, state, action, fps, video) per episode."""
    import pyarrow.parquet as pq

    info = json.loads((root / "meta/info.json").read_text())
    names = info["features"]["observation.state"]["names"]
    names = names[0] if isinstance(names[0], list) else names
    action_names = info["features"]["action"]["names"]
    action_names = action_names[0] if isinstance(action_names[0], list) else action_names
    fps = float(info["fps"])
    episodes = pq.read_table(root / "meta/episodes").to_pylist()
    tables = {}
    for p in sorted((root / "data").rglob("*.parquet")):
        t = pq.read_table(
            p, columns=["observation.state", "action", "episode_index", "frame_index"]
        )
        tables[p] = t
    by_episode = {}
    for t in tables.values():
        ep = np.asarray(t.column("episode_index").to_pylist())
        st = np.asarray(t.column("observation.state").to_pylist(), float)
        ac = np.asarray(t.column("action").to_pylist(), float)
        fi = np.asarray(t.column("frame_index").to_pylist())
        for e in np.unique(ep):
            m = ep == e
            order = np.argsort(fi[m])
            by_episode[int(e)] = (st[m][order], ac[m][order])
    key = "videos/observation.images.cam_left_high"
    for row in episodes:
        e = int(row["episode_index"])
        state, action = by_episode[e]
        video = None
        if f"{key}/file_index" in row:
            video = {
                "path": root
                / key
                / f"chunk-{row[f'{key}/chunk_index']:03d}"
                / f"file-{row[f'{key}/file_index']:03d}.mp4",
                "from": float(row[f"{key}/from_timestamp"]),
            }
        task = (row.get("tasks") or [""])[0]
        if action_names != names:
            raise ValueError("unitree action and state names differ")
        yield f"{e:06d}", task, rc.map_names(names, provider="unitree"), state, action, fps, video


def load_nvidia(root: Path):
    import pyarrow.parquet as pq

    folders = sorted(p.parent.parent for p in root.rglob("meta/info.json"))
    for folder in folders:
        info = json.loads((folder / "meta/info.json").read_text())
        fps = float(info["fps"])
        names = info["features"]["observation.state"]["names"]
        if len(names) != 43 or not all(str(n).endswith("_joint") for n in names):
            modality = json.loads((folder / "meta/modality.json").read_text())["state"]
            names = names_from_modality(modality, 43)
        names = rc.map_names(names, provider="nvidia")
        tasks = {}
        for line in (folder / "meta/tasks.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                tasks[row["task_index"]] = row["task"]
        prefix = "" if folder == root else f"{folder.name}/"
        for p in sorted((folder / "data").rglob("episode_*.parquet")):
            t = pq.read_table(p, columns=["observation.state", "action", "task_index"])
            state = np.asarray(t.column("observation.state").to_pylist(), float)
            action = np.asarray(t.column("action").to_pylist(), float)[:, :43]
            index = int(p.stem.split("_")[-1])
            video = (
                folder
                / "videos"
                / p.parent.name
                / "observation.images.ego_view"
                / (p.stem + ".mp4")
            )
            task = tasks.get(int(t.column("task_index")[0].as_py()), "")
            yield (
                f"{prefix}{index:06d}",
                task,
                list(names),
                state,
                action,
                fps,
                {"path": video, "from": 0.0} if video.exists() else None,
            )


def names_from_modality(modality: dict, width: int) -> list:
    """Joint names from GR00T modality groups (state groups only; legs ignored)."""
    names = [None] * width
    for group, span in modality.items():
        if span.get("original_key", "observation.state") != "observation.state":
            continue
        if group == "waist":
            parts = WAIST
        elif "_" in group and group.split("_", 1)[1] in NVIDIA_GROUP_ORDER:
            side, kind = group.split("_", 1)
            parts = [f"{side}_{p}" for p in NVIDIA_GROUP_ORDER[kind]]
        else:
            continue  # legs and anything else: not used
        if span["end"] - span["start"] != len(parts):
            raise ValueError(f"modality group {group} has {span['end'] - span['start']} joints")
        names[span["start"] : span["end"]] = parts
    return names


def frames_at(video: dict, indices: np.ndarray, fps: float) -> np.ndarray:
    """Decode the source frames at ``indices`` (episode-relative), resized to 112 px."""
    import av

    wanted = {int(round(video["from"] * fps)) + int(i): k for k, i in enumerate(indices)}
    out = np.zeros((len(indices), rc.IMAGE_SIZE, rc.IMAGE_SIZE, 3), np.uint8)
    got = np.zeros(len(indices), bool)
    last = max(wanted)
    with av.open(str(video["path"])) as container:
        stream = container.streams.video[0]
        if stream.average_rate and abs(float(stream.average_rate) - fps) > 0.01:
            raise ValueError(f"{video['path']}: video {float(stream.average_rate)} fps != {fps}")
        start = min(wanted) / fps
        container.seek(int(max(start - 1.0, 0) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            n = int(round(float(frame.pts * stream.time_base) * fps))
            if n in wanted:
                img = rc.square_resize(frame.to_ndarray(format="rgb24"))
                for k in [k for i, k in wanted.items() if i == n]:
                    out[k], got[k] = img, True
            if n >= last:
                break
    if not got.all():
        raise ValueError(f"{video['path']}: {int((~got).sum())} frames not decoded")
    return out


# ----- converting -------------------------------------------------------------------------------
class Pool:
    """Pooled counts for the decision (protocol §5)."""

    def __init__(self):
        self.steps = self.valid_steps = self.arm_out = self.grasp_out = 0
        self.invalid_frames = self.kept_steps = 0
        self.residual = {"left": [], "right": []}
        self.t = {"left": [], "right": []}
        self.abs_actions = []

    def add(self, c: rc.Converted):
        s = rc.episode_stats(c)
        for k in ("steps", "valid_steps", "arm_out", "grasp_out", "invalid_frames", "kept_steps"):
            setattr(self, k, getattr(self, k) + s[k])
        for i, side in enumerate(("left", "right")):
            self.residual[side].append(c.residual[c.valid, i])
            self.t[side].append(c.t[c.valid, i])
        ok = c.valid[:-1] & c.valid[1:]
        self.abs_actions.append(np.abs(c.actions[ok]))

    def summary(self) -> dict:
        out = {
            k: getattr(self, k)
            for k in (
                "steps",
                "valid_steps",
                "arm_out",
                "grasp_out",
                "invalid_frames",
                "kept_steps",
            )
        }
        out["arm_out_fraction"] = self.arm_out / max(self.valid_steps, 1)
        out["kept_hours"] = self.kept_steps / rc.TARGET_FPS / 3600
        for side in ("left", "right"):
            r = np.concatenate(self.residual[side]) if self.residual[side] else np.zeros(0)
            t = np.concatenate(self.t[side]) if self.t[side] else np.zeros(0)
            lo, hi = rc.GRASP_OUTSIDE
            out[f"{side}_median_residual"] = float(np.median(r)) if len(r) else float("nan")
            out[f"{side}_outside_fraction"] = float(((t < lo) | (t > hi)).mean()) if len(t) else 0.0
            out[f"{side}_t_percentiles"] = (
                np.percentile(t, [1, 5, 50, 95, 99]).round(3).tolist() if len(t) else []
            )
        a = np.concatenate(self.abs_actions) if self.abs_actions else np.zeros((0, 14))
        if len(a):
            out["abs_action_percentiles_50_95_99"] = (
                np.percentile(a, [50, 95, 99], axis=0).round(3).tolist()
            )
            out["dim_out_fraction"] = (a > 1).mean(axis=0).round(4).tolist()
        return out


def _schema(kin):
    from embodied_jepa.contracts import StateSchema

    # Our 43 joints; only the arm and hand joints are required (legs and waist may be missing).
    required = tuple(
        (
            "_hip_" not in n
            and "_knee_" not in n
            and "_ankle_" not in n
            and not n.startswith("waist_")
        )
        for n in kin.joint_names
    )
    return StateSchema(
        tuple(f"{n}.position" for n in kin.joint_names)
        + tuple(f"{n}.velocity" for n in kin.joint_names),
        ("rad",) * len(kin.joint_names) + ("rad/s",) * len(kin.joint_names),
        "g1_dex3_proprio_v0_real",
        required + required,
    )


def _loader(spec):
    return load_unitree if spec["provider"] == "unitree" else load_nvidia


def measure(raw: Path, names, kin, limit=None) -> dict:
    """Pass 1: every converted step of every downloaded set (no images)."""
    downloads = json.loads((raw / "downloads.json").read_text())
    pooled, pooled_target = Pool(), Pool()
    report = {"sets": {}}
    for name in names:
        spec, dl = SOURCES[name], downloads.get(name, {})
        if dl.get("status") != "downloaded":
            report["sets"][name] = {"status": dl.get("status", "missing")}
            continue
        per, per_target, tasks, started = Pool(), Pool(), {}, time.time()
        for n_ep, (_, task, jnames, state, action, fps, _) in enumerate(_loader(spec)(raw / name)):
            if limit and n_ep >= limit:
                break
            tasks[task] = tasks.get(task, 0) + 1
            c = rc.convert_episode(kin, jnames, state, fps)
            ct = rc.convert_episode(kin, jnames, action, fps)
            per.add(c)
            per_target.add(ct)
            if spec["role"] == "train":
                pooled.add(c)
                pooled_target.add(ct)
        summary = per.summary()
        report["sets"][name] = {
            "role": spec["role"],
            "status": "measured",
            "state": summary,
            "action_target": per_target.summary(),
            "tasks": tasks,
            "episodes": sum(tasks.values()),
            "seconds": time.time() - started,
            "excluded_range": spec["role"] == "train"
            and summary["arm_out_fraction"] > rc.RANGE_BAR,
            "waist": "0 (not in the source)" if spec["provider"] == "unitree" else "from data",
        }
    report["pooled_train_eligible"] = pooled.summary()
    report["pooled_action_target"] = pooled_target.summary()
    void = [
        f"{n} not downloaded"
        for n, spec in SOURCES.items()
        if spec["role"] == "train" and downloads.get(n, {}).get("status") != "downloaded"
    ]
    report["decision"] = rc.decide(pooled.summary(), void="; ".join(void))
    return report


def write_store(raw: Path, out: Path, name: str, kin, revision: str, limit=None) -> dict:
    """Pass 2: one DatasetStore for a set with video (train sealed, holdout unsealed)."""
    from embodied_jepa.data import DatasetStore, Episode

    spec = SOURCES[name]
    schema = _schema(kin)
    store = DatasetStore.create(
        out / name,
        fps=rc.TARGET_FPS,
        state_schema=schema,
        action_manifest=kin.manifest,
        provenance={
            "task": "TASK-086",
            "protocol": "docs/experiments/real_g1_dex3_prestep.md",
            "repo": spec["repo"],
            "revision": revision,
            "license": spec["license"],
            "role": spec["role"],
            "camera": rc.CAMERA,
            "waist": "0 (not in the source)" if spec["provider"] == "unitree" else "from data",
            "converter": "real_g1_convert v0",
        },
    )
    sessions = []
    for n_ep, (key, task, jnames, state, _, fps, video) in enumerate(_loader(spec)(raw / name)):
        if limit and n_ep >= limit:
            break
        if video is None:
            continue
        c = rc.convert_episode(kin, jnames, state, fps)
        segs = rc.segments(c.flagged)
        if not segs:
            continue
        lo, hi = segs[0][0], segs[-1][1]
        images = frames_at(video, c.image_index[lo : hi + 1], fps)
        for s, e in segs:
            values, mask = c.state(kin, s, e)
            store.write_episode(
                Episode(
                    episode_id=f"{name}-{key.replace('/', '-')}-{s:05d}",
                    session_id=f"{name}-{key}",
                    task=task or name,
                    object_id=f"real_{name}",
                    container_id="real",
                    observations={rc.CAMERA: images[s - lo : e - lo + 1]},
                    robot_states=values,
                    state_mask=mask,
                    actions=c.actions[s:e].astype(np.float32),
                    timestamps=np.arange(s, e + 1, dtype=np.float64) / rc.TARGET_FPS,
                    state_schema=schema,
                    terminated=False,
                    truncated=True,
                    metadata={
                        "source_episode": key,
                        "steps": [s, e],
                        "residual_median": [
                            float(np.median(c.residual[s : e + 1, i])) for i in range(2)
                        ],
                    },
                )
            )
        sessions.append(key)
    entry = {"episodes": len(store.manifest["episodes"]), "source_episodes": len(sessions)}
    if spec["role"] == "train":
        split = rc.split_for(spec["repo"], sorted(set(sessions)))
        assign = {f"{name}-{k}": part for part, keys in split.items() for k in keys}
        by_split = {"train": [], "val": [], "test": [], "holdout": []}
        for row in store.manifest["episodes"]:
            by_split[assign[row["session_id"]]].append(row["episode_id"])
        store.freeze_split_assignments(
            by_split,
            provenance={"rule": "protocol §4, §9", "salt": rc.SPLIT_SALT},
            heldout_combinations=(),
        )
        entry.update(role="train", splits={k: len(v) for k, v in split.items()})
    else:
        entry.update(role="real_test_holdout", splits="unsealed")
    entry["manifest_hash"] = store.manifest_hash
    return entry


def duplicate_check(raw: Path) -> dict:
    """GraspSquare against BlockStacking on decoded states and actions (reported only)."""
    a = list(load_unitree(raw / "unitree-blockstacking"))
    b = list(load_unitree(raw / "unitree-graspsquare"))
    same = len(a) == len(b) and all(
        x[3].shape == y[3].shape and np.array_equal(x[3], y[3]) and np.array_equal(x[4], y[4])
        for x, y in zip(a, b, strict=False)
    )
    return {"episodes": [len(a), len(b)], "identical_states_and_actions": bool(same)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--raw", required=True)
    d.add_argument("--only", nargs="*")
    d.add_argument("--floor-gib", type=float, default=12.0)
    c = sub.add_parser("convert")
    c.add_argument("--raw", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--report", required=True)
    c.add_argument("--only", nargs="*")
    c.add_argument("--limit", type=int, default=None, help="episodes per set (debug only)")
    args = parser.parse_args(argv)
    raw = Path(args.raw)
    names = args.only or list(SOURCES)
    if args.cmd == "download":
        raw.mkdir(parents=True, exist_ok=True)
        path = raw / "downloads.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        for name in names:
            if record.get(name, {}).get("status") == "downloaded":
                names = [n for n in names if n != name]
        record.update(download(raw, names, args.floor_gib))
        write_json(path, record)
        print(json.dumps({k: v.get("status") for k, v in record.items()}, indent=1))
        return 0
    out, report_path = Path(args.out), Path(args.report)
    if report_path.exists():
        raise SystemExit(f"refusing to overwrite {report_path}")
    out.mkdir(parents=True, exist_ok=True)
    kin = rc.Kinematics()
    report = measure(raw, names, kin, limit=args.limit)
    report["downloads_sha256"] = sha256(raw / "downloads.json")
    write_json(report_path.with_suffix(".measure.json"), report)  # kept if pass 2 fails
    try:
        if "unitree-graspsquare" in names and "unitree-blockstacking" in names:
            report["graspsquare_vs_blockstacking"] = duplicate_check(raw)
    except Exception as error:  # reported only
        report["graspsquare_vs_blockstacking"] = {"error": repr(error)}
    downloads = json.loads((raw / "downloads.json").read_text())
    report["stores"] = {}
    for name in names:
        spec, entry = SOURCES[name], report["sets"].get(name, {})
        if not spec["video"] or entry.get("status") != "measured":
            continue
        if spec["role"] not in ("train", "holdout") or entry.get("excluded_range"):
            continue
        report["stores"][name] = write_store(
            raw, out, name, kin, downloads[name]["revision"], limit=args.limit
        )
    write_json(report_path, report)
    corpus = out / "corpus.json"
    if not corpus.exists():
        write_json(
            corpus,
            {"task": "TASK-086", "stores": report["stores"], "report_sha256": sha256(report_path)},
        )
    print(json.dumps(report["decision"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
