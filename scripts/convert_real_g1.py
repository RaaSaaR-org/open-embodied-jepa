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
        "patterns": ["README*", "LICEN*", "NOTICE*", "*/meta/**", "*/data/**", "*/videos/**"],
        "video": True,
    },
    "nvidia-appletoplate": {
        "repo": "nvidia/GR00T-N1.7-AppleToPlate",
        "provider": "nvidia",
        "license": "cc-by-4.0",
        "role": "holdout",
        "patterns": BASE + ["data/**", "videos/**"],
        "video": True,
    },
}
# AppleToPlate's state names are not joint names: its modality.json groups and, inside a group,
# the order of NVIDIA's Teleop set (arm in MJCF order; hand index, middle, thumb). An assumption
# for the held-out set only; it is outside the decision.
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
            names = [None] * 43
            for group, span in modality.items():
                side, kind = group.split("_", 1)
                if kind in NVIDIA_GROUP_ORDER:
                    for i, part in enumerate(NVIDIA_GROUP_ORDER[kind]):
                        names[span["start"] + i] = f"{side}_{part}"
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


def frames_at(video: dict, indices: np.ndarray, fps: float) -> np.ndarray:
    """Decode the source frames at ``indices`` (episode-relative), resized to 112 px."""
    import av

    wanted = {int(round(video["from"] * fps)) + int(i): k for k, i in enumerate(indices)}
    out = np.zeros((len(indices), rc.IMAGE_SIZE, rc.IMAGE_SIZE, 3), np.uint8)
    got = np.zeros(len(indices), bool)
    last = max(wanted)
    with av.open(str(video["path"])) as container:
        stream = container.streams.video[0]
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


def convert(raw: Path, out: Path, names, limit=None) -> dict:
    from embodied_jepa.contracts import StateSchema
    from embodied_jepa.data import DatasetStore, Episode

    downloads = json.loads((raw / "downloads.json").read_text())
    kin = rc.Kinematics()
    schema = StateSchema(
        tuple(f"{n}.position" for n in kin.joint_names)
        + tuple(f"{n}.velocity" for n in kin.joint_names),
        ("rad",) * len(kin.joint_names) + ("rad/s",) * len(kin.joint_names),
        "g1_dex3_proprio_v0",
    )
    pooled, pooled_target = Pool(), Pool()
    report = {"sets": {}, "stores": {}}
    for name in names:
        spec, dl = SOURCES[name], downloads.get(name, {})
        if dl.get("status") != "downloaded":
            report["sets"][name] = {"status": dl.get("status", "missing")}
            continue
        root = raw / name
        loader = load_unitree if spec["provider"] == "unitree" else load_nvidia
        per, per_target = Pool(), Pool()
        store = None
        store_episodes, sessions = [], []
        tasks = {}
        started = time.time()
        for n_ep, (key, task, jnames, state, action, fps, video) in enumerate(loader(root)):
            if limit and n_ep >= limit:
                break
            tasks[task] = tasks.get(task, 0) + 1
            c = rc.convert_episode(kin, jnames, state, fps)
            per.add(c)
            ct = rc.convert_episode(kin, jnames, action, fps)
            per_target.add(ct)
            if spec["role"] == "train":
                pooled.add(c)
                pooled_target.add(ct)
            want_store = (
                spec["video"] and video is not None and spec["role"] in ("train", "holdout")
            )
            if not want_store:
                continue
            segs = rc.segments(c.flagged)
            if not segs:
                continue
            if store is None:
                store = DatasetStore.create(
                    out / name,
                    fps=rc.TARGET_FPS,
                    state_schema=schema,
                    action_manifest=kin.manifest,
                    provenance={
                        "task": "TASK-086",
                        "protocol": "docs/experiments/real_g1_dex3_prestep.md",
                        "repo": spec["repo"],
                        "revision": dl["revision"],
                        "license": spec["license"],
                        "role": spec["role"],
                        "camera": rc.CAMERA,
                        "converter": "real_g1_convert v0",
                    },
                )
            lo, hi = segs[0][0], segs[-1][1]
            images = frames_at(video, c.image_index[lo : hi + 1], fps)
            for s, e in segs:
                eid = f"{name}-{key.replace('/', '-')}-{s:05d}"
                episode = Episode(
                    episode_id=eid,
                    session_id=f"{name}-{key}",
                    task=task or name,
                    object_id=f"real_{name}",
                    container_id="real",
                    observations={rc.CAMERA: images[s - lo : e - lo + 1]},
                    robot_states=c.state[s : e + 1],
                    state_mask=c.state_mask[s : e + 1],
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
                store.write_episode(episode)
                store_episodes.append(eid)
            sessions.append(key)
        entry = {
            "role": spec["role"],
            "state": per.summary(),
            "action_target": per_target.summary(),
            "tasks": tasks,
            "seconds": time.time() - started,
        }
        if store is not None:
            if spec["role"] == "train":
                keys = sorted(set(sessions))
                split = rc.split_for(spec["repo"], keys)
                assign = {f"{name}-{s}": k for k, v in split.items() for s in v}
                by_split = {"train": [], "val": [], "test": [], "holdout": []}
                for row in store.manifest["episodes"]:
                    by_split[assign[row["session_id"]]].append(row["episode_id"])
                store.freeze_split_assignments(
                    by_split,
                    provenance={"rule": "protocol §4", "salt": rc.SPLIT_SALT},
                    heldout_combinations=(),
                )
                entry["splits"] = {k: len(v) for k, v in split.items()}
            else:
                entry["splits"] = "unsealed: real_test_holdout"
            report["stores"][name] = {
                "episodes": len(store_episodes),
                "manifest_hash": store.manifest_hash,
                "role": "real_test_holdout" if spec["role"] == "holdout" else "train",
            }
        report["sets"][name] = entry
    report["pooled_train_eligible"] = pooled.summary()
    report["pooled_action_target"] = pooled_target.summary()
    void = []
    for name, spec in SOURCES.items():
        if spec["role"] == "train" and downloads.get(name, {}).get("status") != "downloaded":
            void.append(f"{name} not downloaded")
    report["decision"] = rc.decide(pooled.summary(), void="; ".join(void))
    return report


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
    report = convert(raw, out, names, limit=args.limit)
    if "unitree-graspsquare" in names and "unitree-blockstacking" in names:
        report["graspsquare_vs_blockstacking"] = duplicate_check(raw)
    report["downloads_sha256"] = sha256(raw / "downloads.json")
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
