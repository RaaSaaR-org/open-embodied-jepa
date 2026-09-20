"""Small-corpus LeRobot v3 PNG/Parquet storage with canonical JEPA sequences.

Optional Arrow/Pillow/pandas imports occur only at storage calls. The format
targets LeRobot v0.4.4, commit 8fff0fde7c79f23a93d845d1a50e985de01f8b8a.
One immutable data shard per episode favors crash diagnosis over throughput.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from embodied_jepa.contracts import (
    ACTION_DIM,
    EE_DELTA_GRASP_V0,
    ContractError,
    SequenceBatch,
    StateSchema,
)

LEROBOT_REVISION = "8fff0fde7c79f23a93d845d1a50e985de01f8b8a"
MANIFEST = "meta/jepa_manifest.json"
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
EPISODES_PATH = "meta/episodes/chunk-000/file-000.parquet"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _storage():
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image
    except ImportError as error:
        raise ImportError("Dataset storage requires the project 'data' extra") from error
    return pa, pq, pd, Image


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ContractError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Episode:
    episode_id: str
    session_id: str
    task: str
    object_id: str
    container_id: str
    observations: Mapping[str, np.ndarray]
    robot_states: np.ndarray
    state_mask: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray
    state_schema: StateSchema
    terminated: bool = False
    truncated: bool = True
    raw_actions: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in ("episode_id", "session_id", "task", "object_id", "container_id"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ContractError(f"{key} must be a nonempty string")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ContractError("terminated and truncated must be booleans")
        if self.terminated == self.truncated:
            raise ContractError("a stored episode must have exactly one terminal/truncation reason")
        if not isinstance(self.actions, np.ndarray) or self.actions.ndim != 2:
            raise ContractError("episode actions must be an array [T,14]")
        boundary = np.zeros((1, self.actions.shape[0]), dtype=np.bool_)
        if boundary.shape[1]:
            boundary[0, -1] = True
        # Validate complete arrays before slicing: never hide extra/missing frames.
        batch = SequenceBatch(
            observations={key: value[None] for key, value in self.observations.items()},
            robot_states=self.robot_states[None],
            state_mask=self.state_mask[None],
            actions=self.actions[None],
            timestamps=self.timestamps[None],
            terminated=boundary,
            episode_ids=(self.episode_id,),
            state_schema=self.state_schema,
        )
        object.__setattr__(
            self, "observations", {name: frames[0] for name, frames in batch.observations.items()}
        )
        for target, source in (
            ("robot_states", "robot_states"),
            ("state_mask", "state_mask"),
            ("actions", "actions"),
            ("timestamps", "timestamps"),
        ):
            object.__setattr__(self, target, getattr(batch, source)[0])
        if self.raw_actions is not None:
            raw = self.raw_actions
            if (
                not isinstance(raw, np.ndarray)
                or raw.dtype != np.float32
                or raw.shape != self.actions.shape
                or not np.isfinite(raw).all()
            ):
                raise ContractError("raw_actions must be finite float32 [T,14] physical commands")
            raw = raw.copy()
            raw.flags.writeable = False
            object.__setattr__(self, "raw_actions", raw)
        # Ensure metadata can be saved before writing any files; disallow NaN.
        object.__setattr__(self, "metadata", json.loads(_json_bytes(dict(self.metadata))))

    @property
    def transitions(self) -> int:
        return self.actions.shape[0]

    def sequence(self, start: int = 0, horizon: int | None = None) -> SequenceBatch:
        horizon = self.actions.shape[0] if horizon is None else horizon
        _positive(horizon, "horizon")
        if type(start) is not int or start < 0 or start + horizon > self.actions.shape[0]:
            raise ContractError("sequence window exceeds episode boundaries")
        boundary = np.zeros((1, horizon), dtype=np.bool_)
        boundary[0, -1] = start + horizon == self.actions.shape[0]
        end = start + horizon
        return SequenceBatch(
            observations={
                key: value[None, start : end + 1] for key, value in self.observations.items()
            },
            robot_states=self.robot_states[None, start : end + 1],
            state_mask=self.state_mask[None, start : end + 1],
            actions=self.actions[None, start:end],
            timestamps=self.timestamps[None, start : end + 1],
            terminated=boundary,
            episode_ids=(self.episode_id,),
            state_schema=self.state_schema,
        )


def concatenate_batches(batches: list[SequenceBatch]) -> SequenceBatch:
    if not batches:
        raise ContractError("cannot concatenate an empty list of windows")
    first = batches[0]
    for batch in batches:
        if (
            batch.state_schema != first.state_schema
            or batch.action_schema != first.action_schema
            or batch.horizon != first.horizon
            or set(batch.observations) != set(first.observations)
        ):
            raise ContractError("batch schemas, horizons, and cameras must agree")
    values = first.as_mapping()
    values["observations"] = {
        key: np.concatenate([batch.observations[key] for batch in batches])
        for key in first.observations
    }
    for name in ("robot_states", "state_mask", "actions", "timestamps", "terminated"):
        values[name] = np.concatenate([getattr(batch, name) for batch in batches])
    values["episode_ids"] = sum((batch.episode_ids for batch in batches), ())
    return SequenceBatch.from_mapping(values)


class DatasetStore:
    """Single-writer local v3 store. Existing content is hash-checked on every open.

    Appends are allowed until splits are frozen. Never edit a sealed corpus in
    place: create another version so all runs retain identical manifest hashes.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / MANIFEST).read_text())
        if self.manifest.get("format") != "jepa_lerobot_v3_png_v0":
            raise ContractError("unsupported dataset schema")
        self.state_schema = StateSchema(**self.manifest["state_schema"])
        if self.manifest["action_schema"] != json.loads(_json_bytes(asdict(EE_DELTA_GRASP_V0))):
            raise ContractError("dataset action ordering/schema is incompatible")
        self.verify()

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        fps: int,
        state_schema: StateSchema,
        action_manifest: Mapping[str, Any],
        provenance: Mapping[str, Any],
        robot_type: str = "unitree_g1_dex3",
    ) -> DatasetStore:
        _positive(fps, "fps")
        root = Path(root)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty dataset {root}")
        if not action_manifest or not provenance:
            raise ContractError("action conversion and source/license provenance are required")
        manifest = {
            "format": "jepa_lerobot_v3_png_v0",
            "lerobot_revision": LEROBOT_REVISION,
            "fps": fps,
            "robot_type": robot_type,
            "state_schema": asdict(state_schema),
            "action_schema": asdict(EE_DELTA_GRASP_V0),
            "action_manifest": dict(action_manifest),
            "provenance": dict(provenance),
            "episodes": [],
            "sha256": {},
            "splits": None,
        }
        _write_json(root / MANIFEST, manifest)
        return cls(root)

    @property
    def manifest_hash(self) -> str:
        return _hash(self.root / MANIFEST)

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(row["episode_id"] for row in self.manifest["episodes"])

    def verify(self) -> None:
        for relative, expected in self.manifest["sha256"].items():
            path = self.root / relative
            if not path.is_file() or _hash(path) != expected:
                raise ContractError(f"dataset hash mismatch or missing file: {relative}")

    def _record_hash(self, relative: str) -> None:
        self.manifest["sha256"][relative] = _hash(self.root / relative)

    def _persist(self) -> None:
        _write_json(self.root / MANIFEST, self.manifest)

    def _features(self, episode: Episode) -> dict:
        features = {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "observation.timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "observation.state": {
                "dtype": "float32",
                "shape": [self.state_schema.dimension],
                "names": list(self.state_schema.names),
            },
            "observation.state_mask": {
                "dtype": "bool",
                "shape": [self.state_schema.dimension],
                "names": list(self.state_schema.names),
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": list(EE_DELTA_GRASP_V0.names),
            },
            "action.physical": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": list(EE_DELTA_GRASP_V0.names),
            },
        }
        for key in ("index", "episode_index", "frame_index", "task_index"):
            features[key] = {"dtype": "int64", "shape": [1], "names": None}
        for key in (
            "action_valid",
            "action.physical_valid",
            "next.done",
            "next.terminated",
            "next.truncated",
        ):
            features[key] = {"dtype": "bool", "shape": [1], "names": None}
        for name, frames in episode.observations.items():
            features[f"observation.images.{name}"] = {
                "dtype": "image",
                "shape": list(frames.shape[1:]),
                "names": ["height", "width", "channels"],
            }
        return features

    def write_episode(self, episode: Episode) -> None:
        if self.manifest["splits"] is not None:
            raise ContractError("dataset is sealed; create a new version before adding episodes")
        if episode.episode_id in self.episode_ids:
            raise ContractError("duplicate episode_id")
        if episode.state_schema != self.state_schema:
            raise ContractError("episode state order/units/schema differs from dataset")
        delta = np.diff(episode.timestamps)
        if not np.allclose(delta, 1 / self.manifest["fps"], atol=1e-6, rtol=1e-3):
            raise ContractError(
                "dropped/misaligned frames: timestamps disagree with configured FPS"
            )
        self.verify()
        pa, pq, pd, image = _storage()
        features = self._features(episode)
        if self.manifest["episodes"]:
            previous = json.loads((self.root / "meta/info.json").read_text())["features"]
            if features != previous:
                raise ContractError("camera names/resolutions or feature schemas changed")
        episodes = self.manifest["episodes"]
        index = len(episodes)
        tasks = list(dict.fromkeys([row["task"] for row in episodes] + [episode.task]))
        n = episode.transitions + 1
        offset = sum(row["length"] for row in episodes)
        actions = np.concatenate([episode.actions, np.zeros((1, ACTION_DIM), dtype=np.float32)])
        physical = np.zeros_like(actions)
        if episode.raw_actions is not None:
            physical[:-1] = episode.raw_actions
        valid = np.arange(n) < n - 1
        done = np.arange(n) == n - 2
        columns = {
            "timestamp": pa.array((episode.timestamps - episode.timestamps[0]).astype(np.float32)),
            "observation.timestamp": pa.array(episode.timestamps),
            "observation.state": pa.array(
                episode.robot_states[:, 0]
                if self.state_schema.dimension == 1
                else episode.robot_states.tolist(),
                type=pa.float32() if self.state_schema.dimension == 1 else pa.list_(pa.float32()),
            ),
            "observation.state_mask": pa.array(
                episode.state_mask[:, 0]
                if self.state_schema.dimension == 1
                else episode.state_mask.tolist(),
                type=pa.bool_() if self.state_schema.dimension == 1 else pa.list_(pa.bool_()),
            ),
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32())),
            "action.physical": pa.array(physical.tolist(), type=pa.list_(pa.float32())),
            "index": pa.array(np.arange(offset, offset + n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, index, dtype=np.int64)),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "task_index": pa.array(np.full(n, tasks.index(episode.task), dtype=np.int64)),
            "action_valid": pa.array(valid),
            "action.physical_valid": pa.array(valid & (episode.raw_actions is not None)),
            "next.done": pa.array(done),
            "next.terminated": pa.array(done & episode.terminated),
            "next.truncated": pa.array(done & episode.truncated),
        }
        for name, frames in episode.observations.items():
            encoded = []
            for frame in frames:
                stream = io.BytesIO()
                image.fromarray(frame).save(stream, format="PNG")
                encoded.append({"bytes": stream.getvalue(), "path": None})
            columns[f"observation.images.{name}"] = pa.array(
                encoded, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])
            )
        relative = DATA_PATH.format(chunk_index=index // 1000, file_index=index % 1000)
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"uncommitted shard exists; inspect before recovery: {target}")
        pq.write_table(pa.table(columns), target, compression="zstd")
        episodes.append(
            {
                "episode_index": index,
                "episode_id": episode.episode_id,
                "session_id": episode.session_id,
                "task": episode.task,
                "object_id": episode.object_id,
                "container_id": episode.container_id,
                "terminated": episode.terminated,
                "truncated": episode.truncated,
                "has_raw_actions": episode.raw_actions is not None,
                "metadata": dict(episode.metadata),
                "length": n,
                "path": relative,
                "dataset_from_index": offset,
                "dataset_to_index": offset + n,
            }
        )
        metadata_rows = [
            {
                "episode_index": row["episode_index"],
                "tasks": [row["task"]],
                "length": row["length"],
                "dataset_from_index": row["dataset_from_index"],
                "dataset_to_index": row["dataset_to_index"],
                "data/chunk_index": row["episode_index"] // 1000,
                "data/file_index": row["episode_index"] % 1000,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            for row in episodes
        ]
        (self.root / EPISODES_PATH).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(metadata_rows), self.root / EPISODES_PATH)
        pd.DataFrame({"task_index": range(len(tasks))}, index=tasks).to_parquet(
            self.root / "meta/tasks.parquet", engine="pyarrow"
        )
        _write_json(
            self.root / "meta/info.json",
            {
                "codebase_version": "v3.0",
                "robot_type": self.manifest["robot_type"],
                "total_episodes": len(episodes),
                "total_frames": offset + n,
                "total_tasks": len(tasks),
                "chunks_size": 1000,
                "data_files_size_in_mb": 100,
                "video_files_size_in_mb": 500,
                "fps": self.manifest["fps"],
                "splits": {"all": f"0:{len(episodes)}"},
                "data_path": DATA_PATH,
                "video_path": None,
                "features": features,
            },
        )
        for path in (relative, EPISODES_PATH, "meta/tasks.parquet", "meta/info.json"):
            self._record_hash(path)
        self._persist()

    def read_episode(self, episode_id: str) -> Episode:
        row = next(
            (row for row in self.manifest["episodes"] if row["episode_id"] == episode_id), None
        )
        if row is None:
            raise KeyError(episode_id)
        path = self.root / row["path"]
        if not path.exists() or _hash(path) != self.manifest["sha256"][row["path"]]:
            raise ContractError(f"episode hash mismatch or missing file: {episode_id}")
        pa, pq, _, image = _storage()
        table = pq.read_table(path)
        info = json.loads((self.root / "meta/info.json").read_text())
        if set(table.column_names) != set(info["features"]):
            raise ContractError("missing or unknown Parquet feature columns")
        for key, feature in info["features"].items():
            if feature["dtype"] == "image":
                continue  # Image nulls and encoded content get explicit diagnostics below.
            column = table[key]
            dtype = pa.type_for_alias(feature["dtype"])
            if feature["shape"] != [1]:
                if not (pa.types.is_list(column.type) or pa.types.is_fixed_size_list(column.type)):
                    raise ContractError(f"incorrect vector storage for {key}")
                if column.type.value_type != dtype:
                    raise ContractError(f"incorrect vector dtype for {key}")
            elif column.type != dtype:
                raise ContractError(f"incorrect scalar dtype for {key}")
            if column.null_count:
                raise ContractError(f"missing values in required column {key}")
            if feature["shape"] != [1] and any(
                len(value) != feature["shape"][0] or any(item is None for item in value)
                for value in column.to_pylist()
            ):
                raise ContractError(f"missing values or incorrect vector width in {key}")
        data = table.to_pydict()
        n = row["length"]
        tasks = list(dict.fromkeys(item["task"] for item in self.manifest["episodes"]))
        for key, expected in {
            "frame_index": list(range(n)),
            "episode_index": [row["episode_index"]] * n,
            "index": list(range(row["dataset_from_index"], row["dataset_to_index"])),
            "action_valid": [True] * (n - 1) + [False],
            "next.done": [False] * (n - 2) + [True, False],
            "next.terminated": [False] * (n - 2) + [row["terminated"], False],
            "next.truncated": [False] * (n - 2) + [row["truncated"], False],
            "action.physical_valid": [row["has_raw_actions"]] * (n - 1) + [False],
            "task_index": [tasks.index(row["task"])] * n,
        }.items():
            if data.get(key) != expected:
                raise ContractError(f"corrupt {key} or episode boundary in {episode_id}")
        if any(value != 0 for value in data["action"][-1]):
            raise ContractError("end observation row contains a non-placeholder action")
        images = {}
        for key, feature in info["features"].items():
            if feature["dtype"] != "image":
                continue
            frames = []
            for item in data[key]:
                if not item or not item.get("bytes"):
                    raise ContractError(f"missing embedded RGB frame: {episode_id}/{key}")
                try:
                    with image.open(io.BytesIO(item["bytes"])) as decoded:
                        if (
                            decoded.mode != "RGB"
                            or list(np.asarray(decoded).shape) != feature["shape"]
                        ):
                            raise ContractError("RGB mode/resolution differs from feature schema")
                        frames.append(np.asarray(decoded).copy())
                except (OSError, ValueError) as error:
                    raise ContractError(f"invalid RGB frame: {episode_id}/{key}") from error
            images[key.removeprefix("observation.images.")] = np.stack(frames)
        timestamps = np.asarray(data["observation.timestamp"], dtype=np.float64)
        if not np.allclose(data["timestamp"], timestamps - timestamps[0], atol=1e-6, rtol=1e-6):
            raise ContractError("LeRobot and raw timestamp fields disagree")
        if not np.allclose(np.diff(timestamps), 1 / self.manifest["fps"], atol=1e-6, rtol=1e-3):
            raise ContractError("dropped/misaligned frames in episode")
        return Episode(
            episode_id=row["episode_id"],
            session_id=row["session_id"],
            task=row["task"],
            object_id=row["object_id"],
            container_id=row["container_id"],
            observations=images,
            robot_states=np.asarray(data["observation.state"], dtype=np.float32).reshape(n, -1),
            state_mask=np.asarray(data["observation.state_mask"], dtype=np.bool_).reshape(n, -1),
            actions=np.asarray(data["action"][:-1], dtype=np.float32),
            timestamps=timestamps,
            state_schema=self.state_schema,
            terminated=row["terminated"],
            truncated=row["truncated"],
            raw_actions=(
                np.asarray(data["action.physical"][:-1], dtype=np.float32)
                if row["has_raw_actions"]
                else None
            ),
            metadata=row["metadata"],
        )

    def freeze_splits(
        self,
        *,
        seed: int = 0,
        ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        heldout_combinations: tuple[tuple[str, str], ...] = (("apple", "plate"),),
    ) -> Mapping[str, list[str]]:
        """Deterministic whole-session split; any held-out pairing reserves its entire session."""
        if self.manifest["splits"] is not None:
            raise ContractError("splits already frozen")
        if (
            len(ratios) != 3
            or any(not np.isfinite(x) or x <= 0 for x in ratios)
            or not np.isclose(sum(ratios), 1)
        ):
            raise ContractError("three positive train/val/test ratios must sum to one")
        self.verify()
        sessions: dict[str, list[dict]] = {}
        for row in self.manifest["episodes"]:
            sessions.setdefault(row["session_id"], []).append(row)
        reserved = set(heldout_combinations)
        holdout = {
            session
            for session, rows in sessions.items()
            if any((row["object_id"], row["container_id"]) in reserved for row in rows)
        }
        available = sorted(set(sessions) - holdout)
        if len(available) < 3:
            raise ContractError("at least three non-held-out sessions required for train/val/test")
        rng = np.random.default_rng(seed)
        rng.shuffle(available)
        n = len(available)
        n_val = max(1, int(round(n * ratios[1])))
        n_test = max(1, int(round(n * ratios[2])))
        if n_val + n_test >= n:
            raise ContractError("split ratios leave no training sessions")
        groups = {
            "val": available[:n_val],
            "test": available[n_val : n_val + n_test],
            "train": available[n_val + n_test :],
            "holdout": sorted(holdout),
        }
        splits = {
            name: sorted(row["episode_id"] for session in selected for row in sessions[session])
            for name, selected in groups.items()
        }
        self.manifest["splits"] = splits
        self.manifest["split_policy"] = {
            "seed": seed,
            "ratios": ratios,
            "group_by": "session_id",
            "heldout_combinations": list(heldout_combinations),
            "sessions": groups,
        }
        _write_json(
            self.root / "meta/jepa_splits.json",
            self.manifest["split_policy"] | {"episodes": splits},
        )
        self._record_hash("meta/jepa_splits.json")
        self._persist()
        return splits

    def freeze_split_assignments(
        self,
        splits: Mapping[str, list[str]],
        *,
        provenance: Mapping[str, Any],
        heldout_combinations: tuple[tuple[str, str], ...] = (("apple", "plate"),),
    ) -> Mapping[str, list[str]]:
        """Seal inherited partitions without moving prior test sessions into training."""
        if self.manifest["splits"] is not None:
            raise ContractError("splits already frozen")
        names = {"train", "val", "test", "holdout"}
        if set(splits) != names or not provenance:
            raise ContractError("explicit splits require four partitions and provenance")
        if any(
            not isinstance(ids, list) or any(not isinstance(i, str) for i in ids)
            for ids in splits.values()
        ):
            raise ContractError("split assignments must be lists of episode IDs")
        flat = [episode_id for ids in splits.values() for episode_id in ids]
        if len(flat) != len(set(flat)) or set(flat) != set(self.episode_ids):
            raise ContractError("split assignments must cover every episode exactly once")
        if any(not splits[name] for name in ("train", "val", "test")):
            raise ContractError("train/val/test partitions must be nonempty")
        self.verify()
        assignment = {episode_id: name for name, ids in splits.items() for episode_id in ids}
        sessions = {}
        reserved = set(heldout_combinations)
        for row in self.manifest["episodes"]:
            name = assignment[row["episode_id"]]
            previous = sessions.setdefault(row["session_id"], name)
            if previous != name:
                raise ContractError("session cannot span split partitions")
            if (row["object_id"], row["container_id"]) in reserved and name != "holdout":
                raise ContractError("held-out pairing must remain in holdout")
        # Validate serialization before mutating the manifest or writing files.
        policy = json.loads(
            json.dumps(
                {
                    "group_by": "session_id",
                    "method": "preserved_explicit_assignments",
                    "provenance": dict(provenance),
                    "heldout_combinations": list(heldout_combinations),
                    "sessions": {
                        name: sorted(s for s, p in sessions.items() if p == name) for name in names
                    },
                },
                allow_nan=False,
            )
        )
        selected = {name: sorted(splits[name]) for name in sorted(names)}
        self.manifest["splits"], self.manifest["split_policy"] = selected, policy
        _write_json(self.root / "meta/jepa_splits.json", policy | {"episodes": selected})
        self._record_hash("meta/jepa_splits.json")
        self._persist()
        return selected

    def _selected(self, split: str | None) -> list[str]:
        if split is None:
            return list(self.episode_ids)
        if self.manifest["splits"] is None or split not in self.manifest["splits"]:
            raise ContractError(f"unknown or unfrozen split {split!r}")
        return self.manifest["splits"][split]

    def windows(self, horizon: int, *, split: str | None = None) -> Iterator[SequenceBatch]:
        _positive(horizon, "horizon")
        for episode_id in self._selected(split):
            episode = self.read_episode(episode_id)
            for start in range(episode.transitions - horizon + 1):
                yield episode.sequence(start, horizon)

    def sample_batch(
        self, batch_size: int, horizon: int, *, split: str = "train", seed: int = 0
    ) -> SequenceBatch:
        _positive(batch_size, "batch_size")
        _positive(horizon, "horizon")
        selected = set(self._selected(split))
        windows = [
            (row["episode_id"], start)
            for row in self.manifest["episodes"]
            if row["episode_id"] in selected
            for start in range(row["length"] - horizon)
        ]
        if not windows:
            raise ContractError("split has no complete windows for the requested horizon")
        indices = np.random.default_rng(seed).integers(len(windows), size=batch_size)
        cache = {}
        batches = []
        for index in indices:
            episode_id, start = windows[index]
            if episode_id not in cache:
                cache[episode_id] = self.read_episode(episode_id)
            batches.append(cache[episode_id].sequence(start, horizon))
        return concatenate_batches(batches)

    def fit_normalization(self) -> dict[str, Any]:
        """Fit only frozen training episodes; optional missing state is excluded per field."""
        selected = self._selected("train")
        if not selected:
            raise ContractError("normalization requires nonempty frozen training data")
        count = np.zeros(self.state_schema.dimension, dtype=np.int64)
        mean, m2 = np.zeros_like(count, dtype=np.float64), np.zeros_like(count, dtype=np.float64)
        minimum, maximum = np.full_like(mean, np.inf), np.full_like(mean, -np.inf)
        actions = []
        for episode_id in selected:
            episode = self.read_episode(episode_id)
            actions.append(episode.actions)
            for values, mask in zip(episode.robot_states, episode.state_mask, strict=True):
                count[mask] += 1
                delta = values[mask] - mean[mask]
                mean[mask] += delta / count[mask]
                m2[mask] += delta * (values[mask] - mean[mask])
                minimum[mask] = np.minimum(minimum[mask], values[mask])
                maximum[mask] = np.maximum(maximum[mask], values[mask])
        std = np.sqrt(m2 / np.maximum(count, 1))
        std[std < 1e-6] = 1
        minimum[count == 0], maximum[count == 0] = 0, 0
        action_values = np.concatenate(actions).astype(np.float64)
        action_std = action_values.std(axis=0)
        action_std[action_std < 1e-6] = 1
        stats = {
            "observation.state": {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "min": minimum.tolist(),
                "max": maximum.tolist(),
                "count": [int(count.max())],
            },
            "action": {
                "mean": action_values.mean(axis=0).tolist(),
                "std": action_std.tolist(),
                "min": action_values.min(axis=0).tolist(),
                "max": action_values.max(axis=0).tolist(),
                "count": [len(action_values)],
            },
        }
        normalization = {
            "fit_split": "train",
            "episode_ids": selected,
            "state_valid_count": count.tolist(),
            "stats": stats,
        }
        _write_json(self.root / "meta/stats.json", stats)
        _write_json(self.root / "meta/jepa_normalization.json", normalization)
        for relative in ("meta/stats.json", "meta/jepa_normalization.json"):
            self._record_hash(relative)
        self.manifest["normalization"] = normalization
        self._persist()
        return normalization


def deterministic_fixture(
    episode_id: str = "fixture-0", session_id: str = "fixture-session-0"
) -> Episode:
    """Apache-2.0 procedural pixels and signals; software fixture, never manipulation evidence."""
    schema = StateSchema(
        ("fixture.q", "fixture.optional"), ("rad", "1"), "fixture_v0", (True, False)
    )
    images = np.arange(5 * 8 * 8 * 3, dtype=np.uint16).reshape(5, 8, 8, 3).astype(np.uint8)
    states = np.stack([np.arange(5) / 10, np.zeros(5)], axis=-1).astype(np.float32)
    masks = np.ones_like(states, dtype=np.bool_)
    masks[:, 1] = False
    actions = np.linspace(-0.5, 0.5, 4 * ACTION_DIM, dtype=np.float32).reshape(4, ACTION_DIM)
    return Episode(
        episode_id,
        session_id,
        "fixture reach",
        "cube",
        "target",
        {"head": images},
        states,
        masks,
        actions,
        np.arange(5, dtype=np.float64) / 20,
        schema,
        raw_actions=actions * np.float32(0.01),
        metadata={"software_fixture_only": True},
    )
