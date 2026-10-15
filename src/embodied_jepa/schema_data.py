"""The dataset store for any registered action schema (SO-101 S0, TASK-096).

``data.py`` sizes ``Episode`` and ``DatasetStore`` by the 14-D ``ee_delta_grasp_v0`` and is
byte-pinned (TASK-065's frozen manifest, checked by ``tests/test_latent_dynamics.py``), so it stays
as it is. ``SchemaEpisode`` and ``SchemaDatasetStore`` are the same LeRobot v3 PNG/Parquet profile
with the action width, names and final-row placeholder taken from a declared, registered
``ActionSchema``. Their ``__post_init__``, ``sequence``, ``write_episode`` and ``read_episode``
mirror ``data.py`` line for line apart from the schema; a test holds a ``SchemaDatasetStore`` on
``ee_delta_grasp_v0`` to byte-identical files with ``DatasetStore``.

``open_dataset`` opens an ``ee_delta_grasp_v0`` corpus with the pinned ``DatasetStore`` and any
other registered schema with ``SchemaDatasetStore``. Encoded forks (``fork_unsealed``) are not yet
available for schema stores. NumPy at import; storage imports stay lazy.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embodied_jepa.action_schemas import action_schema_from_record, schema_record
from embodied_jepa.contracts import (
    EE_DELTA_GRASP_V0,
    ActionSchema,
    ContractError,
    SequenceBatch,
    StateSchema,
)
from embodied_jepa.data import (
    DATA_PATH,
    EPISODES_PATH,
    LEROBOT_REVISION,
    MANIFEST,
    DatasetStore,
    Episode,
    _hash,
    _json_bytes,
    _positive,
    _storage,
    _write_json,
)


@dataclass(frozen=True)
class SchemaEpisode(Episode):
    """``Episode`` whose actions follow ``action_schema`` (default ``ee_delta_grasp_v0``)."""

    action_schema: ActionSchema = EE_DELTA_GRASP_V0

    def __post_init__(self) -> None:
        for key in ("episode_id", "session_id", "task", "object_id", "container_id"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ContractError(f"{key} must be a nonempty string")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ContractError("terminated and truncated must be booleans")
        if self.terminated == self.truncated:
            raise ContractError("a stored episode must have exactly one terminal/truncation reason")
        if not isinstance(self.action_schema, ActionSchema):
            raise ContractError("episode action_schema must be an ActionSchema")
        action_schema_from_record(schema_record(self.action_schema))  # registered schemas only
        width = self.action_schema.dimension
        if not isinstance(self.actions, np.ndarray) or self.actions.ndim != 2:
            raise ContractError(f"episode actions must be an array [T,{width}]")
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
            action_schema=self.action_schema,
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
                raise ContractError(
                    f"raw_actions must be finite float32 [T,{width}] physical commands"
                )
            raw = raw.copy()
            raw.flags.writeable = False
            object.__setattr__(self, "raw_actions", raw)
        # Ensure metadata can be saved before writing any files; disallow NaN.
        object.__setattr__(self, "metadata", json.loads(_json_bytes(dict(self.metadata))))

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
            action_schema=self.action_schema,
        )


class SchemaDatasetStore(DatasetStore):
    """``DatasetStore`` sized by the corpus's registered action schema; see the module docstring."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / MANIFEST).read_text())
        if self.manifest.get("format") != "jepa_lerobot_v3_png_v0":
            raise ContractError("unsupported dataset schema")
        self.state_schema = StateSchema(**self.manifest["state_schema"])
        try:
            self.action_schema = action_schema_from_record(self.manifest["action_schema"])
        except ContractError as error:
            raise ContractError("dataset action ordering/schema is incompatible") from error
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
        robot_type: str,
        action_schema: ActionSchema,
    ) -> SchemaDatasetStore:
        _positive(fps, "fps")
        if not isinstance(action_schema, ActionSchema):
            raise ContractError("action_schema must be an ActionSchema")
        record = schema_record(action_schema)
        action_schema_from_record(record)  # registered schemas only
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
            "action_schema": record,
            "action_manifest": dict(action_manifest),
            "provenance": dict(provenance),
            "episodes": [],
            "sha256": {},
            "splits": None,
        }
        _write_json(root / MANIFEST, manifest)
        return cls(root)

    def fork_unsealed(self, root, *, provenance):
        raise ContractError("encoded forks are not yet available for schema-sized stores")

    def _features(self, episode) -> dict:
        features = super()._features(episode)
        for key in ("action", "action.physical"):
            features[key] = {
                "dtype": "float32",
                "shape": [self.action_schema.dimension],
                "names": list(self.action_schema.names),
            }
        return features

    def write_episode(self, episode: SchemaEpisode) -> None:
        if self.manifest["splits"] is not None:
            raise ContractError("dataset is sealed; create a new version before adding episodes")
        if episode.episode_id in self.episode_ids:
            raise ContractError("duplicate episode_id")
        if episode.state_schema != self.state_schema:
            raise ContractError("episode state order/units/schema differs from dataset")
        if not isinstance(episode, SchemaEpisode) or episode.action_schema != self.action_schema:
            raise ContractError("episode action schema differs from dataset")
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
        width = self.action_schema.dimension
        actions = np.concatenate([episode.actions, np.zeros((1, width), dtype=np.float32)])
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

    def read_episode(self, episode_id: str) -> SchemaEpisode:
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
        return SchemaEpisode(
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
            action_schema=self.action_schema,
        )


def open_dataset(root: str | Path) -> DatasetStore:
    """The pinned ``DatasetStore`` for ``ee_delta_grasp_v0``; ``SchemaDatasetStore`` otherwise."""
    manifest = json.loads((Path(root) / MANIFEST).read_text())
    try:
        schema = action_schema_from_record(manifest.get("action_schema"))
    except ContractError:
        schema = EE_DELTA_GRASP_V0  # let the pinned store report its own error
    if schema == EE_DELTA_GRASP_V0:
        return DatasetStore(root)
    return SchemaDatasetStore(root)
