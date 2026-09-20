"""Data alignment, provenance, leakage, and optional upstream compatibility checks."""

import hashlib
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa.contracts import ContractError, StateSchema
from embodied_jepa.data import (
    LEROBOT_REVISION,
    DatasetStore,
    concatenate_batches,
    deterministic_fixture,
)

pytest.importorskip("pyarrow")
pytest.importorskip("pandas")
pytest.importorskip("PIL")


@pytest.fixture
def episode():
    return deterministic_fixture()


def create_store(path, episode):
    return DatasetStore.create(
        path,
        fps=20,
        state_schema=episode.state_schema,
        action_manifest={"physical_scale": [0.01] * 14, "purpose": "software fixture only"},
        provenance={"source": "procedural fixture", "license": "Apache-2.0"},
        robot_type="software_fixture",
    )


@pytest.fixture
def store(tmp_path, episode):
    return create_store(tmp_path / "dataset", episode)


def populate(store, episode):
    for session in range(6):
        for index in range(2):
            store.write_episode(
                replace(
                    episode,
                    episode_id=f"episode-{session}-{index}",
                    session_id=f"session-{session}",
                    robot_states=episode.robot_states + np.float32(session),
                )
            )
    for object_id in ("apple", "cube"):
        store.write_episode(
            replace(
                episode,
                episode_id=f"holdout-{object_id}",
                session_id="heldout-session",
                object_id=object_id,
                container_id="plate",
                robot_states=episode.robot_states + 10000,
            )
        )


def test_lossless_roundtrip_and_final_observation_alignment(store, episode):
    store.write_episode(episode)
    reloaded = DatasetStore(store.root)
    restored = reloaded.read_episode(episode.episode_id)
    for key in ("robot_states", "state_mask", "actions", "timestamps", "raw_actions"):
        np.testing.assert_array_equal(getattr(restored, key), getattr(episode, key))
    np.testing.assert_array_equal(restored.observations["head"], episode.observations["head"])
    assert restored.metadata == episode.metadata
    assert restored.truncated and not restored.terminated
    windows = list(reloaded.windows(3))
    assert len(windows) == 2
    assert windows[0].terminated.tolist() == [[False, False, False]]
    assert windows[1].terminated.tolist() == [[False, False, True]]
    np.testing.assert_array_equal(
        windows[1].observations["head"][0, -1], episode.observations["head"][-1]
    )
    np.testing.assert_array_equal(windows[1].actions[0], episode.actions[1:])


def test_no_window_crosses_episodes_and_sampling_is_deterministic(store, episode):
    populate(store, episode)
    splits = store.freeze_splits(seed=5)
    windows = list(store.windows(4, split="train"))
    assert len(windows) == len(splits["train"])
    for batch in windows:
        assert len(batch.episode_ids) == 1
        np.testing.assert_array_equal(batch.timestamps[0], episode.timestamps)
    a = store.sample_batch(8, 3, seed=10)
    b = DatasetStore(store.root).sample_batch(8, 3, seed=10)
    assert a.episode_ids == b.episode_ids
    np.testing.assert_array_equal(a.robot_states, b.robot_states)
    with pytest.raises(ContractError, match="no complete windows"):
        store.sample_batch(2, 50)


def test_split_sessions_and_holdout_pairings_never_leak(store, episode, tmp_path):
    populate(store, episode)
    first = store.freeze_splits(seed=19)
    assert first["holdout"] == ["holdout-apple", "holdout-cube"]
    session_split = {}
    for split, ids in first.items():
        for episode_id in ids:
            item = store.read_episode(episode_id)
            if item.session_id in session_split:
                assert session_split[item.session_id] == split
            session_split[item.session_id] = split
    other = create_store(tmp_path / "other", episode)
    populate(other, episode)
    assert other.freeze_splits(seed=19) == first
    assert other.manifest_hash == store.manifest_hash
    with pytest.raises(ContractError, match="sealed"):
        store.write_episode(replace(episode, episode_id="new"))
    with pytest.raises(ContractError, match="already frozen"):
        store.freeze_splits(seed=20)


def test_normalization_uses_only_training_and_masks(store, episode):
    populate(store, episode)
    splits = store.freeze_splits(seed=2)
    stats = store.fit_normalization()
    training = [store.read_episode(episode_id) for episode_id in splits["train"]]
    values = np.concatenate([item.robot_states for item in training])
    actual = stats["stats"]["observation.state"]
    assert actual["mean"][0] == pytest.approx(float(values[:, 0].mean()), abs=1e-6)
    assert actual["mean"][0] < 10  # the reserved 10,000-valued observations never entered fitting
    assert actual["mean"][1] == 0 and actual["std"][1] == 1
    assert stats["state_valid_count"] == [len(values), 0]
    assert stats["episode_ids"] == splits["train"]
    assert set(stats["episode_ids"]).isdisjoint(splits["val"] + splits["test"] + splits["holdout"])
    first_hash = store.manifest_hash
    assert store.fit_normalization() == stats
    assert store.manifest_hash == first_hash
    DatasetStore(store.root).verify()


def test_fixtures_and_manifest_hashes_reproduce(store, episode, tmp_path):
    store.write_episode(episode)
    other = create_store(tmp_path / "copy", episode)
    other.write_episode(deterministic_fixture())
    assert other.manifest_hash == store.manifest_hash
    assert other.manifest["sha256"] == store.manifest["sha256"]


@pytest.mark.parametrize(
    "relative", ["meta/info.json", "meta/tasks.parquet", "data/chunk-000/file-000.parquet"]
)
def test_corrupt_or_missing_file_is_rejected_on_open(store, episode, relative):
    store.write_episode(episode)
    path = store.root / relative
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ContractError, match="hash mismatch"):
        DatasetStore(store.root)
    path.unlink()
    with pytest.raises(ContractError, match="missing"):
        DatasetStore(store.root)


def rewrite_shard(store, mutation):
    import pyarrow as pa
    import pyarrow.parquet as pq

    relative = store.manifest["episodes"][0]["path"]
    path = store.root / relative
    table = pq.read_table(path)
    values = table.to_pydict()
    mutation(values)
    pq.write_table(pa.Table.from_pydict(values, schema=table.schema), path)
    # Rebind the hash deliberately to exercise semantic checks separately from integrity.
    store.manifest["sha256"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "key,bad",
    [
        ("episode_index", 99),
        ("frame_index", 99),
        ("index", 99),
        ("action_valid", False),
        ("next.done", True),
    ],
)
def test_corrupt_alignment_rejected_even_with_rebound_hash(store, episode, key, bad):
    store.write_episode(episode)
    rewrite_shard(store, lambda values: values[key].__setitem__(0, bad))
    with pytest.raises(ContractError):
        store.read_episode(episode.episode_id)


@pytest.mark.parametrize("bad", [None, {"bytes": b"not a png", "path": None}])
def test_missing_or_invalid_image_rejected(store, episode, bad):
    store.write_episode(episode)
    rewrite_shard(store, lambda values: values["observation.images.head"].__setitem__(0, bad))
    with pytest.raises(ContractError, match="RGB frame"):
        store.read_episode(episode.episode_id)


@pytest.mark.parametrize(
    "change",
    [
        lambda ep: {"actions": np.full((4, 14), 1.1, dtype=np.float32)},
        lambda ep: {"timestamps": np.array([0, 0.1, 0.1, 0.2, 0.3], dtype=np.float64)},
        lambda ep: {"robot_states": np.zeros((6, 2), dtype=np.float32)},
        lambda ep: {"state_mask": np.zeros((5, 2), dtype=np.bool_)},
        lambda ep: {"raw_actions": np.full((4, 14), np.nan, dtype=np.float32)},
        lambda ep: {"terminated": True, "truncated": True},
        lambda ep: {"terminated": False, "truncated": False},
        lambda ep: {"observations": {"head": ep.observations["head"][:-1]}},
    ],
)
def test_invalid_episode_fails_before_disk_writes(episode, change):
    with pytest.raises(ContractError):
        replace(episode, **change(episode))


def test_dropped_frames_duplicate_ids_and_schema_mismatch_rejected(store, episode):
    with pytest.raises(ContractError, match="FPS"):
        store.write_episode(replace(episode, timestamps=episode.timestamps * 2))
    store.write_episode(episode)
    with pytest.raises(ContractError, match="duplicate"):
        store.write_episode(episode)
    wrong = StateSchema(
        ("another.q", "fixture.optional"), ("rad", "1"), "fixture_v0", (True, False)
    )
    with pytest.raises(ContractError, match="state order"):
        store.write_episode(replace(episode, episode_id="another", state_schema=wrong))
    with pytest.raises(ContractError, match="camera"):
        store.write_episode(
            replace(
                episode, episode_id="another", observations={"new": episode.observations["head"]}
            )
        )


def test_unfrozen_normalization_and_insufficient_sessions_fail(store, episode):
    store.write_episode(episode)
    with pytest.raises(ContractError, match="unfrozen"):
        store.fit_normalization()
    with pytest.raises(ContractError, match="three"):
        store.freeze_splits()
    with pytest.raises(FileExistsError):
        create_store(store.root, episode)


def test_concatenate_keeps_schema_and_masks(episode):
    combined = concatenate_batches([episode.sequence(0, 2), episode.sequence(1, 2)])
    assert combined.batch_size == 2
    assert not combined.state_mask[..., 1].any()
    with pytest.raises(ContractError):
        concatenate_batches([episode.sequence(0, 2), episode.sequence(0, 3)])


@pytest.mark.parametrize("dimension", [1, 86])
def test_state_dimensions_are_schema_driven_not_hardcoded(tmp_path, episode, dimension):
    schema = StateSchema(
        tuple(f"joint-{i}" for i in range(dimension)), ("rad",) * dimension, "test_v0"
    )
    states = np.arange(5 * dimension, dtype=np.float32).reshape(5, dimension)
    changed = replace(
        episode,
        state_schema=schema,
        robot_states=states,
        state_mask=np.ones_like(states, dtype=np.bool_),
    )
    store = create_store(tmp_path / "dimension", changed)
    store.write_episode(changed)
    restored = store.read_episode(changed.episode_id)
    np.testing.assert_array_equal(restored.robot_states, states)
    assert restored.sequence().robot_states.shape == (1, 5, dimension)


@pytest.mark.parametrize("dimension", [1, 2, 86])
def test_official_lerobot_reader(tmp_path, episode, monkeypatch, dimension):
    """Opt in with LEROBOT_SOURCE=third_party/lerobot and the data-compat extra."""
    source = os.environ.get("LEROBOT_SOURCE")
    if not source:
        pytest.skip("set LEROBOT_SOURCE to the pinned upstream checkout for compatibility evidence")
    source = Path(source).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == LEROBOT_REVISION
    monkeypatch.syspath_prepend(str(source / "src"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    schema = StateSchema(
        tuple(f"joint-{i}" for i in range(dimension)), ("rad",) * dimension, "test_v0"
    )
    states = np.arange(5 * dimension, dtype=np.float32).reshape(5, dimension)
    episode = replace(
        episode,
        state_schema=schema,
        robot_states=states,
        state_mask=np.ones_like(states, dtype=np.bool_),
    )
    store = create_store(tmp_path / "official", episode)
    store.write_episode(episode)
    store.write_episode(
        replace(episode, episode_id="second", session_id="second", task="second task")
    )
    official = LeRobotDataset(repo_id="local/jepa-fixture", root=store.root, video_backend="pyav")
    assert len(official) == 10
    assert official.meta.total_episodes == 2
    assert official.meta.get_data_file_path(1).as_posix() == "data/chunk-000/file-001.parquet"
    frame = official[0]
    assert frame["task"] == episode.task
    assert official[5]["task"] == "second task"
    rgb = frame["observation.images.head"].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(rgb * 255, episode.observations["head"][0], atol=1e-5)
    np.testing.assert_array_equal(
        np.atleast_1d(frame["observation.state"].numpy()), episode.robot_states[0]
    )
    np.testing.assert_array_equal(frame["action"].numpy(), episode.actions[0])
    assert not official[4]["action_valid"].item()
    assert official[3]["next.truncated"].item()
    # Upstream hf_transform_to_torch casts scalar Python floats to torch's default float32.
    assert official[4]["observation.timestamp"].item() == pytest.approx(episode.timestamps[-1])
