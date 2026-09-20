"""Software fixtures for split-safe matched collection, not physics evidence."""

import copy
import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pandas")
pytest.importorskip("PIL")
from embodied_jepa.contracts import ContractError
from embodied_jepa.data import DatasetStore, deterministic_fixture

spec = importlib.util.spec_from_file_location(
    "apple_branches", Path(__file__).resolve().parents[1] / "scripts/collect_apple_branches.py"
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def test_plan_roots_are_phase_contained_and_parent_split_preserved():
    phases = [p for p in collector.PHASES for _ in range(30)]
    manifest = {
        "splits": {
            "train": [f"t{i}" for i in range(9)],
            "val": ["v0", "v1", "v2"],
            "test": ["sealed"],
        },
        "episodes": [
            {
                "episode_id": name,
                "session_id": name,
                "metadata": {"variant": "nominal", "phase_labels": phases},
            }
            for name in [*[f"t{i}" for i in range(9)], "v0", "v1", "v2", "sealed"]
        ],
    }
    plan = collector.make_plan(manifest)
    assert len(plan["roots"]) == 66 and len(plan["attempts"]) == 528
    assert all(r["parent_episode_id"] not in ("sealed", "t8") for r in plan["roots"])
    for row in plan["roots"]:
        assert phases[row["frame_index"] : row["frame_index"] + 16] == [row["phase"]] * 16
    assert plan == collector.make_plan(manifest)


def test_branch_actions_have_sustained_declared_effect_and_no_input_mutation():
    recorded = np.linspace(-0.5, 0.5, 14, dtype=np.float32)
    saved = recorded.copy()
    grasp = np.array([-0.7, 0.2], np.float32)
    hold = collector.branch_action(recorded, "hold", grasp)
    assert not hold[:12].any()
    np.testing.assert_array_equal(hold[12:], grasp)
    reverse = collector.branch_action(recorded, "reverse_translation", grasp)
    np.testing.assert_array_equal(reverse[6:9], -recorded[6:9])
    np.testing.assert_array_equal(reverse[9:], recorded[9:])
    assert collector.branch_action(recorded, "open", grasp)[13] == -1
    assert collector.branch_action(recorded, "closed", grasp)[13] == 1
    up = collector.branch_action(recorded, "up", grasp)
    assert up[8] == 0.25 and not up[9:12].any()
    np.testing.assert_array_equal(recorded, saved)


def sealed_store(tmp_path):
    fixture = deterministic_fixture()
    store = DatasetStore.create(
        tmp_path / "source",
        fps=20,
        state_schema=fixture.state_schema,
        action_manifest={"synthetic": True},
        provenance={"synthetic": True},
    )
    for name in ("train", "val", "test"):
        store.write_episode(replace(fixture, episode_id=name, session_id=name))
    splits = {x: [x] for x in ("train", "val", "test")} | {"holdout": []}
    store.freeze_split_assignments(splits, provenance={"synthetic": True}, heldout_combinations=())
    store.fit_normalization()
    return store, fixture, splits


def test_encoded_fork_never_decodes_test_preserves_bytes_and_parent_groups(tmp_path, monkeypatch):
    source, fixture, splits = sealed_store(tmp_path)
    original_hash = source.manifest_hash

    def no_decode(*args):
        raise AssertionError("encoded fork must not decode any episodes")

    monkeypatch.setattr(DatasetStore, "read_episode", no_decode)
    fork = source.fork_unsealed(tmp_path / "fork", provenance={"synthetic": True})
    assert fork.manifest["splits"] is None and "normalization" not in fork.manifest
    assert source.manifest_hash == original_hash
    for row in source.manifest["episodes"]:
        assert (source.root / row["path"]).read_bytes() == (fork.root / row["path"]).read_bytes()
    fork.write_episode(replace(fixture, episode_id="sibling", session_id="train"))
    wrong = copy.deepcopy(splits)
    wrong["val"].append("sibling")
    with pytest.raises(ContractError, match="parent session"):
        fork.freeze_split_assignments(
            wrong, provenance={"synthetic": True}, heldout_combinations=()
        )
    right = copy.deepcopy(splits)
    right["train"].append("sibling")
    fork.freeze_split_assignments(right, provenance={"synthetic": True}, heldout_combinations=())
    fork.verify()


def test_encoded_fork_refuses_corruption_and_overwrite(tmp_path):
    source, _, _ = sealed_store(tmp_path)
    with pytest.raises(FileExistsError):
        source.fork_unsealed(source.root, provenance={"synthetic": True})
    path = source.root / source.manifest["episodes"][0]["path"]
    path.write_bytes(b"corrupt")
    with pytest.raises(ContractError, match="hash mismatch"):
        source.fork_unsealed(tmp_path / "fork", provenance={"synthetic": True})
    assert not (tmp_path / "fork").exists()


def test_supervisor_failure_preserves_all_attempts(tmp_path):
    output = tmp_path / "out"
    attempts = [
        {"episode_id": "a", "status": "not_started"},
        {"episode_id": "b", "status": "not_started"},
    ]
    collector.write(
        output / "branch_report.json", {"attempts": [{"episode_id": "a", "status": "running"}]}
    )
    assert collector.finalize(output, {"attempts": attempts}, -9, True) == 2
    import json

    report = json.loads((output / "branch_report.json").read_text())
    assert report["training_ready"] is False and len(report["attempts"]) == 2
    assert report["attempts"][0]["status"] == "interrupted_unknown_execution"
    assert report["attempts"][1]["status"] == "not_started_supervisor_timeout"


def test_pair_distance_reports_real_future_and_action_contrast():
    frame = (np.zeros((8, 8, 3), np.uint8), np.zeros(2, np.float32), np.ones(2, bool), 0.0)
    changed = (np.full((8, 8, 3), 2, np.uint8), np.ones(2, np.float32), frame[2], 0.8)
    endpoints = {
        "a": {16: (frame, np.zeros((16, 14), np.float32))},
        "b": {16: (changed, np.ones((16, 14), np.float32))},
    }
    metric = collector.paired_metrics("root", "train", "close", endpoints)["pairs"][0]
    assert metric["visually_distinct"] and metric["rgb_rms"] == pytest.approx(2 / 255)
    assert metric["applied_action_rms"] == 1 and metric["measured_state_rms"] == 1


def test_journal_keeps_complete_transition_and_pending_command(tmp_path):
    frame = (np.zeros((8, 8, 3), np.uint8), np.zeros(2, np.float32), np.ones(2, bool), 0.05)
    collector.journal_frame(tmp_path, 0, frame)
    collector.write(tmp_path / "command.json", {"index": 1, "status": "pending"})
    collector.journal_frame(tmp_path, 1, frame, action=np.ones(14, np.float32))
    with np.load(tmp_path / "frame-001.npz") as data:
        assert data["action"].shape == (14,) and data["rgb"].shape == (8, 8, 3)
    assert not list(tmp_path.glob("*.tmp"))


def test_encoded_fork_cannot_refit_partition(tmp_path):
    source, _, _ = sealed_store(tmp_path)
    fork = source.fork_unsealed(tmp_path / "fork", provenance={"synthetic": True})
    with pytest.raises(ContractError, match="preserved explicit"):
        fork.freeze_splits(seed=99)


def test_low_contrast_preserves_completed_collection_but_is_not_training_ready(tmp_path):
    output = tmp_path / "out"
    records = [{"episode_id": "a", "status": "completed", "executed_steps": 16}]
    collector.write(
        output / "branch_report.json",
        {
            "status": "completed_not_training_ready",
            "training_ready": False,
            "attempts": records,
            "action_contrast_gate_passed": False,
        },
    )
    assert collector.finalize(output, {"attempts": records}, 2, False) == 2
    import json

    result = json.loads((output / "branch_report.json").read_text())
    assert result["status"] == "completed_not_training_ready"
    assert result["attempts"][0]["status"] == "completed" and not result["training_ready"]
