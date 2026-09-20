"""Independent branch-artifact audit fixtures; no physics or inference."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pandas")
pytest.importorskip("PIL")
from embodied_jepa.data import DatasetStore, deterministic_fixture

spec = importlib.util.spec_from_file_location(
    "branch_audit", Path(__file__).resolve().parents[1] / "scripts/audit_apple_branches.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_missing_branches_remain_in_denominator_and_test_payloads_never_decode(
    tmp_path, monkeypatch
):
    episode = deterministic_fixture()
    source = DatasetStore.create(
        tmp_path / "source",
        fps=20,
        state_schema=episode.state_schema,
        action_manifest={"fixture": True},
        provenance={"fixture": True},
    )
    splits = {"train": ["train"], "val": ["val"], "test": ["test"], "holdout": []}
    for name in ("train", "val", "test"):
        source.write_episode(replace(episode, episode_id=name, session_id=name))
    source.freeze_split_assignments(splits, provenance={"fixture": True}, heldout_combinations=())
    source.fit_normalization()
    fork = source.fork_unsealed(tmp_path / "fork", provenance={"fixture": True})
    fork.freeze_split_assignments(splits, provenance={"fixture": True}, heldout_combinations=())
    fork.fit_normalization()
    root = {
        "root_id": "r",
        "parent_episode_id": "train",
        "split": "train",
        "frame_index": 0,
        "phase": "orient",
    }
    plan = {
        "roots": [root],
        "attempts": [root | {"episode_id": "missing", "branch": "hold"}],
        "parent_dataset_sha256": source.manifest_hash,
        "identity": {"fixture": True},
    }
    report = {
        "status": "incomplete",
        "training_ready": False,
        "attempts": [{"episode_id": "missing", "status": "not_started_budget"}],
        "identity": plan["identity"],
        "supervisor_wall_seconds": 595.1,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    (fork.root / "branch_report.json").write_text(json.dumps(report))
    original = DatasetStore.read_episode

    def checked_read(self, name):
        assert name != "test", "TEST image decoding is forbidden"
        return original(self, name)

    monkeypatch.setattr(DatasetStore, "read_episode", checked_read)
    audit = module.audit(source.root, fork.root, plan_path)
    assert audit["status"] == "passed" and not audit["training_ready"]
    assert audit["planned_attempts"] == 1
    assert audit["status_counts"] == {"not_started_budget": 1}
    assert audit["byte_checks"] == {
        "test_payloads_byte_equal": 1,
        "original_payloads_byte_equal": 3,
    }
    assert not audit["test_images_decoded"] and not audit["decoded_parent_ids"]


def test_contrast_gate_requires_four_complete_branches_and_two_visible_pairs():
    roots = [{"root_id": "r", "split": "val", "frame_index": 5}]
    blank = (np.zeros((2, 2, 3), np.uint8), np.zeros(2, np.float32), np.ones(2, bool), 0.8)
    visible = (np.ones((2, 2, 3), np.uint8) * 2, *blank[1:])
    actions = np.zeros((16, 14), np.float32)
    siblings = {key: {16: (blank, actions)} for key in ("a", "b", "c", "d")}
    _, gates = module.contrast(roots, {"r": siblings})
    assert not gates["val"]["passed"]
    siblings["d"] = {16: (visible, actions)}
    metrics, gates = module.contrast(roots, {"r": siblings})
    assert gates["val"]["passed"] and metrics[0]["distinct_h16_pairs"] == 3
    del siblings["c"]
    _, gates = module.contrast(roots, {"r": siblings})
    assert not gates["val"]["passed"]
