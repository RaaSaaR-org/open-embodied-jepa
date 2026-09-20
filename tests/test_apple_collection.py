"""Synthetic task-specific collection checks; never evidence of physical manipulation."""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pandas")
pytest.importorskip("PIL")

from embodied_jepa.data import DatasetStore, deterministic_fixture  # noqa: E402

path = Path(__file__).resolve().parents[1] / "scripts/collect_apple.py"
spec = importlib.util.spec_from_file_location("collect_apple", path)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def test_fixed_plan_and_deterministic_perturbation_recovery_blocks():
    plan = collector.make_plan(-0.035)
    assert len(plan["attempts"]) == 32
    assert Counter(row["variant"] for row in plan["attempts"]) == {
        "nominal": 16,
        "position_burst": 8,
        "closure_burst": 8,
    }
    assert plan["heldout_combinations"] == [] and plan["split_seed"] == 42
    base = np.zeros(14, np.float32)
    for variant in ("nominal", "position_burst", "closure_burst"):
        assert np.array_equal(collector.perturb_action(base, 0, 42, variant), base)
        assert np.array_equal(collector.perturb_action(base, 8, 42, variant), base)
    position = collector.perturb_action(base, 4, 42, "position_burst")
    np.testing.assert_array_equal(position, collector.perturb_action(base, 7, 42, "position_burst"))
    assert np.any(position[6:12]) and not position[:6].any() and not position[12:].any()
    assert np.max(np.abs(position[6:9])) <= 0.04 and np.max(np.abs(position[9:12])) <= 0.02
    closure = collector.perturb_action(base, 4, 42, "closure_burst")
    assert not closure[:13].any() and abs(closure[13]) <= 0.08
    assert not np.array_equal(position, collector.perturb_action(base, 12, 42, "position_burst"))


def test_task_specific_canonical_storage_and_train_only_waypoints(tmp_path, monkeypatch):
    """Exercise actual DatasetStore writes/sealing with synthetic sensor/control fixtures."""
    source = tmp_path / "mechanics.json"
    source.write_text(
        json.dumps({"selection": {"selected": {"successes": 2, "transfer_x_shift": -0.035}}})
    )
    real_plan = collector.make_plan

    def short_plan(shift):
        plan = real_plan(shift)
        plan["attempts"] = plan["attempts"][:3]
        return plan

    monkeypatch.setattr(collector, "make_plan", short_plan)

    class Sim:
        def __init__(self, **kwargs):
            self.step = 0

    class Robot:
        state_schema = deterministic_fixture().state_schema
        manifest = {"synthetic_fixture_only": True}

        def __init__(self, sim):
            self.sim = sim

        def reset(self, seed, **kwargs):
            return {"array_truth": np.array([1, 2]), "synthetic_fixture_only": True}

        def observe(self):
            return SimpleNamespace(
                images={"onboard_rgb": np.full((1, 8, 8, 3), self.sim.step, np.uint8)},
                robot_state=np.full((1, 2), self.sim.step, np.float32),
                state_mask=np.ones((1, 2), np.bool_),
                timestamps=np.array([self.sim.step * 0.05], np.float64),
            )

        def execute(self, action):
            self.sim.step += 1
            return SimpleNamespace(applied_action=action.copy())

        def denormalize_action(self, action):
            return action.copy()

        def stop(self, reason):
            pass

        def close(self):
            pass

    class Policy:
        step_count = 0
        phase = "synthetic_phase"

        @property
        def done(self):
            return self.step_count == 10

        def action(self, robot):
            return np.zeros(14, np.float32)

        def advance(self, result):
            self.step_count += 1

    class Scorer:
        def __init__(self, robot):
            self.robot = robot

        def evaluate(self):
            return {"success": self.robot.sim.step == 10, "synthetic_fixture_only": True}

    monkeypatch.setattr(collector, "MuJoCoSimulation", Sim)
    monkeypatch.setattr(collector, "G1Embodiment", Robot)
    monkeypatch.setattr(collector, "AppleToPlateTask", Scorer)
    monkeypatch.setattr(collector, "make_policy", lambda *args: Policy())
    output = tmp_path / "corpus"
    collector.worker(output, source, collector.digest(source))
    store = DatasetStore(output)
    assert len(store.episode_ids) == 3
    assert not store.manifest["splits"]["holdout"]
    assert store.manifest["normalization"]["episode_ids"] == store.manifest["splits"]["train"]
    for episode_id in store.episode_ids:
        episode = store.read_episode(episode_id)
        assert episode.transitions == 10 and len(episode.robot_states) == 11
        assert episode.task == "apple_to_plate" and episode.terminated
        assert len(episode.metadata["phase_labels"]) == 10
        assert len(episode.metadata["policy_labels"]) == 10
        assert episode.metadata["model_scored_actions"] is None
        assert episode.metadata["reset_truth"]["array_truth"] == [1, 2]
        assert episode.sequence(horizon=4).robot_states.shape == (1, 5, 2)
    goals = json.loads((output / "train_waypoints.json").read_text())
    assert goals["source_manifest_sha256"] == store.manifest_hash
    assert {row["episode_id"] for row in goals["successful_train_episodes"]} == set(
        store.manifest["splits"]["train"]
    )
    for episode in goals["successful_train_episodes"]:
        for image in episode["images"]:
            assert image["sha256"] == collector.digest(output / image["image"])
            assert "timestamp" in image and "robot_state" not in image
    with pytest.raises(FileExistsError):
        collector.worker(output, source, collector.digest(source))


def test_supervisor_marks_active_and_unstarted_trials_incomplete(tmp_path):
    plan = collector.make_plan(-0.035)
    rows = [dict(row) for row in plan["attempts"]]
    rows[0].update(status="success", executed_steps=501)
    rows[1].update(status="attempted")
    collector.write(tmp_path / "collection_report.json", {"attempts": rows})
    report = collector.finalize_supervision(tmp_path, plan, returncode=-9, timed_out=True)
    assert report["status"] == "incomplete" and not report["training_ready"]
    assert report["attempts"][0]["executed_steps"] == 501
    assert report["attempts"][1]["status"] == "interrupted_unknown_execution"
    assert report["attempts"][1]["executed_steps"] is None
    assert all(row["status"] == "not_started_supervisor_budget" for row in report["attempts"][2:])


def test_snapshot_verification_detects_modified_source(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("frozen")
    manifest = tmp_path / "snapshot_manifest.json"
    collector.write(manifest, {"files": {"source.py": collector.digest(source)}})
    expected = collector.digest(manifest)
    collector.verify_snapshot(tmp_path, expected)
    source.write_text("changed")
    with pytest.raises(ValueError, match="frozen snapshot changed"):
        collector.verify_snapshot(tmp_path, expected)


def test_main_child_failure_returns_nonzero_and_preserves_prephysics_plan(tmp_path, monkeypatch):
    mechanics = tmp_path / "mechanics.json"
    mechanics.write_text(
        json.dumps(
            {
                "selection": {
                    "selected": {
                        "successes": 2,
                        "transfer_x_shift": -0.035,
                    }
                }
            }
        )
    )
    output = tmp_path / "corpus"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_apple.py",
            "--output",
            str(output),
            "--mechanics-report",
            str(mechanics),
            "--mechanics-sha256",
            collector.digest(mechanics),
        ],
    )

    class FailedChild:
        returncode = 7

        def __init__(self, command, **kwargs):
            snapshot = Path(command[1]).parents[1]
            assert (snapshot / "frozen_plan.json").is_file()
            assert len(json.loads((snapshot / "frozen_plan.json").read_text())["attempts"]) == 32

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    monkeypatch.setattr(collector.subprocess, "Popen", FailedChild)
    assert collector.main() == 2
    report = json.loads((output / "collection_report.json").read_text())
    assert len(report["attempts"]) == 32 and not report["training_ready"]
    assert report["returncode"] == 7
