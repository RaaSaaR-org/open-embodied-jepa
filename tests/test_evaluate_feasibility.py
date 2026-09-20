"""Synthetic runner and supervision checks; no robot/model experiments."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_feasibility.py"
    spec = importlib.util.spec_from_file_location("evaluate_feasibility", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_script()


def fixture_episode():
    return {
        "termination_reason": "stopped",
        "executed_steps": 1,
        "final_score": {"reach": True, "success": False},
        "trace": [
            {
                "sampled_action": [1.0] * 14,
                "projected_action": [0.2] * 14,
                "requested_action": [0.2] * 14,
                "action": [0.15] * 14,
                "planning_seconds": 0.2,
                "projection_seconds": 0.1,
                "control_seconds": 0.3,
                "reason": "",
                "executed": True,
            },
            {
                "sampled_action": [1.0] * 14,
                "projected_action": [0.2] * 14,
                "requested_action": [0.2] * 14,
                "action": None,
                "reason": "right joint rate limit",
                "executed": False,
            },
        ],
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    protocol = root / "docs/experiments/feasibility_v1.md"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("synthetic protocol")
    monkeypatch.setattr(runner, "verify_inputs", lambda root: None)
    monkeypatch.setattr(runner, "source_hash", lambda: "frozen")
    monkeypatch.setattr(runner, "source_identity", lambda: {"synthetic_fixture_only": True})

    def prepare(root, output, records):
        goal = output / "goal.json"
        goal.write_text("{}")
        for row in records:
            path = output / f"{row['run_id']}.json"
            path.write_text(json.dumps({"task": {"goal_manifest": str(goal)}}))
            row.update(
                config=str(path),
                config_sha256=runner.digest(path),
                goal_manifest_sha256=runner.digest(goal),
            )

    monkeypatch.setattr(runner, "prepare", prepare)
    return root, tmp_path / "result"


def test_metrics_separate_sampled_projected_applied_and_missing():
    metrics = runner.episode_metrics(fixture_episode())
    assert metrics["joint_rate_stop"] and metrics["any_guard_or_runtime_stop"]
    assert metrics["executed_steps"] == 1 and metrics["stages"]["reach"]
    errors = metrics["action_disagreement"]
    assert errors["sampled_vs_projected"]["paired_steps"] == 2
    assert errors["sampled_vs_projected"]["mean_absolute_error"] == pytest.approx(0.8)
    assert errors["execute_requested_vs_applied"]["paired_steps"] == 1
    assert errors["execute_requested_vs_applied"]["mean_absolute_error"] == pytest.approx(0.05)
    episode = fixture_episode()
    episode["termination_reason"] = "step_limit"
    for row in episode["trace"]:
        row.pop("projected_action")
    metrics = runner.episode_metrics(episode)
    assert not metrics["any_guard_or_runtime_stop"]
    assert metrics["action_disagreement"]["sampled_vs_projected"]["mean_absolute_error"] is None


def test_twenty_paired_attempts_failures_retained_and_no_overwrite(setup, monkeypatch):
    root, output = setup
    calls = []

    def supervise(command, log_path, seconds, root):
        run_id = command[-1]
        calls.append(run_id)
        path = output / "runs" / run_id / f"episode-{run_id.split('-')[0]}" / "episode.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(fixture_episode()))
        return {"returncode": 7 if len(calls) == 2 else 0, "timeout": False}

    monkeypatch.setattr(runner, "supervise", supervise)
    report = runner.run(output, "frozen", root=root)
    assert len(calls) == 20 and report["status"] == "incomplete"
    assert report["runs"][1]["status"] == "process_failed"
    assert report["paired_summary"]["native_jepa"]["completed_pairs"] == 4
    assert report["paired_summary"]["native_jepa"]["incomplete_pairs"] == 1
    assert report["paired_summary"]["leworldmodel"]["completed_pairs"] == 5
    for left, right in zip(report["runs"][::2], report["runs"][1::2], strict=True):
        assert (left["seed"], left["backend"]) == (right["seed"], right["backend"])
        assert not left["project_candidates"] and right["project_candidates"]
    with pytest.raises(FileExistsError):
        runner.run(output, "frozen", root=root)


def test_true_wall_budget_lists_unstarted_without_fabricated_metrics(setup, monkeypatch):
    root, output = setup
    wall = [0.0]
    real_clock = runner.RunClock
    monkeypatch.setattr(
        runner,
        "RunClock",
        lambda: real_clock(
            wall_clock=lambda: wall[0],
            monotonic_clock=lambda: 0.0,
            cpu_clock=lambda: 0.0,
        ),
    )

    def supervise(*args):
        wall[0] = 600.0  # Suspension consumes budget even when monotonic time pauses.
        return {"returncode": -9, "timeout": True}

    monkeypatch.setattr(runner, "supervise", supervise)
    report = runner.run(output, "frozen", root=root)
    assert report["runs"][0]["status"] == "incomplete_total_budget"
    assert all(row["status"] == "not_started_total_budget" for row in report["runs"][1:])
    assert all(row["metrics"] is None for row in report["runs"])
    assert report["paired_summary"]["native_jepa"]["completed_pairs"] == 0


def test_source_change_after_attempt_invalidates_comparison(setup, monkeypatch):
    root, output = setup

    def supervise(*args):
        monkeypatch.setattr(runner, "source_hash", lambda: "changed")
        return {"returncode": 0, "timeout": False}

    monkeypatch.setattr(runner, "supervise", supervise)
    report = runner.run(output, "frozen", root=root)
    assert report["runs"][0]["status"] == "failed_integrity_or_orchestration"
    assert all(row["status"] == "not_started_integrity_changed" for row in report["runs"][1:])


def test_configuration_change_during_attempt_is_rejected_afterward(setup, monkeypatch):
    root, output = setup

    def supervise(command, *args):
        path = Path(command[command.index("--config") + 1])
        path.write_text("{}")
        return {"returncode": 0, "timeout": False}

    monkeypatch.setattr(runner, "supervise", supervise)
    report = runner.run(output, "frozen", root=root)
    assert report["runs"][0]["status"] == "failed_integrity_or_orchestration"
    assert "configuration changed" in report["runs"][0]["error"]
    assert all(row["status"] == "not_started_integrity_changed" for row in report["runs"][1:])


def test_setup_failure_preserves_all_planned_records(setup, monkeypatch):
    root, output = setup
    monkeypatch.setattr(runner, "source_hash", lambda: "wrong")
    report = runner.run(output, "frozen", root=root)
    assert report["status"] == "setup_failed" and len(report["runs"]) == 20
    assert all(row["status"] == "not_started_setup_failure" for row in report["runs"])


def test_real_subprocess_timeout_is_bounded(tmp_path):
    result = runner.supervise(
        [sys.executable, "-c", "import time; time.sleep(5)"], tmp_path / "log", 0.1, tmp_path
    )
    assert result["timeout"] and result["returncode"] != 0
    assert result["timing"]["elapsed_seconds"] < 3


def test_goal_reuse_preserves_original_bytes_reset_and_lineage(tmp_path, monkeypatch):
    root = tmp_path / "root"
    original_dir = root / "data/goals/mvp-v0"
    original_dir.mkdir(parents=True)
    image = original_dir / "goal.png"
    image.write_bytes(b"synthetic image fixture")
    source = root / "src/embodied_jepa/fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("revised source")
    original = {
        "seeds": [20000, 20001],
        "runtime_source_hashes": {"fixture.py": "old"},
        "episodes": [
            {
                "seed": 20000,
                "image": "goal.png",
                "sha256": runner.digest(image),
                "reset": {"seed": 20000},
                "target_base": None,
            }
        ],
    }
    parent = original_dir / "manifest.json"
    parent.write_text(json.dumps(original))
    before = parent.read_bytes()
    monkeypatch.setattr(runner, "RUNTIME_SOURCES", ("fixture.py",))
    monkeypatch.setattr(runner, "load_goals", lambda *args, **kwargs: None)
    path = runner.derive_goal(root, tmp_path / "derived", 20000)
    derived = json.loads(path.read_text())
    assert parent.read_bytes() == before
    assert (path.parent / "goal.png").read_bytes() == image.read_bytes()
    assert derived["episodes"][0] == original["episodes"][0]
    assert derived["seeds"] == [20000]
    assert derived["runtime_source_hashes"]["fixture.py"] == runner.digest(source)
    assert derived["development_reuse"]["parent_manifest_sha256"] == runner.digest(parent)
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="image hash"):
        runner.derive_goal(root, tmp_path / "changed", 20000)


def test_action_trace_shape_and_finiteness_are_not_silently_averaged():
    episode = fixture_episode()
    episode["trace"][0]["action"] = [np.nan] * 14
    with pytest.raises(ValueError, match="invalid action"):
        runner.episode_metrics(episode)
