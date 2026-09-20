"""Evaluation protocol/accounting fixtures; no physics runs or trained-model claims."""

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa.contracts import ContractError, StateSchema

SPEC = importlib.util.spec_from_file_location(
    "apple_evaluation", Path(__file__).parents[1] / "scripts/evaluate_apple.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_plan_pairs_modes_resets_and_freezes_stage_budgets():
    plan = module.make_plan(
        seeds=[43000, 43001], modes=["learned", "persistence", "dynamics_shuffle"]
    )
    assert len(plan["attempts"]) == 6 and plan["waypoint_dwell"] == 3
    assert plan == module.make_plan(
        seeds=[43000, 43001], modes=["learned", "persistence", "dynamics_shuffle"]
    )
    for seed in plan["seeds"]:
        resets = [x["reset"] for x in plan["attempts"] if x["seed"] == seed]
        assert resets[0] == resets[1] == resets[2]
        assert np.max(np.abs(np.array(resets[0]["object_xy"]) - [0.34, -0.18])) <= 0.006
    with pytest.raises(ValueError):
        module.make_plan(max_seconds=1801)
    with pytest.raises(ValueError):
        module.make_plan(seeds=[44000])
    with pytest.raises(ValueError):
        module.make_plan(stage="final", seeds=[44000])
    final = module.make_plan(stage="final")
    assert final["seeds"] == list(range(44000, 44020))


def episode(name="train-a", *, success=True, object_id="apple", terminal=True):
    frames = np.arange(24, dtype=np.uint8)[:, None, None, None]
    return SimpleNamespace(
        episode_id=name,
        object_id=object_id,
        container_id="plate",
        terminated=terminal,
        metadata={"collection_success": success, "variant": "nominal"},
        observations={"onboard_rgb": np.broadcast_to(frames, (24, 2, 2, 3)).copy()},
        actions=np.repeat((np.arange(23, dtype=np.float32) / 100)[:, None], 14, axis=1),
    )


def test_demo_selection_uses_only_successful_training_episodes_in_lexical_order():
    episodes = {
        "train-a": episode("train-a", success=False),
        "train-b": episode("train-b"),
        "train-c": episode("train-c"),
        "test-0": episode("test-0"),
    }
    read = []

    def load(name):
        assert name != "test-0"
        read.append(name)
        return episodes[name]

    store = SimpleNamespace(
        manifest={"splits": {"train": ["train-c", "train-b", "train-a"], "test": ["test-0"]}},
        read_episode=load,
    )
    assert module.select_demonstration(store).episode_id == "train-b"
    assert read == ["train-a", "train-b"]
    store.manifest["splits"]["train"] = ["train-a"]
    with pytest.raises(ContractError, match="no successful"):
        module.select_demonstration(store)


class ImageMetric:
    def observed_distance(self, current, goal):
        return np.square(current["onboard_rgb"].astype(np.float32) - goal["onboard_rgb"]).mean(
            (1, 2, 3)
        )


def test_calibration_records_training_neighbors_and_only_prior_action_windows():
    demo = episode()
    waypoints, calibration = module.calibrate_waypoints(ImageMetric(), demo)
    assert [r["frame"] for r in calibration["waypoints"]] == [0, 10, 20, 23]
    assert calibration["adjacent_distance_floor"] == 1
    assert all(w.threshold == pytest.approx(4.4) for w in waypoints)
    assert all(w.dwell_observations == 3 for w in waypoints)
    assert waypoints[0].action_proposals is None
    for waypoint, record in zip(waypoints[1:], calibration["waypoints"][1:], strict=True):
        for plan, start in zip(
            waypoint.action_proposals, record["proposal_start_frames"], strict=True
        ):
            assert max(0, record["frame"] - 10) <= start and start + 4 <= record["frame"]
            np.testing.assert_array_equal(plan, demo.actions[start : start + 4])
    without, _ = module.calibrate_waypoints(ImageMetric(), demo, proposals=False)
    assert all(w.action_proposals is None for w in without)


def test_frozen_input_hash_verification_rejects_checkpoint_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "assets").mkdir()
    (tmp_path / "data/meta").mkdir(parents=True)
    files = {
        "checkpoint": tmp_path / "model.pt",
        "dataset_manifest": tmp_path / "data/meta/jepa_manifest.json",
        "source": tmp_path / "source.py",
        "action": tmp_path / "configs/g1_sim_action.json",
        "asset": tmp_path / "assets/manifest.json",
    }
    for path in files.values():
        path.write_text("immutable fixture")
    files["dataset_manifest"].write_text(json.dumps({"sha256": {}}))
    plan = {
        "checkpoint": str(files["checkpoint"]),
        "checkpoint_sha256": module.digest(files["checkpoint"]),
        "dataset": str(tmp_path / "data"),
        "dataset_manifest_sha256": module.digest(files["dataset_manifest"]),
        "source_hashes": {"source.py": module.digest(files["source"])},
        "action_manifest_sha256": module.digest(files["action"]),
        "asset_manifest_sha256": module.digest(files["asset"]),
    }
    module.verify_plan_inputs(plan)
    files["checkpoint"].write_text("changed")
    with pytest.raises(ContractError, match="checkpoint"):
        module.verify_plan_inputs(plan)


def test_checkpoint_provenance_checks_before_model_use(monkeypatch):
    from embodied_jepa import config, data

    schema = StateSchema(("q",), ("rad",), "test_v0")
    manifest = {
        "splits": {"train": ["a"], "val": ["b"]},
        "split_policy": {"seed": 3},
        "action_manifest": {"unit": "fixture"},
    }
    store = SimpleNamespace(manifest=manifest, manifest_hash="dataset123", state_schema=schema)
    expected = {
        "dataset_hash": "dataset123",
        "split_hash": module.json_hash(
            {"splits": manifest["splits"], "policy": manifest["split_policy"]}
        ),
        "action_hash": module.json_hash(manifest["action_manifest"]),
        "normalization": {"episode_ids": ["a"]},
    }
    checkpoint = {
        "metadata": expected,
        "state_schema": asdict(schema),
        "backend": "sensor_wm",
        "seed": 0,
        "config": {},
    }
    fake_torch = SimpleNamespace(
        get_num_threads=lambda: 1, set_num_threads=lambda _: None, load=lambda *a, **kw: checkpoint
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(data, "DatasetStore", lambda root: store)
    loaded = []
    model = SimpleNamespace(
        load=lambda path: loaded.append(path), observed_distance=lambda *_: None
    )
    monkeypatch.setattr(config.MODELS, "create", lambda *a, **kw: model)
    assert module.checkpoint_model("corpus", "checkpoint")[1] is model
    assert loaded == ["checkpoint"]
    checkpoint["metadata"]["split_hash"] = "wrong"
    with pytest.raises(ContractError, match="split_hash"):
        module.checkpoint_model("corpus", "checkpoint")
    assert loaded == ["checkpoint"]


def test_hard_wall_supervisor_kills_process_group_without_blocking(tmp_path, monkeypatch):
    elapsed, killed = [0.0], []
    process = SimpleNamespace(pid=123, returncode=None)
    process.poll = lambda: process.returncode
    process.wait = lambda: process.returncode

    def kill(pid, signal):
        killed.append((pid, signal))
        process.returncode = -9

    def start(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        return process

    monkeypatch.setattr(module.os, "killpg", kill)
    monkeypatch.setattr(module.time, "time", lambda: elapsed[0])
    monkeypatch.setattr(module.time, "monotonic", lambda: elapsed[0])
    result = module.supervise(
        ["fake"],
        cwd=tmp_path,
        log=tmp_path / "log",
        remaining=0.3,
        popen=start,
        sleep=lambda amount: elapsed.__setitem__(0, elapsed[0] + amount),
    )
    assert result == {"status": "hard_wall_timeout", "returncode": -9}
    assert killed == [(123, 9)] and elapsed[0] >= 0.3


def test_interrupted_attempt_keeps_partial_results_and_uncertain_command(tmp_path):
    trial = {"attempt_id": "43000-learned", "seed": 43000, "mode": "learned"}
    folder = tmp_path / "attempts" / trial["attempt_id"]
    folder.mkdir(parents=True)
    (folder / "started.json").write_text("{}")
    rows = [
        {"event": "command_pending"},
        {"event": "result", "executed": True, "score": {"success": False, "grasp": True}},
        {"event": "command_pending"},
    ]
    (folder / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + '\n{"partial')
    record = module.summarize_attempt(tmp_path, trial, "hard_wall_timeout")
    assert record["executed_steps"] == 1 and record["execution_uncertain"]
    assert record["success"] is False and record["score"]["grasp"] is True
    assert record["status"] == "hard_wall_timeout"
    missing = module.summarize_attempt(
        tmp_path, trial | {"attempt_id": "not-started"}, "not_started_budget"
    )
    assert missing["executed_steps"] == 0 and not missing["started"]


def test_cross_training_calibration_uses_frame_coverage_and_excludes_test():
    first, second = episode("train-a"), episode("train-b")
    second.observations["onboard_rgb"] += 5
    second.observations["onboard_rgb"] = second.observations["onboard_rgb"][:21]
    goals, record = module.calibrate_waypoints(
        ImageMetric(),
        first,
        calibration_episodes=[first, second],
        training_episode_ids=["train-a", "train-b"],
    )
    assert goals[0].threshold == pytest.approx(1.1 * 22.5)
    assert record["waypoints"][0]["cross_training_reference_count"] == 2
    assert record["waypoints"][-1]["cross_training_reference_count"] == 1
    assert record["waypoints"][-1]["cross_training_distances"][0]["frame"] == 23
    assert "adjacent_waypoint_separability" in record["waypoints"][0]
    with pytest.raises(ContractError, match="non-training"):
        module.calibrate_waypoints(
            ImageMetric(),
            first,
            calibration_episodes=[episode("test-a")],
            training_episode_ids=["train-a"],
        )


@pytest.mark.parametrize(
    "status,termination,expected",
    [
        ("completed", "step_limit", 0),
        ("completed", "runtime_error", 2),
        ("hard_wall_timeout", None, 2),
        ("preparation_failed", None, 2),
    ],
)
def test_cli_exit_distinguishes_physical_failure_from_incomplete_infrastructure(
    monkeypatch, status, termination, expected
):
    monkeypatch.setattr(
        sys, "argv", ["evaluate", "--dataset", "data", "--checkpoint", "model", "--output", "new"]
    )
    monkeypatch.setattr(
        module,
        "run",
        lambda *a, **kw: [{"status": status, "termination_reason": termination, "success": False}],
    )
    assert module.main() == expected


@pytest.mark.parametrize("failure", ["input_drift", "child_exit"])
def test_parent_verifies_after_attempt_and_retains_worker_failure(tmp_path, monkeypatch, failure):
    root = tmp_path / "repo"
    for directory in ("src", "configs", "assets", "third_party", "dataset/meta"):
        (root / directory).mkdir(parents=True)
    (root / "src/fake.py").write_text("# fixture source")
    (root / "configs/g1_sim_action.json").write_text("{}")
    (root / "assets/manifest.json").write_text("{}")
    (root / "dataset/meta/jepa_manifest.json").write_text(json.dumps({"sha256": {}}))
    checkpoint = root / "model.pt"
    checkpoint.write_text("immutable checkpoint fixture")
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "fixture-sha")
    output = tmp_path / "output"
    attempted = []

    def supervise(command, **kwargs):
        if command[-1] == "prepare":
            plan = json.loads((output / "plan.json").read_text())
            (output / "calibration.json").write_text("{}")
            (output / "waypoints.npz").write_bytes(b"fixture")
            module.write(
                output / "resolved_plan.json",
                plan
                | {
                    "calibration_sha256": module.digest(output / "calibration.json"),
                    "waypoints_sha256": module.digest(output / "waypoints.npz"),
                },
            )
            return {"status": "exited", "returncode": 0}
        name = command[-1]
        attempted.append(name)
        folder = output / "attempts" / name
        folder.mkdir(parents=True)
        module.write(
            folder / "report.json",
            {
                "status": "completed",
                "success": True,
                "termination_reason": "success",
                "executed_steps": 5,
            },
        )
        if failure == "input_drift":
            checkpoint.write_text("changed after successful attempt")
            return {"status": "exited", "returncode": 0}
        return {"status": "exited", "returncode": 1}

    monkeypatch.setattr(module, "supervise", supervise)
    args = SimpleNamespace(
        output=output,
        dataset=root / "dataset",
        checkpoint=checkpoint,
        stage="development",
        seeds=[43000],
        modes=["learned", "hold"],
        max_seconds=600,
        max_steps=1000,
        stride=10,
        dwell=3,
        horizon=4,
        candidates=16,
        iterations=2,
        no_proposals=False,
        selection=None,
        control_timeout=5.0,
    )
    records = module.run(args)
    assert len(records) == 2 and not module.clean_completion(records)
    report = json.loads((output / "report.json").read_text())
    assert report["denominator"] == 2 and report["successes"] == 0
    if failure == "input_drift":
        assert len(attempted) == 1
        assert records[0]["status"] == "invalid_inputs"
        assert records[1]["status"] == "not_started_invalid_inputs"
        assert report["provenance_valid"] is False
    else:
        assert len(attempted) == 2
        assert records[0]["supervisor"]["returncode"] == 1
        assert records[0]["status"] == "child_failed"
        assert records[0]["reported_success_before_failure"] is True
        assert records[0]["success"] is False


@pytest.mark.parametrize("planning_seconds", [0.0, 6.0])
def test_worker_deadline_prevents_actuation_and_control_time_survives_ack(
    tmp_path, monkeypatch, planning_seconds
):
    from embodied_jepa import embodiment, simulation, task
    from embodied_jepa.contracts import ExecutionResult

    clock, calls = [0.0], []
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    schema = StateSchema(("q",), ("rad",), "deadline_fixture_v0")
    store = SimpleNamespace(state_schema=schema, manifest={"action_manifest": {"fixture": True}})
    monkeypatch.setattr(module, "verify_plan_inputs", lambda *_: None)
    monkeypatch.setattr(
        module,
        "checkpoint_model",
        lambda *a: (store, SimpleNamespace(observed_distance=lambda *_: None)),
    )
    monkeypatch.setattr(module, "load_waypoints", lambda *a: [])

    class Controller:
        def __init__(self, *a, **kw):
            self.trace = {}

        def step(self, observation, projector):
            clock[0] += planning_seconds
            return SimpleNamespace(action=np.zeros(14, np.float32), trace=self.trace)

        def acknowledge(self, result):
            return self.trace

    class Robot:
        state_schema = schema
        manifest = {"fixture": True}

        def __init__(self, *_):
            pass

        def reset(self, **kwargs):
            pass

        def observe(self):
            return None

        def project_candidates(self, requested):
            raise AssertionError("fixture controller does not need projection")

        def execute(self, command):
            calls.append("execute")
            return ExecutionResult(command, command, "applied", 0.05)

        def stop(self, reason):
            calls.append(reason)

        def close(self):
            pass

    monkeypatch.setattr(module, "WaypointController", Controller)
    monkeypatch.setattr(simulation, "MuJoCoSimulation", lambda **kw: None)
    monkeypatch.setattr(embodiment, "G1Embodiment", Robot)
    monkeypatch.setattr(
        task, "AppleToPlateTask", lambda robot: SimpleNamespace(evaluate=lambda: {"success": True})
    )
    plan = module.make_plan(seeds=[43000], modes=["learned"], control_timeout=5.0)
    plan.update(dataset="unused", checkpoint="unused")
    module.write(tmp_path / "resolved_plan.json", plan)
    module.attempt_worker(tmp_path, "43000-learned")
    folder = tmp_path / "attempts/43000-learned"
    report = json.loads((folder / "report.json").read_text())
    rows = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
    assert rows[-1]["control_seconds"] == planning_seconds
    if planning_seconds > 5:
        assert "execute" not in calls
        assert report["termination_reason"] == "deadline_miss"
        assert report["executed_steps"] == 0
    else:
        assert calls == ["execute", "success"]
        assert report["success"] is True
