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
    files["asset"].write_text(json.dumps({"sha256": {}}))
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


@pytest.mark.parametrize(
    "planning_seconds,initial_elapsed,attempt_seconds",
    [(0.0, 0.0, None), (6.0, 0.0, None), (0.0, 89.0, 90), (2.0, 84.0, 90), (0.0, 0.0, 90)],
)
def test_worker_deadline_prevents_actuation_and_control_time_survives_ack(
    tmp_path, monkeypatch, planning_seconds, initial_elapsed, attempt_seconds
):
    from embodied_jepa import embodiment, simulation, task
    from embodied_jepa.contracts import ExecutionResult

    clock, calls = [initial_elapsed], []
    monkeypatch.setattr(module, "ENTRY_CLOCK", (0.0, 0.0))
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
    module.attempt_worker(tmp_path, "43000-learned", attempt_seconds=attempt_seconds)
    folder = tmp_path / "attempts/43000-learned"
    report = json.loads((folder / "report.json").read_text())
    rows = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
    if attempt_seconds is not None and initial_elapsed + planning_seconds >= 85:
        assert "execute" not in calls
        assert report["status"] == "attempt_timeout"
        assert report["executed_steps"] == 0
        assert report["wall_seconds"] >= 85
        assert rows[-1]["stage"] == (
            "before_observe" if initial_elapsed >= 85 else "before_execute"
        )
        return
    assert rows[-1]["control_seconds"] == planning_seconds
    if planning_seconds > 5:
        assert "execute" not in calls
        assert report["termination_reason"] == "deadline_miss"
        assert report["executed_steps"] == 0
    else:
        assert calls == ["execute", "success"]
        assert report["success"] is True


@pytest.mark.parametrize("cap", [0, -1, float("nan"), float("inf"), 1801, True])
def test_attempt_cap_validation_before_worker_io(cap):
    with pytest.raises(ValueError, match="attempt wall budget"):
        module.make_plan(attempt_max_seconds=cap)
    with pytest.raises(ValueError, match="attempt wall budget"):
        module.attempt_worker("nonexistent", "unused", attempt_seconds=cap)


def test_supervisor_rejects_late_zero_exit_after_civil_clock_jump(tmp_path, monkeypatch):
    wall = [0.0]
    monkeypatch.setattr(module.time, "time", lambda: wall[0])
    monkeypatch.setattr(module.time, "monotonic", lambda: 0.0)
    process = SimpleNamespace(returncode=0)

    def poll():
        wall[0] = 91.0
        return 0

    process.poll = poll
    result = module.supervise(
        ["fake"],
        cwd=tmp_path,
        log=tmp_path / "log",
        remaining=90,
        popen=lambda *a, **kw: process,
    )
    assert result == {"status": "hard_wall_timeout", "returncode": 0}


@pytest.mark.parametrize(
    "cap,total,worker_status,prepare_status,expected",
    [
        (90, 600, "attempt_timeout", "exited", "attempt_timeout"),
        (90, 100, "attempt_timeout", "exited", "hard_wall_timeout"),
        (90, 100, "hard_wall_timeout", "exited", "hard_wall_timeout"),
        (90, 600, "hard_wall_timeout", "exited", "attempt_timeout"),
        (None, 600, "completed", "exited", "completed"),
        (90, 600, "completed", "hard_wall_timeout", "preparation_failed"),
    ],
)
def test_parent_attempt_allocations_status_and_preparation(
    tmp_path, monkeypatch, cap, total, worker_status, prepare_status, expected
):
    root = tmp_path / "repo"
    for directory in ("src", "configs", "assets", "third_party", "dataset/meta"):
        (root / directory).mkdir(parents=True)
    for name in (
        "configs/g1_sim_action.json",
        "assets/manifest.json",
        "dataset/meta/jepa_manifest.json",
    ):
        (root / name).write_text('{"sha256": {}}')
    (root / "src/fake.py").write_text("# fixture")
    (root / "model.pt").write_text("checkpoint")
    clock, allocations, commands = [0.0], [], []
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "fixture")
    output = tmp_path / "output"

    def supervise(command, **kwargs):
        if command[-1] == "prepare":
            clock[0] += 10
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
            return {"status": prepare_status, "returncode": 0}
        commands.append(command)
        allocations.append(kwargs["remaining"])
        clock[0] += kwargs["remaining"] if cap is not None else 1
        folder = output / "attempts" / command[-1]
        folder.mkdir(parents=True)
        module.write(
            folder / "report.json",
            {
                "status": "attempt_timeout" if worker_status == "attempt_timeout" else "completed",
                "termination_reason": "attempt_timeout"
                if worker_status == "attempt_timeout"
                else "step_limit",
                "success": False,
                "executed_steps": 3,
                "score": {"reach": True},
            },
        )
        return {
            "status": "hard_wall_timeout" if worker_status == "hard_wall_timeout" else "exited",
            "returncode": 0,
        }

    monkeypatch.setattr(module, "supervise", supervise)
    args = SimpleNamespace(
        output=output,
        dataset=root / "dataset",
        checkpoint=root / "model.pt",
        stage="development",
        seeds=[43000, 43001],
        modes=list(module.MODES[:3]),
        max_seconds=total,
        max_steps=1000,
        stride=28,
        dwell=3,
        horizon=16,
        candidates=16,
        iterations=2,
        no_proposals=False,
        selection=None,
        control_timeout=5.0,
        commitment_steps=4 if cap else 1,
        attempt_max_seconds=cap,
    )
    records = module.run(args, start_clock=(0.0, 0.0))
    assert len(records) == 6 and records[0]["status"] == expected
    if prepare_status != "exited":
        assert not commands and all(r["status"] == "preparation_failed" for r in records)
        return
    assert records[0]["executed_steps"] == 3 and records[0]["score"]["reach"]
    if cap is None:
        assert len(commands) == 6 and all("--worker-attempt-seconds" not in c for c in commands)
        assert records[0]["attempt_allocation"]["requested_seconds"] is None
        assert module.clean_completion(records)
    else:
        assert allocations[0] == (85 if total == 100 else 90)
        assert (
            float(commands[0][commands[0].index("--worker-attempt-seconds") + 1]) == allocations[0]
        )
        assert records[0]["timeout_scope"] == ("global" if total == 100 else "attempt")
        assert records[0]["attempt_allocation"]["global_shortened"] == (total == 100)
        assert not module.clean_completion(records)
        assert len(commands) == (1 if total == 100 else 6)
        if total == 100:
            assert all(r["status"] == "not_started_budget" for r in records[1:])


@pytest.mark.parametrize(
    "options,commitment,cap",
    [([], 1, None), (["--commitment-steps", "4", "--attempt-max-seconds", "90"], 4, 90.0)],
)
def test_cli_preserves_default_or_explicit_attempt_policy(monkeypatch, options, commitment, cap):
    seen = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "--dataset", "data", "--checkpoint", "model", "--output", "new", *options],
    )
    monkeypatch.setattr(
        module,
        "run",
        lambda args, **kw: (
            seen.append(args) or [{"status": "completed", "termination_reason": "step_limit"}]
        ),
    )
    assert module.main() == 0
    assert seen[0].commitment_steps == commitment and seen[0].attempt_max_seconds == cap


# TASK-043 demonstration-state-goal protocol fixtures (no physics, no trained model).

STATE_SCHEMA = StateSchema(("a", "b", "v"), ("rad", "rad", "rad/s"), "state_goal_fixture_v0")


def state_plan(**overrides):
    options = dict(
        seeds=[43000, 43001, 43002, 43003],
        goal_kind="state",
        stride=16,
        horizon=16,
        dwell=1,
        proposals=False,
        goal_stall_limit=64,
        max_seconds=3600,
        attempt_max_seconds=200,
    )
    return module.make_plan(**(options | overrides))


def test_state_plan_is_reset_major_with_four_modes_and_frozen_rules():
    plan = state_plan()
    assert plan["modes"] == ["learned", "dynamics_shuffle", "persistence", "demo_replay"]
    assert [a["attempt_id"] for a in plan["attempts"][:5]] == [
        "43000-learned",
        "43000-dynamics_shuffle",
        "43000-persistence",
        "43000-demo_replay",
        "43001-learned",
    ]
    assert len(plan["attempts"]) == 16 and plan["goal_kind"] == "state"
    assert plan["controller"]["goal_stall_limit"] == 64
    assert plan["visual_diagnostic_weight"] == 0.0 and len(plan["state_goal_fields"]) == 14
    assert plan["demonstration_proposals"] is False
    for bad in (
        dict(proposals=True),
        dict(stride=28),
        dict(goal_stall_limit=None),
        dict(max_seconds=3601),
        dict(modes=["learned", "hold"]),
    ):
        with pytest.raises(ValueError):
            state_plan(**bad)
    # The default image plan is unchanged: no stall key, no demo replay, 1800-second cap.
    image = module.make_plan(seeds=[43000])
    assert "goal_stall_limit" not in image["controller"] and "goal_kind" not in image
    assert "demo_replay" not in image["modes"]
    for bad in (dict(modes=["demo_replay"]), dict(goal_stall_limit=64), dict(max_seconds=3600)):
        with pytest.raises(ValueError):
            module.make_plan(seeds=[43000], **bad)


class StateMetricModel:
    """Fixture metric over the first two state fields; never a trained model."""

    def select(self, states, schema):
        return np.asarray(states, np.float32)[:, :2].copy()

    def observed_distance(self, robot_state, goal):
        return np.square(self.select(robot_state.values, None) - goal).mean(-1)

    def image_distance(self, current, goal):
        return np.square(current["onboard_rgb"].astype(np.float32) - goal["onboard_rgb"]).mean(
            (1, 2, 3)
        )


def state_episode(name, *, offset=0.0, length=40, first_pixel=0):
    t = np.arange(length, dtype=np.float32)
    states = np.stack([t / 10 + offset, -t / 20, np.zeros_like(t)], 1)
    frames = np.zeros((length, 2, 2, 3), np.uint8)
    frames[0] = first_pixel
    return SimpleNamespace(
        episode_id=name,
        robot_states=states,
        state_mask=np.ones_like(states, bool),
        timestamps=t.astype(np.float64) / 20,
        state_schema=STATE_SCHEMA,
        observations={"onboard_rgb": frames},
        actions=np.full((length - 1, 14), 0.01, np.float32),
        metadata={"variant": "nominal", "collection_success": True},
    )


def test_state_goal_library_is_train_only_with_existing_threshold_recipe():
    model = StateMetricModel()
    demos = [state_episode("train-b", offset=0.1), state_episode("train-a")]
    arrays, calibration = module.build_state_goal_library(
        model, demos, stride=16, dwell=1, training_episode_ids=["train-a", "train-b", "x"]
    )
    assert calibration["calibration_training_episode_ids"] == ["train-a", "train-b"]
    record = calibration["demonstrations"][0]
    assert record["episode_id"] == "train-a" and record["goal_frames"] == [16, 32, 39]
    assert arrays["goals_0"].shape == (3, 2) and arrays["goal_images_0"].shape == (3, 2, 2, 3)
    np.testing.assert_array_equal(arrays["goals_0"][0], demos[1].robot_states[16, :2])
    goal = record["goals"][0]
    adjacent = (0.1**2 + 0.05**2) / 2
    neighborhood = 4 * adjacent  # frames t+-2 on a linear trajectory
    cross = [0.0, 0.1**2 / 2]
    expected = max(1.1 * neighborhood, adjacent, 1.1 * float(np.quantile(cross, 0.9)))
    assert goal["threshold"] == pytest.approx(expected, rel=1e-5)
    assert [c["episode_id"] for c in goal["cross_training_distances"]] == ["train-a", "train-b"]
    assert goal["adjacent_goal_separability"][0]["frame"] == 0
    assert goal["source_id"] == "train-a/frame-16" and goal["dwell"] == 1
    with pytest.raises(ContractError, match="non-training"):
        module.build_state_goal_library(
            model,
            demos + [state_episode("val-a")],
            stride=16,
            dwell=1,
            training_episode_ids=["train-a", "train-b"],
        )
    with pytest.raises(ContractError):
        module.build_state_goal_library(
            model, [demos[0], demos[0]], stride=16, dwell=1, training_episode_ids=["train-b"]
        )
    broken = state_episode("train-a")
    broken.actions = broken.actions[:-1]
    with pytest.raises(ContractError, match="T actions"):
        module.build_state_goal_library(
            model, [broken], stride=16, dwell=1, training_episode_ids=["train-a"]
        )


def test_retrieval_uses_initial_rgb_only_with_lexical_ties_and_train_waypoints():
    model = StateMetricModel()
    demos = [
        state_episode("train-c", first_pixel=50, offset=5.0),
        state_episode("train-b", first_pixel=10),
        state_episode("train-a", first_pixel=10, offset=1.0),
    ]
    arrays, calibration = module.build_state_goal_library(
        model, demos, stride=16, dwell=1, training_episode_ids=[d.episode_id for d in demos]
    )
    library = calibration | {"arrays": arrays}
    image = {"onboard_rgb": np.full((1, 2, 2, 3), 12, np.uint8)}
    record, retrieval = module.retrieve_demonstration(model, image, library)
    assert record["episode_id"] == "train-a"  # tie with train-b resolves lexically
    assert retrieval["distances"]["train-a"] == retrieval["distances"]["train-b"] == 4.0
    far = {"onboard_rgb": np.full((1, 2, 2, 3), 49, np.uint8)}
    assert module.retrieve_demonstration(model, far, library)[0]["episode_id"] == "train-c"
    waypoints = module.state_waypoints(library, record)
    assert [w.source_id for w in waypoints] == [
        "train-a/frame-16",
        "train-a/frame-32",
        "train-a/frame-39",
    ]
    assert all(w.source_split == "train" and w.images is not None for w in waypoints)


def test_state_gate_counts_missing_and_failed_attempts_as_zero():
    seeds = [43000, 43001, 43002, 43003]
    full = dict.fromkeys(module.STAGES, True) | {"success": True}
    grasp = dict(reach=True, grasp=True, transport=False, place=False, release=False)
    records = [
        dict(seed=43000, mode="learned", score=grasp, success=False),
        dict(seed=43001, mode="learned", score=full, success=True),
        dict(seed=43000, mode="dynamics_shuffle", score=dict(reach=True), success=False),
        dict(seed=43000, mode="persistence", score={}, success=False),
        dict(seed=43000, mode="demo_replay", score=full, success=True),
    ]
    gate = module.state_gate(records, seeds)
    assert gate["learned_grasp_resets"] == 2 and gate["primary_gate_passed"]
    assert gate["summed_ordered_stages"] == dict(
        learned=7, dynamics_shuffle=1, persistence=0, demo_replay=5
    )
    assert gate["learned_full_successes"] == 1
    records[1]["score"] = dict(reach=True, grasp=False, transport=True)
    gate = module.state_gate(records, seeds)
    assert not gate["primary_gate_passed"] and gate["learned_grasp_resets"] == 1
    assert module.ordered_stage_count(dict(reach=True, grasp=False, transport=True)) == 1
    tied = [dict(seed=s, mode=m, score=grasp) for s in seeds[:2] for m in module.STATE_MODES]
    assert not module.state_gate(tied, seeds)["primary_gate_passed"]


def state_worker_fixture(tmp_path, monkeypatch, controller_decisions):
    from embodied_jepa import embodiment, simulation, task
    from embodied_jepa.constraints import CandidateProjection
    from embodied_jepa.contracts import ExecutionResult

    calls, predictions = [], []
    store = SimpleNamespace(state_schema=STATE_SCHEMA, manifest={"action_manifest": {"f": 1}})
    model = StateMetricModel()
    model.predict = lambda *a: predictions.append(a)
    model.pop_diagnostics = lambda: [{"visual_weight": 0.0, "visual_endpoint_costs": [1.0]}]
    demos = [state_episode("train-a")]
    arrays, calibration = module.build_state_goal_library(
        model, demos, stride=16, dwell=1, training_episode_ids=["train-a"]
    )
    monkeypatch.setattr(module, "verify_plan_inputs", lambda *_: None)
    monkeypatch.setattr(module, "checkpoint_model", lambda *a, **kw: (store, model))
    monkeypatch.setattr(module, "load_state_library", lambda *a: calibration | {"arrays": arrays})

    class Controller:
        def __init__(self, model, waypoints, **kwargs):
            assert all(w.source_split == "train" for w in waypoints)
            self.decisions = list(controller_decisions)

        def step(self, observation, projector):
            return self.decisions.pop(0)

        def acknowledge(self, result):
            return {"goal_index": 0}

    class Robot:
        state_schema = STATE_SCHEMA
        manifest = {"f": 1}
        clock = 0.0

        def __init__(self, *_):
            pass

        def reset(self, **kwargs):
            pass

        def observe(self):
            Robot.clock += 0.05
            return SimpleNamespace(
                images={"onboard_rgb": np.zeros((1, 2, 2, 3), np.uint8)},
                timestamps=np.array([Robot.clock]),
            )

        def project_candidates(self, requested):
            return CandidateProjection(requested, np.ones(requested.shape[:2], bool))

        def execute(self, command):
            calls.append(np.asarray(command).copy())
            return ExecutionResult(command, command, "applied", Robot.clock + 0.01)

        def stop(self, reason):
            calls.append(reason)

        def close(self):
            pass

    monkeypatch.setattr(module, "WaypointController", Controller)
    monkeypatch.setattr(simulation, "MuJoCoSimulation", lambda **kw: None)
    monkeypatch.setattr(embodiment, "G1Embodiment", Robot)
    monkeypatch.setattr(
        task,
        "AppleToPlateTask",
        lambda robot: SimpleNamespace(evaluate=lambda: dict(reach=True, success=False)),
    )
    plan = state_plan(seeds=[43000], max_steps=1000)
    plan.update(dataset="unused", checkpoint="unused")
    module.write(tmp_path / "resolved_plan.json", plan)
    return calls, predictions


def test_state_worker_records_goal_stall_as_failure(tmp_path, monkeypatch):
    stall = SimpleNamespace(
        action=None,
        trace={"goal_index": 3, "goal_commands": 64, "goal_stall_limit": 64},
        termination_reason="goal_stall",
    )
    go = SimpleNamespace(action=np.zeros(14, np.float32), trace={"goal_index": 3})
    calls, _ = state_worker_fixture(tmp_path, monkeypatch, [go, stall])
    module.attempt_worker(tmp_path, "43000-learned", attempt_seconds=200)
    folder = tmp_path / "attempts/43000-learned"
    report = json.loads((folder / "report.json").read_text())
    rows = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
    assert report["termination_reason"] == "goal_stall" and report["success"] is False
    assert report["last_goal_index"] == 3 and report["goal_count"] == 3
    assert report["demonstration_episode_id"] == "train-a"
    assert report["ordered_stage_count"] == 1 and report["executed_steps"] == 1
    assert rows[0]["visual_diagnostic"][0]["visual_weight"] == 0.0
    result = next(r for r in rows if r["event"] == "result")
    assert result["visual_diagnostic"] == rows[0]["visual_diagnostic"]
    assert rows[-1]["event"] == "controller_termination" and rows[-1]["goal_commands"] == 64
    assert calls[-1] == "goal_stall"
    retrieval = json.loads((folder / "retrieval.json").read_text())
    assert retrieval["demonstration_episode_id"] == "train-a"


def test_demo_replay_is_open_loop_non_learned_and_exhausts(tmp_path, monkeypatch):
    calls, predictions = state_worker_fixture(tmp_path, monkeypatch, [])
    module.attempt_worker(tmp_path, "43000-demo_replay", attempt_seconds=200)
    folder = tmp_path / "attempts/43000-demo_replay"
    report = json.loads((folder / "report.json").read_text())
    rows = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
    assert report["termination_reason"] == "demo_exhausted"
    assert report["executed_steps"] == 39 and not predictions
    assert all(r["control"] == "demo_replay_non_learned" for r in rows if "control" in r)
    assert rows[-1]["replay_frame"] == 38
    # Recorded actions are clipped to the same bounds as MPC candidates (left pose fixed 0).
    np.testing.assert_allclose(calls[0][:6], 0.0)
    np.testing.assert_allclose(calls[0][6:12], 0.01, rtol=1e-6)
    assert calls[0][12] == -1.0
