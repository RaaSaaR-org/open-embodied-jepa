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
    pytest.importorskip("torch")  # state plans resolve fields via the torch model
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


def test_close_goal_index_comes_from_recorded_train_right_grasp_only():
    model = StateMetricModel()
    demo = state_episode("train-a")
    demo.actions[:, 12:] = -1.0
    demo.actions[20, 13] = 0.5  # right grasp: preceding window of the frame-32 goal
    demo.actions[5, 12] = 1.0  # a left grasp command is not a close command
    _, calibration = module.build_state_goal_library(
        model, [demo], stride=16, dwell=1, training_episode_ids=["train-a"]
    )
    assert calibration["demonstrations"][0]["first_close_goal_index"] == 1
    demo.actions[:, 13] = -1.0
    _, calibration = module.build_state_goal_library(
        model, [demo], stride=16, dwell=1, training_episode_ids=["train-a"]
    )
    assert calibration["demonstrations"][0]["first_close_goal_index"] is None


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


def test_state_gate_scores_only_clean_provenance_valid_attempts_and_readings():
    seeds = [43000, 43001, 43002, 43003]
    grasp = dict(reach=True, grasp=True, transport=False, place=False, release=False)
    reach = dict(reach=True, grasp=False)

    def learned(seed, score, **extra):
        return dict(seed=seed, mode="learned", score=score, status="completed", **extra)

    records = [learned(43000, grasp), learned(43001, grasp)]
    assert module.state_gate(records, seeds)["primary_gate_passed"]
    for failure in (
        dict(status="child_failed"),
        dict(status="hard_wall_timeout"),
        dict(termination_reason="runtime_error"),
        dict(termination_reason="deadline_miss"),
        dict(termination_reason="attempt_timeout"),
    ):
        failed = [records[0], records[1] | failure]
        gate = module.state_gate(failed, seeds)
        assert gate["learned_grasp_resets"] == 1 and not gate["primary_gate_passed"]
        assert gate["summed_ordered_stages"]["learned"] == 2
    stalled = learned(43001, grasp, termination_reason="goal_stall")
    assert module.state_gate([records[0], stalled], seeds)["primary_gate_passed"]
    invalid = [records[0], records[1] | {"provenance_valid": False}]
    gate = module.state_gate(invalid, seeds)
    assert not gate["provenance_valid"] and not gate["primary_gate_passed"]

    early = [
        learned(
            s, {}, termination_reason="goal_stall", last_goal_index=0, first_close_goal_index=13
        )
        for s in seeds[:3]
    ] + [learned(43003, reach, last_goal_index=14, first_close_goal_index=13)]
    readings = module.state_gate(early, seeds)["falsification"]
    assert readings["learned_stalled_before_close_goal_resets"] == 3
    assert readings["model_or_planner_failure"]
    assert readings["learned_reach_resets"] == 1 and readings["grasp_precision_bottleneck"]
    assert not readings["scaffold_explains_result"]
    shuffle = dict(seed=43003, mode="dynamics_shuffle", score=reach)
    assert module.state_gate(early + [shuffle], seeds)["falsification"]["scaffold_explains_result"]


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
    pytest.importorskip("torch")  # state plans resolve fields via the torch model
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
    assert report["first_close_goal_index"] == 0  # fixture actions all command grasp 0.01
    assert report["ordered_stage_count"] == 1 and report["executed_steps"] == 1
    assert rows[0]["visual_diagnostic"][0]["visual_weight"] == 0.0
    result = next(r for r in rows if r["event"] == "result")
    assert result["visual_diagnostic"] == rows[0]["visual_diagnostic"]
    assert rows[-1]["event"] == "controller_termination" and rows[-1]["goal_commands"] == 64
    assert calls[-1] == "goal_stall"
    retrieval = json.loads((folder / "retrieval.json").read_text())
    assert retrieval["demonstration_episode_id"] == "train-a"


def test_demo_replay_is_open_loop_non_learned_and_exhausts(tmp_path, monkeypatch):
    pytest.importorskip("torch")  # state plans resolve fields via the torch model
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


# TASK-044 NON-LEARNED privileged simulator-rollout ceiling (no physics here).


def ceiling_plan(**overrides):
    return state_plan(
        modes=["privileged_rollout"], attempt_max_seconds=840, max_seconds=3600, **overrides
    )


def test_privileged_ceiling_runs_alone_in_development_and_is_labelled():
    pytest.importorskip("torch")  # state plans resolve fields via the torch model
    plan = ceiling_plan()
    assert plan["privileged_ceiling"] is True and plan["modes"] == ["privileged_rollout"]
    assert plan["result_label"].startswith("NON-LEARNED privileged")
    assert plan["planning_dynamics"] == "privileged_mujoco_rollout_v1"
    assert [a["attempt_id"] for a in plan["attempts"]] == [
        f"{s}-privileged_rollout" for s in (43000, 43001, 43002, 43003)
    ]
    # The controller configuration is the TASK-043 state-goal scaffold, unchanged.
    assert plan["controller"] == state_plan()["controller"]
    assert "privileged_ceiling" not in state_plan() and "result_label" not in state_plan()
    for bad in (
        dict(modes=["privileged_rollout", "learned"]),
        dict(modes=["learned", "privileged_rollout"]),
        dict(modes=["privileged_rollout", "demo_replay"]),
    ):
        with pytest.raises(ValueError, match="runs alone"):
            state_plan(**bad)
    with pytest.raises(ValueError, match="runs alone"):
        module.make_plan(
            stage="final",
            modes=["privileged_rollout"],
            goal_kind="state",
            stride=16,
            horizon=16,
            dwell=1,
            proposals=False,
            goal_stall_limit=64,
            max_seconds=3600,
        )
    with pytest.raises(ValueError, match="state-goal scaffold"):
        module.make_plan(seeds=[43000], modes=["privileged_rollout"])


def test_privileged_records_never_enter_learned_state_gate_or_reports():
    seeds = [43000, 43001, 43002, 43003]
    full = dict.fromkeys(module.STAGES, True) | {"success": True}
    privileged = [
        dict(seed=s, mode="privileged_rollout", score=full, success=True, status="completed")
        for s in seeds
    ]
    learned = [dict(seed=s, mode="learned", score={}, status="completed") for s in seeds]
    gate = module.state_gate(learned + privileged, seeds)
    assert gate["learned_grasp_resets"] == 0 and not gate["primary_gate_passed"]
    assert gate["learned_full_successes"] == 0
    assert "privileged_rollout" not in gate["summed_ordered_stages"]
    ceiling = module.gate_summary(
        {"privileged_ceiling": True, "goal_kind": "state", "seeds": seeds}, privileged
    )
    assert set(ceiling) == {"ceiling_gate"}
    assert ceiling["ceiling_gate"]["label"].startswith("NON-LEARNED")
    learned_report = module.gate_summary({"goal_kind": "state", "seeds": seeds}, learned)
    assert set(learned_report) == {"state_gate"}
    assert module.gate_summary({"seeds": seeds}, learned) == {}
    # A ceiling gate ignores any learned record, even a successful one.
    ignored = module.ceiling_gate([r | {"score": full} for r in learned], seeds)
    assert ignored["privileged_grasp_resets"] == 0 and ignored["counted_attempts"] == 0


def test_ceiling_gate_counts_only_clean_attempts_and_computes_readings():
    seeds = [43000, 43001, 43002, 43003]
    grasp = dict(reach=True, grasp=True, transport=False, place=False, release=False)

    def ceiling(seed, score, **extra):
        return (
            dict(
                seed=seed,
                mode="privileged_rollout",
                score=score,
                status="completed",
                rollout_parity_checks=10,
                rollout_parity_mismatches=0,
            )
            | extra
        )

    records = [ceiling(43000, grasp), ceiling(43001, grasp, termination_reason="goal_stall")]
    gate = module.ceiling_gate(records, seeds)
    assert gate["primary_gate_passed"] and gate["privileged_grasp_resets"] == 2
    assert gate["summed_ordered_stages"] == 4 and gate["counted_attempts"] == 2
    assert gate["per_reset"]["43002"]["counted"] is False  # missing counts as zero
    assert "learned" in gate["readings"]["interpretation"]
    for failure in (
        dict(status="child_failed"),
        dict(status="hard_wall_timeout"),
        dict(termination_reason="runtime_error"),
        dict(termination_reason="deadline_miss"),
        dict(termination_reason="attempt_timeout"),
    ):
        failed = module.ceiling_gate([records[0], records[1] | failure], seeds)
        assert failed["privileged_grasp_resets"] == 1 and not failed["primary_gate_passed"]
        assert failed["per_reset"]["43001"]["reported_ordered_stages_uncounted"] == 2
    invalid = module.ceiling_gate([records[0], records[1] | {"provenance_valid": False}], seeds)
    assert not invalid["provenance_valid"] and not invalid["primary_gate_passed"]
    inexact = module.ceiling_gate(
        [records[0], records[1] | {"rollout_parity_mismatches": 1}], seeds
    )
    assert not inexact["rollouts_exact"] and not inexact["primary_gate_passed"]
    assert "inconclusive" in inexact["readings"]["interpretation"]
    # Missing or failed attempts never trigger a design reading.
    failures = [ceiling(s, {}, termination_reason="runtime_error") for s in seeds]
    readings = module.ceiling_gate(failures, seeds)["readings"]
    assert readings["stalled_before_close_goal_resets"] == 0 and not readings["conclusive"]
    assert not readings["state_goal_tracking_inadequate"]
    assert "inconclusive" in readings["interpretation"]
    early = [
        ceiling(
            s, {}, termination_reason="goal_stall", last_goal_index=2, first_close_goal_index=13
        )
        for s in seeds
    ]
    readings = module.ceiling_gate(early, seeds)["readings"]
    assert readings["stalled_before_close_goal_resets"] == 4
    assert readings["state_goal_tracking_inadequate"]
    assert not readings["arm_pose_goals_insufficient_for_grasp"]
    assert "planner/goal design must change" in readings["interpretation"]
    on_close = [r | {"last_goal_index": 13} for r in early]
    readings = module.ceiling_gate(on_close, seeds)["readings"]
    assert readings["advanced_past_close_goal_without_grasp_resets"] == 0
    assert readings["stalled_before_close_goal_resets"] == 0
    late = [r | {"last_goal_index": 15} for r in early]
    readings = module.ceiling_gate(late, seeds)["readings"]
    assert readings["advanced_past_close_goal_without_grasp_resets"] == 4
    assert readings["arm_pose_goals_insufficient_for_grasp"]
    assert not readings["state_goal_tracking_inadequate"]


def test_privileged_worker_uses_acknowledged_ceiling_and_logs_rollout_diagnostics(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")  # state plans resolve fields via the torch model
    from embodied_jepa import privileged_rollout

    built = []

    class Ceiling:
        def __init__(self, model, robot, *, acknowledge_privileged_ceiling=False):
            assert acknowledge_privileged_ceiling is True
            built.append((model, robot))
            self.closed = False

        def pop_diagnostics(self):
            return [
                {
                    "planning_dynamics": "privileged_mujoco_rollout_v1",
                    "candidates": 16,
                    "previous_search_first_step_exact_match": True,
                }
            ]

        def close(self):
            built.append("closed")

    monkeypatch.setattr(privileged_rollout, "PrivilegedRolloutModel", Ceiling)
    go = SimpleNamespace(action=np.zeros(14, np.float32), trace={"goal_index": 2})
    stall = SimpleNamespace(
        action=None, trace={"goal_index": 2, "goal_commands": 64}, termination_reason="goal_stall"
    )
    calls, predictions = state_worker_fixture(tmp_path, monkeypatch, [go, stall])
    plan = ceiling_plan(seeds=[43000]) | {"dataset": "unused", "checkpoint": "unused"}
    module.write(tmp_path / "resolved_plan.json", plan)
    module.attempt_worker(tmp_path, "43000-privileged_rollout", attempt_seconds=840)
    folder = tmp_path / "attempts/43000-privileged_rollout"
    report = json.loads((folder / "report.json").read_text())
    rows = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
    assert report["termination_reason"] == "goal_stall" and report["last_goal_index"] == 2
    assert report["executed_steps"] == 1 and built[-1] == "closed" and len(built) == 2
    assert report["rollout_parity_checks"] == 1 and report["rollout_parity_mismatches"] == 0
    result = next(r for r in rows if r["event"] == "result")
    assert result["privileged_rollout_diagnostic"][0]["candidates"] == 16
    assert all("visual_diagnostic" not in r for r in rows)


def test_privileged_worker_refuses_an_undeclared_plan(tmp_path, monkeypatch):
    pytest.importorskip("torch")  # state plans resolve fields via the torch model
    state_worker_fixture(tmp_path, monkeypatch, [])
    plan = ceiling_plan(seeds=[43000]) | {"dataset": "unused", "checkpoint": "unused"}
    del plan["privileged_ceiling"]
    module.write(tmp_path / "resolved_plan.json", plan)
    module.attempt_worker(tmp_path, "43000-privileged_rollout", attempt_seconds=840)
    report = json.loads((tmp_path / "attempts/43000-privileged_rollout/report.json").read_text())
    assert report["termination_reason"] == "runtime_error"
    assert "declared ceiling plan" in report["error"] and report["executed_steps"] == 0


# TASK-045 time-indexed trajectory tracking (no physics here).


def tracking_plan(**overrides):
    return state_plan(goal_kind="trajectory", attempt_max_seconds=600, **overrides)


def test_trajectory_plan_freezes_tracking_rules_and_keeps_other_plans_unchanged():
    pytest.importorskip("torch")  # state plans resolve fields via the torch model
    plan = tracking_plan()
    assert plan["modes"] == ["learned", "dynamics_shuffle", "persistence"]
    assert plan["goal_kind"] == "trajectory" and plan["backend"] == module.STATE_BACKEND
    assert plan["tracking_window"] == 16 and plan["tracking_stall_commands"] == 64
    assert plan["tracking_stall_min_advance"] == 16
    assert plan["keyframe_threshold_fraction"] == 0.1
    assert plan["controller"] == state_plan()["controller"]
    ceiling = tracking_plan(modes=["privileged_rollout"])
    assert (
        ceiling["privileged_ceiling"] is True
        and "trajectory-tracking" in ceiling["privileged_rule"]
    )
    # The TASK-044 endpoint ceiling plan and the TASK-043 state plan carry no tracking keys.
    endpoint = state_plan(modes=["privileged_rollout"], attempt_max_seconds=840)
    assert "horizon endpoint" in endpoint["privileged_rule"]
    for other in (endpoint, state_plan(), module.make_plan(seeds=[43000])):
        assert not any(key.startswith("tracking") for key in other)
    for bad in (
        dict(modes=["demo_replay"]),
        dict(modes=["privileged_rollout", "learned"]),
        dict(commitment_steps=4),
        dict(proposals=True),
        dict(goal_stall_limit=None),
        dict(stride=28),
    ):
        with pytest.raises(ValueError):
            tracking_plan(**bad)


def test_tracking_references_are_train_keyframes_with_close_and_grasp_rows():
    model = StateMetricModel()
    demo = state_episode("train-a", length=12)
    demo.robot_states[4:8] = demo.robot_states[4]  # dead time: frames 5-7 do not move
    demo.actions[:] = 0.0
    demo.actions[:, 12:] = -1.0
    demo.actions[8:, 13] = 1.0  # first closing command at action 8 -> frame 9
    demo.metadata["stage_scores"] = [{"grasp": i >= 9} for i in range(11)]  # frame 10
    arrays, library = module.build_state_goal_library(
        model, [demo], stride=4, dwell=1, training_episode_ids=["train-a"]
    )
    references, tracking = module.build_tracking_references(
        model, [demo], library, fraction=0.1, training_episode_ids=["train-a"]
    )
    row = tracking["references"][0]
    frames = references["frames_0"].tolist()
    assert frames == [0, 1, 2, 3, 4, 8, 9, 10, 11] and row["dropped_frames"] == 3
    np.testing.assert_array_equal(references["reference_0"], demo.robot_states[frames, :2])
    assert row["first_close_frame"] == 9 and row["first_close_reference_index"] == 6
    assert row["demo_grasp_frame"] == 10 and row["demo_grasp_reference_index"] == 7
    assert row["keyframe_threshold"] == pytest.approx(
        0.1 * library["demonstrations"][0]["adjacent_distance_floor"]
    )
    with pytest.raises(ContractError):
        module.build_tracking_references(
            model, [demo], library, fraction=0.1, training_episode_ids=["other"]
        )
    reference, _ = module.tracking_reference(
        tracking | {"arrays": references}, library["demonstrations"][0]
    )
    assert reference.frames == tuple(frames) and reference.source_split == "train"


def tracking_record(seed, score, mode="privileged_rollout", **extra):
    return (
        dict(
            seed=seed,
            mode=mode,
            score=score,
            status="completed",
            termination_reason="reference_stall",
            rollout_parity_checks=10,
            rollout_parity_mismatches=0,
            last_reference_index=20,
            first_close_reference_index=142,
            demo_grasp_reference_index=198,
        )
        | extra
    )


def test_tracking_ceiling_gate_counts_clean_attempts_and_readings():
    seeds = [43000, 43001, 43002, 43003]
    grasp = dict(reach=True, grasp=True, transport=False, place=False, release=False)
    records = [tracking_record(43000, grasp), tracking_record(43001, grasp)]
    summary = module.gate_summary(
        {"goal_kind": "trajectory", "privileged_ceiling": True, "seeds": seeds}, records
    )
    gate = summary["tracking_ceiling_gate"]
    assert set(summary) == {"tracking_ceiling_gate"}
    assert gate["primary_gate_passed"] and gate["learned_stage_authorized"]
    assert gate["per_reset"]["43002"]["counted"] is False
    failed = module.tracking_ceiling_gate(
        [records[0], records[1] | {"termination_reason": "attempt_timeout"}], seeds
    )
    assert not failed["primary_gate_passed"] and failed["privileged_grasp_resets"] == 1
    inexact = module.tracking_ceiling_gate(
        [records[0], records[1] | {"rollout_parity_mismatches": 1}], seeds
    )
    assert not inexact["primary_gate_passed"] and not inexact["rollouts_exact"]
    stalled = [tracking_record(s, {}) for s in seeds]
    readings = module.tracking_ceiling_gate(stalled, seeds)["readings"]
    assert readings["conclusive"] and readings["trajectory_tracking_inadequate"]
    assert readings["stalled_before_close_reference_resets"] == 4
    assert not readings["arm_pose_tracking_insufficient_for_grasp"]
    late = [r | {"last_reference_index": 198} for r in stalled]
    readings = module.tracking_ceiling_gate(late, seeds)["readings"]
    assert readings["arm_pose_tracking_insufficient_for_grasp"]
    assert not readings["trajectory_tracking_inadequate"]
    # Learned or endpoint-ceiling records never enter the tracking ceiling gate.
    learned = [tracking_record(s, grasp, mode="learned") for s in seeds]
    assert module.tracking_ceiling_gate(learned, seeds)["counted_attempts"] == 0


def test_tracking_learned_gate_requires_grasps_and_beating_both_controls():
    seeds = [43000, 43001, 43002, 43003]
    grasp = dict(reach=True, grasp=True, transport=False, place=False, release=False)
    reach = dict(reach=True, grasp=False)
    records = [
        tracking_record(43000, grasp, mode="learned"),
        tracking_record(43001, grasp, mode="learned"),
        tracking_record(43000, reach, mode="dynamics_shuffle"),
    ]
    summary = module.gate_summary({"goal_kind": "trajectory", "seeds": seeds}, records)
    gate = summary["tracking_gate"]
    assert set(summary) == {"tracking_gate"} and gate["primary_gate_passed"]
    assert gate["summed_ordered_stages"] == dict(learned=4, dynamics_shuffle=1, persistence=0)
    tied = records + [tracking_record(s, grasp, mode="persistence") for s in seeds[:2]]
    assert not module.tracking_gate(tied, seeds)["primary_gate_passed"]
    assert not module.tracking_gate(tied, seeds)["readings"]["no_model_contribution"]  # partial
    complete = [
        tracking_record(s, grasp if s == 43000 else {}, mode=m)
        for s in seeds
        for m in module.TRACKING_MODES
    ]
    readings = module.tracking_gate(complete, seeds)["readings"]
    assert readings["conclusive"] and readings["no_model_contribution"]
    privileged = [tracking_record(s, grasp) for s in seeds]
    assert module.tracking_gate(privileged, seeds)["learned_grasp_resets"] == 0


@pytest.mark.parametrize("gate_name", ["tracking_ceiling_gate", "tracking_gate"])
def test_tracking_pass_with_missing_attempts_does_not_make_readings_conclusive(gate_name):
    seeds = [43000, 43001, 43002, 43003]
    mode = "privileged_rollout" if gate_name == "tracking_ceiling_gate" else "learned"
    grasp = dict(reach=True, grasp=True)
    records = [tracking_record(seed, grasp, mode=mode) for seed in seeds[:2]]
    gate = getattr(module, gate_name)(records, seeds)
    # Keep the preregistered numeric gate and authorization unchanged. The
    # broader reading needs the complete planned denominator, including controls.
    assert gate["primary_gate_passed"]
    assert gate["counted_attempts"] == 2
    assert not gate["readings"]["conclusive"]
    if gate_name == "tracking_ceiling_gate":
        assert gate["learned_stage_authorized"]
        assert "inconclusive" in gate["readings"]["interpretation"]
        complete = records + [tracking_record(seed, {}, mode=mode) for seed in seeds[2:]]
    else:
        complete = records + [
            tracking_record(seed, {}, mode=current_mode)
            for current_mode in module.TRACKING_MODES
            for seed in seeds
            if current_mode != "learned" or seed not in seeds[:2]
        ]
    completed_gate = getattr(module, gate_name)(complete, seeds)
    assert completed_gate["primary_gate_passed"]
    assert completed_gate["readings"]["conclusive"]


def test_trajectory_worker_builds_tracking_controller_and_records_reference_progress(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")  # state plans resolve fields via the torch model
    from embodied_jepa import privileged_rollout, trajectory_tracking

    built = []

    class Ceiling:
        def __init__(self, model, robot, *, acknowledge_privileged_ceiling=False):
            assert acknowledge_privileged_ceiling is True

        def pop_diagnostics(self):
            return [{"previous_search_first_step_exact_match": True}]

        def close(self):
            built.append("closed")

    class Tracker:
        def __init__(self, model, reference, *, progress_distance, config, rule):
            built.append((type(model).__name__, reference.source_id, rule))
            self.decisions = [
                SimpleNamespace(action=np.zeros(14, np.float32), trace={"reference_index": 5}),
                SimpleNamespace(
                    action=None,
                    trace={"reference_index": 7},
                    termination_reason="reference_stall",
                ),
            ]

        def step(self, observation, projector):
            return self.decisions.pop(0)

        def acknowledge(self, result):
            return {"reference_index": 5}

    monkeypatch.setattr(privileged_rollout, "PrivilegedRolloutModel", Ceiling)
    monkeypatch.setattr(trajectory_tracking, "TrackingController", Tracker)
    state_worker_fixture(tmp_path, monkeypatch, [])
    model = StateMetricModel()
    demo = state_episode("train-a")
    arrays, library = module.build_state_goal_library(
        model, [demo], stride=16, dwell=1, training_episode_ids=["train-a"]
    )
    references, tracking = module.build_tracking_references(
        model, [demo], library, fraction=0.1, training_episode_ids=["train-a"]
    )
    monkeypatch.setattr(
        module, "load_tracking_references", lambda *a: tracking | {"arrays": references}
    )
    plan = tracking_plan(seeds=[43000], modes=["privileged_rollout"])
    module.write(tmp_path / "resolved_plan.json", plan | {"dataset": "u", "checkpoint": "u"})
    module.attempt_worker(tmp_path, "43000-privileged_rollout", attempt_seconds=600)
    report = json.loads((tmp_path / "attempts/43000-privileged_rollout/report.json").read_text())
    assert report["termination_reason"] == "reference_stall" and report["executed_steps"] == 1
    assert report["last_reference_index"] == 7 and report["reference_length"] == 40
    assert report["rollout_parity_checks"] == 1 and report["rollout_parity_mismatches"] == 0
    name, source, rule = built[0]
    assert name == "Ceiling" and source == "train-a/keyframes"
    assert (rule.window, rule.stall_commands, rule.stall_min_advance) == (16, 64, 16)
    assert built[-1] == "closed"


# TASK-046 hybrid phase control under the privileged ceiling (no physics here).


def hybrid_plan(**overrides):
    return state_plan(goal_kind="hybrid", attempt_max_seconds=840, **overrides)


def test_hybrid_plan_is_mode_major_non_learned_and_leaves_other_plans_unchanged():
    pytest.importorskip("torch")  # state plans resolve fields via the torch model
    plan = hybrid_plan()
    assert plan["modes"] == ["privileged_hybrid", "demo_replay", "privileged_hybrid_early"]
    assert [a["attempt_id"] for a in plan["attempts"][:5]] == [
        "43000-privileged_hybrid",
        "43001-privileged_hybrid",
        "43002-privileged_hybrid",
        "43003-privileged_hybrid",
        "43000-demo_replay",
    ]
    assert plan["attempts"][-1]["attempt_id"] == "43003-privileged_hybrid_early"
    assert plan["hybrid_handoff_rows_before_close"] == {
        "privileged_hybrid": 0,
        "privileged_hybrid_early": 16,
    }
    assert plan["privileged_ceiling"] is True and "NON-LEARNED" in plan["result_label"]
    trajectory = tracking_plan()
    for key in ("tracking_window", "tracking_stall_commands", "keyframe_threshold_fraction"):
        assert plan[key] == trajectory[key]
    assert plan["controller"] == trajectory["controller"]
    # Resets are identical to every other development plan.
    resets = {a["seed"]: a["reset"] for a in plan["attempts"]}
    assert resets == {a["seed"]: a["reset"] for a in trajectory["attempts"]}
    for other in (trajectory, state_plan(), module.make_plan(seeds=[43000])):
        assert not any(key.startswith(("hybrid", "handoff", "replay_rule")) for key in other)
        assert "attempt_order" not in other
    for bad in (
        dict(modes=["learned"]),
        dict(modes=["privileged_hybrid", "persistence"]),
        dict(modes=["privileged_rollout"]),
        dict(commitment_steps=4),
        dict(proposals=True),
        dict(goal_stall_limit=None),
    ):
        with pytest.raises(ValueError):
            hybrid_plan(**bad)
    with pytest.raises(ValueError):
        hybrid_plan(stage="final", seeds=None)
    with pytest.raises(ValueError):
        tracking_plan(modes=["privileged_hybrid"])
    with pytest.raises(ValueError):
        module.make_plan(seeds=[43000], modes=["privileged_hybrid_early"])


def hybrid_record(seed, score, mode="privileged_hybrid", **extra):
    fields = dict(
        termination_reason="demo_exhausted",
        handed_off=True,
        handoff_row=142,
        handoff_reference_index=143,
        handoff_frame=213,
        handoff_command=240,
        tracked_commands=240,
        replayed_commands=289,
    )
    return tracking_record(seed, score, mode=mode, **(fields | extra))


def test_hybrid_ceiling_gate_counts_only_the_primary_arm():
    seeds = [43000, 43001, 43002, 43003]
    grasp = dict(reach=True, grasp=True, transport=True, place=False, release=False)
    records = [hybrid_record(43000, grasp), hybrid_record(43001, grasp)]
    summary = module.gate_summary({"goal_kind": "hybrid", "seeds": seeds}, records)
    gate = summary["hybrid_ceiling_gate"]
    assert set(summary) == {"hybrid_ceiling_gate"}
    assert gate["primary_gate_passed"] and gate["privileged_hybrid_grasp_resets"] == 2
    assert gate["summed_ordered_stages"] == 6 and not gate["readings"]["conclusive"]
    row = gate["arms"]["privileged_hybrid"]["per_reset"]["43000"]
    assert row["handoff_reference_index"] == 143 and "stalled_before_close_reference" not in row
    assert "inconclusive" in gate["readings"]["interpretation"]
    # Secondary and reference arms never pass the primary gate.
    others = [
        hybrid_record(s, grasp, mode=m)
        for s in seeds
        for m in ("privileged_hybrid_early", "demo_replay")
    ]
    assert not module.hybrid_ceiling_gate(others, seeds)["primary_gate_passed"]
    timeout = module.hybrid_ceiling_gate(
        [records[0], records[1] | {"termination_reason": "attempt_timeout"}], seeds
    )
    assert not timeout["primary_gate_passed"]
    inexact = module.hybrid_ceiling_gate(
        [records[0], records[1] | {"rollout_parity_mismatches": 1}], seeds
    )
    assert not inexact["primary_gate_passed"] and not inexact["rollouts_exact"]
    invalid = module.hybrid_ceiling_gate(
        records + [records[0] | {"provenance_valid": False}], seeds
    )
    assert not invalid["primary_gate_passed"]


def test_hybrid_ceiling_readings_and_handoff_comparison():
    seeds = [43000, 43001, 43002, 43003]
    grasp = dict(reach=True, grasp=True)
    reach = dict(reach=True, grasp=False)
    replayed = [hybrid_record(s, reach) for s in seeds]
    early_grasp = [hybrid_record(s, grasp, mode="privileged_hybrid_early") for s in seeds]
    replay = [hybrid_record(s, grasp, mode="demo_replay") for s in seeds]
    gate = module.hybrid_ceiling_gate(replayed + early_grasp + replay, seeds)
    readings = gate["readings"]
    assert readings["conclusive"] and not gate["primary_gate_passed"]
    assert readings["replay_from_tracked_state_insufficient_for_grasp"]
    assert not readings["tracking_failed_before_handoff"]
    assert readings["secondary_early_handoff_grasp_on_two"]
    assert readings["handoff_comparison"].startswith("only the earlier handoff")
    assert readings["demo_replay_conclusive"] and gate["arms"]["demo_replay"]["grasp_resets"] == 4
    stalled = [
        hybrid_record(s, {}, handed_off=False, termination_reason="reference_stall") for s in seeds
    ]
    early_none = [hybrid_record(s, reach, mode="privileged_hybrid_early") for s in seeds]
    readings = module.hybrid_ceiling_gate(stalled + early_none, seeds)["readings"]
    assert readings["tracking_failed_before_handoff"]
    assert not readings["replay_from_tracked_state_insufficient_for_grasp"]
    assert readings["handoff_comparison"].startswith("neither handoff")
    assert not readings["demo_replay_conclusive"]
    passed = [hybrid_record(s, grasp) for s in seeds]
    gate = module.hybrid_ceiling_gate(passed + early_none, seeds)
    assert gate["primary_gate_passed"] and gate["readings"]["conclusive"]
    assert "attributable to the replayed demonstration" in gate["readings"]["interpretation"]
    assert gate["readings"]["handoff_comparison"].startswith("only the close-row handoff")
    # An incomplete secondary arm never makes the comparison conclusive.
    partial = module.hybrid_ceiling_gate(passed + early_grasp[:3], seeds)["readings"]
    assert partial["handoff_comparison"].startswith("inconclusive")


@pytest.mark.parametrize("mode,offset", [("privileged_hybrid", 0), ("privileged_hybrid_early", 16)])
def test_hybrid_worker_wraps_the_tracker_and_records_the_handoff(
    tmp_path, monkeypatch, mode, offset
):
    pytest.importorskip("torch")  # state plans resolve fields via the torch model
    from embodied_jepa import hybrid_phase, privileged_rollout, trajectory_tracking

    built = []

    class Ceiling:
        def __init__(self, model, robot, *, acknowledge_privileged_ceiling=False):
            assert acknowledge_privileged_ceiling is True

        def pop_diagnostics(self):
            return []

        def close(self):
            built.append("closed")

    class Tracker:
        def __init__(self, model, reference, *, progress_distance, config, rule):
            built.append((type(model).__name__, reference.source_id, config.ablation))

    class Hybrid:
        def __init__(self, tracker, actions, *, handoff_row, lower_bounds, upper_bounds):
            built.append(("hybrid", type(tracker).__name__, actions.shape, handoff_row))
            self.decisions = [
                SimpleNamespace(action=np.zeros(14, np.float32), trace={"reference_index": 5}),
                SimpleNamespace(action=np.zeros(14, np.float32), trace={"phase": "replay"}),
                SimpleNamespace(action=None, trace={}, termination_reason="demo_exhausted"),
            ]

        def step(self, observation, projector):
            return self.decisions.pop(0)

        def acknowledge(self, result):
            return {"phase": "x"}

        def summary(self):
            return {"handed_off": True, "handoff_reference_index": 6, "replayed_commands": 1}

    monkeypatch.setattr(privileged_rollout, "PrivilegedRolloutModel", Ceiling)
    monkeypatch.setattr(trajectory_tracking, "TrackingController", Tracker)
    monkeypatch.setattr(hybrid_phase, "HybridPhaseController", Hybrid)
    state_worker_fixture(tmp_path, monkeypatch, [])
    model = StateMetricModel()
    demo = state_episode("train-a")
    demo.actions[:, 13] = -1.0
    demo.actions[20:, 13] = 1.0  # first closing command at action 20 -> frame 21
    arrays, library = module.build_state_goal_library(
        model, [demo], stride=16, dwell=1, training_episode_ids=["train-a"]
    )
    references, tracking = module.build_tracking_references(
        model, [demo], library, fraction=0.1, training_episode_ids=["train-a"]
    )
    monkeypatch.setattr(module, "load_state_library", lambda *a: library | {"arrays": arrays})
    monkeypatch.setattr(
        module, "load_tracking_references", lambda *a: tracking | {"arrays": references}
    )
    plan = hybrid_plan(seeds=[43000], modes=[mode])
    module.write(tmp_path / "resolved_plan.json", plan | {"dataset": "u", "checkpoint": "u"})
    module.attempt_worker(tmp_path, f"43000-{mode}", attempt_seconds=840)
    report = json.loads((tmp_path / f"attempts/43000-{mode}/report.json").read_text())
    close = tracking["references"][0]["first_close_reference_index"]
    assert close == 21
    assert built[0] == ("Ceiling", "train-a/keyframes", "learned")
    assert built[1] == ("hybrid", "Tracker", (39, 14), max(0, close - offset))
    assert report["termination_reason"] == "demo_exhausted" and report["executed_steps"] == 2
    assert report["handed_off"] and report["handoff_reference_index"] == 6
    assert report["last_reference_index"] == 5 and report["first_close_reference_index"] == 21
    assert report["rollout_parity_checks"] == 0 and built[-1] == "closed"


def test_hybrid_worker_refuses_an_undeclared_plan(tmp_path, monkeypatch):
    pytest.importorskip("torch")  # state plans resolve fields via the torch model
    state_worker_fixture(tmp_path, monkeypatch, [])
    monkeypatch.setattr(module, "load_tracking_references", lambda *a: {})
    plan = tracking_plan(seeds=[43000], modes=["privileged_rollout"])
    plan["attempts"] = [
        plan["attempts"][0] | {"attempt_id": "43000-privileged_hybrid", "mode": "privileged_hybrid"}
    ]
    module.write(tmp_path / "resolved_plan.json", plan | {"dataset": "u", "checkpoint": "u"})
    module.attempt_worker(tmp_path, "43000-privileged_hybrid", attempt_seconds=840)
    report = json.loads((tmp_path / "attempts/43000-privileged_hybrid/report.json").read_text())
    assert report["termination_reason"] == "runtime_error"
    assert "declared hybrid ceiling" in report["error"] and report["executed_steps"] == 0


# TASK-047 wide-jitter distribution and object-aware privileged ceiling (no physics here).


def object_plan(**overrides):
    options = dict(
        goal_kind="object",
        stride=16,
        dwell=1,
        horizon=6,
        candidates=24,
        iterations=2,
        proposals=False,
        max_seconds=8400,
        attempt_max_seconds=1500,
    )
    return module.make_plan(**(options | overrides))


def test_object_plan_uses_new_wide_resets_and_leaves_other_plans_unchanged():
    plan = object_plan()
    assert plan["seeds"] == list(range(45000, 45008))
    assert plan["modes"] == ["demo_replay", "scripted_oracle", "privileged_object"]
    assert [a["attempt_id"] for a in plan["attempts"][:2]] == [
        "45000-demo_replay",
        "45001-demo_replay",
    ]
    assert plan["attempts"][-1]["attempt_id"] == "45007-privileged_object"
    assert plan == object_plan()
    offsets = []
    for attempt in plan["attempts"]:
        reset = attempt["reset"]
        assert reset == module.wide_reset(attempt["seed"])
        obj = np.array(reset["object_xy"]) - [0.34, -0.18]
        plate = np.array(reset["plate_xy"]) - [0.49, -0.09]
        assert np.abs(obj).max() <= 0.03 and np.abs(plate).max() <= 0.02
        offsets.append(np.abs(obj).max())
    assert max(offsets) > 0.006  # genuinely wider than the narrow distribution
    assert plan["object_ceiling"]["horizon"] == 6 and "seed" not in plan["object_ceiling"]
    # Existing narrow plans neither gain object fields nor accept object modes.
    narrow = module.make_plan(seeds=[43000], modes=["learned"])
    assert narrow["reset_jitter_m"] == 0.006 and "object_ceiling" not in narrow
    for bad in (
        dict(stage="final"),
        dict(seeds=[43000]),
        dict(seeds=[45000, 45000]),
        dict(modes=["learned"]),
        dict(proposals=True),
        dict(max_seconds=9001),
        dict(goal_stall_limit=64),
        dict(commitment_steps=2),
    ):
        with pytest.raises(ValueError):
            object_plan(**bad)
    with pytest.raises(ValueError, match="goal-kind object"):
        module.make_plan(seeds=[43000], modes=["scripted_oracle"])


def object_record(seed, score, mode="privileged_object", **extra):
    record = {
        "seed": seed,
        "mode": mode,
        "status": "completed",
        "termination_reason": "success",
        "score": score,
    }
    if mode == "privileged_object":
        record.update(
            rollout_parity_checks=10,
            rollout_parity_mismatches=0,
            rollout_full_state_parity_checks=10,
            rollout_full_state_parity_mismatches=0,
            phase="retreat",
        )
    return record | extra


def arms(ceiling, replay, scripted=8, seeds=tuple(range(45000, 45008))):
    grasp = {"reach": True, "grasp": True}
    reach = {"reach": True}
    records = []
    for mode, count in (
        ("privileged_object", ceiling),
        ("demo_replay", replay),
        ("scripted_oracle", scripted),
    ):
        for n, seed in enumerate(seeds):
            records.append(
                object_record(seed, grasp if n < count else reach, mode=mode, phase="lift")
            )
    return records


def test_object_ceiling_gate_outcomes_and_next_steps():
    seeds = list(range(45000, 45008))

    def gate(records):
        summary = module.gate_summary({"goal_kind": "object", "seeds": seeds}, records)
        assert set(summary) == {"object_ceiling_gate"}
        return summary["object_ceiling_gate"]

    passed = gate(arms(6, 3))
    assert passed["primary_gate_passed"] and passed["readings"]["outcome"] == "separated"
    assert passed["readings"]["preregistered_next_step"].startswith("T2")
    assert gate(arms(8, 5))["primary_gate_passed"]
    assert gate(arms(6, 4))["readings"]["outcome"] == "ceiling_adequate_replay_not_separated"
    assert gate(arms(5, 0))["readings"]["outcome"] == "ceiling_inadequate_task_feasible"
    failed = gate(arms(5, 0, scripted=2))
    assert failed["readings"]["outcome"] == "ceiling_inadequate_scripted_also_fails"
    assert failed["arms"]["privileged_object"]["failed_reset_furthest_phase"] == {"lift": 3}
    # The scripted reference never changes the gate itself.
    assert gate(arms(6, 3, scripted=0))["primary_gate_passed"]
    # Inexact rollouts, a failed ceiling attempt, an incomplete replay arm or invalid
    # provenance can never pass and make the reading inconclusive.
    records = arms(8, 0)
    records[0]["rollout_full_state_parity_mismatches"] = 1
    assert gate(records)["readings"]["outcome"] == "inconclusive"
    assert not gate(records)["primary_gate_passed"]
    records = arms(8, 0)
    records[8]["termination_reason"] = "runtime_error"  # a demo_replay attempt
    assert not gate(records)["primary_gate_passed"]
    assert gate(records)["readings"]["outcome"] == "inconclusive"
    records = arms(8, 0)
    records[0]["termination_reason"] = "attempt_timeout"
    assert gate(records)["privileged_object_grasp_resets"] == 7
    assert gate(records)["readings"]["outcome"] == "inconclusive"
    records = arms(8, 0)
    records[3]["provenance_valid"] = False
    assert not gate(records)["primary_gate_passed"]
    # Clean ceiling stops (drop, phase stall, guard refusal) are counted failures.
    records = arms(8, 0)
    records[0]["termination_reason"] = "object_dropped"
    records[0]["score"] = {"reach": True}
    assert gate(records)["readings"]["conclusive"]


def test_object_worker_builds_the_ceiling_and_records_parity(tmp_path, monkeypatch):
    pytest.importorskip("torch")  # the fixture reuses the state-worker plumbing
    from embodied_jepa import object_ceiling

    built = []

    class Rollouts:
        def __init__(self, model, robot, *, acknowledge_privileged_ceiling=False):
            assert acknowledge_privileged_ceiling is True
            self.rows = [
                [],
                [
                    {
                        "previous_search_first_step_exact_match": True,
                        "previous_search_first_step_full_state_exact_match": False,
                    }
                ],
            ]

        def pop_diagnostics(self):
            return self.rows.pop(0) if self.rows else []

        def close(self):
            built.append("closed")

    class Ceiling:
        def __init__(self, rollouts, robot, config, *, acknowledge_privileged_ceiling=False):
            assert acknowledge_privileged_ceiling is True
            built.append((type(rollouts).__name__, config.seed, config.horizon))
            self.decisions = [
                SimpleNamespace(action=np.zeros(14, np.float32), trace={"phase": "approach"}),
                SimpleNamespace(action=np.zeros(14, np.float32), trace={"phase": "approach"}),
                SimpleNamespace(action=None, trace={}, termination_reason="phase_stall"),
            ]

        def step(self, observation, projector):
            return self.decisions.pop(0)

        def acknowledge(self, result):
            return {"phase": "approach"}

        def summary(self):
            return {"phase": "approach", "furthest_phase_index": 0, "phase_log": []}

    monkeypatch.setattr(object_ceiling, "ObjectRolloutModel", Rollouts)
    monkeypatch.setattr(object_ceiling, "ObjectCeilingController", Ceiling)
    state_worker_fixture(tmp_path, monkeypatch, [])
    plan = object_plan(seeds=[45003], modes=["privileged_object"])
    module.write(tmp_path / "resolved_plan.json", plan | {"dataset": "u", "checkpoint": "u"})
    module.attempt_worker(tmp_path, "45003-privileged_object", attempt_seconds=1500)
    report = json.loads((tmp_path / "attempts/45003-privileged_object/report.json").read_text())
    assert built[0] == ("Rollouts", 45003, 6) and built[-1] == "closed"
    assert report["termination_reason"] == "phase_stall" and report["executed_steps"] == 2
    assert report["rollout_parity_checks"] == 1 and report["rollout_parity_mismatches"] == 0
    assert report["rollout_full_state_parity_checks"] == 1
    assert report["rollout_full_state_parity_mismatches"] == 1
    assert report["furthest_phase_index"] == 0 and "last_goal_index" not in report


def test_object_modes_refuse_an_undeclared_plan(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    state_worker_fixture(tmp_path, monkeypatch, [])
    plan = state_plan(seeds=[43000], modes=["demo_replay"])
    plan["attempts"] = [
        plan["attempts"][0] | {"attempt_id": "43000-scripted_oracle", "mode": "scripted_oracle"}
    ]
    module.write(tmp_path / "resolved_plan.json", plan | {"dataset": "u", "checkpoint": "u"})
    module.attempt_worker(tmp_path, "43000-scripted_oracle", attempt_seconds=200)
    report = json.loads((tmp_path / "attempts/43000-scripted_oracle/report.json").read_text())
    assert report["termination_reason"] == "runtime_error"
    assert "declared object plan" in report["error"]


def test_object_gate_requires_every_ceiling_attempt_and_non_vacuous_parity():
    seeds = list(range(45000, 45008))
    records = arms(7, 0)
    records[0]["termination_reason"] = "runtime_error"  # 45000 ceiling: 6 counted grasps
    gate = module.object_ceiling_gate(records, seeds)
    assert gate["privileged_object_grasp_resets"] == 6
    assert not gate["primary_gate_passed"] and gate["readings"]["outcome"] == "inconclusive"
    vacuous = arms(8, 0)
    vacuous[0].update(executed_steps=10, rollout_parity_checks=0)
    gate = module.object_ceiling_gate(vacuous, seeds)
    assert not gate["rollouts_exact"] and not gate["primary_gate_passed"]
    enough = arms(8, 0)
    enough[0].update(executed_steps=11)  # 10 checks >= 11 - 1
    assert module.object_ceiling_gate(enough, seeds)["primary_gate_passed"]


def test_open_loop_guard_refusal_is_clean_only_for_object_plans():
    from embodied_jepa.constraints import CandidateProjection

    class Robot:
        def __init__(self, message=None):
            self.message = message

        def project_candidates(self, requested):
            if self.message:
                raise ContractError(self.message)
            return CandidateProjection(requested, np.ones(requested.shape[:2], bool))

    command = np.zeros(14, np.float32)
    guard = "measured joint velocity limit exceeded"
    assert module.project_open_loop(Robot(guard), command, guarded=True) is None
    with pytest.raises(ContractError, match="velocity"):
        module.project_open_loop(Robot(guard), command, guarded=False)
    with pytest.raises(ContractError, match="unconsumed"):
        module.project_open_loop(Robot("unconsumed"), command, guarded=True)
    assert module.project_open_loop(Robot(), command, guarded=True).actions.shape == (1, 1, 1, 14)


# TASK-049 object-aware ceiling v2 on fresh wide-jitter resets (no physics here).


def test_object_v2_plan_adds_fresh_primary_resets_and_leaves_the_v1_plan_unchanged():
    v1 = object_plan()
    assert v1 == object_plan(object_ceiling_version=1)
    assert "object_ceiling_version" not in v1 and "primary_seeds" not in v1
    plan = object_plan(object_ceiling_version=2, max_seconds=16800)
    primary, secondary = list(range(45100, 45108)), list(range(45000, 45008))
    assert plan["seeds"] == primary + secondary
    assert plan["primary_seeds"] == primary and plan["secondary_seeds"] == secondary
    assert plan["object_ceiling_version"] == 2
    assert plan["attempts"][0]["attempt_id"] == "45100-demo_replay"
    assert plan["attempts"][-1]["attempt_id"] == "45007-privileged_object"
    assert len(plan["attempts"]) == 48
    for attempt in plan["attempts"]:
        assert attempt["reset"] == module.wide_reset(attempt["seed"])
    assert plan["object_ceiling"]["release_ramp"] == 0.08
    assert "seed" not in plan["object_ceiling"]
    # Fresh draws differ from every TASK-047 draw.
    old = {tuple(module.wide_reset(s)["object_xy"]) for s in secondary}
    assert not old & {tuple(module.wide_reset(s)["object_xy"]) for s in primary}
    for bad in (
        dict(object_ceiling_version=4),
        dict(object_ceiling_version=2, max_seconds=21601),
        dict(object_ceiling_version=2, seeds=[44000]),
        dict(seeds=[45100]),  # fresh resets are not v1 resets
    ):
        with pytest.raises(ValueError):
            object_plan(**bad)
    with pytest.raises(ValueError, match="goal-kind object"):
        module.make_plan(seeds=[43000], modes=["learned"], object_ceiling_version=2)


def v2_records(successes, grasps, scripted=8, replay=2):
    primary = tuple(range(45100, 45108))
    records = arms(grasps, replay, scripted=scripted, seeds=primary)
    full = {k: True for k in ("reach", "grasp", "transport", "place", "release", "success")}
    for n, record in enumerate(r for r in records if r["mode"] == "privileged_object"):
        if n < successes:
            record["score"] = full
    for n, record in enumerate(r for r in records if r["mode"] == "scripted_oracle"):
        if n < scripted:
            record["score"] = full
    return records + arms(0, 0, scripted=0, seeds=tuple(range(45000, 45008)))


def test_object_v2_gate_outcomes_use_the_primary_resets_only():
    primary, secondary = list(range(45100, 45108)), list(range(45000, 45008))
    plan = {
        "goal_kind": "object",
        "object_ceiling_version": 2,
        "seeds": primary + secondary,
        "primary_seeds": primary,
        "secondary_seeds": secondary,
    }

    def gate(records):
        summary = module.gate_summary(plan, records)
        assert set(summary) == {"object_ceiling_v2_gate"}
        return summary["object_ceiling_v2_gate"]

    passed = gate(v2_records(6, 7))
    assert passed["primary_gate_passed"] and passed["readings"]["outcome"] == "ceiling_adequate"
    assert passed["readings"]["preregistered_next_step"].startswith("T4")
    assert passed["readings"]["replay_separated_diagnostic"]
    assert passed["secondary_arms"]["privileged_object"]["grasp_resets"] == 0
    assert not gate(v2_records(5, 8))["primary_gate_passed"]
    assert gate(v2_records(5, 8))["readings"]["outcome"] == "grasp_adequate_place_inadequate"
    assert not gate(v2_records(6, 6))["primary_gate_passed"]
    assert gate(v2_records(6, 6))["readings"]["outcome"] == "ceiling_inadequate_task_feasible"
    failed = gate(v2_records(2, 3, scripted=4))
    assert failed["readings"]["outcome"] == "ceiling_inadequate_scripted_also_fails"
    assert not gate(v2_records(8, 8, replay=6))["readings"]["replay_separated_diagnostic"]
    assert gate(v2_records(8, 8, replay=6))["primary_gate_passed"]  # replay never gates
    records = v2_records(8, 8)
    records[0]["termination_reason"] = "runtime_error"  # a primary ceiling attempt
    assert gate(records)["readings"]["outcome"] == "inconclusive"
    records = v2_records(8, 8)
    records[0]["rollout_full_state_parity_mismatches"] = 1
    assert not gate(records)["primary_gate_passed"]
    records = v2_records(8, 8)
    records[-1]["termination_reason"] = "runtime_error"  # a secondary attempt never gates
    assert gate(records)["primary_gate_passed"]
    records = v2_records(8, 8)
    records[-1]["provenance_valid"] = False
    assert not gate(records)["primary_gate_passed"]


def test_object_v2_worker_builds_the_v2_ceiling(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from embodied_jepa import object_ceiling_v2

    built = []

    class Rollouts:
        def __init__(self, model, robot, *, acknowledge_privileged_ceiling=False):
            assert acknowledge_privileged_ceiling is True

        def pop_diagnostics(self):
            return []

        def close(self):
            built.append("closed")

    class Ceiling:
        def __init__(self, rollouts, robot, config, *, acknowledge_privileged_ceiling=False):
            built.append((type(rollouts).__name__, type(config).__name__, config.seed))

        def step(self, observation, projector):
            return SimpleNamespace(action=None, trace={}, termination_reason="phase_stall")

        def summary(self):
            return {"phase": "approach", "furthest_phase_index": 0, "ceiling_version": 2}

    monkeypatch.setattr(object_ceiling_v2, "ObjectRolloutV2Model", Rollouts)
    monkeypatch.setattr(object_ceiling_v2, "ObjectCeilingV2Controller", Ceiling)
    state_worker_fixture(tmp_path, monkeypatch, [])
    plan = object_plan(
        object_ceiling_version=2, seeds=[45104], modes=["privileged_object"], max_seconds=16800
    )
    module.write(tmp_path / "resolved_plan.json", plan | {"dataset": "u", "checkpoint": "u"})
    module.attempt_worker(tmp_path, "45104-privileged_object", attempt_seconds=960)
    report = json.loads((tmp_path / "attempts/45104-privileged_object/report.json").read_text())
    assert built == [("Rollouts", "ObjectCeilingV2Config", 45104), "closed"]
    assert report["termination_reason"] == "phase_stall" and report["ceiling_version"] == 2


# TASK-051 object-aware ceiling v3 (scheduled grasp closure) on fresh wide-jitter resets.


def test_object_v3_plan_adds_fresh_primary_resets_and_leaves_v1_and_v2_unchanged():
    v1, v2 = object_plan(), object_plan(object_ceiling_version=2, max_seconds=16800)
    plan = object_plan(object_ceiling_version=3, max_seconds=30000)
    primary = list(range(45200, 45208))
    secondary = list(range(45100, 45108)) + list(range(45000, 45008))
    assert plan["seeds"] == primary + secondary
    assert plan["primary_seeds"] == primary and plan["secondary_seeds"] == secondary
    assert plan["object_ceiling_version"] == 3
    assert len(plan["attempts"]) == 72
    assert plan["attempts"][0]["attempt_id"] == "45200-demo_replay"
    assert plan["attempts"][-1]["attempt_id"] == "45007-privileged_object"
    for attempt in plan["attempts"]:
        assert attempt["reset"] == module.wide_reset(attempt["seed"])
    ceiling = plan["object_ceiling"]
    assert ceiling["release_ramp"] == 0.08  # the v2 release predictor carries over
    assert 0 < ceiling["close_descent_bound"] <= ceiling["arm_bound"]
    assert "seed" not in ceiling
    assert "v3" in plan["result_label"] and "NON-LEARNED" in plan["result_label"]
    # Earlier plans are untouched by the new option.
    assert v1 == object_plan()
    assert v2 == object_plan(object_ceiling_version=2, max_seconds=16800)
    assert "close_descent_bound" not in v2["object_ceiling"]
    # Fresh draws differ from every earlier wide-jitter draw.
    old = {tuple(module.wide_reset(s)["object_xy"]) for s in secondary}
    assert not old & {tuple(module.wide_reset(s)["object_xy"]) for s in primary}
    for bad in (
        dict(object_ceiling_version=3, max_seconds=32401),
        dict(object_ceiling_version=3, seeds=[44000]),
        dict(object_ceiling_version=2, seeds=[45200]),  # v3 resets are not v2 resets
    ):
        with pytest.raises(ValueError):
            object_plan(**bad)


def v3_records(successes, grasps, scripted=8, replay=2):
    primary = tuple(range(45200, 45208))
    records = arms(grasps, replay, scripted=scripted, seeds=primary)
    full = {k: True for k in ("reach", "grasp", "transport", "place", "release", "success")}
    for n, record in enumerate(r for r in records if r["mode"] == "privileged_object"):
        if n < successes:
            record["score"] = full
    for n, record in enumerate(r for r in records if r["mode"] == "scripted_oracle"):
        if n < scripted:
            record["score"] = full
    secondary = tuple(range(45100, 45108)) + tuple(range(45000, 45008))
    return records + arms(0, 0, scripted=0, seeds=secondary)


def test_object_v3_gate_uses_the_same_rules_on_its_own_primary_resets():
    primary = list(range(45200, 45208))
    secondary = list(range(45100, 45108)) + list(range(45000, 45008))
    plan = {
        "goal_kind": "object",
        "object_ceiling_version": 3,
        "seeds": primary + secondary,
        "primary_seeds": primary,
        "secondary_seeds": secondary,
    }

    def gate(records):
        summary = module.gate_summary(plan, records)
        assert set(summary) == {"object_ceiling_v3_gate"}
        return summary["object_ceiling_v3_gate"]

    passed = gate(v3_records(6, 7))
    assert passed["primary_gate_passed"] and passed["readings"]["outcome"] == "ceiling_adequate"
    assert "v3" in passed["label"] and "v2" not in passed["label"]
    assert "v3" in passed["readings"]["interpretation"]
    assert passed["rule"].startswith("primary resets only: privileged_object (v3)")
    assert passed["secondary_seeds"] == secondary
    assert not gate(v3_records(6, 6))["primary_gate_passed"]
    assert gate(v3_records(6, 6))["readings"]["outcome"] == "ceiling_inadequate_task_feasible"
    assert gate(v3_records(5, 8))["readings"]["outcome"] == "grasp_adequate_place_inadequate"
    records = v3_records(8, 8)
    records[-1]["termination_reason"] = "runtime_error"  # a secondary attempt never gates
    assert gate(records)["primary_gate_passed"]
    records = v3_records(8, 8)
    records[0]["rollout_full_state_parity_mismatches"] = 1
    assert not gate(records)["primary_gate_passed"]
    with pytest.raises(ValueError, match="versions 2 and 3"):
        module.object_ceiling_v2_gate([], primary, secondary, version=1)


def test_object_v3_worker_builds_the_v3_ceiling(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from embodied_jepa import object_ceiling_v2, object_ceiling_v3

    built = []

    class Rollouts:
        def __init__(self, model, robot, *, acknowledge_privileged_ceiling=False):
            assert acknowledge_privileged_ceiling is True

        def pop_diagnostics(self):
            return []

        def close(self):
            built.append("closed")

    class Ceiling:
        def __init__(self, rollouts, robot, config, *, acknowledge_privileged_ceiling=False):
            built.append((type(rollouts).__name__, type(config).__name__, config.seed))

        def step(self, observation, projector):
            return SimpleNamespace(action=None, trace={}, termination_reason="phase_stall")

        def summary(self):
            return {"phase": "approach", "furthest_phase_index": 0, "ceiling_version": 3}

    monkeypatch.setattr(object_ceiling_v2, "ObjectRolloutV2Model", Rollouts)
    monkeypatch.setattr(object_ceiling_v3, "ObjectCeilingV3Controller", Ceiling)
    state_worker_fixture(tmp_path, monkeypatch, [])
    plan = object_plan(
        object_ceiling_version=3, seeds=[45204], modes=["privileged_object"], max_seconds=30000
    )
    module.write(tmp_path / "resolved_plan.json", plan | {"dataset": "u", "checkpoint": "u"})
    module.attempt_worker(tmp_path, "45204-privileged_object", attempt_seconds=960)
    report = json.loads((tmp_path / "attempts/45204-privileged_object/report.json").read_text())
    assert built == [("Rollouts", "ObjectCeilingV3Config", 45204), "closed"]
    assert report["termination_reason"] == "phase_stall" and report["ceiling_version"] == 3
