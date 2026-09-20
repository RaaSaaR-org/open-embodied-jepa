"""Collector accounting uses a cheap simulated transport, not robot-performance evidence."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa import manipulation_collection as module
from embodied_jepa.contracts import ExecutionResult, Observation, StateSchema


def test_frozen_plan_is_deterministic_separated_and_excludes_heldout():
    plan = module.make_plan()
    assert plan == module.make_plan()
    assert len(plan["trials"]) == 100
    assert [t["seed"] for t in plan["trials"]] == list(range(1000, 1100))
    for trial in plan["trials"]:
        assert (trial["object"], trial["container"]) != ("apple", "plate")
        assert np.linalg.norm(np.array(trial["object_xy"]) - trial["container_xy"]) > 0.15
    with pytest.raises(ValueError):
        module.make_plan(max_seconds=601)


class FakeSimulation:
    def __init__(self, **kwargs):
        self.kind = kwargs["object_kind"], kwargs["container_kind"]
        self.model = SimpleNamespace(camera=lambda name: SimpleNamespace(fovy=[75]))


class FakeRobot:
    state_schema = StateSchema(("q",), ("rad",), "test_robot_v0")
    manifest = {"action_schema": "ee_delta_grasp_v0", "fixture": True}

    def __init__(self, sim):
        self.sim = sim

    def reset(self, seed, **kwargs):
        self.seed, self.steps = seed, 0
        return {"fixture": True, "seed": seed}

    def observe(self):
        if self.seed == 1002 and self.steps == 9:
            raise RuntimeError("lost RGB after a real command")
        values = np.array([[self.steps]], dtype=np.float32)
        return Observation(
            {"onboard_rgb": np.zeros((1, 8, 8, 3), dtype=np.uint8)},
            values,
            np.ones_like(values, dtype=bool),
            np.array([self.steps * 0.05]),
            self.state_schema,
        )

    def execute(self, action):
        self.steps += 1
        return ExecutionResult(action, action, "applied", self.steps * 0.05)

    def denormalize_action(self, action):
        return action.copy()

    def stop(self, reason):
        pass

    def close(self):
        pass


class FakePolicy:
    def __init__(self, truth):
        self.steps, self.failure = 0, ""

    @property
    def done(self):
        return self.steps == 10

    @property
    def phase(self):
        return "lift"

    def action(self, robot):
        return np.zeros(14, dtype=np.float32)

    def advance(self, result):
        self.steps += 1


class FakeTask:
    def __init__(self, robot):
        self.robot = robot

    def evaluate(self):
        return {
            "success": self.robot.steps >= 10,
            "reach": True,
            "grasp": self.robot.steps >= 5,
            "transport": False,
            "place": False,
            "release": False,
        }


def test_collection_keeps_prefix_and_failure_denominator(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    pytest.importorskip("PIL")
    monkeypatch.setattr(module, "MuJoCoSimulation", FakeSimulation)
    monkeypatch.setattr(module, "G1Embodiment", FakeRobot)
    monkeypatch.setattr(module, "OracleManipulationPolicy", FakePolicy)
    monkeypatch.setattr(module, "AppleToPlateTask", FakeTask)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "0.0.0")
    root = tmp_path / "corpus"
    report = module.collect(root, episodes=12, max_commands=10, width=8)
    assert report["attempted_episodes"] == report["success_denominator"] == 12
    assert report["stored_episodes"] == 12 and report["successes"] == 11
    failed = report["attempts"][2]
    assert failed["complete_transitions"] == 8
    assert failed["incomplete_execution_count"] == 1
    assert "observe_following" in failed["reasons"][0]
    assert {key: len(value) for key, value in report["splits"].items()} == {
        "train": 10,
        "val": 1,
        "test": 1,
        "holdout": 0,
    }
    store = module.DatasetStore(root)
    episode = store.read_episode(failed["episode_id"])
    assert episode.actions.shape == (8, 14)
    assert episode.observations["onboard_rgb"].shape[0] == 9
    assert len(episode.metadata["phase_labels"]) == 8
    assert episode.truncated and not episode.terminated
    assert json.loads((root / "collection_plan.json").read_text())["requested_episodes"] == 12
    with pytest.raises(FileExistsError):
        module.collect(root, episodes=12, max_commands=10, width=8)


def test_global_wall_budget_reports_unattempted_and_partial_trial(tmp_path, monkeypatch):
    import itertools

    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    pytest.importorskip("PIL")
    monkeypatch.setattr(module, "MuJoCoSimulation", FakeSimulation)
    monkeypatch.setattr(module, "G1Embodiment", FakeRobot)
    monkeypatch.setattr(module, "OracleManipulationPolicy", FakePolicy)
    monkeypatch.setattr(module, "AppleToPlateTask", FakeTask)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "0.0.0")
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)
    ticks = itertools.count(step=0.001)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    report = module.collect(
        tmp_path / "bounded", episodes=12, max_commands=10, max_seconds=0.02, width=8
    )
    assert report["attempted_episodes"] + report["unattempted_budget_episodes"] == 12
    assert report["unattempted_budget_episodes"] > 0
    assert any("wall_budget" in row["reasons"] for row in report["attempts"])
    assert report["success_denominator"] == report["attempted_episodes"]


def test_clock_budget_includes_suspension_and_resists_backward_adjustment(monkeypatch):
    monkeypatch.setattr(module.time, "time", lambda: 2800.0)
    monkeypatch.setattr(module.time, "monotonic", lambda: 330.0)
    assert module.elapsed_seconds(1000.0, 30.0) == 1800.0
    monkeypatch.setattr(module.time, "time", lambda: 900.0)
    assert module.elapsed_seconds(1000.0, 30.0) == 300.0


def test_interrupt_preserves_current_observed_prefix_and_seals_report(tmp_path, monkeypatch):
    import signal

    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    pytest.importorskip("PIL")

    class InterruptedRobot(FakeRobot):
        def execute(self, action):
            result = super().execute(action)
            if self.steps == 9:
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            return result

    original_handler = signal.getsignal(signal.SIGINT)
    monkeypatch.setattr(module, "MuJoCoSimulation", FakeSimulation)
    monkeypatch.setattr(module, "G1Embodiment", InterruptedRobot)
    monkeypatch.setattr(module, "OracleManipulationPolicy", FakePolicy)
    monkeypatch.setattr(module, "AppleToPlateTask", FakeTask)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "0.0.0")
    report = module.collect(
        tmp_path / "interrupted", episodes=12, seed=2000, max_commands=10, width=8
    )
    assert report["attempted_episodes"] == report["stored_episodes"] == 1
    assert report["attempts"][0]["complete_transitions"] == 9
    assert report["attempts"][0]["reasons"] == ["interrupt_requested"]
    assert report["unattempted_interrupted_episodes"] == 11
    assert report["unattempted_budget_episodes"] == 0
    assert signal.getsignal(signal.SIGINT) == original_handler


def test_early_release_policy_dispatch_and_invalid_reset_count(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    pytest.importorskip("PIL")
    calls = []

    class EarlyPolicy(FakePolicy):
        def __init__(self, truth, *, opening_ramp):
            super().__init__(truth)
            calls.append(opening_ramp)

    class InvalidResetRobot(FakeRobot):
        def reset(self, seed, **kwargs):
            if seed == 3001:
                raise ValueError("initial overlap")
            return super().reset(seed, **kwargs)

    monkeypatch.setattr(module, "MuJoCoSimulation", FakeSimulation)
    monkeypatch.setattr(module, "G1Embodiment", InvalidResetRobot)
    monkeypatch.setattr(module, "EarlyReleaseOracleManipulationPolicy", EarlyPolicy)
    monkeypatch.setattr(module, "AppleToPlateTask", FakeTask)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "0.0.0")
    root = tmp_path / "release"
    report = module.collect(
        root,
        episodes=4,
        seed=3000,
        max_commands=10,
        width=8,
        policy_name="early_release_v1",
        container_center=(0.45, -0.10),
    )
    assert calls == [0.08, 0.08, 0.08]
    assert report["attempted_episodes"] == report["success_denominator"] == 4
    assert report["stored_episodes"] == 3
    assert not report["attempts"][1]["stored"]
    assert "initial overlap" in report["attempts"][1]["reasons"][0]
    store = module.DatasetStore(root)
    store.verify()
    assert "meta/collection_source/scripted.py" in store.manifest["sha256"]
    assert report["plan"]["policy_options"] == {"opening_ramp": 0.08}
    with pytest.raises(ValueError):
        module.make_plan(policy_name="early_release_v1", max_commands=746)


def test_early_release_ramp_tracks_accepted_command_and_stops_on_rejection():
    from embodied_jepa.scripted import EarlyReleaseOracleManipulationPolicy

    truth = {
        "position_frame": "world",
        "base_position_world": [0, 0, 0.793],
        "base_rotation_world": np.eye(3),
        "object_position": [0.34, -0.18, 0.766],
        "plate_position": [0.45, -0.10, 0.746],
        "container_surface_z": 0.752,
        "object_support_height": 0.023,
    }
    policy = EarlyReleaseOracleManipulationPolicy(truth, opening_ramp=0.08)
    robot = SimpleNamespace(
        ee_pose=lambda side: (np.array([0.42, -0.1, 0.18]), np.eye(3)),
        manifest={"translation_per_step_m": 0.015, "rotation_per_step_rad": 0.06},
    )
    policy.phase_index = 5
    policy.accepted_grasp = 1.0
    requested = policy.action(robot)
    assert requested[13] == pytest.approx(0.92)
    applied = requested.copy()
    applied[13] = 0.96  # A transport can accept a smaller safe opening step.
    policy.advance(ExecutionResult(requested, applied, "clipped", 0.05, "joint rate limit"))
    assert policy.action(robot)[13] == pytest.approx(0.88)
    policy.advance(ExecutionResult(requested, None, "stopped", 0.05, "velocity stop"))
    assert policy.done and policy.failure == "velocity stop"
    assert policy.accepted_grasp == pytest.approx(0.96)
