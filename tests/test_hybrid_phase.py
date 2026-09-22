"""TASK-046 hybrid phase controller: opaque toy models, no simulator or claims."""

import numpy as np
import pytest

from embodied_jepa.constraints import CandidateProjection
from embodied_jepa.contracts import (
    EE_DELTA_GRASP_V0,
    Capabilities,
    ContractError,
    ExecutionResult,
    Observation,
    StateSchema,
)
from embodied_jepa.hybrid_phase import HybridPhaseController, handoff_row
from embodied_jepa.trajectory_tracking import TrackingController, TrackingReference
from embodied_jepa.waypoint_planning import WaypointConfig

SCHEMA = StateSchema(("q",), ("rad",), "hybrid_test_v0")
LOWER, UPPER = (-1.0,) * 14, (1.0,) * 14


class Opaque:
    def __init__(self, value):
        self.value = value


class ToyModel:
    """Scalar state; action dimension 6 moves q by its value each step."""

    capabilities = Capabilities(EE_DELTA_GRASP_V0, SCHEMA, 16)

    def __init__(self):
        self.calls = []

    def encode(self, camera_images, robot_state):
        return Opaque(robot_state.values[:, 0].astype(np.float64))

    def encode_goal(self, goal):
        return Opaque(goal["state"][:, 0].astype(np.float64))

    def predict(self, z, actions):
        self.calls.append(actions.copy())
        return Opaque(z.value[:, None, None] + np.cumsum(actions[..., 6], axis=2))

    def distance(self, predicted, goal):
        return np.square(predicted.value - goal.value[:, None, None]).astype(np.float32)


def progress(robot_state, goal_rows):
    return np.square(robot_state.values - goal_rows).mean(-1).astype(np.float32)


def observation(q, timestamp):
    return Observation(
        {"onboard_rgb": np.zeros((1, 2, 2, 3), np.uint8)},
        np.array([[q]], np.float32),
        np.ones((1, 1), bool),
        np.array([timestamp], np.float64),
        SCHEMA,
    )


def projection(requested):
    return CandidateProjection(requested, np.ones(requested.shape[:2], bool))


def tracker(model=None, frames=(0, 2, 5, 6, 9, 12)):
    values = np.arange(len(frames), dtype=np.float32)[:, None] * 0.1
    config = WaypointConfig(
        horizon=4, candidates=8, iterations=2, seed=5, lower_bounds=LOWER, upper_bounds=UPPER
    )
    return TrackingController(
        model or ToyModel(),
        TrackingReference(values, frames, "train-demo/keyframes"),
        progress_distance=progress,
        config=config,
    )


def demo_actions(length=12):
    actions = np.zeros((length, 14), np.float32)
    actions[:, 6] = np.arange(length) / 100  # distinct, recognizable per frame
    actions[:, 13] = 2.0  # clipped to the upper bound by the replay bounds check
    return actions


def hybrid(row=2, **kwargs):
    return HybridPhaseController(
        tracker(**kwargs), demo_actions(), handoff_row=row, lower_bounds=LOWER, upper_bounds=UPPER
    )


def applied(decision, time):
    return ExecutionResult(decision.action, decision.action, "applied", time)


def test_handoff_row_is_close_row_minus_offset_clamped_at_zero():
    assert handoff_row(142, 0) == 142 and handoff_row(142, 16) == 126
    assert handoff_row(10, 16) == 0
    for close, offset in ((0, 0), (None, 0), (5, -1), (5, 1.0)):
        with pytest.raises(ContractError):
            handoff_row(close, offset)


def test_pre_handoff_commands_equal_the_unwrapped_tracker():
    plain, wrapped = tracker(), hybrid(row=3)
    clock = 0.0
    for q in (0.0, 0.05, 0.1):  # measured rows 0, 0/1, 1: all below the handoff row
        a = plain.step(observation(q, clock), projection)
        b = wrapped.step(observation(q, clock), projection)
        np.testing.assert_array_equal(a.action, b.action)
        assert b.trace["phase"] == "tracking" and b.trace["handoff_row"] == 3
        plain.acknowledge(applied(a, clock + 0.01))
        assert wrapped.acknowledge(applied(b, clock + 0.01))["phase"] == "tracking"
        clock += 0.05
    assert wrapped.summary()["handed_off"] is False and wrapped.tracker.steps == 3


def test_handoff_replays_recorded_actions_from_the_matched_frame_until_exhausted():
    model = ToyModel()
    controller = hybrid(row=2, model=model)
    first = controller.step(observation(0.0, 0.0), projection)
    controller.acknowledge(applied(first, 0.01))
    calls = len(model.calls)
    # Measured row 3 (frame 6) is past the handoff row 2: replay starts at action 6.
    decision = controller.step(observation(0.3, 0.05), projection)
    assert decision.trace["phase"] == "replay" and decision.trace["replay_frame"] == 6
    assert decision.trace["handoff_reference_index"] == 3 and decision.trace["handoff_frame"] == 6
    assert decision.trace["handoff_command"] == 1
    expected = demo_actions()[6].copy()
    expected[13] = 1.0
    np.testing.assert_array_equal(decision.action, expected)
    clock = 0.05
    frames = []
    while decision.action is not None:
        frames.append(decision.trace["replay_frame"])
        trace = controller.acknowledge(applied(decision, clock + 0.01))
        assert trace["status"] == "applied"
        clock += 0.05
        decision = controller.step(observation(9.0, clock), projection)
    assert frames == list(range(6, 12)) and decision.termination_reason == "demo_exhausted"
    assert len(model.calls) == calls  # no model call after the handoff
    summary = controller.summary()
    assert summary["handed_off"] and summary["replayed_commands"] == 6
    assert summary["replay_start_frame"] == 6 and summary["tracked_commands"] == 1
    assert controller.step(observation(9.0, clock + 1), projection).action is None


def test_replay_enforces_acknowledgement_freshness_and_feasibility():
    controller = hybrid(row=0)
    decision = controller.step(observation(0.0, 0.0), projection)
    assert decision.trace["phase"] == "replay" and decision.trace["replay_frame"] == 0
    with pytest.raises(ContractError):
        controller.step(observation(0.0, 0.1), projection)  # unacknowledged
    with pytest.raises(ContractError):
        controller.acknowledge(
            ExecutionResult(np.zeros(14, np.float32), None, "rejected", 0.01, reason="x")
        )
    controller.acknowledge(applied(decision, 0.01))
    with pytest.raises(ContractError):
        controller.step(observation(0.0, 0.0), projection)  # stale observation
    infeasible = hybrid(row=0)
    with pytest.raises(ContractError):
        infeasible.step(
            observation(0.0, 0.0),
            lambda r: CandidateProjection(r, np.zeros(r.shape[:2], bool)),
        )
    rejected = hybrid(row=0)
    decision = rejected.step(observation(0.0, 0.0), projection)
    rejected.acknowledge(ExecutionResult(decision.action, None, "rejected", 0.0, reason="guard"))
    assert rejected.step(observation(0.0, 0.1), projection).termination_reason == (
        "execution_rejected"
    )


def test_tracker_termination_before_handoff_is_propagated():
    controller = hybrid(row=4)
    controller.tracker.termination_reason = "reference_stall"
    decision = controller.step(observation(0.0, 0.0), projection)
    assert decision.action is None and decision.termination_reason == "reference_stall"
    assert controller.summary()["handed_off"] is False


def test_construction_rejects_bad_inputs():
    for row in (5, 6, -1, 1.0):  # the final row (5) and beyond cannot be a handoff
        with pytest.raises(ContractError):
            hybrid(row=row)
    with pytest.raises(ContractError):  # actions must match the reference demonstration
        HybridPhaseController(
            tracker(), demo_actions(11), handoff_row=1, lower_bounds=LOWER, upper_bounds=UPPER
        )
    with pytest.raises(ContractError):
        HybridPhaseController(
            object(), demo_actions(), handoff_row=1, lower_bounds=LOWER, upper_bounds=UPPER
        )
