"""TASK-045 trajectory-tracking controller: opaque toy models, no simulator or claims."""

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
from embodied_jepa.trajectory_tracking import (
    TrackingController,
    TrackingReference,
    TrackingRule,
    keyframe_indices,
)
from embodied_jepa.waypoint_planning import (
    StateWaypoint,
    WaypointConfig,
    WaypointController,
)

SCHEMA = StateSchema(("q",), ("rad",), "tracking_test_v0")


class Opaque:
    def __init__(self, value):
        self.value = value


class ToyModel:
    """Scalar state; action dimension 6 moves q by its value each step."""

    capabilities = Capabilities(EE_DELTA_GRASP_V0, SCHEMA, 16)

    def __init__(self):
        self.calls, self.goals = [], []

    def encode(self, camera_images, robot_state):
        return Opaque(robot_state.values[:, 0].astype(np.float64))

    def encode_goal(self, goal):
        assert set(goal) == {"state"} and goal["state"].shape == (1, 1)
        self.goals.append(float(goal["state"][0, 0]))
        return Opaque(goal["state"][:, 0].astype(np.float64))

    def predict(self, z, actions):
        self.calls.append(actions.copy())
        return Opaque(z.value[:, None, None] + np.cumsum(actions[..., 6], axis=2))

    def distance(self, predicted, goal):
        return np.square(predicted.value - goal.value[:, None, None]).astype(np.float32)


def progress(robot_state, goal_rows):
    assert robot_state.schema == SCHEMA and robot_state.values.shape[0] == len(goal_rows)
    return np.square(robot_state.values - goal_rows).mean(-1).astype(np.float32)


def observation(q=0.0, timestamp=0.0):
    return Observation(
        {"onboard_rgb": np.zeros((1, 2, 2, 3), np.uint8)},
        np.array([[q]], np.float32),
        np.ones((1, 1), bool),
        np.array([timestamp], np.float64),
        SCHEMA,
    )


def projection(requested):
    return CandidateProjection(requested, np.ones(requested.shape[:2], bool))


def reference(values, frames=None):
    values = np.asarray(values, np.float32)[:, None]
    return TrackingReference(values, frames or tuple(range(len(values))), "train-demo/keyframes")


def config(**overrides):
    options = dict(
        horizon=4,
        candidates=8,
        iterations=2,
        seed=5,
        lower_bounds=(-1.0,) * 14,
        upper_bounds=(1.0,) * 14,
    )
    return WaypointConfig(**(options | overrides))


def acknowledge(controller, decision, time):
    return controller.acknowledge(
        ExecutionResult(decision.action, decision.action, "applied", time)
    )


def test_keyframes_drop_only_dead_time_and_keep_both_ends():
    q = np.array([0.0, 0.1, 0.1, 0.1, 0.2, 0.3, 0.3], np.float32)
    kept = keyframe_indices(len(q), lambda i, j: float((q[i] - q[j]) ** 2), 1e-6)
    assert kept == [0, 1, 4, 5, 6]  # final frame kept even though it is stationary
    assert keyframe_indices(3, lambda i, j: 0.0, 0.0) == [0, 2]
    with pytest.raises(ContractError):
        keyframe_indices(1, lambda i, j: 0.0, 0.0)
    with pytest.raises(ContractError):
        keyframe_indices(3, lambda i, j: float("nan"), 0.0)


def test_reference_requires_train_provenance_and_increasing_frames():
    with pytest.raises(ContractError):
        TrackingReference(np.zeros((3, 1), np.float32), (0, 1, 2), "x", source_split="val")
    with pytest.raises(ContractError):
        TrackingReference(np.zeros((3, 1), np.float32), (0, 2, 2), "x")
    with pytest.raises(ContractError):
        TrackingReference(np.zeros((3, 1), np.float32), (1, 2, 3), "x")
    with pytest.raises(ContractError):
        TrackingReference(np.zeros((1, 1), np.float32), (0,), "x")
    ref = reference([0.0, 1.0, 2.0])
    assert not ref.states.flags.writeable and len(ref) == 3


def test_cost_is_horizon_mean_against_advancing_reference_rows_clamped_at_end():
    model = ToyModel()
    ref = reference(np.arange(6) * 0.1)
    controller = TrackingController(model, ref, progress_distance=progress, config=config())
    decision = controller.step(observation(0.0), projection)
    assert decision.trace["target_reference_indices"] == [1, 2, 3, 4]
    # Recompute the selected candidate's cost by hand from the recorded prediction.
    actions = model.calls[-1][0]
    predicted = np.cumsum(actions[..., 6], axis=1)
    manual = np.square(predicted - np.array([0.1, 0.2, 0.3, 0.4])).mean(1)
    costs = decision.trace["rounds"][-1]["candidate_costs"]
    np.testing.assert_allclose(costs, manual, rtol=1e-5)
    controller.index = 4
    assert controller.targets() == [5, 5, 5, 5]


def test_measured_progress_is_monotone_windowed_and_prefers_later_ties():
    model = ToyModel()
    ref = reference([0.0, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    controller = TrackingController(
        model, ref, progress_distance=progress, config=config(), rule=TrackingRule(window=3)
    )
    decision = controller.step(observation(0.1, 0.0), projection)
    assert controller.index == 3 and decision.trace["reference_advance"] == 3  # tie -> later
    acknowledge(controller, decision, 0.05)
    decision = controller.step(observation(0.6, 0.1), projection)
    assert controller.index == 6  # bounded by the forward window, not the nearest row (8)
    acknowledge(controller, decision, 0.15)
    decision = controller.step(observation(0.0, 0.2), projection)
    assert controller.index == 6 and decision.trace["reference_advance"] == 0  # never decreases


def test_stall_is_a_recorded_failure_and_completion_terminates():
    model = ToyModel()
    ref = reference(np.arange(40) * 0.1)
    rule = TrackingRule(window=16, stall_commands=3, stall_min_advance=2)
    controller = TrackingController(
        model, ref, progress_distance=progress, config=config(), rule=rule
    )
    clock = 0.0
    for q in (0.0, 0.1, 0.1):  # index 0 -> 1 -> 1
        decision = controller.step(observation(q, clock), projection)
        assert decision.action is not None
        acknowledge(controller, decision, clock + 0.01)
        clock += 0.05
    decision = controller.step(observation(0.1, clock), projection)
    assert decision.action is None and decision.termination_reason == "reference_stall"
    assert decision.trace["reference_index"] == 1
    assert controller.step(observation(3.9, clock + 1), projection).action is None
    done = TrackingController(ToyModel(), ref, progress_distance=progress, config=config())
    done.index = 30
    finished = done.step(observation(3.9, 0.0), projection)
    assert finished.termination_reason == "reference_complete"
    assert finished.trace["reference_index"] == 39


@pytest.mark.parametrize("ablation", ["dynamics_shuffle", "persistence"])
def test_ablations_are_seeded_and_persistence_never_predicts(ablation):
    ref = reference(np.arange(10) * 0.1)
    runs = []
    for _ in range(2):
        model = ToyModel()
        controller = TrackingController(
            model, ref, progress_distance=progress, config=config(ablation=ablation)
        )
        decision = controller.step(observation(0.0), projection)
        runs.append((decision.action.copy(), len(model.calls)))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    assert (runs[0][1] == 0) == (ablation == "persistence")


def test_candidate_sampling_mirrors_the_waypoint_cem_with_the_same_seed():
    """Same sampling, labels and projection as WaypointController; only the cost differs."""
    tracking_model, waypoint_model = ToyModel(), ToyModel()
    tracker = TrackingController(
        tracking_model, reference([0.0, 0.5, 1.0]), progress_distance=progress, config=config()
    )
    waypoint = WaypointController(
        waypoint_model,
        [StateWaypoint(np.array([[1.0]], np.float32), 0.0, 1, "train-demo/frame-1")],
        progress_distance=lambda state, goal: progress(state, goal),
        config=config(),
    )
    tracker.step(observation(0.0), projection)
    waypoint.step(observation(0.0), projection)
    np.testing.assert_array_equal(tracking_model.calls[0], waypoint_model.calls[0])


def test_commitment_other_than_one_and_bad_rules_are_rejected():
    ref = reference([0.0, 1.0])
    with pytest.raises(ContractError):
        TrackingController(
            ToyModel(), ref, progress_distance=progress, config=config(commitment_steps=2)
        )
    with pytest.raises(ContractError):
        TrackingController(ToyModel(), "ref", progress_distance=progress, config=config())
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ContractError):
            TrackingRule(window=bad)
