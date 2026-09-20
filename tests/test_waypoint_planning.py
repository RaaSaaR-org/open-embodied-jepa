"""Cheap opaque-model tests; no simulator, training or manipulation-performance claims."""

from dataclasses import replace

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
from embodied_jepa.waypoint_planning import ImageWaypoint, WaypointConfig, WaypointController

SCHEMA = StateSchema(("q",), ("rad",), "waypoint_test_v0")


def images(value):
    return {"onboard_rgb": np.full((1, 2, 2, 3), value, np.uint8)}


def observation(value=0, timestamp=0):
    return Observation(
        images(value),
        np.zeros((1, 1), np.float32),
        np.ones((1, 1), bool),
        np.array([timestamp], np.float64),
        SCHEMA,
    )


def progress(current, goal):
    assert set(current) == set(goal) == {"onboard_rgb"}
    return np.square(current["onboard_rgb"].astype(np.float32) - goal["onboard_rgb"]).mean(
        (1, 2, 3)
    )


def projection(requested):
    return CandidateProjection(requested, np.ones(requested.shape[:2], bool))


class Opaque:
    def __init__(self, value):
        self.value = value


class ToyModel:
    capabilities = Capabilities(EE_DELTA_GRASP_V0, SCHEMA, 16)

    def __init__(self, direction=1):
        self.direction = direction
        self.calls = []

    def encode(self, camera_images, robot_state):
        assert robot_state.schema == SCHEMA
        return Opaque(camera_images["onboard_rgb"].mean((1, 2, 3)))

    def encode_goal(self, goal):
        # The controller can only pass an image mapping: no goal robot state.
        return Opaque(goal["onboard_rgb"].mean((1, 2, 3)))

    def predict(self, z, actions):
        self.calls.append(actions.copy())
        return Opaque(z.value[:, None, None] + self.direction * np.cumsum(actions[..., 0], axis=2))

    def distance(self, predicted, goal):
        return np.square(predicted.value - goal.value[:, None, None]).astype(np.float32)


def waypoint(goal=1, dwell=1, proposals=None):
    return ImageWaypoint(
        images(goal),
        threshold=0,
        dwell_observations=dwell,
        source_id="train-episode/frame-10",
        action_proposals=proposals,
    )


def acknowledge(controller, decision, time=0.05):
    return controller.acknowledge(
        ExecutionResult(decision.action, decision.action, "applied", time)
    )


def two_proposals():
    plans = np.zeros((2, 1, 14), np.float32)
    plans[:, :, 12:] = -1
    plans[0, :, 0], plans[1, :, 0] = 1, -1
    return plans


def test_learned_dynamics_choose_opposite_actions_from_identical_proposals():
    selected = []
    for direction in (1, -1):
        model = ToyModel(direction)
        controller = WaypointController(
            model,
            [waypoint(proposals=two_proposals())],
            progress_distance=progress,
            config=WaypointConfig(horizon=1, candidates=6, iterations=2, seed=7),
        )
        decision = controller.step(observation(), projection)
        selected.append(decision.action[0])
        assert decision.trace["selected_cost"] == 0
        assert decision.trace["selected_origin"] in (
            "demonstration_proposal",
            "warm_or_search_best",
        )
        assert any(r["cost_spread"] > 0 for r in decision.trace["rounds"])
        assert len(model.calls) == 2
        assert decision.trace["candidate_evaluations"] == 12
    assert selected == [1, -1]


def test_projection_precedes_prediction_and_invalid_best_proposal_cannot_win():
    model = ToyModel()
    controller = WaypointController(
        model,
        [waypoint(proposals=two_proposals())],
        progress_distance=progress,
        config=WaypointConfig(horizon=1, candidates=5, iterations=1),
    )

    def projected(requested):
        values = requested.copy()
        values[..., 0] = 0.2
        feasible = np.ones(requested.shape[:2], bool)
        feasible[:, 2] = False
        values[:, 2, :, 0] = 1  # Better cost, but this candidate is infeasible.
        return CandidateProjection(values, feasible)

    decision = controller.step(observation(), projected)
    assert decision.action[0] == pytest.approx(0.2)
    assert decision.trace["rounds"][0]["candidate_costs"][2] is None
    assert model.calls[0][0, 0, 0, 0] == pytest.approx(0.2)
    assert decision.trace["selected_cost"] == pytest.approx(0.64)


def test_progress_requires_observed_image_match_and_consecutive_dwell():
    controller = WaypointController(
        ToyModel(),
        [waypoint(2, dwell=2), waypoint(3)],
        progress_distance=progress,
        config=WaypointConfig(horizon=1, candidates=3, iterations=1),
    )
    # Time alone and action execution cannot advance a waypoint.
    for step, value in enumerate((0, 2, 0, 2)):
        decision = controller.step(observation(value, step), projection)
        assert controller.goal_index == 0
        acknowledge(controller, decision, step + 0.05)
    decision = controller.step(observation(2, 4), projection)
    assert controller.goal_index == 1 and decision.trace["waypoint_advanced"]
    acknowledge(controller, decision, 4.05)
    done = controller.step(observation(3, 5), projection)
    assert done.action is None and done.termination_reason == "waypoints_complete"


def test_fresh_observation_acknowledgement_and_budget_are_required():
    controller = WaypointController(
        ToyModel(),
        [waypoint()],
        progress_distance=progress,
        config=WaypointConfig(horizon=1, candidates=3, iterations=1, max_steps=2),
    )
    first = controller.step(observation(), projection)
    with pytest.raises(ContractError, match="acknowledge"):
        controller.step(observation(timestamp=1), projection)
    wrong = first.action.copy()
    wrong[0] = 0.7
    with pytest.raises(ContractError, match="match"):
        controller.acknowledge(ExecutionResult(wrong, wrong, "applied", 0.05))
    acknowledge(controller, first)
    with pytest.raises(ContractError, match="fresh"):
        controller.step(observation(), projection)
    second = controller.step(observation(timestamp=1), projection)
    acknowledge(controller, second, 1.05)
    done = controller.step(observation(timestamp=2), projection)
    assert done.action is None and done.termination_reason == "step_limit"


def test_no_feasible_candidate_fails_without_dynamics_or_hidden_action():
    model = ToyModel()
    controller = WaypointController(
        model,
        [waypoint()],
        progress_distance=progress,
        config=WaypointConfig(horizon=1, candidates=3, iterations=1),
    )
    with pytest.raises(ContractError, match="no feasible"):
        controller.step(
            observation(), lambda a: CandidateProjection(a, np.zeros(a.shape[:2], bool))
        )
    assert model.calls == [] and controller.pending is None


def test_execution_rejection_is_terminal_and_applied_grasp_drives_next_hold():
    controller = WaypointController(
        ToyModel(),
        [waypoint()],
        progress_distance=progress,
        config=WaypointConfig(horizon=1, candidates=3, iterations=1),
    )
    first = controller.step(observation(), projection)
    applied = first.action.copy()
    applied[12:] = -0.5
    trace = controller.acknowledge(ExecutionResult(first.action, applied, "clipped", 0.05, "rate"))
    assert trace["applied_action"][12:] == [-0.5, -0.5]
    second = controller.step(observation(timestamp=1), projection)
    assert np.all(controller.model.calls[-1][0, 0, :, 12:] == -0.5)
    controller.acknowledge(ExecutionResult(second.action, None, "stopped", 1.0, "velocity"))
    assert (
        controller.step(observation(timestamp=2), projection).termination_reason
        == "execution_rejected"
    )


@pytest.mark.parametrize("ablation", ["dynamics_shuffle", "persistence"])
def test_ablation_is_seeded_reproducible_and_persistence_does_not_call_dynamics(ablation):
    decisions = []
    for _ in range(2):
        model = ToyModel()
        controller = WaypointController(
            model,
            [waypoint(proposals=two_proposals())],
            progress_distance=progress,
            config=WaypointConfig(
                horizon=1, candidates=6, iterations=2, ablation=ablation, seed=41
            ),
        )
        decisions.append(controller.step(observation(), projection))
        assert len(model.calls) == (0 if ablation == "persistence" else 2)
    np.testing.assert_array_equal(decisions[0].action, decisions[1].action)
    assert decisions[0].trace["selected_cost"] == decisions[1].trace["selected_cost"]
    if ablation == "persistence":
        assert all(r["cost_spread"] == 0 for r in decisions[0].trace["rounds"])


def test_nonfinite_metrics_and_nontraining_sources_rejected():
    with pytest.raises(ContractError, match="training-source"):
        replace(waypoint(), source_split="test")
    with pytest.raises(ContractError, match="threshold"):
        replace(waypoint(), threshold=np.nan)
    controller = WaypointController(
        ToyModel(),
        [waypoint()],
        progress_distance=lambda *_: np.array([np.nan]),
        config=WaypointConfig(horizon=1),
    )
    with pytest.raises(ContractError, match="finite"):
        controller.step(observation(), projection)

    class Broken(ToyModel):
        def distance(self, predicted, goal):
            return np.full_like(predicted.value, np.nan, dtype=np.float32)

    controller = WaypointController(
        Broken(),
        [waypoint()],
        progress_distance=progress,
        config=WaypointConfig(horizon=1, candidates=3, iterations=1),
    )
    with pytest.raises(ContractError, match="finite"):
        controller.step(observation(), projection)


def test_proposal_exploration_resets_each_step_instead_of_collapsing_to_warm_start():
    controller = WaypointController(
        ToyModel(),
        [waypoint()],
        progress_distance=progress,
        config=WaypointConfig(
            horizon=1, candidates=4, iterations=2, proposal_std=0.2, minimum_std=0.05
        ),
    )
    for step in range(3):
        decision = controller.step(observation(timestamp=step), projection)
        assert [r["proposal_std"] for r in decision.trace["rounds"]] == [0.2, 0.1]
        acknowledge(controller, decision, step + 0.05)


def test_observation_must_not_precede_last_acknowledged_execution():
    controller = WaypointController(
        ToyModel(),
        [waypoint()],
        progress_distance=progress,
        config=WaypointConfig(horizon=1, candidates=3, iterations=1),
    )
    first = controller.step(observation(timestamp=0), projection)
    acknowledge(controller, first, 0.05)
    with pytest.raises(ContractError, match="fresh"):
        controller.step(observation(timestamp=0.025), projection)
    assert controller.step(observation(timestamp=0.05), projection).action is not None
