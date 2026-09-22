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
from embodied_jepa.waypoint_planning import (
    ImageWaypoint,
    StateWaypoint,
    WaypointConfig,
    WaypointController,
)

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


def chunk_controller(*, waypoints=None, mode="learned"):
    return WaypointController(
        ToyModel(),
        waypoints or [waypoint(3)],
        progress_distance=progress,
        config=WaypointConfig(
            horizon=16, candidates=6, iterations=2, seed=17, commitment_steps=4, ablation=mode
        ),
    )


@pytest.mark.parametrize("mode", ["learned", "persistence", "dynamics_shuffle"])
def test_commit_four_preserves_full_warm_and_draws_no_cached_rng(mode):
    import copy

    controller = chunk_controller(mode=mode)
    root = controller.step(observation(), projection)
    original = controller.pending[1].copy()
    rng = copy.deepcopy(controller.rng.bit_generator.state)
    calls = len(controller.model.calls)
    acknowledge(controller, root)
    for offset in range(1, 4):
        shapes = []

        def fresh(actions, shapes=shapes):
            shapes.append(actions.shape)
            return projection(actions)

        decision = controller.step(observation(timestamp=offset * 0.05), fresh)
        assert shapes == [(1, 1, 1, 14)]
        assert controller.rng.bit_generator.state == rng
        assert len(controller.model.calls) == calls
        np.testing.assert_array_equal(decision.action, original[offset])
        assert controller.pending[1].shape == (16, 14)
        assert decision.trace["decision_kind"] == "commitment"
        assert decision.trace["plan_step"] == 0
        assert decision.trace["commitment_offset"] == offset
        assert decision.trace["selected_cost"] == root.trace["selected_cost"]
        assert decision.trace["candidate_evaluations"] == 0
        assert decision.trace["selected_round"] is None
        trace = acknowledge(controller, decision, (offset + 1) * 0.05)
        assert trace["commitment_remaining_after_ack"] == 3 - offset
        if offset == 3:
            assert trace["commitment_clear_reason"] == "commitment_completed"
        assert controller.warm.shape == (16, 14)
    assert controller._commitment is None
    next_decision = controller.step(observation(timestamp=0.2), projection)
    assert next_decision.trace["decision_kind"] == "search"
    assert next_decision.trace["plan_step"] == 4
    assert controller.pending[1].shape == (16, 14)


@pytest.mark.parametrize("change", ["altered", "infeasible"])
def test_cached_projection_fallback_has_one_search_and_one_dwell(change):
    controller = chunk_controller(waypoints=[waypoint(0, dwell=20)])
    root = controller.step(observation(), projection)
    acknowledge(controller, root)
    shapes = []

    def fresh(actions):
        shapes.append(actions.shape)
        if actions.shape[1:3] == (1, 1):
            result = actions.copy()
            if change == "altered":
                result[..., 0] = 0.9 if result[..., 0] != 0.9 else 0.8
            return CandidateProjection(result, np.array([[change != "infeasible"]]))
        return projection(actions)

    decision = controller.step(observation(timestamp=0.05), fresh)
    assert controller.dwell == 2
    assert shapes == [(1, 1, 1, 14), (1, 6, 16, 14), (1, 6, 16, 14)]
    assert decision.trace["decision_kind"] == "search"
    assert decision.trace["plan_step"] == 1
    assert decision.trace["cache_validation"]["abort_reason"] == (
        "changed_cached_command" if change == "altered" else "infeasible_cached_command"
    )


def test_goal_advance_discards_cache_before_projection():
    controller = chunk_controller(waypoints=[waypoint(0, dwell=2), waypoint(3)])
    root = controller.step(observation(), projection)
    acknowledge(controller, root)
    shapes = []

    def fresh(actions):
        shapes.append(actions.shape)
        return projection(actions)

    decision = controller.step(observation(timestamp=0.05), fresh)
    assert decision.trace["waypoint_advanced"]
    assert decision.trace["goal_index"] == 1
    assert decision.trace["cache_validation"]["abort_reason"] == "waypoint_advanced"
    assert decision.trace["cache_validation"]["plan_step"] == 0
    assert decision.trace["decision_kind"] == "search"
    assert all(shape == (1, 6, 16, 14) for shape in shapes)


def test_changed_applied_action_aborts_commitment_and_rejection_is_terminal():
    controller = chunk_controller()
    decision = controller.step(observation(), projection)
    applied = decision.action.copy()
    applied[12:] = 0.5
    trace = controller.acknowledge(
        ExecutionResult(decision.action, applied, "clipped", 0.05, "transport_clip")
    )
    assert trace["commitment_abort_reason"] == "applied_action_changed"
    assert controller._commitment is None
    np.testing.assert_array_equal(controller.last_grasps, [0.5, 0.5])
    decision = controller.step(observation(timestamp=0.05), projection)
    controller.acknowledge(ExecutionResult(decision.action, None, "rejected", 0.05, "guard"))
    assert controller._commitment is None
    assert (
        controller.step(observation(timestamp=0.1), projection).termination_reason
        == "execution_rejected"
    )


@pytest.mark.parametrize("value", [0, -1, True, 1.5, 17])
def test_invalid_commitment(value):
    with pytest.raises(ContractError):
        WaypointConfig(horizon=16, commitment_steps=value)


def test_default_one_matches_historical_actions_rng_and_warm():
    import hashlib

    controller = WaypointController(
        ToyModel(),
        [waypoint(3)],
        progress_distance=progress,
        config=WaypointConfig(horizon=4, candidates=6, iterations=2, seed=17),
    )
    first_coordinates, costs = [], []
    for step in range(5):
        decision = controller.step(observation(timestamp=step * 0.05), projection)
        first_coordinates.append(float(decision.action[0]))
        costs.append(decision.trace["selected_cost"])
        acknowledge(controller, decision, (step + 1) * 0.05)
    assert first_coordinates == [
        0.2443234622478485,
        0.40660780668258667,
        0.11599817872047424,
        0.7389726042747498,
        0.39713600277900696,
    ]
    assert costs == [
        7.186814308166504,
        4.13236141204834,
        3.4269065856933594,
        1.613690972328186,
        1.8827123641967773,
    ]
    assert (
        controller.rng.bit_generator.state["state"]["state"]
        == 275847063342611265020023383026917730119
    )
    assert (
        hashlib.sha256(controller.warm.tobytes()).hexdigest()
        == "fd3f3493847b669c701e82b999047d113975b3322f3519a337bb2bd3aecb4cd3"
    )


# TASK-043 optional demonstration-state goals; the default image path stays unchanged.


def state_observation(q=0.0, timestamp=0):
    return Observation(
        images(0),
        np.array([[q]], np.float32),
        np.ones((1, 1), bool),
        np.array([timestamp], np.float64),
        SCHEMA,
    )


def state_progress(current, goal):
    assert current.schema == SCHEMA and goal.shape == (1, 1)
    return np.square(current.values - goal).mean(-1).astype(np.float32)


class StateToyModel(ToyModel):
    def encode(self, camera_images, robot_state):
        return Opaque(robot_state.values[:, 0].astype(np.float32))

    def encode_goal(self, goal):
        assert set(goal) <= {"state", "images"} and goal["state"].shape == (1, 1)
        return Opaque(goal["state"][:, 0])


def state_waypoint(goal=1.0, threshold=0.0, **kwargs):
    return StateWaypoint(
        np.array([[goal]], np.float32), threshold, 1, f"train-demo/frame-{goal}", **kwargs
    )


def test_state_waypoints_use_measured_state_progress_and_state_goal_input():
    model = StateToyModel()
    controller = WaypointController(
        model,
        [state_waypoint(0.0), state_waypoint(2.0, images=images(9))],
        progress_distance=state_progress,
        config=WaypointConfig(horizon=2, candidates=6, iterations=1, seed=3),
    )
    decision = controller.step(state_observation(0.0), projection)
    assert controller.goal_index == 1 and decision.trace["waypoint_advanced"]
    assert decision.trace["observed_distance"] == 4.0
    assert "goal_state_sha256" in decision.trace and "goal_image_sha256" not in decision.trace
    assert decision.trace["goal_commands"] == 0
    acknowledge(controller, decision)
    assert controller.goal_commands == 1
    assert np.asarray(controller._goal_latents[1].value).tolist() == [2.0]


def test_goal_stall_terminates_as_failure_without_skipping_and_resets_on_advance():
    controller = WaypointController(
        StateToyModel(),
        [state_waypoint(0.5, threshold=0.01), state_waypoint(9.0)],
        progress_distance=state_progress,
        config=WaypointConfig(horizon=1, candidates=4, iterations=1, goal_stall_limit=2),
    )
    decision = controller.step(state_observation(0.0, 0.0), projection)
    acknowledge(controller, decision, 0.05)
    decision = controller.step(state_observation(0.5, 0.1), projection)  # reaches goal 0
    assert controller.goal_index == 1 and controller.goal_commands == 0
    for step in range(2):
        acknowledge(controller, decision, 0.15 + step * 0.1)
        decision = controller.step(state_observation(0.5, 0.2 + step * 0.1), projection)
    assert decision.action is None and decision.termination_reason == "goal_stall"
    assert decision.trace["goal_index"] == 1 and decision.trace["goal_commands"] == 2
    assert controller.goal_index == 1  # never skipped
    later = controller.step(state_observation(9.0, 1.0), projection)
    assert later.termination_reason == "goal_stall" and controller.goal_index == 1


def test_state_waypoint_and_stall_validation():
    with pytest.raises(ContractError):
        WaypointController(
            StateToyModel(), [waypoint(), state_waypoint()], progress_distance=progress
        )
    for bad in (0, -2, True, 1.5):
        with pytest.raises(ContractError):
            WaypointConfig(goal_stall_limit=bad)
    with pytest.raises(ContractError):
        state_waypoint(source_split="val")
    with pytest.raises(ContractError):
        StateWaypoint(np.array([[np.nan]], np.float32), 0.0, 1, "train/frame-1")
    with pytest.raises(ContractError):
        StateWaypoint(np.array([[1]], np.int64), 0.0, 1, "train/frame-1")
    with pytest.raises(ContractError):
        StateWaypoint(np.zeros((2, 1), np.float32), 0.0, 1, "train/frame-1")
    goal = state_waypoint(images=images(3))
    assert not goal.state.flags.writeable and set(goal.goal_input) == {"state", "images"}


@pytest.mark.parametrize("limit", [None, 1000])
def test_default_image_path_parity_with_or_without_unused_stall_limit(limit):
    baseline = WaypointController(
        ToyModel(),
        [waypoint(3)],
        progress_distance=progress,
        config=WaypointConfig(horizon=4, candidates=6, iterations=2, seed=17),
    )
    variant = WaypointController(
        ToyModel(),
        [waypoint(3)],
        progress_distance=progress,
        config=WaypointConfig(
            horizon=4, candidates=6, iterations=2, seed=17, goal_stall_limit=limit
        ),
    )
    for step in range(5):
        first = baseline.step(observation(timestamp=step * 0.05), projection)
        second = variant.step(observation(timestamp=step * 0.05), projection)
        np.testing.assert_array_equal(first.action, second.action)
        assert first.trace.keys() == second.trace.keys()
        assert "goal_commands" not in first.trace and "goal_image_sha256" in first.trace
        acknowledge(baseline, first, (step + 1) * 0.05)
        acknowledge(variant, second, (step + 1) * 0.05)
    assert baseline.rng.bit_generator.state == variant.rng.bit_generator.state
    np.testing.assert_array_equal(baseline.warm, variant.warm)
