"""TASK-044 privileged-rollout ceiling: exact-parity and isolation checks.

These tests run real MuJoCo physics (headless, fixture RGB). They check that the
NON-LEARNED ceiling's rollouts reproduce live execution bit for bit, never touch
the live simulator, and that the ceiling cannot be selected as a learned model.
"""

import numpy as np
import pytest

from embodied_jepa.contracts import Capabilities, ContractError, RobotState
from embodied_jepa.registry import MODELS


def test_ceiling_requires_explicit_acknowledgement_and_is_not_a_registered_model():
    from embodied_jepa import config  # noqa: F401  (populates the registries)
    from embodied_jepa.privileged_rollout import PrivilegedRolloutModel

    with pytest.raises(ContractError, match="acknowledge_privileged_ceiling"):
        PrivilegedRolloutModel(object(), object())
    with pytest.raises(ContractError, match="acknowledge_privileged_ceiling"):
        PrivilegedRolloutModel(object(), object(), acknowledge_privileged_ceiling="yes")
    for name in ("privileged_rollout", "privileged_mujoco_rollout_v1"):
        with pytest.raises(ValueError, match="unknown world model"):
            MODELS.require(name)
    assert all("privileged" not in name for name in config.BACKEND_KEYS)


class FirstFieldsMetric:
    """Fixture metric over the first two right-arm positions; not a trained model."""

    def __init__(self, schema):
        self.schema = schema
        self.index = [schema.names.index("right_shoulder_pitch_joint.position")] + [
            schema.names.index("right_elbow_joint.position")
        ]
        self.calls = 0

    @property
    def capabilities(self):
        from embodied_jepa.contracts import EE_DELTA_GRASP_V0

        return Capabilities(EE_DELTA_GRASP_V0, self.schema, 16, supported_devices=("cpu",))

    def observed_distance(self, robot_state, goal):
        assert isinstance(robot_state, RobotState)
        self.calls += 1
        values = robot_state.values[:, self.index]
        return np.square(values - np.asarray(goal)[:, :2]).mean(-1).astype(np.float32)


@pytest.fixture
def physics():
    pytest.importorskip("mujoco")
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.privileged_rollout import PrivilegedRolloutModel
    from embodied_jepa.simulation import ROOT, MuJoCoSimulation

    if not (ROOT / "third_party/unitree_mujoco/unitree_robots/g1/g1_29dof_with_hand.xml").exists():
        pytest.skip("pinned robot assets have not been fetched")
    sim = MuJoCoSimulation(render=False)
    sim.render = lambda camera="onboard_rgb": np.zeros((8, 8, 3), np.uint8)
    live = G1Embodiment(sim)
    live.reset(object_xy=[0.345, -0.182], plate_xy=[0.488, -0.093])
    metric = FirstFieldsMetric(live.state_schema)
    model = PrivilegedRolloutModel(metric, live, acknowledge_privileged_ceiling=True)
    yield live, model, metric
    model.close()
    live.close()


def candidates(live, count=4, horizon=3, seed=0):
    rng = np.random.default_rng(seed)
    requested = np.zeros((1, count, horizon, 14), np.float32)
    requested[..., 6:12] = rng.uniform(-0.5, 0.5, (count, horizon, 6))
    requested[..., 12] = -1.0
    requested[..., 13] = rng.uniform(-1, 1, (count, horizon))
    projection = live.project_candidates(requested)
    assert projection.feasible.all()
    return projection.actions


def live_state(live):
    sim = live.sim
    return (
        sim.data.qpos.copy(),
        sim.data.qvel.copy(),
        float(sim.data.time),
        sim.targets.copy(),
    )


@pytest.mark.parametrize("candidate", [0, 2])
def test_rollout_first_step_matches_live_execution_exactly(physics, candidate):
    live, model, _ = physics
    for _ in range(3):  # Start from a moving, non-reset state.
        live.observe()
        live.execute(candidates(live, 1, 1, seed=9)[0, 0, 0])
    observation = live.observe()
    before = live_state(live)
    snapshot = model.encode(observation.images, observation.state)
    actions = candidates(live)
    prediction = model.predict(snapshot, actions)
    after = live_state(live)
    for old, new in zip(before, after, strict=True):
        np.testing.assert_array_equal(old, new)  # The live simulator is untouched.
    assert prediction.valid.all() and prediction.states.shape == (
        1,
        4,
        3,
        live.state_schema.dimension,
    )
    # Execute the same projected command on the live robot: identical next state.
    result = live.execute(actions[0, candidate, 0])
    assert result.applied_action is not None
    measured = live.observe()
    np.testing.assert_array_equal(prediction.states[0, candidate, 0], measured.robot_state[0])
    assert prediction.timestamps[0, candidate, 0] == measured.timestamps[0]
    # The next snapshot records the runtime conformance check.
    model.predict(model.encode(measured.images, measured.state), candidates(live, seed=1))
    diagnostic = model.pop_diagnostics()[-1]
    assert diagnostic["previous_search_first_step_exact_match"] is True
    assert diagnostic["rejected_candidates"] == 0


def test_whole_sequence_matches_step_by_step_live_execution(physics):
    live, model, _ = physics
    observation = live.observe()
    actions = candidates(live, 2, 4, seed=3)
    prediction = model.predict(model.encode(observation.images, observation.state), actions)
    for step in range(4):
        live.observe()
        projected = live.project_candidates(actions[:, 1:2, step : step + 1].copy())
        live.execute(projected.actions[0, 0, 0])
        np.testing.assert_array_equal(prediction.states[0, 1, step], live.observe().robot_state[0])


def test_repeated_rollouts_are_deterministic_and_parity_is_recorded(physics):
    live, model, _ = physics
    observation = live.observe()
    actions = candidates(live, seed=4)
    first = model.predict(model.encode(observation.images, observation.state), actions)
    second = model.predict(model.encode(observation.images, observation.state), actions)
    np.testing.assert_array_equal(first.states, second.states)
    live.execute(actions[0, 3, 0])
    after = live.observe()
    model.predict(model.encode(after.images, after.state), candidates(live, seed=5))
    diagnostic = model.pop_diagnostics()[-1]
    assert diagnostic["previous_search_first_step_exact_match"] is True
    assert diagnostic["previous_search_first_step_min_max_abs_error"] == 0.0


def test_cost_is_the_metric_distance_and_rejections_rank_last(physics, monkeypatch):
    live, model, metric = physics
    observation = live.observe()
    snapshot = model.encode(observation.images, observation.state)
    actions = candidates(live, 3, 3, seed=6)
    original = model.twin.execute
    calls = {"n": 0}

    def reject_candidate_one_step_two(command):
        calls["n"] += 1
        if calls["n"] == 5:  # candidate 1, step 1 (0-based)
            model.twin.sim.stop("fixture rejection")
            from embodied_jepa.contracts import ExecutionResult

            return ExecutionResult(command, None, "stopped", 0.0, "fixture rejection")
        return original(command)

    monkeypatch.setattr(model.twin, "execute", reject_candidate_one_step_two)
    prediction = model.predict(snapshot, actions)
    assert prediction.valid[0, 1].tolist() == [True, False, False]
    assert prediction.valid[0, [0, 2]].all()
    goal = model.encode_goal({"state": np.array([[0.1, 0.2] + [0.0] * 12], np.float64)})
    costs = model.distance(prediction, goal)
    assert costs.dtype == np.float32 and costs.shape == (1, 3, 3)
    reached = FirstFieldsMetric(live.state_schema).observed_distance(
        RobotState(
            prediction.states[0, 0],
            np.ones((3, live.state_schema.dimension), bool),
            prediction.timestamps[0, 0],
            live.state_schema,
        ),
        np.repeat(goal.state, 3, 0),
    )
    np.testing.assert_allclose(costs[0, 0], reached)
    from embodied_jepa.privileged_rollout import REJECTED_ROLLOUT_COST

    last = costs[0, 1, 0]
    np.testing.assert_allclose(costs[0, 1, 1:], REJECTED_ROLLOUT_COST + last)
    assert costs[0, 1, -1] > costs[0, [0, 2], -1].max()
    diagnostic = model.pop_diagnostics()[-1]
    assert diagnostic["rejected_candidates"] == 1
    assert diagnostic["rejection_reasons"] == {"fixture rejection": 1}
    assert metric.calls >= 1


def test_snapshot_must_match_the_live_observation_and_owner(physics):
    live, model, _ = physics
    observation = live.observe()
    live.execute(candidates(live, 1, 1)[0, 0, 0])
    with pytest.raises(ContractError, match="does not match"):
        model.encode(observation.images, observation.state)
    fresh = live.observe()
    snapshot = model.encode(fresh.images, fresh.state)
    with pytest.raises(ContractError, match="another privileged"):
        model.predict(object(), candidates(live))
    prediction = model.predict(snapshot, candidates(live))
    with pytest.raises(ContractError, match="another privileged"):
        model.distance(prediction, object())
    with pytest.raises(ContractError, match="state goal"):
        model.encode_goal({"image": np.zeros((1, 2, 2, 3), np.uint8)})
    with pytest.raises(ContractError, match="not a trainable"):
        model.train_step(None)
