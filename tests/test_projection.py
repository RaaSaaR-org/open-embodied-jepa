"""Projection acceptance uses real headless physics; planner fixtures test contracts only."""

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
from embodied_jepa.planning import MPC, CEMConfig, CEMPlanner


@pytest.fixture
def robot(monkeypatch):
    pytest.importorskip("mujoco")
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import ROOT, MuJoCoSimulation

    if not (ROOT / "third_party/unitree_mujoco/unitree_robots/g1/g1_29dof_with_hand.xml").exists():
        pytest.skip("pinned G1/Dex3 assets unavailable")
    sim = MuJoCoSimulation(render=False)
    # Synthetic RGB only: all joint motion, tracking lag and time remain real physics.
    monkeypatch.setattr(sim, "render", lambda: np.zeros((8, 8, 3), np.uint8))
    embodiment = G1Embodiment(sim)
    embodiment.reset(41)
    embodiment.observe()
    yield embodiment
    embodiment.close()


def requests(candidates=2, horizon=3):
    value = np.zeros((1, candidates, horizon, 14), np.float32)
    value[..., 12:] = 1
    return value


def snapshot(robot):
    return {
        "qpos": robot.sim.data.qpos.copy(),
        "qvel": robot.sim.data.qvel.copy(),
        "ctrl": robot.sim.data.ctrl.copy(),
        "targets": robot.sim.targets.copy(),
        "grasp": robot._grasp.copy(),
        "time": robot.sim.data.time,
        "observation": robot._observation,
    }


def assert_unchanged(robot, before):
    for field in ("qpos", "qvel", "ctrl"):
        np.testing.assert_array_equal(getattr(robot.sim.data, field), before[field])
    np.testing.assert_array_equal(robot.sim.targets, before["targets"])
    np.testing.assert_array_equal(robot._grasp, before["grasp"])
    assert robot.sim.data.time == before["time"]
    assert robot._observation is before["observation"]


def test_projection_is_pure_and_independent_of_object_truth(robot, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("projection polled sensors, accessed task truth or actuated")

    for name in ("task_truth", "read", "render", "send_joint_targets", "stop"):
        monkeypatch.setattr(robot.sim, name, forbidden)
    command = requests()
    command[0, 1, :, 2] = command[0, 1, :, 8] = -0.3
    original = command.copy()
    before = snapshot(robot)
    first = robot.project_candidates(command)
    assert_unchanged(robot, before)
    np.testing.assert_array_equal(command, original)
    assert first.feasible.all()
    assert not first.actions.flags.writeable and not first.feasible.flags.writeable
    # A test-only intervention changes privileged object coordinates, never robot state.
    address = robot.model.joint("apple_free").qposadr[0]
    robot.sim.data.qpos[address : address + 3] = [1.2, 0.9, 0.3]
    robot.mj.mj_forward(robot.model, robot.sim.data)
    changed = snapshot(robot)
    second = robot.project_candidates(command)
    assert_unchanged(robot, changed)
    np.testing.assert_array_equal(first.actions, second.actions)
    np.testing.assert_array_equal(first.feasible, second.feasible)


def test_horizon_ramps_both_mirrored_hands_without_exceeding_joint_target_rates(robot):
    result = robot.project_candidates(requests(candidates=1, horizon=6))
    assert result.feasible[0, 0]
    grasp = result.actions[0, 0, :, 12:]
    assert np.all(np.diff(grasp, axis=0) > 0)
    assert np.all(grasp[0] < 0) and np.all(grasp[-1] > grasp[0])
    np.testing.assert_allclose(grasp[:, 0], grasp[:, 1], atol=1e-7)
    for side_index, side in enumerate(("left", "right")):
        opened = np.array(robot.manifest[f"{side}_open_rad"])
        closed = np.array(robot.manifest[f"{side}_closed_rad"])
        targets = opened + (grasp[:, side_index, None] + 1) / 2 * (closed - opened)
        delta = np.diff(np.vstack((opened, targets)), axis=0)
        limit = robot.manifest["joint_speed_limit_rad_s"] * robot.sim.control_dt
        assert np.max(np.abs(delta)) <= limit + 1e-6
    # The projection must not pretend the full absolute grasp can execute immediately.
    assert not np.array_equal(result.actions, requests(candidates=1, horizon=6))


def test_projection_repeats_and_is_idempotent_with_float_tolerance(robot):
    command = requests(candidates=2, horizon=3)
    command[0, 1, :, :12] = np.array([0, 0, -0.3, 0, 0, 0.2] * 2)
    first = robot.project_candidates(command)
    repeated = robot.project_candidates(command)
    projected_again = robot.project_candidates(first.actions)
    assert first.feasible.all()
    np.testing.assert_array_equal(first.feasible, repeated.feasible)
    np.testing.assert_array_equal(first.actions, repeated.actions)
    np.testing.assert_array_equal(first.feasible, projected_again.feasible)
    np.testing.assert_allclose(first.actions, projected_again.actions, atol=2e-6, rtol=0)


@pytest.mark.parametrize("with_tracking_lag", [False, True])
def test_projected_first_step_matches_execution_with_actual_tracking_lag(robot, with_tracking_lag):
    command = requests(candidates=1, horizon=1)
    command[..., 2] = command[..., 8] = -0.5
    if with_tracking_lag:
        applied = robot.execute(command[0, 0, 0])
        assert applied.applied_action is not None, applied.reason
        assert np.max(np.abs(robot.sim.data.qpos[robot.sim.qadr] - robot.sim.targets)) > 0.001
        robot.observe()
    projection = robot.project_candidates(command)
    assert projection.feasible[0, 0]
    action = projection.actions[0, 0, 0]
    result = robot.execute(action)
    assert result.applied_action is not None, result.reason
    np.testing.assert_allclose(result.applied_action, action, atol=2e-6, rtol=0)


def test_infeasible_zero_delta_is_excluded_instead_of_silent_hold(robot):
    # A lagging joint is farther from the last command than one legal target increment.
    index = robot.sim.joint_names.index("right_shoulder_pitch_joint")
    robot.sim.targets[index] += 0.4
    robot.observe()
    command = requests(candidates=1, horizon=1)
    before = snapshot(robot)
    result = robot.project_candidates(command)
    assert not result.feasible[0, 0]
    assert_unchanged(robot, before)


class SquaredActionModel:
    def __init__(self):
        self.predictions = []

    def predict(self, latent, actions):
        self.predictions.append(actions.copy())
        return actions

    def distance(self, predicted, goal):
        return np.square(predicted).sum(-1).astype(np.float32)


def test_cem_masks_cheaper_invalid_rows_and_scores_projected_actions():
    model = SquaredActionModel()
    config = CEMConfig(project_candidates=True, samples=4, elites=4, iterations=2, horizon=2)

    def project(command):
        actions = np.zeros_like(command)
        actions[:, 2, :, 13] = 0.5
        feasible = np.zeros(command.shape[:2], bool)
        feasible[:, 2] = True
        return CandidateProjection(actions, feasible)

    plan = CEMPlanner(config).plan(model, None, None, batch_size=2, projector=project)
    assert plan.feasible_candidates == 4
    assert len(model.predictions) == 2
    np.testing.assert_allclose(plan.actions[..., 13], 0.5)
    np.testing.assert_allclose(plan.costs, 0.25)
    assert plan.projection_seconds >= 0
    assert not np.array_equal(plan.actions, plan.requested_actions)


def test_cem_all_invalid_fails_before_model_prediction():
    model = SquaredActionModel()
    planner = CEMPlanner(CEMConfig(project_candidates=True, samples=4, elites=2, horizon=1))

    def project(command):
        feasible = np.ones(command.shape[:2], bool)
        feasible[-1] = False
        return CandidateProjection(command, feasible)

    with pytest.raises(ContractError, match="no feasible"):
        planner.plan(model, None, None, batch_size=2, projector=project)
    assert model.predictions == []
    with pytest.raises(ContractError, match="projector"):
        planner.plan(model, None, None)


class TraceRobot:
    state_schema = StateSchema(("joint.q",), ("rad",), "projection_fixture_v0")
    action_schema = EE_DELTA_GRASP_V0

    def __init__(self):
        self.commands = []
        self.stopped = None

    def observe(self):
        return Observation(
            {"onboard_rgb": np.zeros((1, 8, 8, 3), np.uint8)},
            np.zeros((1, 1), np.float32),
            np.ones((1, 1), bool),
            np.array([0.0]),
            self.state_schema,
        )

    def project_candidates(self, requested):
        actions = requested.copy()
        actions[..., 13] = 0.25
        return CandidateProjection(actions, np.ones(actions.shape[:2], bool))

    def execute(self, action):
        self.commands.append(action.copy())
        applied = action.copy()
        applied[13] = 0.125
        return ExecutionResult(action, applied, "clipped", 0.05, "transport changed acceptance")

    def stop(self, reason):
        self.stopped = reason


class TraceModel(SquaredActionModel):
    def __init__(self, robot):
        super().__init__()
        self.capabilities = Capabilities(robot.action_schema, robot.state_schema, 4)

    def encode_goal(self, goal):
        return goal

    def encode(self, images, state):
        return None


def test_mpc_trace_distinguishes_sampled_projected_and_applied():
    robot = TraceRobot()
    fixed = (0.0,) * 13 + (1.0,)
    planner = CEMPlanner(
        CEMConfig(
            project_candidates=True,
            samples=2,
            elites=1,
            iterations=1,
            horizon=1,
            lower_bounds=fixed,
            upper_bounds=fixed,
        )
    )
    result = MPC(TraceModel(robot), planner).run(robot, None, max_steps=1)
    assert result["executed_steps"] == 1
    row = result["trace"][0]
    assert row["sampled_action"][13] == 1
    assert row["projected_action"][13] == row["requested_action"][13] == 0.25
    assert row["action"][13] == 0.125
    assert row["status"] == "clipped" and row["projection_seconds"] >= 0
    assert row["feasible_candidates"] == 2
    np.testing.assert_array_equal(robot.commands[0], row["projected_action"])
    assert robot.stopped == "step_limit"


def test_pose_backtracking_repairs_actual_joint_rate_rejection_at_identical_reset(robot):
    # Six positive pose components on both arms exceed the target-rate limit at this reset.
    request = np.ones(14, np.float32)
    request[12:] = -1
    initial = snapshot(robot)
    rejected = robot.execute(request)
    assert rejected.applied_action is None
    assert rejected.status == "stopped" and "joint rate limit" in rejected.reason
    np.testing.assert_array_equal(robot.sim.data.qpos, initial["qpos"])
    assert robot.sim.data.time == initial["time"]

    # Restore the exact same state and command history, not an easier robot pose.
    robot.reset(41)
    robot.observe()
    np.testing.assert_array_equal(robot.sim.data.qpos, initial["qpos"])
    np.testing.assert_array_equal(robot.sim.targets, initial["targets"])
    projected = robot.project_candidates(request[None, None, None])
    assert projected.feasible[0, 0]
    repaired = projected.actions[0, 0, 0]
    for start in (0, 6):
        assert 0 < np.linalg.norm(repaired[start : start + 6]) < np.sqrt(6)
    np.testing.assert_array_equal(repaired[12:], request[12:])
    previous_targets = robot.sim.targets.copy()
    result = robot.execute(repaired)
    assert result.applied_action is not None, result.reason
    np.testing.assert_allclose(result.applied_action, repaired, atol=2e-6, rtol=0)
    limit = robot.manifest["joint_speed_limit_rad_s"] * robot.sim.control_dt
    assert np.max(np.abs(robot.sim.targets - previous_targets)) <= limit + 1e-6
    assert robot.sim.data.time == pytest.approx(initial["time"] + robot.sim.control_dt)


def test_projection_acceptance_after_physics_refreshes_cached_ee_transforms(robot):
    # mj_step can leave derived EE transforms behind its latest integrated qpos.
    # Both preview and execution must prepare targets from the same fresh snapshot.
    robot.reset(seed=4)
    rng = np.random.default_rng(341)
    for step in range(2):
        robot.observe()
        requested = rng.uniform(-1, 1, (1, 1, 3, 14)).astype(np.float32)
        projection = robot.project_candidates(requested)
        assert projection.feasible[0, 0], f"step {step} unexpectedly infeasible"
        action = projection.actions[0, 0, 0]
        result = robot.execute(action)
        assert result.applied_action is not None, f"step {step}: {result.reason}"
        np.testing.assert_allclose(result.applied_action, action, atol=2e-6, rtol=0)
