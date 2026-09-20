import numpy as np
import pytest

from embodied_jepa.contracts import ContractError
from embodied_jepa.planning import CEMConfig, CEMPlanner


class Integrator:
    def predict(self, latent, actions):
        return latent[:, None, None] + np.cumsum(actions[..., :2], axis=2)

    def distance(self, predicted, goal):
        return np.square(predicted - goal[:, None, None]).sum(-1).astype(np.float32)


def test_cem_reaches_goal_better_than_hold_and_random_same_bounds():
    config = CEMConfig(horizon=4, samples=256, iterations=5, elites=20, seed=14)
    model = Integrator()
    initial = np.zeros((2, 2), np.float32)
    goal = np.array([[1.7, -1.3], [-0.8, 1.2]], np.float32)
    plan = CEMPlanner(config).plan(model, initial, goal, batch_size=2)
    random = np.random.default_rng(14).uniform(-1, 1, (2, 256, 4, 14)).astype(np.float32)
    random_cost = model.distance(model.predict(initial, random), goal)[:, :, -1].mean(1)
    hold_cost = np.square(goal).sum(1)
    assert np.all(plan.costs < hold_cost * 0.01)
    assert np.all(plan.costs < random_cost * 0.01)
    assert plan.actions.shape == (2, 4, 14)
    assert plan.actions.dtype == np.float32
    assert np.all(np.abs(plan.actions) <= 1)
    assert plan.evaluations == 2 * 256 * 5
    assert plan.elapsed_seconds > 0
    repeat = CEMPlanner(config).plan(model, initial, goal, batch_size=2)
    np.testing.assert_array_equal(plan.actions, repeat.actions)


def test_cem_rejects_nonfinite_backend_cost_without_returning_command():
    class Broken(Integrator):
        def distance(self, predicted, goal):
            result = super().distance(predicted, goal)
            result[0, 0, 0] = np.nan
            return result

    with pytest.raises(ContractError, match="finite"):
        CEMPlanner().plan(Broken(), np.zeros((1, 2)), np.zeros((1, 2)))


@pytest.mark.parametrize(
    "options",
    [
        {"horizon": 0},
        {"samples": True},
        {"elites": 65},
        {"action_penalty": -1},
        {"minimum_std": 0},
        {"seed": -1},
    ],
)
def test_invalid_budget_fails_early(options):
    with pytest.raises(ValueError):
        CEMConfig(**options)


class MockEmbodiment:
    def __init__(self):
        from embodied_jepa.contracts import EE_DELTA_GRASP_V0, StateSchema

        self.state_schema = StateSchema(("x", "y"), ("m", "m"), "test_v0")
        self.action_schema = EE_DELTA_GRASP_V0
        self.t = 0.0
        self.xy = np.zeros(2, np.float32)
        self.commands = []
        self.stopped = None

    def observe(self):
        from embodied_jepa.contracts import Observation

        return Observation(
            {"rgb": np.zeros((1, 8, 8, 3), np.uint8)},
            self.xy[None],
            np.ones((1, 2), bool),
            np.array([self.t]),
            self.state_schema,
        )

    def execute(self, action):
        from embodied_jepa.contracts import ExecutionResult

        self.commands.append(action.copy())
        self.xy += action[:2]
        self.t += 0.1
        return ExecutionResult(action, action, "applied", self.t)

    def stop(self, reason):
        self.stopped = reason


class ControlIntegrator(Integrator):
    def __init__(self, embodiment):
        from embodied_jepa.contracts import Capabilities

        self.capabilities = Capabilities(embodiment.action_schema, embodiment.state_schema, 10)

    def encode(self, observation, robot_state):
        return robot_state.values

    def encode_goal(self, goal):
        return goal


def test_mpc_replans_after_each_executed_action():
    from embodied_jepa.planning import MPC

    robot = MockEmbodiment()
    model = ControlIntegrator(robot)
    planner = CEMPlanner(CEMConfig(horizon=1, samples=128, iterations=4, elites=12))
    result = MPC(model, planner).run(
        robot,
        np.array([[0.7, -0.4]], np.float32),
        max_steps=3,
        evaluate=lambda: {"success": bool(np.linalg.norm(robot.xy - [0.7, -0.4]) < 0.05)},
    )
    assert result["termination_reason"] == "success"
    assert result["executed_steps"] == len(robot.commands)
    assert result["executed_steps"] >= 1
    assert all(t["status"] == "applied" for t in result["trace"])


def test_mpc_deadline_stops_without_executing_stale_action():
    from embodied_jepa.planning import MPC

    robot = MockEmbodiment()
    result = MPC(ControlIntegrator(robot), CEMPlanner(), timeout_seconds=1e-12).run(
        robot, np.zeros((1, 2), np.float32), max_steps=2
    )
    assert result["termination_reason"] == "deadline_miss"
    assert robot.stopped
    assert not robot.commands
    assert result["trace"][0]["executed"] is False


def test_mpc_rejects_reused_observation_after_command():
    from embodied_jepa.planning import MPC

    robot = MockEmbodiment()
    original = robot.observe()
    robot.observe = lambda: original
    result = MPC(ControlIntegrator(robot), CEMPlanner()).run(
        robot, np.zeros((1, 2), np.float32), max_steps=2
    )
    assert result["termination_reason"] == "stale_observation"
    assert len(robot.commands) == 1


@pytest.mark.parametrize("stage", ["observe", "encode_goal", "execute", "capabilities"])
def test_mpc_stops_on_every_runtime_failure(stage):
    from embodied_jepa.planning import MPC

    robot = MockEmbodiment()
    model = ControlIntegrator(robot)

    def fail(*args, **kwargs):
        raise ContractError("injected failure")

    if stage in ("observe", "execute"):
        setattr(robot, stage, fail)
    elif stage == "capabilities":

        class BrokenCapabilities:
            require = fail

        model.capabilities = BrokenCapabilities()
    else:
        model.encode_goal = fail
    result = MPC(model, CEMPlanner()).run(robot, np.zeros((1, 2), np.float32), max_steps=2)
    assert result["termination_reason"] == "runtime_error"
    assert robot.stopped
    assert result["trace"][-1]["stage"] == stage
    assert result["trace"][-1]["execution_uncertain"] is (stage == "execute")


def test_mpc_stops_after_rejected_acknowledgement():
    from embodied_jepa.contracts import ExecutionResult
    from embodied_jepa.planning import MPC

    robot = MockEmbodiment()
    robot.execute = lambda action: ExecutionResult(action, None, "rejected", 0.0, "test rejection")
    result = MPC(ControlIntegrator(robot), CEMPlanner()).run(
        robot, np.zeros((1, 2), np.float32), max_steps=2
    )
    assert result["termination_reason"] == "rejected"
    assert robot.stopped == "rejected"
    assert result["executed_steps"] == 0


def test_shared_action_bounds_can_hold_inactive_components():
    lower, upper = np.zeros(14), np.zeros(14)
    lower[:2], upper[:2] = -0.5, 0.5
    lower[12:], upper[12:] = -1, -1
    config = CEMConfig(lower_bounds=tuple(lower), upper_bounds=tuple(upper))
    plan = CEMPlanner(config).plan(Integrator(), np.zeros((1, 2)), np.ones((1, 2)))
    assert np.all(plan.actions[..., 2:12] == 0)
    assert np.all(plan.actions[..., 12:] == -1)
    assert np.all(np.abs(plan.actions[..., :2]) <= 0.5)
