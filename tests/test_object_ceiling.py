"""TASK-047 object-aware privileged ceiling: isolation, phase machine, costs and parity.

The phase-machine and cost tests use a fixture robot (no physics). The parity test
runs real headless MuJoCo and is skipped without the pinned assets. Nothing here
is a learned result.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa.contracts import ContractError, ExecutionResult, Observation
from embodied_jepa.object_ceiling import (
    PHASES,
    TOP_DOWN,
    ObjectCeilingConfig,
    ObjectCeilingController,
    ObjectRollout,
    ObjectRolloutModel,
    rotation_error,
)
from embodied_jepa.registry import MODELS


class FixtureRobot:
    """Settable privileged truth; the base frame equals the world frame."""

    def __init__(self):
        self.palm = np.array([0.30, -0.15, 0.90])
        self.rotation = TOP_DOWN.copy()
        self.apple = np.array([0.34, -0.18, 0.77])
        self.plate = np.array([0.49, -0.09, 0.746])
        self.contact = False
        self.sim = SimpleNamespace(task_truth=self.truth)

    def truth(self):
        return {
            "base_position_world": np.zeros(3),
            "base_rotation_world": np.eye(3),
            "object_position": self.apple.copy(),
            "plate_position": self.plate.copy(),
            "hand_contact": self.contact,
            "dropped": bool(self.apple[2] < 0.70),
            "container_surface_z": 0.75,
            "object_support_height": 0.027,
        }

    def ee_pose(self, side):
        assert side == "right"
        return self.palm.copy(), self.rotation.copy()


def controller(robot=None, **config):
    robot = robot or FixtureRobot()
    model = ObjectRolloutModel.__new__(ObjectRolloutModel)  # no twin needed here
    return (
        ObjectCeilingController(
            model, robot, ObjectCeilingConfig(**config), acknowledge_privileged_ceiling=True
        ),
        robot,
    )


def live(ctl):
    from embodied_jepa.object_ceiling import features

    return features(ctl.robot)


def test_ceiling_requires_acknowledgement_and_is_not_a_registered_model():
    from embodied_jepa import config  # noqa: F401  (populates the registries)

    with pytest.raises(ContractError, match="acknowledge_privileged_ceiling"):
        ObjectRolloutModel(object(), object())
    model = ObjectRolloutModel.__new__(ObjectRolloutModel)
    with pytest.raises(ContractError, match="acknowledge_privileged_ceiling"):
        ObjectCeilingController(model, FixtureRobot(), ObjectCeilingConfig())
    with pytest.raises(ContractError, match="exact object rollouts"):
        ObjectCeilingController(
            object(), FixtureRobot(), ObjectCeilingConfig(), acknowledge_privileged_ceiling=True
        )
    for name in ("privileged_object", "privileged_mujoco_rollout_object_v1"):
        with pytest.raises(ValueError, match="unknown world model"):
            MODELS.require(name)
    with pytest.raises(ContractError):
        ObjectCeilingConfig(candidates=2)
    with pytest.raises(ContractError):
        ObjectCeilingConfig(slip_weight=-1.0)


def test_rotation_error_is_the_geodesic_angle_without_a_90_degree_plateau():
    from embodied_jepa.embodiment import rotation_delta

    assert rotation_error(TOP_DOWN) == pytest.approx(0.0, abs=1e-7)
    for angle in (0.3, np.pi / 2, 2.0):
        rotated = rotation_delta([0, 0, angle]) @ TOP_DOWN
        assert rotation_error(rotated) == pytest.approx(angle, abs=1e-6)


def test_phase_machine_follows_object_state_in_order():
    ctl, robot = controller(close_commands=2, release_commands=1, block_commands=2)
    assert ctl.phase.name == "approach"
    ctl._advance(live(ctl))
    assert ctl.phase.name == "approach"  # palm far from the pre-grasp point
    robot.palm = robot.apple + [-0.015, 0, 0.13]
    ctl._advance(live(ctl))
    assert ctl.phase.name == "descend"
    # A blocked descent (no palm progress over block_commands) ends the descent.
    robot.palm = robot.apple + [-0.015, 0, 0.11]
    robot.palm[:2] += [0.02, 0]  # blocked but away from the grasp point: keep descending
    for _ in range(3):
        ctl.phase.anchor["progress"].append(float(robot.palm[2]))
    ctl._advance(live(ctl))
    assert ctl.phase.name == "descend"
    robot.palm[:2] -= [0.02, 0]
    ctl._advance(live(ctl))
    assert ctl.phase.name == "close"
    assert ctl.phase_log[-1] == {"phase": "close", "command": 0, "previous_ended": "blocked"}
    ctl._advance(live(ctl))
    assert ctl.phase.name == "close"  # fixed command count
    ctl.phase.commands = 2
    ctl._advance(live(ctl))
    assert ctl.phase.name == "lift"
    robot.apple = robot.apple + [0, 0, 0.11]
    ctl._advance(live(ctl))
    assert ctl.phase.name == "lift"  # risen but no hand contact
    robot.contact = True
    ctl._advance(live(ctl))
    assert ctl.phase.name == "transport"
    # Transport is blocked only within the plate radius; then it hands over to lower.
    robot.apple = np.array([robot.plate[0] - 0.03, robot.plate[1], robot.apple[2]])
    for _ in range(3):
        ctl.phase.anchor["progress"].append(0.03)
    ctl._advance(live(ctl))
    assert ctl.phase.name == "lower"
    assert ctl.phase_log[-1]["previous_ended"] == "blocked"
    robot.apple = robot.apple.copy()
    robot.apple[2] = 0.75 + 0.027 + 0.015
    ctl._advance(live(ctl))
    assert ctl.phase.name == "release"
    ctl.phase.commands = 1
    ctl._advance(live(ctl))
    assert ctl.phase.name == "retreat"
    assert [row["phase"] for row in ctl.phase_log] == list(PHASES)


def rollout(palm, apple, contact=True, valid=None):
    palm, apple = np.asarray(palm, float), np.asarray(apple, float)
    shape = palm.shape[:2]
    return ObjectRollout(
        palm,
        np.zeros(shape),
        apple,
        np.full(shape, contact),
        np.zeros(shape, bool),
        np.ones(shape, bool) if valid is None else valid,
        None,
    )


def test_phase_costs_are_object_aware():
    ctl, robot = controller()
    target = robot.apple + [-0.015, 0, 0.13]
    far = robot.apple + [0.05, 0.05, 0.2]
    near = rollout([[target]], [[robot.apple]])
    away = rollout([[far]], [[robot.apple]])
    assert ctl.phase_costs(near, live(ctl))[0, 0] < ctl.phase_costs(away, live(ctl))[0, 0]
    # The approach target follows the apple: the same palm is worse for a moved apple.
    moved = rollout([[target]], [[robot.apple + [0.03, 0, 0]]])
    assert ctl.phase_costs(moved, live(ctl))[0, 0] > ctl.phase_costs(near, live(ctl))[0, 0]
    # Transport scores the apple (not the palm) against the plate.
    ctl._enter("transport", live(ctl))
    high = robot.apple[2] + 0.13
    on_plate = rollout([[robot.palm]], [[[*robot.plate[:2], high]]])
    off_plate = rollout([[robot.palm]], [[[*robot.apple[:2], high]]])
    dropped_contact = rollout([[robot.palm]], [[[*robot.plate[:2], high]]], contact=False)
    live_state = live(ctl)
    assert (
        ctl.phase_costs(on_plate, live_state)[0, 0] < ctl.phase_costs(off_plate, live_state)[0, 0]
    )
    assert (
        ctl.phase_costs(dropped_contact, live_state)[0, 0]
        > ctl.phase_costs(on_plate, live_state)[0, 0]
    )


def test_rejected_rollout_steps_rank_after_executable_candidates():
    ctl, robot = controller(horizon=2)
    target = robot.apple + [-0.015, 0, 0.13]
    palms = np.array([[target, target], [target, target]])
    valid = np.array([[True, True], [True, False]])
    costs = ctl.candidate_costs(
        rollout(palms, np.repeat([[robot.apple]], 2, 0).repeat(2, 1), valid=valid), live(ctl)
    )
    assert costs[1] > 100 * costs[0]


def test_step_refuses_stale_input_and_stops_on_drop_and_stall():
    ctl, robot = controller(phase_stall_commands=1)
    with pytest.raises(ContractError, match="synchronized observation"):
        ctl.step(object(), lambda requested: None)
    robot.apple = robot.apple - [0, 0, 0.2]
    obs = Observation.__new__(Observation)
    object.__setattr__(obs, "timestamps", np.array([0.05]))
    decision = ctl.step(obs, lambda requested: None)
    assert decision.action is None and decision.termination_reason == "object_dropped"
    ctl2, _ = controller(phase_stall_commands=1)
    ctl2.phase.commands = 1
    decision = ctl2.step(obs, lambda requested: None)
    assert decision.termination_reason == "phase_stall"
    assert decision.trace["phase"] == "approach"
    with pytest.raises(ContractError, match="awaits"):
        ctl2.acknowledge(ExecutionResult(np.zeros(14, np.float32), None, "stopped", 0.1, "x"))


@pytest.fixture
def physics():
    pytest.importorskip("mujoco")
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import ROOT, MuJoCoSimulation

    if not (ROOT / "third_party/unitree_mujoco/unitree_robots/g1/g1_29dof_with_hand.xml").exists():
        pytest.skip("pinned robot assets have not been fetched")
    sim = MuJoCoSimulation(render=False)
    sim.render = lambda camera="onboard_rgb": np.zeros((8, 8, 3), np.uint8)
    live_robot = G1Embodiment(sim)
    live_robot.reset(object_xy=[0.36, -0.20], plate_xy=[0.50, -0.08])
    metric = SimpleNamespace(observed_distance=lambda *a: None, capabilities=None)
    model = ObjectRolloutModel(metric, live_robot, acknowledge_privileged_ceiling=True)
    yield live_robot, model
    model.close()
    live_robot.close()


def test_object_rollout_first_step_matches_live_full_state_and_live_is_untouched(physics):
    live_robot, model = physics
    ctl = ObjectCeilingController(
        model,
        live_robot,
        ObjectCeilingConfig(horizon=2, candidates=3, iterations=1),
        acknowledge_privileged_ceiling=True,
    )
    for _ in range(3):
        observation = live_robot.observe()
        before = (live_robot.sim.data.qpos.copy(), live_robot.sim.data.qvel.copy())
        decision = ctl.step(observation, live_robot.project_candidates)
        np.testing.assert_array_equal(before[0], live_robot.sim.data.qpos)
        np.testing.assert_array_equal(before[1], live_robot.sim.data.qvel)
        assert decision.action is not None and decision.trace["phase"] == "approach"
        ctl.acknowledge(live_robot.execute(decision.action))
    diagnostics = model.pop_diagnostics()
    checks = [
        row["previous_search_first_step_full_state_exact_match"]
        for row in diagnostics
        if "previous_search_first_step_full_state_exact_match" in row
    ]
    robot_checks = [
        row["previous_search_first_step_exact_match"]
        for row in diagnostics
        if "previous_search_first_step_exact_match" in row
    ]
    assert checks == [True, True] and robot_checks == [True, True]
    with pytest.raises(ContractError, match="rollout"):
        model.predict(None, None)
