"""TASK-051 object-aware privileged ceiling v3: isolation, the caged grasp closure and
the inheritance of every other v2 behaviour. Fixture-robot tests use no physics.
Nothing here is a learned result.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from test_object_ceiling import FixtureRobot, live
from test_object_ceiling_v2 import landing

from embodied_jepa.contracts import ContractError
from embodied_jepa.object_ceiling import ObjectCeilingConfig
from embodied_jepa.object_ceiling_v2 import (
    ObjectCeilingV2Config,
    ObjectCeilingV2Controller,
    ObjectRolloutV2Model,
)
from embodied_jepa.object_ceiling_v3 import (
    CEILING_VERSION,
    ObjectCeilingV3Config,
    ObjectCeilingV3Controller,
)
from embodied_jepa.registry import MODELS


def controller(robot=None, **config):
    robot = robot or FixtureRobot()
    model = ObjectRolloutV2Model.__new__(ObjectRolloutV2Model)  # no twin needed here
    ctl = ObjectCeilingV3Controller(
        model, robot, ObjectCeilingV3Config(**config), acknowledge_privileged_ceiling=True
    )
    return ctl, robot


def test_v3_requires_acknowledgement_and_its_own_config_and_is_unregistered():
    from embodied_jepa import config  # noqa: F401  (populates the registries)

    model = ObjectRolloutV2Model.__new__(ObjectRolloutV2Model)
    with pytest.raises(ContractError, match="acknowledge_privileged_ceiling"):
        ObjectCeilingV3Controller(model, FixtureRobot(), ObjectCeilingV3Config())
    for config_object in (ObjectCeilingConfig(), ObjectCeilingV2Config()):
        with pytest.raises(ContractError):
            ObjectCeilingV3Controller(
                model, FixtureRobot(), config_object, acknowledge_privileged_ceiling=True
            )
    for name in ("privileged_object_v3", "object_ceiling_v3"):
        with pytest.raises(ValueError, match="unknown world model"):
            MODELS.require(name)


def test_v3_config_rejects_an_unusable_descent_bound():
    for bad in (
        dict(close_descent_bound=0.0),
        dict(close_descent_bound=-0.1),
        dict(close_descent_bound=0.6),  # above arm_bound
        dict(close_descent_bound=0.3, arm_bound=0.2),
    ):
        with pytest.raises(ContractError):
            ObjectCeilingV3Config(**bad)
    # The v2 validation still applies.
    with pytest.raises(ContractError):
        ObjectCeilingV3Config(place_tolerance_m=0.05)


def test_the_close_cages_the_apple_and_only_lets_the_palm_sink():
    ctl, robot = controller()
    v2 = ObjectCeilingV2Controller(
        ObjectRolloutV2Model.__new__(ObjectRolloutV2Model),
        robot,
        ObjectCeilingV2Config(),
        acknowledge_privileged_ceiling=True,
    )
    robot.palm = robot.apple + [-0.015, 0, 0.115]
    ctl._enter("close", live(ctl))
    v2._enter("close", live(v2))
    bound = ctl.config.close_descent_bound
    for commands in (0, 1, 10, ctl.config.close_commands - 1):
        ctl.phase.commands = v2.phase.commands = commands
        lower, upper = ctl._bounds()
        assert lower.dtype == upper.dtype == np.float32
        np.testing.assert_array_equal(lower[:6], 0.0)  # the left arm never moves
        np.testing.assert_array_equal(upper[:6], 0.0)
        np.testing.assert_array_equal(lower[6:8], 0.0)  # no lateral palm command
        np.testing.assert_array_equal(upper[6:8], 0.0)
        np.testing.assert_array_equal(lower[9:12], 0.0)  # no reorientation
        np.testing.assert_array_equal(upper[9:12], 0.0)
        assert (lower[8], upper[8]) == (-bound, 0.0)  # descent only: the palm never rises
        # The grasp schedule is v2's unchanged step to closed (rate-limited downstream).
        v2_lower, v2_upper = v2._bounds()
        np.testing.assert_array_equal(lower[12:], v2_lower[12:])
        np.testing.assert_array_equal(upper[12:], v2_upper[12:])
        assert lower[13] == upper[13] == 1.0
        # v2 would have planned all six arm deltas in +-arm_bound.
        assert (v2_lower[6], v2_upper[6]) == (-ctl.config.arm_bound, ctl.config.arm_bound)


def test_every_other_phase_keeps_the_v2_bounds():
    ctl, robot = controller()
    v2 = ObjectCeilingV2Controller(
        ObjectRolloutV2Model.__new__(ObjectRolloutV2Model),
        robot,
        ObjectCeilingV2Config(),
        acknowledge_privileged_ceiling=True,
    )
    for phase in ("approach", "descend", "lift", "transport", "lower", "release", "retreat"):
        ctl._enter(phase, live(ctl))
        v2._enter(phase, live(v2))
        for commands in (0, 7):
            ctl.phase.commands = v2.phase.commands = commands
            for mine, theirs in zip(ctl._bounds(), v2._bounds(), strict=True):
                np.testing.assert_array_equal(mine, theirs)


def test_v3_inherits_the_v2_transitions_and_release_predictor():
    ctl, robot = controller()
    robot.palm = robot.apple + [-0.015, 0, 0.115]
    ctl._enter("close", live(ctl))
    ctl.phase.commands = ctl.config.close_commands
    ctl._advance(live(ctl))
    assert ctl.phase.name == "lift"
    ctl._enter("transport", live(ctl))
    robot.apple = np.array([robot.plate[0], robot.plate[1], 0.9])
    robot.contact = True
    ctl.landing = landing(robot)
    ctl._advance(live(ctl))
    assert ctl.phase.name == "release"
    assert ctl.phase_log[-1]["previous_ended"] == "release_predicted"


def test_summary_reports_the_v3_identity():
    ctl, _ = controller()
    summary = ctl.summary()
    assert summary["ceiling_version"] == CEILING_VERSION == 3
    assert "v3" in summary["control_label"] and "NON-LEARNED" in summary["control_label"]
    assert "release_probes" in summary


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
    model = ObjectRolloutV2Model(metric, live_robot, acknowledge_privileged_ceiling=True)
    yield live_robot, model
    model.close()
    live_robot.close()


def test_the_live_close_holds_the_palm_laterally_and_only_sinks(physics):
    """Under real physics the v3 close never moves the palm sideways or up, and it keeps
    the exact-rollout parity check non-vacuous (the CEM still runs every command)."""
    from embodied_jepa.object_ceiling import features

    live_robot, model = physics
    ctl = ObjectCeilingV3Controller(
        model,
        live_robot,
        ObjectCeilingV3Config(horizon=2, candidates=3, iterations=1),
        acknowledge_privileged_ceiling=True,
    )
    ctl._enter("close", features(live_robot))
    start = features(live_robot)["palm"]
    commands = []
    for _ in range(6):
        decision = ctl.step(live_robot.observe(), live_robot.project_candidates)
        assert decision.action is not None and decision.trace["phase"] == "close"
        commands.append(np.asarray(decision.action))
        ctl.acknowledge(live_robot.execute(decision.action))
    commands = np.array(commands)
    np.testing.assert_array_equal(commands[:, 6:8], 0.0)  # no lateral command, ever
    np.testing.assert_array_equal(commands[:, 9:12], 0.0)  # no reorientation, ever
    assert (commands[:, 8] <= 0).all()  # descent only
    # v2's grasp schedule: a request for full closure that the embodiment's joint-rate
    # limit spreads over eleven commands (0.1 rad / 0.55 rad per unit = 0.1818 per command).
    grasps = commands[:, 13]
    assert (np.diff(grasps) > 0).all() and grasps[-1] < 1.0
    np.testing.assert_allclose(np.diff(grasps), 2 / 11, atol=1e-5)
    palm = features(live_robot)["palm"]
    assert np.linalg.norm(palm[:2] - start[:2]) < 0.002  # the palm holds its lateral pose
    assert palm[2] < start[2]  # and sinks
    checks = [
        row["previous_search_first_step_full_state_exact_match"]
        for row in model.pop_diagnostics()
        if "previous_search_first_step_full_state_exact_match" in row
    ]
    assert len(checks) == 5 and all(checks)  # non-vacuous parity, every command but the first
