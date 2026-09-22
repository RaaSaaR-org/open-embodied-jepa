"""TASK-049 object-aware privileged ceiling v2: isolation, v2 transitions, costs, release
probe and parity. Fixture-robot tests use no physics; the probe/parity tests run real
headless MuJoCo and skip without the pinned assets. Nothing here is a learned result.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from test_object_ceiling import FixtureRobot, live, rollout

from embodied_jepa.contracts import ContractError
from embodied_jepa.object_ceiling import PHASES, ObjectCeilingConfig, ObjectRolloutModel
from embodied_jepa.object_ceiling_v2 import (
    ObjectCeilingV2Config,
    ObjectCeilingV2Controller,
    ObjectRolloutV2Model,
)
from embodied_jepa.registry import MODELS


def controller(robot=None, **config):
    robot = robot or FixtureRobot()
    model = ObjectRolloutV2Model.__new__(ObjectRolloutV2Model)  # no twin needed here
    ctl = ObjectCeilingV2Controller(
        model, robot, ObjectCeilingV2Config(**config), acknowledge_privileged_ceiling=True
    )
    return ctl, robot


def landing(robot, dx=0.0, contact=False, z=None, speed=0.0, n=60):
    """A probed release whose apple path stays at one point for all ``n`` commands."""
    apple = np.array([robot.plate[0] + dx, robot.plate[1], 0.75 + 0.027 if z is None else z])
    return {
        "apple_path": np.repeat(apple[None], n, 0),
        "speed": np.full(n, speed),
        "contact_path": np.full(n, contact),
        "apple": apple,
        "plate": robot.plate.copy(),
        "rest_z": 0.75 + 0.027,
        "contact": contact,
        "dropped": False,
        "executed": n,
        "complete": True,
    }


def test_v2_requires_acknowledgement_v2_config_and_probe_twin_and_is_unregistered():
    from embodied_jepa import config  # noqa: F401  (populates the registries)

    model = ObjectRolloutV2Model.__new__(ObjectRolloutV2Model)
    with pytest.raises(ContractError, match="acknowledge_privileged_ceiling"):
        ObjectCeilingV2Controller(model, FixtureRobot(), ObjectCeilingV2Config())
    with pytest.raises(ContractError, match="ObjectCeilingV2Config"):
        ObjectCeilingV2Controller(
            model, FixtureRobot(), ObjectCeilingConfig(), acknowledge_privileged_ceiling=True
        )
    v1_model = ObjectRolloutModel.__new__(ObjectRolloutModel)
    with pytest.raises(ContractError, match="release-probe twin"):
        ObjectCeilingV2Controller(
            v1_model, FixtureRobot(), ObjectCeilingV2Config(), acknowledge_privileged_ceiling=True
        )
    with pytest.raises(ContractError, match="acknowledge_privileged_ceiling"):
        ObjectRolloutV2Model(object(), object())
    for bad in (
        dict(place_tolerance_m=0.05),
        dict(descend_xy_tolerance_m=0.02),
        dict(release_ramp=0.0),
        dict(landing_dwell_commands=0),
        dict(place_xy_weight=0.0),
    ):
        with pytest.raises(ContractError):
            ObjectCeilingV2Config(**bad)
    with pytest.raises(ValueError, match="unknown world model"):
        MODELS.require("privileged_object_v2")


def blocked_descent(ctl, robot, xy_offset, xy_history):
    robot.palm = robot.apple + [-0.015 + xy_offset, 0, 0.115]
    ctl.phase.anchor["progress"][:] = [float(robot.palm[2])] * 12
    ctl.phase.anchor["xy_progress"][:] = list(xy_history)
    ctl._advance(live(ctl))
    return ctl.phase.name


def test_descent_blocked_rule_is_keyed_to_the_xy_error():
    ctl, robot = controller()
    ctl._enter("descend", live(ctl))
    # Height blocked, xy 8 mm and still improving: keep re-centring.
    assert blocked_descent(ctl, robot, 0.008, np.linspace(0.02, 0.008, 12)) == "descend"
    # Height blocked, xy stalled at 1.3 cm (outside block_xy_m): keep descending.
    assert blocked_descent(ctl, robot, 0.013, [0.013] * 12) == "descend"
    # Height blocked, xy stalled at 1.1 cm (inside block_xy_m): the descent ends.
    assert blocked_descent(ctl, robot, 0.011, [0.011] * 12) == "close"
    assert ctl.phase_log[-1]["previous_ended"] == "blocked"
    ctl2, robot2 = controller()
    ctl2._enter("descend", live(ctl2))
    # Height blocked and xy within 5 mm: ends at once, whatever the xy history.
    assert blocked_descent(ctl2, robot2, 0.004, np.linspace(0.02, 0.004, 12)) == "close"


def test_descent_cost_rewards_xy_recentring_at_a_blocked_height():
    ctl, robot = controller()
    ctl._enter("descend", live(ctl))
    centred = robot.apple + [-0.015, 0, 0.115]
    off = centred + [0.01, 0, 0]
    costs = ctl.phase_costs(rollout([[centred, off]], [[robot.apple, robot.apple]]), live(ctl))
    # v2: a 1 cm xy error costs descend_xy_weight * 1 cm (v1: < 1 mm at this height).
    assert costs[0, 1] - costs[0, 0] == pytest.approx(0.03, abs=1e-9)


def test_close_keeps_grasp_pressure_and_guards_palm_drift_beyond_a_dead_band():
    ctl, robot = controller()
    robot.palm = robot.apple + [-0.015, 0, 0.115]
    ctl._enter("close", live(ctl))
    np.testing.assert_array_equal(ctl.phase.anchor["hold_xy"], robot.palm[:2])
    np.testing.assert_allclose(ctl.phase.anchor["target"], robot.apple + [-0.015, 0, 0.052])
    small = robot.palm + [-0.008, 0, 0]  # inside the 1 cm dead band: v1 cost only
    drifted = robot.palm + [-0.025, 0, 0]  # the TASK-047 45006 drift
    costs = ctl.phase_costs(rollout([[robot.palm, small, drifted]], [[robot.apple] * 3]), live(ctl))
    v1 = [np.linalg.norm(p - ctl.phase.anchor["target"]) for p in (robot.palm, small, drifted)]
    np.testing.assert_allclose(costs[0, :2], v1[:2], atol=1e-9)
    assert costs[0, 2] == pytest.approx(v1[2] + 0.015, abs=1e-9)


def test_release_is_gated_on_a_carried_apple_over_the_plate_and_a_predicted_placement():
    ctl, robot = controller(block_commands=2)
    robot.contact = True
    robot.apple = np.array([*robot.plate[:2], 0.90])  # carried right over the centre
    ctl._enter("transport", live(ctl))
    for bad in (
        landing(robot, dx=0.035),  # outside the place tolerance
        landing(robot, contact=True),  # still in the hand
        landing(robot, z=0.767),  # on the table, not the plate
        landing(robot, speed=0.3),  # still rolling
    ):
        ctl.landing = bad
        ctl._advance(live(ctl))
        assert ctl.phase.name == "transport"  # centred apple, but no predicted placement
    ctl.landing = None
    ctl.phase.anchor["progress"][:] = [0.0] * 4  # blocked: hand over to lower
    ctl._advance(live(ctl))
    assert ctl.phase.name == "lower" and ctl.phase_log[-1]["previous_ended"] == "blocked"
    ctl.phase.anchor["progress"][:] = [0.9] * 4  # a blocked lower keeps planning
    ctl._advance(live(ctl))
    assert ctl.phase.name == "lower"
    ctl.landing = landing(robot, dx=0.02)
    ctl._advance(live(ctl))
    assert ctl.phase.name == "release"
    assert ctl.phase_log[-1]["previous_ended"] == "release_predicted"
    # A good prediction is not enough while the carried apple is off the plate or the
    # hand has lost contact (the scorer's transport stage needs both).
    for apple, contact in (([0.30, -0.15, 0.90], True), ([*robot.plate[:2], 0.90], False)):
        ctl2, robot2 = controller()
        robot2.apple, robot2.contact = np.array(apple), contact
        ctl2._enter("transport", live(ctl2))
        ctl2.landing = landing(robot2)
        ctl2._advance(live(ctl2))
        assert ctl2.phase.name == "transport"
    # Transport may hand over straight to release (skipping lower).
    robot2.contact = True
    ctl2._advance(live(ctl2))
    assert [row["phase"] for row in ctl2.phase_log][-2:] == ["transport", "release"]


def test_predicted_placement_needs_a_dwell_of_consecutive_resting_commands():
    ctl, robot = controller()
    probe = landing(robot)
    probe["speed"][:] = 0.3
    probe["speed"][[10, 11, 12, 20, 21, 22, 23]] = 0.0
    ctl.landing = probe
    assert ctl._placed_at() == 20  # 10-12 is one command short of the 4-command dwell
    probe["complete"] = False
    assert ctl._placed_at() is None


def test_release_executes_the_probed_gradual_opening_with_arms_held():
    ctl, robot = controller()
    schedule = ctl.release_schedule()
    assert len(schedule) == 60 and schedule[0] == pytest.approx(0.92)
    assert schedule[24] == pytest.approx(-1.0) and schedule[-1] == -1.0
    ctl._enter("release", live(ctl))
    for command in (0, 5, 59, 80):
        ctl.phase.commands = command
        lower, upper = ctl._bounds()
        np.testing.assert_array_equal(lower, upper)
        assert (lower[:12] == 0).all() and lower[12] == -1.0
        assert lower[13] == schedule[min(command, 59)]
    ctl._enter("transport", live(ctl))
    assert ctl._bounds()[1][13] == 1.0  # other phases keep the v1 schedule


def test_transport_cost_is_xy_weighted_towards_the_plate_centre():
    ctl, robot = controller()
    robot.contact = True
    ctl._enter("transport", live(ctl))
    high = robot.apple[2] + 0.13
    centred = rollout([[robot.palm]], [[[*robot.plate[:2], high]]])
    off = rollout([[robot.palm]], [[[robot.plate[0] - 0.01, robot.plate[1], high]]])
    state = live(ctl)
    difference = ctl.phase_costs(off, state)[0, 0] - ctl.phase_costs(centred, state)[0, 0]
    assert difference > 0.035  # place_xy_weight * 1 cm, less a small slip change


def test_step_probes_the_release_only_in_transport_and_lower():
    from test_object_ceiling import StubRollouts, fresh_observation

    from embodied_jepa.constraints import CandidateProjection
    from embodied_jepa.contracts import ExecutionResult

    ctl, robot = controller(horizon=2, candidates=3, iterations=1)
    probes = []

    class Stub(StubRollouts):
        def release_probe(self, state, commands):
            probes.append(len(commands))
            np.testing.assert_array_equal(commands, ctl.release_schedule())
            return landing(robot, dx=0.035)

    ctl.rollouts = Stub(robot)

    def projector(requested):
        return CandidateProjection(requested.copy(), np.ones(requested.shape[:2], bool))

    timestamp = 0.05
    for name in PHASES:
        ctl._enter(name, live(ctl))
        before = len(probes)
        decision = ctl.step(fresh_observation(timestamp), projector)
        assert decision.action is not None
        expected = name in ("transport", "lower")
        assert (len(probes) > before) == expected
        assert ("predicted_landing" in decision.trace) == expected
        timestamp += 0.05
        ctl.acknowledge(ExecutionResult(decision.action, decision.action, "applied", timestamp))
        timestamp += 0.05
    assert probes == [60, 60] and ctl.summary()["release_probes"] == 2
    assert ctl.summary()["ceiling_version"] == 2


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


def test_release_probe_leaves_live_state_and_parity_bookkeeping_untouched(physics):
    live_robot, model = physics
    observation = live_robot.observe()
    before = (live_robot.sim.data.qpos.copy(), live_robot.sim.data.qvel.copy())
    result = model.release_probe(observation.state, np.full(5, -1.0))
    np.testing.assert_array_equal(before[0], live_robot.sim.data.qpos)
    np.testing.assert_array_equal(before[1], live_robot.sim.data.qvel)
    assert result["complete"] and result["executed"] == 5
    # The apple rests on the table, not the plate, so this is not a plate landing.
    np.testing.assert_allclose(result["apple"][:2], [0.36, -0.20], atol=0.002)
    assert not result["contact"] and result["apple"][2] < result["rest_z"] - 0.01
    assert model._search_first_steps == [] and model._search_full == []
    assert model.pop_diagnostics() == []
    for bad in ([], [2.0], [np.nan]):
        with pytest.raises(ContractError, match="grasp schedule"):
            model.release_probe(observation.state, bad)


def test_v2_first_step_matches_live_full_state(physics):
    live_robot, model = physics
    ctl = ObjectCeilingV2Controller(
        model,
        live_robot,
        ObjectCeilingV2Config(horizon=2, candidates=3, iterations=1),
        acknowledge_privileged_ceiling=True,
    )
    for _ in range(3):
        decision = ctl.step(live_robot.observe(), live_robot.project_candidates)
        assert decision.action is not None
        ctl.acknowledge(live_robot.execute(decision.action))
    checks = [
        row["previous_search_first_step_full_state_exact_match"]
        for row in model.pop_diagnostics()
        if "previous_search_first_step_full_state_exact_match" in row
    ]
    assert checks == [True, True]
