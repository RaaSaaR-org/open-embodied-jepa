"""Physics checks use no graphics; RGB checks explicitly opt in on a graphics host."""

import os

import numpy as np
import pytest

pytest.importorskip("mujoco")
from embodied_jepa.contracts import ContractError
from embodied_jepa.simulation import ROOT, MuJoCoSimulation

pytestmark = pytest.mark.skipif(
    not (ROOT / "third_party/unitree_mujoco/unitree_robots/g1/g1_29dof_with_hand.xml").exists(),
    reason="pinned robot assets have not been fetched",
)


@pytest.fixture
def sim():
    instance = MuJoCoSimulation(render=False)
    yield instance
    instance.close()


def test_reset_and_torque_steps_are_exactly_reproducible(sim):
    traces = []
    for _ in range(2):
        sim.reset(12)
        targets = sim.targets.astype(np.float32)
        targets[sim.joint_names.index("right_elbow_joint")] += 0.04
        trace = []
        for _ in range(5):
            sim.send_joint_targets(targets, joint_names=sim.joint_names, deadline=sim.data.time)
            trace.append(sim.data.qpos.copy())
        traces.append(trace)
    np.testing.assert_array_equal(traces[0], traces[1])
    assert np.isfinite(traces).all()
    assert not np.array_equal(sim.data.qpos[sim.qadr], targets)  # Dynamics, not teleporting.
    assert sim.model.nu == 43
    assert sim.data.time == pytest.approx(0.25)


def test_transport_rejects_deadline_order_limits_and_bad_resets(sim):
    start = sim.data.qpos.copy()
    ack = sim.send_joint_targets(sim.targets, joint_names=sim.joint_names, deadline=-1)
    assert ack["status"] == "rejected"
    np.testing.assert_array_equal(start, sim.data.qpos)
    with pytest.raises(ContractError, match="named robot"):
        sim.send_joint_targets(sim.targets, joint_names=sim.joint_names[::-1], deadline=1)
    targets = sim.targets.copy()
    targets[0] = 100
    with pytest.raises(ContractError, match="limits"):
        sim.send_joint_targets(targets, joint_names=sim.joint_names, deadline=1)
    with pytest.raises(ContractError, match="tabletop"):
        sim.reset(object_xy=[100, 0])
    with pytest.raises(RuntimeError, match="disabled"):
        sim.read()


def test_reset_objects_seed_and_scoring_are_explicit(sim):
    a = sim.reset(4)["object_position"]
    np.testing.assert_array_equal(a, sim.reset(4)["object_position"])
    assert not np.array_equal(a, sim.reset(5)["object_position"])
    assert not sim.task_truth()["placed"]
    sim.reset(object_xy=[0.48, -0.15], plate_xy=[0.48, -0.15], object_on_container=True)
    for _ in range(10):
        sim.send_joint_targets(sim.targets, joint_names=sim.joint_names, deadline=sim.data.time)
    assert sim.task_truth()["placed"]


@pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")
def test_rgb_is_real_repeatable_and_resettable():
    from embodied_jepa.embodiment import G1Embodiment

    sim = MuJoCoSimulation(width=64, height=64)
    try:
        embodiment = G1Embodiment(sim)
        embodiment.reset(9)
        first = embodiment.observe()
        assert first.images["onboard_rgb"].shape == (1, 64, 64, 3)
        assert first.images["onboard_rgb"].std() > 5
        np.testing.assert_array_equal(first.robot_state, embodiment.state().values)
        embodiment.reset(9)
        np.testing.assert_allclose(
            first.images["onboard_rgb"], embodiment.observe().images["onboard_rgb"], rtol=0, atol=1
        )
        assert all("object" not in n and "plate" not in n for n in embodiment.state_schema.names)
    finally:
        sim.close()


@pytest.mark.parametrize(
    "object_kind,container_kind",
    [("cube", "target"), ("banana", "bowl"), ("apple", "bowl"), ("cube", "plate")],
)
def test_procedural_pair_geometry_is_explicit(object_kind, container_kind):
    sim = MuJoCoSimulation(render=False, object_kind=object_kind, container_kind=container_kind)
    try:
        truth = sim.reset(3, object_kind=object_kind, container_kind=container_kind)
        assert truth["object_kind"] == object_kind and truth["container_kind"] == container_kind
        assert truth["position_frame"] == "world"
        assert sim.model.body("apple").mass[0] == pytest.approx(0.08)
        with pytest.raises(ContractError, match="fixed per simulator"):
            sim.reset(object_kind="invalid")
    finally:
        sim.close()


@pytest.mark.parametrize(
    "object_kind,container_kind",
    [
        ("apple", "bowl"),
        ("banana", "bowl"),
        ("cube", "target"),
        ("cube", "plate"),
        ("apple", "plate"),
    ],
)
def test_reset_rejects_penetration_and_supported_goal_has_correct_height(
    object_kind, container_kind
):
    sim = MuJoCoSimulation(render=False, object_kind=object_kind, container_kind=container_kind)
    try:
        # Every default scene loads without object/container contacts.
        initial = sim.data.qpos.copy()
        initial_plate = sim.model.body("plate").pos.copy()
        with pytest.raises(ContractError, match="overlaps collision envelopes"):
            sim.reset(object_xy=[0.43, -0.16], plate_xy=[0.43, -0.16])
        np.testing.assert_array_equal(initial, sim.data.qpos)
        np.testing.assert_array_equal(initial_plate, sim.model.body("plate").pos)
        truth = sim.reset(plate_xy=[0.48, -0.1], object_on_container=True)
        expected = sim.container_surface_z + sim.object_support_height + 0.001
        assert truth["object_position"][2] == pytest.approx(expected)
        apple = sim.model.geom("apple_geom").id
        assert all(
            c.dist >= -1e-8
            for c in sim.data.contact[: sim.data.ncon]
            if apple in (c.geom1, c.geom2)
        )
        with pytest.raises(ContractError, match="fit inside container"):
            sim.reset(object_xy=[0.52, -0.1], plate_xy=[0.48, -0.1], object_on_container=True)
    finally:
        sim.close()


def test_prior_overlapping_bowl_reset_is_rejected():
    sim = MuJoCoSimulation(render=False, object_kind="apple", container_kind="bowl")
    try:
        with pytest.raises(ContractError, match="overlaps collision envelopes"):
            sim.reset(object_xy=[0.34, -0.18], plate_xy=[0.43, -0.16])
    finally:
        sim.close()
