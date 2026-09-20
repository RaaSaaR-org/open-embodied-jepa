"""Headless control tests use explicit fixture RGB, never claim renderer coverage."""

import json
import time

import numpy as np
import pytest

pytest.importorskip("mujoco")
from embodied_jepa.contracts import ContractError
from embodied_jepa.embodiment import G1Embodiment
from embodied_jepa.simulation import ROOT, MuJoCoSimulation

pytestmark = pytest.mark.skipif(
    not (ROOT / "third_party/unitree_mujoco/unitree_robots/g1/g1_29dof_with_hand.xml").exists(),
    reason="pinned robot assets have not been fetched",
)


@pytest.fixture
def robot(monkeypatch):
    sim = MuJoCoSimulation(render=False)
    # Only replace RGB in this control unit test; qpos/qvel/time are actual physics.
    monkeypatch.setattr(sim, "render", lambda: np.zeros((8, 8, 3), dtype=np.uint8))
    embodiment = G1Embodiment(sim)
    embodiment.reset()
    yield embodiment
    embodiment.close()


def opened_action():
    a = np.zeros(14, dtype=np.float32)
    a[12:] = -1
    return a


def test_action_roundtrip_and_input_failures(robot):
    action = np.linspace(-1, 1, 14, dtype=np.float32)
    np.testing.assert_allclose(
        robot.normalize_action(robot.denormalize_action(action)), action, atol=1e-7
    )
    before = robot.sim.data.qpos.copy()
    with pytest.raises(ContractError):
        robot.execute(np.zeros(14, dtype=np.float64))
    bad = opened_action()
    bad[0] = np.nan
    with pytest.raises(ContractError):
        robot.execute(bad)
    np.testing.assert_array_equal(before, robot.sim.data.qpos)


def test_both_arms_progress_through_actual_torque_dynamics(robot):
    starts = {side: robot.ee_pose(side)[0].copy() for side in ("left", "right")}
    action = opened_action()
    action[2] = action[8] = -0.5
    for _ in range(6):
        robot.observe()
        result = robot.execute(action)
        assert result.status == "applied", result.reason
    for side in ("left", "right"):
        assert robot.ee_pose(side)[0][2] < starts[side][2] - 0.012
    assert np.isfinite(robot.sim.data.qpos).all()
    assert np.all(
        np.abs(robot.sim.data.ctrl) <= np.max(np.abs(robot.model.actuator_ctrlrange), axis=1)
    )


def test_both_arm_rotations_and_public_stop(robot):
    starts = {side: robot.ee_pose(side)[1].copy() for side in ("left", "right")}
    action = opened_action()
    action[5] = action[11] = 0.5
    for _ in range(5):
        robot.observe()
        result = robot.execute(action)
        assert result.status == "applied", result.reason
    for side in starts:
        assert np.linalg.norm(robot.ee_pose(side)[1] - starts[side]) > 0.05
    before = robot.sim.data.qpos.copy()
    robot.stop("planner deadline")
    np.testing.assert_array_equal(before, robot.sim.data.qpos)
    assert robot.sim.stopped_reason == "planner deadline"
    assert robot.execute(action).status == "stopped"


def test_mirrored_named_grasp_rate_and_snapshot_consumption(robot):
    action = opened_action()
    action[12:] = 1
    robot.observe()
    result = robot.execute(action)
    assert result.status == "clipped"
    assert np.all(result.applied_action[12:] < 1)
    left = robot.sim.targets[robot.sim.joint_names.index("left_hand_index_0_joint")]
    right = robot.sim.targets[robot.sim.joint_names.index("right_hand_index_0_joint")]
    assert left < 0 < right
    assert left == pytest.approx(-right)
    now = robot.sim.data.time
    rejected = robot.execute(action)
    assert rejected.status == "stopped" and "consumed" in rejected.reason
    assert rejected.applied_action is None and robot.sim.data.time == now


def test_ik_failure_and_stale_deadline_hold_without_teleport(robot, monkeypatch):
    robot.observe()
    before = robot.sim.data.qpos.copy()
    monkeypatch.setattr(robot, "solve_ik", lambda *args, **kwargs: None)
    result = robot.execute(opened_action())
    assert result.status == "stopped" and "IK" in result.reason
    np.testing.assert_array_equal(before, robot.sim.data.qpos)
    robot._observation_wall = time.monotonic() - 100
    assert "deadline" in robot.execute(opened_action()).reason
    robot.sim.data.time += 1
    assert "stale" in robot.execute(opened_action()).reason


def test_ik_scratch_never_changes_live_state_and_rejects_unreachable(robot):
    position, rotation = robot.ee_pose("right")
    before = robot.sim.data.qpos.copy()
    solution = robot.solve_ik("right", position + [0, 0, -0.01], rotation)
    assert solution is not None
    assert robot.solve_ik("right", [10, 10, 10], rotation) is None
    assert robot.solve_ik("right", position, np.zeros((3, 3))) is None
    np.testing.assert_array_equal(before, robot.sim.data.qpos)


def test_measured_velocity_and_joint_target_rate_limits_stop(robot, monkeypatch):
    robot.observe()
    robot.sim.data.qvel[robot.sim.vadr[0]] = 6
    assert "velocity limit" in robot.execute(opened_action()).reason
    robot.reset()
    robot.observe()
    solution = robot.sim.data.qpos[robot.model.jnt_qposadr[robot.arm_ids["left"]]].copy()
    solution[0] += 0.2
    monkeypatch.setattr(robot, "solve_ik", lambda *args, **kwargs: solution)
    assert "joint rate limit" in robot.execute(opened_action()).reason


def test_incompatible_manifest_fails_before_commands(robot, tmp_path):
    manifest = dict(robot.manifest)
    manifest["control_dt_s"] = 0.1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ContractError, match="timestep"):
        G1Embodiment(robot.sim, action_manifest=path)


def test_stop_resume_grasp_rate_uses_actual_transport_targets(robot):
    action = opened_action()
    action[12:] = 1
    robot.observe()
    assert robot.execute(action).applied_action is not None
    robot.stop("pause during closure")
    previous = robot.sim.targets.copy()
    robot.observe()
    result = robot.execute(action)
    if result.applied_action is not None:
        np.testing.assert_array_less(np.abs(robot.sim.targets - previous), 0.100001)
    else:
        assert "rate-feasible synergy" in result.reason


def test_ik_does_not_accept_opposite_orientation_as_zero_error(robot):
    from embodied_jepa.embodiment import rotation_delta

    position, rotation = robot.ee_pose("right")
    initial = robot.sim.data.qpos[robot.model.jnt_qposadr[robot.arm_ids["right"]]].copy()
    result = robot.solve_ik("right", position, rotation_delta([np.pi, 0, 0]) @ rotation)
    assert result is None or not np.allclose(result, initial)
