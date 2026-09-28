"""TASK-068 resting expert and harness pieces that run without MuJoCo."""

import numpy as np
import pytest

from embodied_jepa import first_policy as fp
from embodied_jepa import resting_expert as rx
from embodied_jepa.contracts import ContractError
from embodied_jepa.resting_expert import RIGHT_SHOULDER_BASE, RestingPlaceExpert

TRUTH = {
    "position_frame": "world",
    "base_position_world": np.zeros(3),
    "base_rotation_world": np.eye(3),
    "object_position": np.array([0.34, -0.18, 0.77]),
    "plate_position": np.array([0.49, -0.09, 0.746]),
    "container_surface_z": 0.752,
    "object_support_height": 0.027,
}


def test_development_seeds_avoid_every_spent_or_reserved_range():
    rx.check_seeds(rx.DEV_SEEDS)
    assert rx.DEV_SEEDS == tuple(range(50000, 50100))


@pytest.mark.parametrize("seed", [46000, 46415, 46850, 46999, 47050, 45300, 45003, 48500, 49100])
def test_check_seeds_refuses_task067_block_corpora_and_cohorts(seed):
    with pytest.raises(fp.GuardError):
        rx.check_seeds((seed,))


def test_check_seeds_refuses_overlap_with_another_declared_range():
    with pytest.raises(fp.GuardError, match="other_0"):
        rx.check_seeds((50099, 50100), others=(rx.DEV_SEEDS,))
    with pytest.raises(fp.GuardError, match="distinct"):
        rx.check_seeds((50100, 50100))


def test_plate_offsets_have_the_declared_size_and_fixed_directions():
    offsets = rx.plate_offsets(32, 1.0, rx.DEV_DIRECTION_SEED)
    assert np.allclose(np.linalg.norm(offsets, axis=1), 0.01)
    assert np.array_equal(offsets, rx.plate_offsets(32, 1.0, rx.DEV_DIRECTION_SEED))
    # the same direction per seed at every level (paired conditions)
    assert np.allclose(rx.plate_offsets(32, 1.5, rx.DEV_DIRECTION_SEED), 1.5 * offsets)
    assert not np.any(rx.plate_offsets(32, 0.0, rx.DEV_DIRECTION_SEED))


def test_release_pose_is_on_the_reach_sphere_or_its_floor():
    pose = RestingPlaceExpert.release_pose(
        [0.46, -0.09], reach_radius_m=0.485, floor=0.10, ceiling=0.26
    )
    assert pose[:2] == pytest.approx([0.46, -0.09])
    assert np.linalg.norm(pose - RIGHT_SHOULDER_BASE) == pytest.approx(0.485)
    near = RestingPlaceExpert.release_pose(
        [0.30, -0.09], reach_radius_m=0.485, floor=0.10, ceiling=0.26
    )
    assert near[2] == 0.10  # the sphere would allow lower; the floor binds


def test_release_pose_pulls_x_back_when_the_xy_is_beyond_reach():
    pose = RestingPlaceExpert.release_pose(
        [0.52, -0.09], reach_radius_m=0.485, floor=0.10, ceiling=0.26
    )
    assert pose[0] < 0.52 and pose[2] == pytest.approx(0.26)
    assert np.linalg.norm(pose - RIGHT_SHOULDER_BASE) == pytest.approx(0.485)


def test_expert_keeps_the_collector_pick_and_fits_the_budget():
    from embodied_jepa.scripted import apple_collector_policy

    expert = RestingPlaceExpert(TRUTH)
    collector = apple_collector_policy(TRUTH)
    names = [p.name for p in expert.phases]
    assert names == [
        "orient",
        "descend",
        "close",
        "lift",
        "transfer",
        "lower",
        "steady",
        "open",
        "clear",
        "retreat",
    ]
    for mine, theirs in zip(expert.phases[:4], collector.phases[:4], strict=True):
        assert np.allclose(mine.target_base, theirs.target_base)
        assert (mine.grasp, mine.commands) == (theirs.grasp, theirs.commands)
    assert expert.max_steps <= rx.EXPERT_BUDGET
    assert rx.EXPERT_BUDGET + rx.SETTLE_STEPS == fp.MAX_POLICY_STEPS
    release = expert.release_target
    assert release[0] == pytest.approx(0.49 - 0.015)  # the held apple over the believed centre
    assert all(np.allclose(p.target_base, release) for p in expert.phases[5:9])


def test_expert_rejects_bad_parameters():
    with pytest.raises(ContractError):
        RestingPlaceExpert(TRUTH, opening_ramp=0.0)
    with pytest.raises(ContractError):
        RestingPlaceExpert(TRUTH, release_z_floor_m=0.3, release_z_ceiling_m=0.2)
    with pytest.raises(ContractError):
        RestingPlaceExpert(TRUTH, closure=-1.0)


class _Robot:
    manifest = {"translation_per_step_m": 0.015, "rotation_per_step_rad": 0.06}

    def __init__(self, position):
        self.position = np.asarray(position, float)

    def ee_pose(self, side):
        from embodied_jepa.embodiment import rotation_delta

        return self.position, rotation_delta([-np.pi / 2, 0, 0])


class _Result:
    def __init__(self, action):
        self.applied_action = action
        self.reason = ""


def test_expert_opens_on_its_ramp_from_the_accepted_grasp():
    expert = RestingPlaceExpert(TRUTH, opening_ramp=0.25)
    robot = _Robot(expert.release_target)
    grasps = []
    while not expert.done:
        action = expert.action(robot)
        grasps.append((expert.phases[expert.phase_index].name, float(action[13])))
        expert.advance(_Result(action))
    opening = [g for name, g in grasps if name == "open"]
    assert opening[:5] == pytest.approx([0.75, 0.5, 0.25, 0.0, -0.25])
    assert all(g == 1.0 for name, g in grasps if name in ("transfer", "lower", "steady"))
    assert grasps[-1] == ("retreat", -1.0)


def test_release_events_read_open_last_hand_and_landing_at_known_steps():
    steps = 12
    names = ["transfer", "open", "clear"]
    phase = np.array([0] * 4 + [1] * 4 + [2] * 4)
    counts = np.zeros((steps, len(rx.PAIR_TYPES)), int)
    hand = rx.PAIR_TYPES.index("apple_hand")
    base_contact = rx.PAIR_TYPES.index("apple_plate_base")
    rim = rx.PAIR_TYPES.index("apple_plate_rim")
    counts[:6, hand] = 1  # hand contact until step 5
    counts[8:, base_contact] = 1  # first plate-base contact at step 8
    counts[10, rim] = 1
    plate = np.array([0.49, -0.09, 0.746])
    rest_z = 0.779
    apple = np.tile([0.49, -0.09, rest_z], (steps, 1))
    apple[:8, 2] = rest_z + 0.10  # 10 cm above rest while held and falling
    apple[8:, 0] += 0.01
    lin = np.zeros((steps, 3))
    lin[6] = [0.2, 0.0, -1.0]  # just after the last hand contact
    lin[8] = [0.03, 0.04, 0.0]  # at landing
    arrays = {
        "phase": phase,
        "counts": counts,
        "palm": np.tile([0.475, -0.09, 0.2], (steps, 1)),
        "apple_pos": apple,
        "apple_lin": lin,
        "apple_ang": np.zeros((steps, 3)),
    }
    out = rx.release_events(arrays, names, plate, np.zeros(3), rest_z)
    assert out["open_step"] == 4 and out["last_apple_hand_step"] == 5
    assert out["apple_height_above_rest_at_open_cm"] == pytest.approx(10.0)
    assert out["apple_horizontal_speed_after_hand_m_s"] == pytest.approx(0.2)
    assert out["first_base_contact_step"] == 8
    assert out["landing_xy_minus_plate_cm"] == pytest.approx([1.0, 0.0])
    assert out["horizontal_speed_at_landing_m_s"] == pytest.approx(0.05)
    assert out["rim_contact_after_open"] and out["hand_plate_contact_steps"] == 0
