from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa.task import (
    AppleAtRestCheck,
    AppleToPlateTask,
    AtRestThresholds,
    ReachTask,
    TaskThresholds,
    apple_at_rest,
)


class ScoringRobot:
    def __init__(self):
        self.sim = SimpleNamespace(data=SimpleNamespace(time=0.0), task_truth=self.truth)
        self.object = np.array([0.3, -0.1, 0.77])
        self.plate = np.array([0.4, -0.1, 0.746])
        self.ee = self.object.copy()
        self.contact = False

    def ee_pose(self, side):
        return self.ee, np.eye(3)

    def truth(self):
        return {
            "timestamp": self.sim.data.time,
            "object_position": self.object.copy(),
            "plate_position": self.plate.copy(),
            "base_position_world": np.zeros(3),
            "base_rotation_world": np.eye(3),
            "hand_contact": self.contact,
            "object_support_height": 0.027,
            "container_surface_z": 0.752,
            "object_velocity": np.zeros(3),
            "dropped": False,
        }


def test_reach_requires_continuous_dwell_and_resets_on_exit():
    robot = ScoringRobot()
    scorer = ReachTask(robot, robot.ee)
    assert not scorer.evaluate()["success"]
    robot.sim.data.time = 0.1
    assert not scorer.evaluate()["success"]
    robot.ee += [0.1, 0, 0]
    robot.sim.data.time = 0.2
    assert not scorer.evaluate()["success"]
    robot.ee -= [0.1, 0, 0]
    robot.sim.data.time = 0.3
    assert not scorer.evaluate()["success"]
    robot.sim.data.time = 0.5
    assert scorer.evaluate()["success"]
    robot.sim.data.time = 0
    with pytest.raises(ValueError, match="reset"):
        scorer.evaluate()


def test_initially_on_plate_cannot_skip_lift_transport_and_release():
    robot = ScoringRobot()
    robot.object = np.array([0.4, -0.1, 0.779])
    scorer = AppleToPlateTask(robot)
    for t in [0, 1, 2]:
        robot.sim.data.time = t
        assert not scorer.evaluate()["success"]


def test_full_success_requires_contact_lift_then_transport_and_release_dwell():
    robot = ScoringRobot()
    scorer = AppleToPlateTask(robot)
    scorer.evaluate()
    robot.object[2] = 0.85
    robot.contact = True
    robot.sim.data.time = 0.1
    assert scorer.evaluate()["grasp"]
    robot.object[0] = 0.4
    robot.sim.data.time = 0.2
    assert scorer.evaluate()["transport"]
    robot.object[2] = 0.779
    robot.contact = False
    robot.sim.data.time = 0.3
    assert not scorer.evaluate()["success"]
    robot.sim.data.time = 0.46
    assert scorer.evaluate()["success"]
    robot.contact = True
    robot.sim.data.time = 0.5
    assert not scorer.evaluate()["success"]


def test_finger_contact_counts_as_reach_despite_palm_offset():
    robot = ScoringRobot()
    robot.ee = robot.object + np.array([0.0, 0.0, 0.11])
    scorer = AppleToPlateTask(robot)
    robot.object[2] += 0.08
    robot.ee[2] += 0.08
    robot.contact = True
    score = scorer.evaluate()
    assert score["reach"] and score["grasp"]


# ----- TASK-068 at-rest check -----------------------------------------------------------------
REST_Z = 0.752 + 0.027


def _rest_arrays(steps=30, *, offset=(0.01, 0.0), speed=0.0, contact_at=None, z=REST_Z):
    plate = np.tile([0.49, -0.09, 0.746], (steps, 1))
    obj = plate.copy()
    obj[:, :2] += offset
    obj[:, 2] = z
    velocity = np.zeros((steps, 3))
    velocity[:, 0] = speed
    contact = np.zeros(steps, bool)
    if contact_at is not None:
        contact[contact_at] = True
    return obj, plate, velocity, contact


def test_at_rest_defaults_keep_the_scorer_radius_and_support_tolerance():
    rest, scorer = AtRestThresholds(), TaskThresholds()
    assert rest.plate_radius_m == scorer.plate_radius_m == 0.04
    assert rest.support_tolerance_m == scorer.support_tolerance_m
    assert rest.version == "apple_at_rest_v0" and scorer.version == "tabletop_proxy_v0"
    assert rest.window_steps <= rest.settle_steps


def test_at_rest_passes_a_still_supported_apple_inside_the_radius():
    verdict = apple_at_rest(*_rest_arrays(), expected_z=REST_Z)
    assert verdict["at_rest"] and verdict["max_speed_m_s"] == 0.0
    assert verdict["final_distance_m"] == pytest.approx(0.01)


@pytest.mark.parametrize(
    "kwargs, failed",
    [
        ({"offset": (0.041, 0.0)}, "inside_all"),  # just outside 4 cm (the rim ring is ~4.6 cm)
        ({"speed": 0.002}, "still_all"),  # a slow roll is not rest
        ({"contact_at": 25}, "no_hand_contact_all"),
        ({"z": REST_Z + 0.02}, "supported_all"),  # held above the plate
    ],
)
def test_at_rest_fails_each_condition_on_its_own(kwargs, failed):
    verdict = apple_at_rest(*_rest_arrays(**kwargs), expected_z=REST_Z)
    assert not verdict["at_rest"] and not verdict[failed]


def test_at_rest_reads_only_the_final_window():
    obj, plate, velocity, contact = _rest_arrays(steps=40)
    velocity[:20, 0] = 0.5  # rolling fast before the window: irrelevant
    obj[:20, 0] += 0.1  # and outside the radius before the window: irrelevant
    contact[:20] = True
    assert apple_at_rest(obj, plate, velocity, contact, expected_z=REST_Z)["at_rest"]
    velocity[20, 0] = 0.0011  # the first step of the window counts
    assert not apple_at_rest(obj, plate, velocity, contact, expected_z=REST_Z)["at_rest"]


def test_at_rest_is_not_a_latch_a_transient_crossing_fails():
    obj, plate, velocity, contact = _rest_arrays(steps=40, offset=(0.046, 0.0), speed=0.01)
    obj[25:28, 0] -= 0.02  # passes within 4 cm mid-window, still rolling
    verdict = apple_at_rest(obj, plate, velocity, contact, expected_z=REST_Z)
    assert not verdict["at_rest"] and not verdict["inside_all"]


def test_at_rest_rejects_short_or_malformed_input():
    obj, plate, velocity, contact = _rest_arrays(steps=19)
    with pytest.raises(ValueError, match="at least 20"):
        apple_at_rest(obj, plate, velocity, contact, expected_z=REST_Z)
    obj, plate, velocity, contact = _rest_arrays()
    with pytest.raises(ValueError, match=r"\[T, 3\]"):
        apple_at_rest(obj[:, :2], plate, velocity, contact, expected_z=REST_Z)
    velocity[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        apple_at_rest(obj, plate, velocity, contact, expected_z=REST_Z)


@pytest.mark.parametrize(
    "kwargs",
    [{"max_speed_m_s": 0.0}, {"window_steps": 0}, {"window_steps": 61}, {"settle_steps": 1.5}],
)
def test_at_rest_thresholds_are_validated(kwargs):
    with pytest.raises(ValueError):
        AtRestThresholds(**kwargs)


def test_at_rest_check_records_truth_per_step_and_never_touches_the_scorer():
    robot = ScoringRobot()
    robot.object = np.array([0.41, -0.1, REST_Z])
    scorer = AppleToPlateTask(robot)
    check = AppleAtRestCheck(robot)
    with pytest.raises(ValueError, match="no step"):
        check.verdict()
    for step in range(1, 21):
        robot.sim.data.time = step * 0.05
        check.record()
    assert check.verdict()["at_rest"]
    assert not any(scorer.stage.values())  # the default scorer is untouched
    with pytest.raises(ValueError, match="once per executed step"):
        check.record()  # same timestamp again
