from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa.task import AppleToPlateTask, ReachTask


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
