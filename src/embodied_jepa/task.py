"""Frozen task scoring. Privileged simulator truth never enters model or planner inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TaskThresholds:
    version: str = "tabletop_proxy_v0"
    reach_tolerance_m: float = 0.025
    dwell_s: float = 0.15
    lift_height_m: float = 0.05
    plate_radius_m: float = 0.04
    support_tolerance_m: float = 0.012
    resting_speed_m_s: float = 0.1

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name != "version" and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")


class ReachTask:
    def __init__(self, embodiment, target_base, thresholds=None):
        self.robot = embodiment
        self.target = np.array(target_base, dtype=float)
        if self.target.shape != (3,) or not np.isfinite(self.target).all():
            raise ValueError("reach target must be finite xyz in robot-base frame")
        self.thresholds = thresholds or TaskThresholds()
        self.inside_since = None
        self.last_time = -1.0

    def evaluate(self):
        now = float(self.robot.sim.data.time)
        if now < self.last_time:
            raise ValueError("reset the scorer after resetting the simulation")
        self.last_time = now
        error = float(np.linalg.norm(self.robot.ee_pose("right")[0] - self.target))
        if error <= self.thresholds.reach_tolerance_m:
            if self.inside_since is None:
                self.inside_since = now
        else:
            self.inside_since = None
        success = (
            self.inside_since is not None
            and now - self.inside_since >= self.thresholds.dwell_s - 1e-9
        )
        return {
            "reach": bool(success),
            "success": bool(success),
            "distance_m": error,
            "dwell_s": 0.0 if self.inside_since is None else now - self.inside_since,
        }


class AppleToPlateTask:
    """Require contact lift, transport, release and sustained support in order."""

    def __init__(self, embodiment, thresholds=None):
        self.robot = embodiment
        self.thresholds = thresholds or TaskThresholds()
        self.initial = embodiment.sim.task_truth()
        self.initial_height = float(self.initial["object_position"][2])
        self.stage = {key: False for key in ("reach", "grasp", "transport", "place", "release")}
        self.placed_since = None
        self.last_time = -1.0

    def evaluate(self):
        truth = self.robot.sim.task_truth()
        cfg = self.thresholds
        now = float(truth["timestamp"])
        if now < self.last_time:
            raise ValueError("reset the scorer after resetting the simulation")
        self.last_time = now
        obj, plate = np.asarray(truth["object_position"]), np.asarray(truth["plate_position"])
        base = np.asarray(truth["base_position_world"])
        rotation = np.asarray(truth["base_rotation_world"])
        ee_world = base + rotation @ self.robot.ee_pose("right")[0]
        # The EE site is on the palm; finger contacts can occur >6.5cm from it.
        near = np.linalg.norm(ee_world - obj) <= 0.065 or truth["hand_contact"]
        lifted = obj[2] - self.initial_height >= cfg.lift_height_m
        over_plate = np.linalg.norm(obj[:2] - plate[:2]) <= cfg.plate_radius_m
        self.stage["reach"] |= bool(near)
        self.stage["grasp"] |= bool(self.stage["reach"] and lifted and truth["hand_contact"])
        self.stage["transport"] |= bool(
            self.stage["grasp"] and over_plate and lifted and truth["hand_contact"]
        )
        expected_z = truth["container_surface_z"] + truth["object_support_height"]
        supported = abs(obj[2] - expected_z) <= cfg.support_tolerance_m
        resting = np.linalg.norm(truth["object_velocity"]) <= cfg.resting_speed_m_s
        released_on_plate = bool(
            self.stage["transport"]
            and over_plate
            and supported
            and resting
            and not truth["hand_contact"]
        )
        if released_on_plate:
            if self.placed_since is None:
                self.placed_since = now
        else:
            self.placed_since = None
        success = self.placed_since is not None and now - self.placed_since >= cfg.dwell_s - 1e-9
        self.stage["place"] |= bool(success)
        self.stage["release"] |= bool(success)
        return {
            **self.stage,
            "success": bool(success),
            "object_height_m": float(obj[2]),
            "object_plate_distance_m": float(np.linalg.norm(obj[:2] - plate[:2])),
            "hand_contact": bool(truth["hand_contact"]),
            "dropped": bool(truth["dropped"]),
            "support_dwell_s": 0.0 if self.placed_since is None else now - self.placed_since,
        }
