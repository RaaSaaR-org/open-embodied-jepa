"""TASK-068 at-rest success check: a separately named, end-of-episode check.

It does not replace or change ``task.AppleToPlateTask`` (thresholds ``tabletop_proxy_v0``),
whose latched ``place``/``release`` stages stay the default scorer. It reads simulator truth for
scoring only; nothing here enters a controller's inputs. NumPy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AtRestThresholds:
    """TASK-068 at-rest success: a separate, end-of-episode check. It does not replace or change
    ``AppleToPlateTask``, whose latched ``place``/``release`` stages remain the default scorer.

    The radius and the support tolerance are the default scorer's (``tabletop_proxy_v0``). The
    speed bar is far below the scorer's 0.1 m/s because the apple is a sphere with condim-3
    contacts: it has no rolling resistance, so a rolling apple does not slow down on the plate
    (a 0.002 m/s roll persisted unchanged for 10 s in the TASK-068 check) and ends at the rim.
    """

    version: str = "apple_at_rest_v0"
    plate_radius_m: float = 0.04
    support_tolerance_m: float = 0.012
    max_speed_m_s: float = 0.001
    window_steps: int = 20  # the final control steps (1.0 s at 20 Hz), all inside the settle
    settle_steps: int = 60  # the harness's settle after the expert ends: arm still, hands open

    def __post_init__(self):
        for name in ("plate_radius_m", "support_tolerance_m", "max_speed_m_s"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("window_steps", "settle_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.window_steps > self.settle_steps:
            raise ValueError("the at-rest window must lie inside the settle period")


def apple_at_rest(
    object_position,
    plate_position,
    object_velocity,
    hand_contact,
    *,
    expected_z: float,
    thresholds: AtRestThresholds | None = None,
) -> dict:
    """At rest on the plate over the final ``window_steps`` steps of per-step truth arrays.

    Every step in the window must have the apple centre within ``plate_radius_m`` of the plate
    centre (world xy), its height within ``support_tolerance_m`` of the resting height
    ``expected_z``, its linear speed at most ``max_speed_m_s``, and no hand contact. Scoring
    only: these are simulator truth and never enter a controller's inputs.
    """
    cfg = thresholds or AtRestThresholds()
    obj = np.asarray(object_position, float)
    plate = np.asarray(plate_position, float)
    velocity = np.asarray(object_velocity, float)
    contact = np.asarray(hand_contact, bool)
    steps = len(obj)
    if (
        obj.ndim != 2
        or obj.shape[1] != 3
        or plate.shape != obj.shape
        or velocity.shape != obj.shape
        or contact.shape != (steps,)
    ):
        raise ValueError("at-rest arrays must be [T, 3] positions/velocities and [T] contacts")
    if steps < cfg.window_steps:
        raise ValueError(f"at-rest needs at least {cfg.window_steps} recorded steps")
    if not (np.isfinite(obj).all() and np.isfinite(plate).all() and np.isfinite(velocity).all()):
        raise ValueError("at-rest arrays must be finite")
    window = slice(steps - cfg.window_steps, steps)
    distance = np.linalg.norm(obj[window, :2] - plate[window, :2], axis=1)
    height_error = np.abs(obj[window, 2] - float(expected_z))
    speed = np.linalg.norm(velocity[window], axis=1)
    inside = distance <= cfg.plate_radius_m
    supported = height_error <= cfg.support_tolerance_m
    still = speed <= cfg.max_speed_m_s
    free = ~contact[window]
    return {
        "version": cfg.version,
        "at_rest": bool(np.all(inside & supported & still & free)),
        "inside_all": bool(inside.all()),
        "supported_all": bool(supported.all()),
        "still_all": bool(still.all()),
        "no_hand_contact_all": bool(free.all()),
        "final_distance_m": float(distance[-1]),
        "max_distance_m": float(distance.max()),
        "max_speed_m_s": float(speed.max()),
        "max_height_error_m": float(height_error.max()),
        "window_steps": cfg.window_steps,
    }


class AppleAtRestCheck:
    """Records the truth ``apple_at_rest`` needs, one call per control step, for the end verdict.

    Construct it after the reset; call ``record()`` after every executed step (including the
    settle) and ``verdict()`` once at the end. It never alters ``AppleToPlateTask``."""

    def __init__(self, embodiment, thresholds: AtRestThresholds | None = None):
        self.robot = embodiment
        self.thresholds = thresholds or AtRestThresholds()
        self.rows: dict[str, list] = {k: [] for k in ("obj", "plate", "vel", "contact")}
        self.expected_z: float | None = None
        self.last_time = -1.0

    def record(self) -> None:
        truth = self.robot.sim.task_truth()
        now = float(truth["timestamp"])
        if now <= self.last_time:
            raise ValueError("record once per executed step, after resetting the simulation")
        self.last_time = now
        self.expected_z = float(truth["container_surface_z"] + truth["object_support_height"])
        self.rows["obj"].append(np.asarray(truth["object_position"], float).copy())
        self.rows["plate"].append(np.asarray(truth["plate_position"], float).copy())
        self.rows["vel"].append(np.asarray(truth["object_velocity"], float).copy())
        self.rows["contact"].append(bool(truth["hand_contact"]))

    def verdict(self) -> dict:
        if self.expected_z is None:
            raise ValueError("no step was recorded")
        return apple_at_rest(
            np.asarray(self.rows["obj"]),
            np.asarray(self.rows["plate"]),
            np.asarray(self.rows["vel"]),
            np.asarray(self.rows["contact"]),
            expected_z=self.expected_z,
            thresholds=self.thresholds,
        )
