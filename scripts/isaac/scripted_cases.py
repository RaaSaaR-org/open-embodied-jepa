"""Scripted physics checks shared by the Isaac and MuJoCo sides (development, TASK-025).

NumPy only. No learned policy is involved: each case is a fixed apple start state and a fixed
joint-target sequence (the robot holds its reset pose except in ``press``). These are physics
parity checks of the ``apple-to-plate-v2`` scene objects, not manipulation results, and a
scripted joint trajectory touching the apple is not a learned or task result.

Cases (0.05 s control intervals, the transports' own):

* ``drop_table``: apple released at rest 10 cm above its table resting height at (0.40, -0.26);
* ``drop_plate_centre``: apple released 10 cm above its resting height over the plate centre;
* ``drop_plate_offcentre``: the same 5 cm off the plate centre in +x, so it lands on the base
  next to the rim capsules;
* ``roll_table``: apple on the table at (0.30, -0.26) given 0.2 m/s in +x with the matching
  rolling spin (0.2 / 0.027 rad/s about +y); v2's rolling friction should stop it in MuJoCo;
* ``press``: the right arm moves to shoulder pitch -0.5, roll -0.25, wrist roll -0.3, elbow 0.6
  (intervals 0-20), then the elbow ramps to 0.95 (intervals 30-60), pressing the right middle
  finger on an apple resting at (0.471, -0.233); the plate is moved away to (0.60, 0.20).
"""

from __future__ import annotations

import numpy as np

APPLE_R = 0.027
TABLE_REST_Z = 0.74 + APPLE_R
PLATE_REST_Z = 0.752 + APPLE_R
DEFAULT_PLATE = (0.48, -0.10)
SPEED_AT_REST = 0.001  # m/s, the apple_at_rest_v0 speed bar


def _ramp(k: int, start: int, length: int) -> float:
    x = min(max((k - start) / length, 0.0), 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


CASES = {
    "drop_table": {
        "plate_xy": DEFAULT_PLATE,
        "reset_object_xy": (0.40, -0.26),
        "start_pos": (0.40, -0.26, TABLE_REST_Z + 0.10),
        "lin_vel": (0.0, 0.0, 0.0),
        "ang_vel": (0.0, 0.0, 0.0),
        "intervals": 60,
    },
    "drop_plate_centre": {
        "plate_xy": DEFAULT_PLATE,
        "reset_object_xy": (0.40, -0.26),
        "start_pos": (0.48, -0.10, PLATE_REST_Z + 0.10),
        "lin_vel": (0.0, 0.0, 0.0),
        "ang_vel": (0.0, 0.0, 0.0),
        "intervals": 60,
    },
    "drop_plate_offcentre": {
        "plate_xy": DEFAULT_PLATE,
        "reset_object_xy": (0.40, -0.26),
        "start_pos": (0.53, -0.10, PLATE_REST_Z + 0.10),
        "lin_vel": (0.0, 0.0, 0.0),
        "ang_vel": (0.0, 0.0, 0.0),
        "intervals": 60,
    },
    "roll_table": {
        "plate_xy": DEFAULT_PLATE,
        "reset_object_xy": (0.30, -0.26),
        "start_pos": (0.30, -0.26, TABLE_REST_Z),
        "lin_vel": (0.2, 0.0, 0.0),
        "ang_vel": (0.0, 0.2 / APPLE_R, 0.0),
        "intervals": 60,
    },
    "press": {
        "plate_xy": (0.60, 0.20),
        "reset_object_xy": (0.471, -0.233),
        "start_pos": None,  # the reset placement, settled during the arm's approach
        "lin_vel": (0.0, 0.0, 0.0),
        "ang_vel": (0.0, 0.0, 0.0),
        "intervals": 80,
    },
}


def targets(case: str, names, reset_q) -> np.ndarray:
    """Joint targets per interval (canonical order) for ``case``."""
    reset_q = np.asarray(reset_q, dtype=float)
    n = CASES[case]["intervals"]
    if case != "press":
        return np.tile(reset_q, (n, 1))
    idx = {name: i for i, name in enumerate(names)}
    out = []
    for k in range(n):
        q = reset_q.copy()
        a, b = _ramp(k, 0, 20), _ramp(k, 30, 30)
        q[idx["right_shoulder_pitch_joint"]] = -0.5 * a
        q[idx["right_shoulder_roll_joint"]] = -0.25 * a
        q[idx["right_wrist_roll_joint"]] = -0.3 * a
        e0 = reset_q[idx["right_elbow_joint"]]
        q[idx["right_elbow_joint"]] = e0 + (0.6 - e0) * a + (0.95 - 0.6) * b
        out.append(q)
    return np.array(out)


def summarise(times, pos, lin_vel, contacts) -> dict:
    """Resting position, settle time and contact history of one simulator's case trace."""
    times = np.asarray(times, dtype=float)
    pos = np.asarray(pos, dtype=float)
    speed = np.linalg.norm(np.asarray(lin_vel, dtype=float), axis=1)
    above = np.flatnonzero(speed > SPEED_AT_REST)
    if above.size == 0:
        settle = float(times[0])
    elif above[-1] == speed.size - 1:
        settle = None
    else:
        settle = float(times[above[-1] + 1])
    pairs: dict[str, dict] = {}
    for k, rows in enumerate(contacts):
        for r in rows:
            if "apple" not in r["bodies"]:
                continue
            key = "|".join(r["bodies"])
            p = pairs.setdefault(key, {"first_read": k, "reads": 0, "max_force_n": 0.0})
            p["reads"] += 1
            p["max_force_n"] = max(p["max_force_n"], float(r["force_n"]))
    return {
        "final_pos": pos[-1].tolist(),
        "final_speed_m_s": float(speed[-1]),
        "settle_time_s": settle,
        "left_table": bool(pos[:, 2].min() < 0.70),
        "max_z": float(pos[:, 2].max()),
        "path_length_xy_m": float(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum()),
        "apple_contact_pairs": pairs,
    }
