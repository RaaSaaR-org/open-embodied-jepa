"""Deterministic joint-target trajectory for the Isaac/MuJoCo transport parity check.

NumPy only; used by ``parity_isaac.py`` (container) and checked by ``parity_mujoco.py``
(host). Targets are in the joint manifest's canonical order. Phases, 0.05 s per interval:

* 0-9: hold the reset pose (gravity hold / settling);
* 10-49: both arms, smooth cosine ramp to shoulder pitch -0.3, shoulder roll 0.25 outward,
  elbow +0.6 and wrist roll +-0.3 (legs and waist held at 0);
* 50-69: both Dex3 hands ramp from the reset pose (all finger joints 0) to the closed grasp
  synergy (``configs/g1_sim_action.json``), arms held;
* 70-89: hands and arms return to the reset pose.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FINGERS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")


def ramp(k: int, start: int, length: int) -> float:
    x = min(max((k - start) / length, 0.0), 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


def trajectory(names: list[str], reset_q: np.ndarray, action_manifest: Path) -> np.ndarray:
    action = json.loads(Path(action_manifest).read_text())
    index = {n: i for i, n in enumerate(names)}
    arm = np.zeros(len(names))  # deltas from the reset pose
    hand_closed = np.array(reset_q, dtype=float)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        arm[index[f"{side}_shoulder_pitch_joint"]] = -0.3
        arm[index[f"{side}_shoulder_roll_joint"]] = 0.25 * sign
        arm[index[f"{side}_elbow_joint"]] = 0.6
        arm[index[f"{side}_wrist_roll_joint"]] = 0.3 * sign
        for f, closed in zip(FINGERS, action[f"{side}_closed_rad"], strict=True):
            hand_closed[index[f"{side}_hand_{f}_joint"]] = closed
    out = []
    for k in range(90):
        a = ramp(k, 10, 40) - ramp(k, 70, 20)
        h = ramp(k, 50, 20) - ramp(k, 70, 20)
        q = reset_q + a * arm + h * (hand_closed - reset_q)
        out.append(q)
    return np.array(out)
