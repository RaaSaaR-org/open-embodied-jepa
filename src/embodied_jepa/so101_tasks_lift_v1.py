"""``so101_cube_lift_v1`` (TASK-101 amendment A1, ``docs/experiments/so101_closed_loop_lift_v1.md``
§18.2): ``so101_cube_lift_v0`` with its inert target marker drawn per reset from the play corpus's
marker distribution instead of v0's fixed point x = 0.30 m, y = -0.20 m.

* **Cube and arm:** v0's draws, in v0's order, from ``default_rng([9801, seed])``
  (``so101_tasks.sample_reset``), so for every seed v1's cube pose equals v0's bit for bit; the
  arm starts at ``home_rad``.
* **Marker:** from its own stream ``default_rng([10107, seed])``: x uniform on [0.16, 0.22] m,
  y uniform on [-0.14, 0.14] m, redrawn until its centre is at least 0.08 m from the cube centre
  (at most 1 000 draws, else an error). The box and the distance are the corpus's own constants,
  ``so101_play_corpus.TARGET_REGION`` and ``MIN_CUBE_TARGET_M``, imported, not copied, and the
  rule is ``so101_play_corpus.sample_reset``'s: uniform on the box (x, then y), redrawn on the
  distance.
* **Scorer, thresholds, step budget, scene, cameras and embodiment:** v0's, unchanged
  (``LIFT_V0``, ``lift_outcome``); the marker has ``contype = conaffinity = 0`` and plays no part
  in lift scoring.

NumPy only at import. No registry entry: ``config.py`` is byte-pinned by G1 manifests, and the run
configuration names this task as a module constant (TASK-066's and TASK-100's precedent).
``so101_tasks.py`` and ``so101_simulation.py`` (pinned by ``so101-play-v1``) are unchanged.
"""

from __future__ import annotations

import numpy as np

from embodied_jepa import so101_play_corpus as play
from embodied_jepa import so101_tasks as tasks

LIFT_V1_TASK = "so101_cube_lift_v1"
MARKER_SALT = 10107
MARKER_REGION = play.TARGET_REGION
MIN_CUBE_MARKER_M = play.MIN_CUBE_TARGET_M
MAX_MARKER_DRAWS = 1000
V0_MARKER_XY = (0.30, -0.20)


def sample_marker(seed: int, cube_xy) -> list[float]:
    """The marker for ``seed`` given its cube: the corpus's rule on its own stream (salt 10107)."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cube = np.asarray(cube_xy, dtype=float)
    if cube.shape != (2,) or not np.isfinite(cube).all():
        raise ValueError("cube_xy must be a finite pair")
    rng = np.random.default_rng([MARKER_SALT, seed])
    for _ in range(MAX_MARKER_DRAWS):
        marker = np.array([rng.uniform(*MARKER_REGION[0]), rng.uniform(*MARKER_REGION[1])])
        if np.linalg.norm(marker - cube) >= MIN_CUBE_MARKER_M:
            return marker.tolist()
    raise ValueError("could not draw a marker far enough from the cube")


def sample_reset_lift_v1(
    seed: int, distribution: tasks.ResetDistribution = tasks.RESETS_V0
) -> dict:
    """v0's declared lift reset for ``seed`` with ``target_xy`` replaced by the v1 marker."""
    spec = tasks.sample_reset(tasks.LIFT_TASK, seed, distribution)
    spec["target_xy"] = sample_marker(seed, spec["cube_xy"])
    return spec


class SO101CubeLiftTaskV1(tasks.SO101CubeLiftTask):
    """``so101_cube_lift_v1``: v0's lift task; only the name and the reset's marker differ."""

    name = LIFT_V1_TASK

    def sample_reset(self, seed: int) -> dict:
        return sample_reset_lift_v1(seed, self.reset_distribution)
