"""TASK-069: development-only feasibility scan toward an ``apple-to-plate-v2`` task definition.

No gate, nothing frozen. Privileged scripted engineering, not a learned result. This module holds
the scan's declared seeds, the two candidate task changes it can apply without touching the v1
code (``simulation.py``, ``task.py`` and ``scripted.py`` stay byte-identical to ``main``), and
the reset distribution for plates inside the set-down region. NumPy only at import.

* **(b3) apple rolling friction.** ``apply_apple_friction`` edits the compiled MuJoCo model of one
  simulator instance in place: the apple geom's contact dimensionality (``condim``) and its
  torsional and rolling friction coefficients. The v1 scene declares ``friction="1 .01 .001"`` on
  the apple and the plate base but leaves ``condim`` at MuJoCo's default 3, so only the sliding
  coefficient acts. ``condim`` 6 makes the declared torsional (0.01 m) and rolling (0.001 m)
  coefficients act; MuJoCo uses the larger condim of the two geoms in a contact.
* **(b2) plates in the set-down region.** ``setdown_reset`` keeps the v1 apple distribution
  (TASK-047 wide jitter) and samples the plate centre uniformly inside a declared box, measured
  with ``scripts/measure_v2_reach.py``, rejecting draws that are too close to the apple.
"""

from __future__ import annotations

import numpy as np

from embodied_jepa import resting_expert as rx

# Declared on the TASK-069 card before any TASK-069 episode ran.
DEV_SEEDS = tuple(range(50100, 50200))
DEV_DIRECTION_SEED = 6840
PLATE_LEVELS_CM = (0.0, 1.0)
APPLE_CENTER = (0.34, -0.18)  # scripts/evaluate_apple.py RESET_CENTERS, TASK-047
APPLE_JITTER_M = 0.03
PLATE_CENTER_V1 = (0.49, -0.09)
PLATE_JITTER_V1_M = 0.02
# Reset contract (simulation.reset): plate radius 0.071 + apple 0.027 + 0.005 margin.
MIN_SEPARATION_M = 0.071 + 0.027 + 0.005


def check_seeds(seeds) -> None:
    rx.check_seeds(seeds, others=(rx.DEV_SEEDS,))


def apply_apple_friction(model, *, condim: int = 6, torsional=0.01, rolling=0.001) -> dict:
    """(b3) Set the apple geom's condim and torsional/rolling friction in a compiled model."""
    if condim not in (1, 3, 4, 6):
        raise ValueError("condim must be 1, 3, 4 or 6")
    for value in (torsional, rolling):
        if not np.isfinite(value) or value < 0:
            raise ValueError("friction coefficients must be finite and non-negative")
    geom = model.geom("apple_geom").id
    before = {
        "condim": int(model.geom_condim[geom]),
        "friction": np.asarray(model.geom_friction[geom], float).tolist(),
    }
    model.geom_condim[geom] = condim
    model.geom_friction[geom, 1] = torsional
    model.geom_friction[geom, 2] = rolling
    after = {
        "condim": int(model.geom_condim[geom]),
        "friction": np.asarray(model.geom_friction[geom], float).tolist(),
    }
    return {"before": before, "after": after}


def v1_reset(seed: int) -> dict:
    """The TASK-047 wide-jitter reset (``scripts/evaluate_apple.py:wide_reset``), reimplemented
    here so the scan can draw it without importing that script; a test pins the equality."""
    rng = np.random.default_rng(seed)
    apple = np.array(APPLE_CENTER) + rng.uniform(-APPLE_JITTER_M, APPLE_JITTER_M, 2)
    plate = np.array(PLATE_CENTER_V1) + rng.uniform(-PLATE_JITTER_V1_M, PLATE_JITTER_V1_M, 2)
    return {"object_xy": apple.tolist(), "plate_xy": plate.tolist()}


def setdown_reset(seed: int, box) -> dict:
    """(b2) The v1 apple draw, then a plate centre uniform in ``box`` = ((x0, x1), (y0, y1)),
    redrawn (same generator) until it is at least ``MIN_SEPARATION_M`` from the apple."""
    (x0, x1), (y0, y1) = box
    if not (x0 < x1 and y0 < y1):
        raise ValueError("plate box must be ordered")
    rng = np.random.default_rng(seed)
    apple = np.array(APPLE_CENTER) + rng.uniform(-APPLE_JITTER_M, APPLE_JITTER_M, 2)
    rng.uniform(-PLATE_JITTER_V1_M, PLATE_JITTER_V1_M, 2)  # keep the v1 stream aligned
    for _ in range(1000):
        plate = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        if np.linalg.norm(plate - apple) >= MIN_SEPARATION_M:
            return {"object_xy": apple.tolist(), "plate_xy": plate.tolist()}
    raise ValueError("no plate draw in the box clears the apple")
