"""Opt-in plate colour for the ``apple-to-plate-v2`` scene (development; owner request 2026-10-01).

The v1 scene (``simulation._scene``) draws the plate, for ``container_kind="plate"``, as one
cylinder (``plate_base``, radius 7 cm) and 16 rim capsules (``plate_rim_*``), all with
``rgba=".15 .35 .85 1"`` (blue). ``apple_to_plate_v2.apply_v2_scene`` changes only the apple's
contact. This module recolours the plate of an already compiled scene in place and changes
nothing else: ``geom_rgba`` is a visual property, so physics, geometry, seeds and trajectories
are untouched; only rendered pixels change.

Why a separate module: ``simulation.py`` and ``apple_to_plate_v2.py`` are sha256-pinned by
frozen manifests (TASK-070 onwards), so the parameter lives here and is applied at run time,
the way ``apply_v2_scene`` is. The default (``rgba=None``) is a no-op: it leaves the compiled
model, and therefore every render and hash, exactly as today.

``WHITE_PLATE_RGBA`` is an off-white like a ceramic plate. It is a development variant used by
``docs/experiments/apple_white_plate_dev.md``; it is not part of any benchmark.

NumPy only at import.
"""

from __future__ import annotations

import numpy as np

PLATE_BODY = "plate"
V2_PLATE_RGBA = (0.15, 0.35, 0.85, 1.0)  # simulation._scene: color["plate"] = ".15 .35 .85 1"
WHITE_PLATE_RGBA = (0.92, 0.92, 0.90, 1.0)  # development variant: off-white, ceramic


def plate_geoms(model) -> list[int]:
    """The geom ids of the ``plate`` body (its base and its rim capsules)."""
    body = model.body(PLATE_BODY).id
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body]


def apply_plate_rgba(model, rgba=None) -> dict:
    """Recolour the plate of a compiled v1/v2 scene in place; ``rgba=None`` changes nothing.

    Refuses a scene whose plate is not the v1/v2 blue (so it is never applied twice, nor to the
    bowl or target variants). Returns a record of what was applied."""
    geoms = plate_geoms(model)
    if not geoms:
        raise ValueError("the scene has no plate geoms")
    current = np.asarray(model.geom_rgba[geoms], np.float64)
    if not np.allclose(current, np.asarray(V2_PLATE_RGBA)[None], atol=1e-6):
        raise ValueError(f"not the v2 plate colour: {current.tolist()}")
    record = {"plate_geoms": len(geoms), "default_rgba": list(V2_PLATE_RGBA)}
    if rgba is None:
        return record | {"plate_rgba": list(V2_PLATE_RGBA), "changed": False}
    value = np.asarray(rgba, np.float64).reshape(-1)
    if value.shape != (4,) or not np.all((value >= 0.0) & (value <= 1.0)):
        raise ValueError(f"rgba must be four values in [0, 1], got {rgba!r}")
    model.geom_rgba[geoms] = value[None]
    return record | {"plate_rgba": [float(v) for v in value], "changed": True}
