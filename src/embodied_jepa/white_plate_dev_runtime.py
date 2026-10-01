"""Development harness worker: TASK-075's multi-view re-render with an opt-in plate colour.

Development only (``docs/experiments/apple_white_plate_dev.md``); not a gated run and not a
preregistered gate. The worker is TASK-075's (``obs_ceiling_v2_runtime``), unmodified: after its
initializer has built the v2 robot, ``plate_color.apply_plate_rgba`` recolours the plate of that
robot's compiled model (``config["plate_rgba"]``; ``None`` keeps the scene's blue). A ``views``
task runs TASK-075's ``run_views_task`` as is; this module only keeps the 112 px onboard frames
that the task re-renders for its own check against the sealed root (TASK-075 reads the reference
view from the sealed corpus, which is blue) and writes them beside the views file.

NumPy at import; MuJoCo and torch are imported by the worker initializer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from embodied_jepa import obs_ceiling_v2_runtime as ort
from embodied_jepa import plate_color as pc
from embodied_jepa import wm_critic_v2_runtime as rtm

_D: dict = {}


def worker_init(config: dict) -> None:
    ort.worker_init(config)
    _D.clear()
    _D["plate"] = pc.apply_plate_rgba(rtm._W["robot"].model, config.get("plate_rgba"))


def ref_path(folder, episode_id: str) -> Path:
    return Path(folder) / f"{episode_id}.onboard112.npz"


def run_views_task(task: dict) -> dict:
    kept: dict = {}
    original = ort.frame_check

    def keep(ours, stored):
        kept["onboard112"] = np.asarray(ours, np.uint8).copy()
        return original(ours, stored)

    ort.frame_check = keep
    try:
        out = ort.run_views_task(task)
    finally:
        ort.frame_check = original
    path = ref_path(task["folder"], task["episode_id"])
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    np.savez_compressed(path, onboard112=kept["onboard112"])
    out["onboard112_npz_sha256"] = ort.sha256_file(path)
    out["onboard112_bytes"] = int(path.stat().st_size)
    out["plate"] = dict(_D["plate"])
    return out


def run_task(task: dict) -> dict:
    if task["kind"] == "views":
        return run_views_task(task)
    return ort.run_task(task)
