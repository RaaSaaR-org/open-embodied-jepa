"""``apple-to-plate-v2``: the v1 Apple→Plate task with the apple's contact at condim 6.

Owner ruling R12 (TASK-070; verbatim in ``docs/experiments/apple_to_plate_v2_expert.md``):
v2 = v1 + apple contact condim 6, and nothing else. The friction values are **not** chosen from
any scan: they are the v1 scene's own declared apple friction ``1 .01 .001`` (sliding 1,
torsional 0.01 m, rolling 0.001 m; ``simulation._scene``), which v1 leaves inactive because
MuJoCo's default condim is 3. v2 makes them act. The robot, the fixed pelvis, the action schema
``ee_delta_grasp_v0``, the plate geometry and distribution, the 4 cm radius and the look are v1's.

The v1 code stays untouched (``simulation.py``, ``task.py``, ``scripted.py``). v2 is applied at
run time to one simulator instance's compiled model by ``apply_v2_scene``, which first checks
that the model is the v1 scene it expects. Success in v2 is ``at_rest.apple_at_rest``
(``apple_at_rest_v0``); the latched v1 scorer is reported beside it.

NumPy only at import. Privileged scripted engineering in this module is not a learned result.
"""

from __future__ import annotations

import numpy as np

from embodied_jepa import resting_expert as rx

SCENE_VERSION = "apple_to_plate_v2"
APPLE_GEOM = "apple_geom"
V1_APPLE_CONDIM = 3
V1_APPLE_FRICTION = (1.0, 0.01, 0.001)  # simulation._scene: friction="1 .01 .001"
V2_APPLE_CONDIM = 6
V2_APPLE_FRICTION = V1_APPLE_FRICTION  # R12 (i): the scene's own declared values

# ----- TASK-070 seeds ---------------------------------------------------------------------------
# Declared on the TASK-070 card before any TASK-070 episode. Disjoint from TASK-068 (50000-50099)
# and TASK-069 (50100-50199) development ranges and every earlier declared range.
DEV_SEEDS = tuple(range(50200, 50300))
DEV_DIRECTION_SEED = 6850
TASK069_DEV_RANGE = (50100, 50199)


def check_seeds(seeds, *, others=()) -> None:
    rx.check_seeds(seeds, others=(rx.DEV_SEEDS, range(50100, 50200), *others))


# ----- the frozen gate (TASK-070 preregistration; docs/experiments/apple_to_plate_v2_expert.md) ---
# The expert: TASK-070 development design e9 (RestingPlaceExpert keyword arguments).
GATE_EXPERT = {"release_pitch_rad": 0.45, "release_dx": 0.015}
GATE_SEEDS = tuple(range(50600, 50632))  # fresh; never simulated before the gated run
GATE_DIRECTION_SEED = 6860
GATE_LEVELS_CM = (0.0, 1.0, 1.5)  # 1.5 cm is reported only
GATE_BAR = {"0.0": 28, "1.0": 28}  # at rest (apple_at_rest_v0), of 32, at each gated level


def check_gate_seeds() -> None:
    """The gated seeds avoid every declared range, including all TASK-068/069/070 dev seeds."""
    check_seeds(GATE_SEEDS, others=(DEV_SEEDS,))
    if len(GATE_SEEDS) != 32:
        raise ValueError("the gate uses 32 seeds")


def gate_row(cells: dict) -> str:
    """VOID if any attempt raised; else PASS iff at rest >= 28/32 at 0 and at 1.0 cm; else FAIL.

    A guard stop is a completed-but-failed attempt (not at rest), not a void."""
    if any(cells[level]["errors"] for level in cells):
        return "VOID"
    for level, bar in GATE_BAR.items():
        if cells[level]["attempts"] != 32:
            return "VOID"
        if cells[level]["at_rest"] < bar:
            return "FAIL"
    return "PASS"


def apply_v2_scene(model) -> dict:
    """Turn a compiled v1 scene into v2 in place. Refuses anything that is not the v1 apple."""
    geom = model.geom(APPLE_GEOM).id
    condim = int(model.geom_condim[geom])
    friction = np.asarray(model.geom_friction[geom], float)
    if condim != V1_APPLE_CONDIM or not np.allclose(friction, V1_APPLE_FRICTION):
        raise ValueError(
            f"not the v1 apple contact (condim {condim}, friction {friction.tolist()})"
        )
    model.geom_condim[geom] = V2_APPLE_CONDIM
    model.geom_friction[geom] = V2_APPLE_FRICTION
    return {
        "scene_version": SCENE_VERSION,
        "apple_condim": int(model.geom_condim[geom]),
        "apple_friction": np.asarray(model.geom_friction[geom], float).tolist(),
    }


def make_v2_robot():
    """The first-policy runtime's robot (112 px onboard, look-ready), switched to v2."""
    from embodied_jepa import first_policy_runtime as rt

    robot = rt.make_robot()
    record = apply_v2_scene(robot.model)
    return robot, record
