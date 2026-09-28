"""TASK-069 feasibility-scan pieces (development only)."""

import numpy as np
import pytest

from embodied_jepa import first_policy as fp
from embodied_jepa import resting_expert as rx
from embodied_jepa import v2_feasibility as v2


def test_development_seeds_are_declared_and_disjoint_from_task068():
    assert v2.DEV_SEEDS == tuple(range(50100, 50200))
    v2.check_seeds(v2.DEV_SEEDS)
    with pytest.raises(fp.GuardError):
        v2.check_seeds((50099,))  # TASK-068's development range
    assert not set(v2.DEV_SEEDS) & set(rx.DEV_SEEDS)


def test_setdown_reset_keeps_the_v1_apple_and_puts_the_plate_in_the_box():
    box = ((0.30, 0.37), (-0.25, -0.04))
    for seed in v2.DEV_SEEDS:
        v1 = v2.v1_reset(seed)
        r = v2.setdown_reset(seed, box)
        assert r["object_xy"] == v1["object_xy"]  # same apple draw as v1
        x, y = r["plate_xy"]
        assert box[0][0] <= x <= box[0][1] and box[1][0] <= y <= box[1][1]
        sep = np.linalg.norm(np.subtract(r["plate_xy"], r["object_xy"]))
        assert sep >= v2.MIN_SEPARATION_M
    assert v2.setdown_reset(50100, box) == v2.setdown_reset(50100, box)


def test_setdown_reset_refuses_bad_boxes():
    with pytest.raises(ValueError, match="ordered"):
        v2.setdown_reset(50100, ((0.37, 0.30), (-0.25, -0.06)))
    with pytest.raises(ValueError, match="clears"):
        v2.setdown_reset(50100, ((0.34, 0.341), (-0.18, -0.179)))  # on top of the apple


def test_apply_apple_friction_changes_only_the_apple_geom():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
        <geom name="floor" type="plane" size="1 1 .1" friction="1 .01 .001"/>
        <body><freejoint/><geom name="apple_geom" type="sphere" size=".027"
          friction="1 .01 .001"/></body>
        </worldbody></mujoco>"""
    )
    change = v2.apply_apple_friction(model, condim=6, torsional=0.02, rolling=0.003)
    assert change["before"] == {"condim": 3, "friction": pytest.approx([1.0, 0.01, 0.001])}
    assert change["after"]["condim"] == 6
    assert change["after"]["friction"] == pytest.approx([1.0, 0.02, 0.003])
    floor = model.geom("floor").id
    assert model.geom_condim[floor] == 3
    with pytest.raises(ValueError):
        v2.apply_apple_friction(model, condim=5)
    with pytest.raises(ValueError):
        v2.apply_apple_friction(model, rolling=-1.0)
