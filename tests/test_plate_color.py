"""Opt-in plate colour for the v2 scene (development): the default changes nothing."""

import hashlib
import os

import numpy as np
import pytest

from embodied_jepa import plate_color as pc
from embodied_jepa.simulation import ROOT

ASSETS = ROOT / "third_party/unitree_mujoco/unitree_robots/g1/g1_29dof_with_hand.xml"
assets = pytest.mark.skipif(not ASSETS.exists(), reason="pinned robot assets have not been fetched")
render = pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")
BLUE = ".15 .35 .85 1"


def _model(color=BLUE):
    mujoco = pytest.importorskip("mujoco")
    return mujoco.MjModel.from_xml_string(
        f"""<mujoco><worldbody>
        <geom name="table" type="box" size=".3 .3 .04" rgba=".55 .4 .25 1"/>
        <body name="plate" pos="0 0 .05">
          <geom name="plate_base" type="cylinder" size=".07 .006" rgba="{color}"/>
          <geom name="plate_rim_0" type="capsule" fromto="0 .067 .009 .067 0 .009" size=".004"
            rgba="{color}"/>
        </body>
        <body name="apple" pos="0 .2 .1"><freejoint/>
          <geom name="apple_geom" type="sphere" size=".027" rgba=".85 .07 .04 1"/></body>
        </worldbody></mujoco>"""
    )


def _arrays(model) -> dict:
    return {
        name: np.array(getattr(model, name))
        for name in dir(model)
        if not name.startswith("_") and isinstance(getattr(model, name, None), np.ndarray)
    }


def test_the_declared_default_is_the_scene_source_colour():
    source = (ROOT / "src/embodied_jepa/simulation.py").read_text()
    assert f'color = {{"plate": "{BLUE}"' in source
    assert pc.V2_PLATE_RGBA == tuple(float(v) for v in BLUE.split())
    assert pc.WHITE_PLATE_RGBA == (0.92, 0.92, 0.90, 1.0)


def test_default_is_a_no_op_on_every_model_array():
    model, fresh = _model(), _model()
    record = pc.apply_plate_rgba(model)
    assert record["changed"] is False and record["plate_geoms"] == 2
    before, after = _arrays(fresh), _arrays(model)
    assert before.keys() == after.keys()
    for name in before:
        assert np.array_equal(before[name], after[name]), name


def test_white_changes_only_the_plate_geoms_rgba():
    model, fresh = _model(), _model()
    record = pc.apply_plate_rgba(model, pc.WHITE_PLATE_RGBA)
    assert record["changed"] is True
    plate = pc.plate_geoms(model)
    np.testing.assert_allclose(model.geom_rgba[plate], [pc.WHITE_PLATE_RGBA] * 2, atol=1e-7)
    others = [g for g in range(model.ngeom) if g not in plate]
    assert np.array_equal(model.geom_rgba[others], fresh.geom_rgba[others])
    before, after = _arrays(fresh), _arrays(model)
    assert [k for k in before if not np.array_equal(before[k], after[k])] == ["geom_rgba"]


def test_refuses_a_non_v2_plate_and_bad_values():
    with pytest.raises(ValueError, match="not the v2 plate colour"):
        pc.apply_plate_rgba(_model(color=".7 .2 .7 1"))  # the bowl's colour
    model = _model()
    pc.apply_plate_rgba(model, pc.WHITE_PLATE_RGBA)
    with pytest.raises(ValueError, match="not the v2 plate colour"):
        pc.apply_plate_rgba(model, pc.WHITE_PLATE_RGBA)  # never twice
    for bad in ((1, 1, 1), (1.2, 1, 1, 1), (-0.1, 0, 0, 1)):
        with pytest.raises(ValueError, match="rgba"):
            pc.apply_plate_rgba(_model(), bad)


def _v2_sim(*, render_on: bool):
    from embodied_jepa import apple_to_plate_v2 as v2
    from embodied_jepa.simulation import MuJoCoSimulation

    sim = MuJoCoSimulation(
        object_kind="apple", container_kind="plate", width=112, height=112, render=render_on
    )
    v2.apply_v2_scene(sim.model)
    return sim


def _rollout(sim, steps=60) -> np.ndarray:
    sim.reset(7, object_xy=[0.40, -0.26], plate_xy=[0.48, -0.10])
    rng = np.random.default_rng(3)
    lo, hi = sim.model.jnt_range[sim.joint_ids].T
    qs = []
    for _ in range(steps):
        target = np.clip(sim.targets + rng.normal(0, 0.02, sim.targets.shape), lo, hi)
        sim.send_joint_targets(target, joint_names=sim.joint_names, deadline=sim.data.time + 1)
        qs.append(sim.data.qpos.copy())
    return np.asarray(qs)


@assets
def test_v2_scene_physics_are_identical_with_a_white_plate():
    pytest.importorskip("mujoco")
    blue, white = _v2_sim(render_on=False), _v2_sim(render_on=False)
    assert pc.apply_plate_rgba(blue.model)["changed"] is False
    pc.apply_plate_rgba(white.model, pc.WHITE_PLATE_RGBA)
    assert len(pc.plate_geoms(white.model)) == 17  # the base and 16 rim capsules
    assert np.array_equal(_rollout(blue), _rollout(white))
    blue.close()
    white.close()


@assets
@render
def test_v2_render_is_byte_identical_by_default_and_differs_when_white():
    pytest.importorskip("mujoco")

    def digest(sim):
        sim.reset(7, object_xy=[0.40, -0.26], plate_xy=[0.48, -0.10])
        return {
            camera: hashlib.sha256(sim.render(camera).tobytes()).hexdigest()
            for camera in ("onboard_rgb", "overview")
        }

    reference = _v2_sim(render_on=True)
    default = _v2_sim(render_on=True)
    pc.apply_plate_rgba(default.model)
    white = _v2_sim(render_on=True)
    pc.apply_plate_rgba(white.model, pc.WHITE_PLATE_RGBA)
    want = digest(reference)
    assert digest(default) == want
    got = digest(white)
    assert all(got[c] != want[c] for c in want)
    for sim in (reference, default, white):
        sim.close()
