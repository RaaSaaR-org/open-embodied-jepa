"""apple-to-plate-v2 scene option (TASK-070, owner ruling R12)."""

import pytest

from embodied_jepa import apple_to_plate_v2 as v2
from embodied_jepa import first_policy as fp


def _model(condim=3, friction="1 .01 .001"):
    mujoco = pytest.importorskip("mujoco")
    return mujoco.MjModel.from_xml_string(
        f"""<mujoco><worldbody>
        <geom name="plate_base" type="cylinder" size=".07 .006" friction="1 .01 .001"/>
        <body pos="0 0 .1"><freejoint/><geom name="apple_geom" type="sphere" size=".027"
          condim="{condim}" friction="{friction}"/></body>
        </worldbody></mujoco>"""
    )


def test_v2_is_v1_plus_condim_6_with_the_scene_declared_friction():
    assert v2.V2_APPLE_CONDIM == 6
    assert v2.V2_APPLE_FRICTION == v2.V1_APPLE_FRICTION == (1.0, 0.01, 0.001)
    model = _model()
    record = v2.apply_v2_scene(model)
    assert record == {
        "scene_version": "apple_to_plate_v2",
        "apple_condim": 6,
        "apple_friction": pytest.approx([1.0, 0.01, 0.001]),
    }
    assert model.geom_condim[model.geom("plate_base").id] == 3  # only the apple changes


def test_apply_v2_scene_refuses_a_non_v1_apple():
    with pytest.raises(ValueError, match="not the v1 apple"):
        v2.apply_v2_scene(_model(condim=6))  # already changed
    with pytest.raises(ValueError, match="not the v1 apple"):
        v2.apply_v2_scene(_model(friction="1 .02 .003"))


def test_v1_scene_source_still_declares_the_values_v2_relies_on():
    from pathlib import Path

    source = (Path(v2.__file__).parent / "simulation.py").read_text()
    assert 'name="apple_geom", mass=".08", friction="1 .01 .001"' in source
    assert "condim" not in source  # v1 leaves condim at MuJoCo's default 3


def test_task070_development_seeds_are_disjoint():
    assert v2.DEV_SEEDS == tuple(range(50200, 50300))
    v2.check_seeds(v2.DEV_SEEDS)
    for seed in (50099, 50150, 46850, 47050):
        with pytest.raises(fp.GuardError):
            v2.check_seeds((seed,))
